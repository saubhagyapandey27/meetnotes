"""
Step 1: Transcribe AMI + ICSI meeting audio datasets using Moonshine-Tiny on Intel Arc GPU.

Uses ONNX Runtime DirectML (via onnxruntime-directml) for GPU-accelerated inference.
Inference is done ONE chunk at a time per file (no batching = no padding issues).
Chunks are 30 seconds long with 2-second overlap at boundaries to avoid losing words.
Overlap deduplication: the leading text of each non-first chunk that is already present
in the previous chunk's tail is stripped before appending.
Each file is saved immediately after transcription to support resume on interruption.
"""

# ============================================================
# CRITICAL: Monkey-patch importlib.metadata BEFORE any imports
# from optimum/onnxruntime so that it recognises onnxruntime-directml.
# ============================================================
import importlib.metadata as _importlib_metadata

_orig_version = _importlib_metadata.version

def _patched_version(pkg_name):
    if pkg_name == "onnxruntime":
        return _orig_version("onnxruntime-directml")
    return _orig_version(pkg_name)

_importlib_metadata.version = _patched_version

try:
    import importlib_metadata as _importlib_metadata2
    _orig_version2 = _importlib_metadata2.version
    def _patched_version2(pkg_name):
        if pkg_name == "onnxruntime":
            return _orig_version2("onnxruntime-directml")
        return _orig_version2(pkg_name)
    _importlib_metadata2.version = _patched_version2
except ImportError:
    pass

# ============================================================
# Standard imports
# ============================================================
import os
import sys
import json
import glob
import time
import wave
import numpy as np

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor

# ===========================================================
# Config
# ===========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AMI_AUDIO_PATTERN  = os.path.join(BASE_DIR, "AMI Dataset Audios", "amicorpus", "*", "audio", "*.Mix-Headset.wav")
ICSI_AUDIO_PATTERN = os.path.join(BASE_DIR, "ICSI Dataset Audios", "Signals", "*", "*.interaction.wav")
OUTPUT_FILE = os.path.join(BASE_DIR, "data_prep", "output", "transcripts.json")
ERROR_LOG   = os.path.join(BASE_DIR, "data_prep", "output", "transcription_errors.log")

MODEL_ID        = "UsefulSensors/moonshine-tiny"
ONNX_MODEL_DIR  = os.path.join(BASE_DIR, "models", "moonshine_tiny_onnx")
CHUNK_SEC  = 30          # seconds per inference window
OVERLAP_SEC = 2          # seconds of overlap between consecutive chunks
TARGET_SR  = 16000       # Moonshine-Tiny expects 16 kHz mono


def log_error(msg: str):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def load_and_resample_wav(wav_path: str) -> np.ndarray:
    """Load a WAV, convert to mono float32, resample to 16 kHz."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    with wave.open(wav_path, "rb") as wf:
        channels     = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate  = wf.getframerate()
        num_frames   = wf.getnframes()
        if num_frames == 0:
            raise ValueError(f"Empty WAV: {wav_path}")
        raw = wf.readframes(num_frames)

    # Raw bytes → float32 in [-1, 1]
    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2_147_483_648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Stereo → mono
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    elif channels > 2:
        audio = audio.reshape(-1, channels)[:, 0]

    # Resample to 16 kHz if needed
    if sample_rate != TARGET_SR:
        target_len = int(len(audio) * TARGET_SR / sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        )

    return audio.astype(np.float32)


def deduplicate_overlap(prev_text: str, new_text: str, overlap_words: int = 8) -> str:
    """
    Remove words from the start of new_text that already appear at the end of prev_text.
    Compares up to `overlap_words` words to find a matching suffix/prefix pair.
    Returns new_text with any duplicated prefix stripped.
    """
    if not prev_text or not new_text:
        return new_text

    prev_words = prev_text.split()
    new_words  = new_text.split()

    # Try decreasing window sizes to find the longest matching overlap
    for window in range(min(overlap_words, len(prev_words), len(new_words)), 0, -1):
        if prev_words[-window:] == new_words[:window]:
            return " ".join(new_words[window:])

    return new_text


def transcribe_file_gpu(audio: np.ndarray, model, processor) -> str:
    """
    Transcribe a full meeting audio by calling model.generate() once per 30-second
    chunk with a 2-second look-back overlap. No batching, no padding issues.
    Overlapping text is deduplicated word-by-word before joining.
    """
    chunk_samples   = CHUNK_SEC  * TARGET_SR
    overlap_samples = OVERLAP_SEC * TARGET_SR
    step_samples    = chunk_samples - overlap_samples  # advance this many samples each iteration

    total     = len(audio)
    all_parts = []
    prev_text = ""

    chunk_idx = 0
    start     = 0

    while start < total:
        end   = min(start + chunk_samples, total)
        chunk = audio[start:end]

        inputs        = processor(chunk, return_tensors="pt", sampling_rate=TARGET_SR)
        generated_ids = model.generate(**inputs)
        raw_text      = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if chunk_idx == 0:
            # First chunk — keep everything
            clean_text = raw_text
        else:
            # Remove words already present at the tail of the previous transcript
            clean_text = deduplicate_overlap(prev_text, raw_text, overlap_words=10)

        if clean_text:
            all_parts.append(clean_text)
            prev_text = raw_text  # store raw (including overlap region) for next dedup

        chunk_idx += 1
        start     += step_samples   # move forward by (chunk - overlap) each step

    return " ".join(all_parts)


def load_model():
    """Load Moonshine-Tiny on Intel Arc GPU via ONNX Runtime DirectML.

    Uses the pre-built ONNX model at ONNX_MODEL_DIR if it exists (fast path).
    Falls back to exporting from HuggingFace and saving to that directory if not.
    """
    # Load processor — always from local dir if available, else HuggingFace
    proc_source = ONNX_MODEL_DIR if os.path.isdir(ONNX_MODEL_DIR) else MODEL_ID
    print(f"Loading processor from '{proc_source}'...")
    processor = AutoProcessor.from_pretrained(proc_source)

    if os.path.isdir(ONNX_MODEL_DIR) and any(
        f.endswith(".onnx") for f in os.listdir(ONNX_MODEL_DIR)
    ):
        # ── Fast path: load from pre-built local ONNX files ──────────────────
        print(f"Found pre-built ONNX model at: {ONNX_MODEL_DIR}")
        print("Loading model → ONNX Runtime DirectML (Intel Arc GPU)...")
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            ONNX_MODEL_DIR,
            provider="DmlExecutionProvider",
        )
    else:
        # ── First-run path: export PyTorch → ONNX, then save for future use ──
        print(f"No pre-built ONNX model found. Exporting from '{MODEL_ID}'...")
        print("(This only happens once — the result will be saved to disk.)")
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            export=True,
            provider="DmlExecutionProvider",
        )
        os.makedirs(ONNX_MODEL_DIR, exist_ok=True)
        model.save_pretrained(ONNX_MODEL_DIR)
        processor.save_pretrained(ONNX_MODEL_DIR)
        print(f"Saved ONNX model to: {ONNX_MODEL_DIR}")

    print("Model ready on GPU.\n")
    return model, processor


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 1. Discover audio files
    ami_files  = glob.glob(AMI_AUDIO_PATTERN)
    icsi_files = glob.glob(ICSI_AUDIO_PATTERN)
    print(f"Discovered {len(ami_files)} AMI meetings.")
    print(f"Discovered {len(icsi_files)} ICSI meetings.")

    all_meetings = []
    for path in ami_files:
        mid = os.path.basename(path).split(".")[0]
        all_meetings.append({"id": mid, "path": path, "source": "AMI"})
    for path in icsi_files:
        mid = os.path.basename(path).split(".")[0]
        all_meetings.append({"id": mid, "path": path, "source": "ICSI"})

    print(f"Total meetings to process: {len(all_meetings)}")
    if not all_meetings:
        print("ERROR: No audio files found. Check path config.")
        sys.exit(1)

    # 2. Load existing progress for resume
    transcripts: dict = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            transcripts = {item["meeting_id"]: item for item in data}
            print(f"Resuming — {len(transcripts)} meetings already transcribed.")
        except Exception as e:
            print(f"Warning: Could not load existing output ({e}). Starting fresh.")

    # 3. Load GPU model once
    model, processor = load_model()

    # 4. Transcription loop
    processed = 0
    skipped   = 0
    t_global  = time.time()

    for idx, meeting in enumerate(all_meetings):
        mid = meeting["id"]

        if mid in transcripts:
            skipped += 1
            continue

        print(f"[{idx+1}/{len(all_meetings)}] Transcribing {mid} ({meeting['source']})...")
        print(f"  Path: {meeting['path']}")

        try:
            t0 = time.time()
            audio = load_and_resample_wav(meeting["path"])

            duration_s = len(audio) / TARGET_SR
            step_s     = CHUNK_SEC - OVERLAP_SEC
            n_chunks   = int(np.ceil(max(0, len(audio) - OVERLAP_SEC * TARGET_SR) / (step_s * TARGET_SR))) + 1
            print(f"  Audio: {duration_s/60:.1f} min → ~{n_chunks} chunks ({CHUNK_SEC}s each, {OVERLAP_SEC}s overlap)")

            transcript = transcribe_file_gpu(audio, model, processor)
            t1 = time.time()

            transcripts[mid] = {
                "meeting_id":       mid,
                "source":           meeting["source"],
                "wav_path":         meeting["path"],
                "transcript":       transcript,
                "duration_seconds": round(t1 - t0, 2),
            }

            # Save immediately after every file
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(list(transcripts.values()), f, indent=2, ensure_ascii=False)

            processed += 1
            rtf = (t1 - t0) / duration_s if duration_s > 0 else 0
            print(f"  Done in {t1-t0:.1f}s  RTF={rtf:.2f}  chars={len(transcript)}\n")

        except Exception as e:
            err = f"Error transcribing {mid}: {e}"
            print(f"  [ERROR] {err}\n")
            log_error(err)

    # 5. Summary
    elapsed = time.time() - t_global
    print("=" * 45)
    print("Transcription complete!")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}  (already done)")
    print(f"  Errors    : {len(all_meetings) - processed - skipped}")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Output    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
