# ============================================================
#  GOOGLE COLAB SETUP — run these cells in order BEFORE this script
# ============================================================
#
# Cell 1 – Switch to T4 GPU runtime first:
#   Runtime -> Change runtime type -> T4 GPU -> Save
#
# Cell 2 – Verify GPU:
#   !nvidia-smi
#
# Cell 3 – Install ONLY what Colab doesn't already have.
#   !! DO NOT upgrade torch / torchvision / torchaudio !!
#   Colab pre-installs a matched set of torch+torchvision+CUDA.
#   Upgrading torch alone breaks torchvision and crashes transformers.
#
#   !pip install -q \
#       "transformers>=4.47.0" \
#       librosa \
#       soundfile \
#       tqdm
#
# Cell 4 – Upload your dataset folders to /content (Colab root):
#   Use the Files panel on the left sidebar to upload, OR unzip from Drive:
#   !unzip -q /content/drive/MyDrive/amicorpus.zip -d /content/
#   !unzip -q /content/drive/MyDrive/signals.zip   -d /content/
#
# Cell 5 – Upload your existing transcripts.json to /content (for resume):
#   Use the Files panel to upload it, or skip if starting fresh.
#
# Cell 6 – Upload this script to /content, then run:
#   !python /content/step1_colab_transcribe.py
#
# DATASET FOLDER STRUCTURE EXPECTED AT /content:
#   AMI:  /content/amicorpus/<meeting_id>/audio/<meeting_id>.Mix-Headset.wav
#   ICSI: /content/signals/<meeting_id>/<meeting_id>.interaction.wav
#
# OUTPUT (saved to /content):
#   /content/transcripts.json      — full transcripts, appended after each meeting
#   /content/transcription_errors.log — any errors
# ============================================================

import os
import glob
import json
import math
import time
import sys
import numpy as np

# Suppress transformers version-mismatch advisory noise
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
import librosa
from tqdm.auto import tqdm

# transformers imports moonshine via its speech models module.
# torchvision is imported inside transformers.image_utils;
# if the installed torchvision is incompatible with the current torch build
# it raises a RuntimeError.  We patch that here so the ASR-only import
# path succeeds even when torchvision is broken.
try:
    import torchvision          # noqa: F401
except Exception:
    import unittest.mock as _mock
    import sys as _sys
    _sys.modules["torchvision"] = _mock.MagicMock()
    _sys.modules["torchvision.io"] = _mock.MagicMock()

from transformers import AutoProcessor, MoonshineForConditionalGeneration

# ─────────────────────────────────────────────────────────────
#  PATHS  (no Drive — everything lives in /content)
# ─────────────────────────────────────────────────────────────
AMI_ROOT    = "/content/amicorpus"
ICSI_ROOT   = "/content/signals"
OUTPUT_FILE = "/content/transcripts.json"
ERROR_LOG   = "/content/transcription_errors.log"

# ─────────────────────────────────────────────────────────────
#  MODEL / CHUNKING CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_ID    = "UsefulSensors/moonshine-tiny"
TARGET_SR   = 16_000     # Moonshine requires 16 kHz mono
CHUNK_SEC   = 28         # seconds per inference window
OVERLAP_SEC = 2          # seconds of look-back overlap at chunk boundaries
STEP_SEC    = CHUNK_SEC - OVERLAP_SEC   # = 26 s advance per chunk

# T4 has 16 GB VRAM; moonshine-tiny is ~50 MB in float16.
# 32 chunks of 28 s each ≈ 1.4 GB input tensors — comfortably safe.
# Raise to 48 if you want a bit more throughput.
BATCH_SIZE  = 32

# ─────────────────────────────────────────────────────────────
#  DEVICE SETUP
# ─────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

print("=" * 60)
print("  Moonshine-Tiny Batch Transcriber — Google Colab T4")
print("=" * 60)
print(f"  Device  : {device}")
print(f"  Dtype   : {dtype}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {props.name}")
    print(f"  VRAM    : {props.total_memory / 1e9:.1f} GB")
else:
    print("  WARNING : No CUDA GPU found — running on CPU (very slow!)")
print("=" * 60)


def log_error(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


# ─────────────────────────────────────────────────────────────
#  LOAD MODEL  (downloaded once from HuggingFace Hub)
# ─────────────────────────────────────────────────────────────
print(f"\nDownloading / loading model: {MODEL_ID}")
t_load = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = (
    MoonshineForConditionalGeneration
    .from_pretrained(MODEL_ID, torch_dtype=dtype)
    .to(device)
    .eval()
)
print(f"Model ready in {time.time() - t_load:.1f}s\n")


# ─────────────────────────────────────────────────────────────
#  AUDIO LOADING
# ─────────────────────────────────────────────────────────────
def load_audio(path: str) -> np.ndarray:
    """Load any WAV/FLAC/MP3 → mono float32 resampled to TARGET_SR."""
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  CHUNKING WITH OVERLAP
# ─────────────────────────────────────────────────────────────
def make_chunks(audio: np.ndarray):
    """
    Yield (chunk_array, chunk_index) for every overlapping window.
    Chunks: 0-28s, 26-54s, 52-80s, ... (2-second look-back overlap).
    """
    chunk_len = int(CHUNK_SEC * TARGET_SR)
    step_len  = int(STEP_SEC  * TARGET_SR)
    total     = len(audio)
    idx, start = 0, 0
    while start < total:
        yield audio[start : start + chunk_len], idx
        start += step_len
        idx   += 1


# ─────────────────────────────────────────────────────────────
#  OVERLAP DEDUPLICATION
# ─────────────────────────────────────────────────────────────
def strip_overlap(prev: str, current: str, window: int = 10) -> str:
    """
    Strip words from the start of `current` that already appear
    at the end of `prev` (caused by the 2-second look-back overlap).
    """
    if not prev or not current:
        return current
    pw = prev.split()
    cw = current.split()
    for n in range(min(window, len(pw), len(cw)), 0, -1):
        if pw[-n:] == cw[:n]:
            return " ".join(cw[n:])
    return current


# ─────────────────────────────────────────────────────────────
#  BATCH GPU INFERENCE
# ─────────────────────────────────────────────────────────────
def transcribe_batch(chunk_arrays: list) -> list:
    """
    Forward-pass a list of float32 audio arrays through the model in one call.
    padding=True handles variable-length chunks safely (Moonshine uses RoPE,
    so padding tokens do not corrupt positional encodings).
    Returns a list of decoded strings.
    """
    inputs = processor(
        chunk_arrays,
        return_tensors="pt",
        sampling_rate=TARGET_SR,
        padding=True,
    )
    input_values = inputs["input_values"].to(device=device, dtype=dtype)

    with torch.no_grad():
        max_new = int(CHUNK_SEC * 6)   # upper-bound: ~6 tokens/second
        generated = model.generate(input_values, max_new_tokens=max_new)

    return [t.strip() for t in processor.batch_decode(generated, skip_special_tokens=True)]


# ─────────────────────────────────────────────────────────────
#  FULL MEETING TRANSCRIPTION
# ─────────────────────────────────────────────────────────────
def transcribe_meeting(audio: np.ndarray, meeting_id: str) -> str:
    """
    1. Slice audio into overlapping 28-second chunks.
    2. Process chunks in GPU batches of BATCH_SIZE.
    3. Strip duplicated boundary words.
    4. Return joined transcript string.
    """
    all_chunks = list(make_chunks(audio))
    n_chunks   = len(all_chunks)
    results    = [""] * n_chunks

    n_batches = math.ceil(n_chunks / BATCH_SIZE)

    # Inner progress bar per meeting (no slowdown — runs between GPU calls)
    batch_bar = tqdm(
        range(0, n_chunks, BATCH_SIZE),
        desc=f"  batches",
        total=n_batches,
        leave=False,
        unit="batch",
        file=sys.stdout,
        dynamic_ncols=True,
    )

    for batch_start in batch_bar:
        batch    = all_chunks[batch_start : batch_start + BATCH_SIZE]
        arrays   = [c[0] for c in batch]
        t_batch  = time.time()
        texts    = transcribe_batch(arrays)
        elapsed  = time.time() - t_batch
        audio_s  = len(arrays) * CHUNK_SEC
        batch_bar.set_postfix(
            chunks=f"{len(arrays)}",
            audio_s=f"{audio_s}s",
            wall_s=f"{elapsed:.1f}s",
            rtf=f"{elapsed/audio_s:.3f}",
        )
        for (_, chunk_idx), text in zip(batch, texts):
            results[chunk_idx] = text

    # Deduplicate overlap and join
    parts, prev = [], ""
    for i, text in enumerate(results):
        clean = text if i == 0 else strip_overlap(prev, text)
        if clean:
            parts.append(clean)
        prev = text

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────
#  DISCOVER MEETINGS
# ─────────────────────────────────────────────────────────────
def discover_meetings():
    meetings = []

    # AMI:  /content/amicorpus/<id>/audio/<id>.Mix-Headset.wav
    for path in sorted(glob.glob(os.path.join(AMI_ROOT, "*", "audio", "*.Mix-Headset.wav"))):
        mid = os.path.basename(path).split(".")[0]
        meetings.append({"id": mid, "path": path, "source": "AMI"})

    # ICSI: /content/signals/<id>/<id>.interaction.wav
    for path in sorted(glob.glob(os.path.join(ICSI_ROOT, "*", "*.interaction.wav"))):
        mid = os.path.basename(path).split(".")[0]
        meetings.append({"id": mid, "path": path, "source": "ICSI"})

    return meetings


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    meetings = discover_meetings()
    n_ami  = sum(1 for m in meetings if m["source"] == "AMI")
    n_icsi = sum(1 for m in meetings if m["source"] == "ICSI")
    print(f"Discovered  : {len(meetings)} meetings  (AMI: {n_ami}  ICSI: {n_icsi})")

    if not meetings:
        print("ERROR: No audio files found.")
        print(f"  Expected AMI  at: {AMI_ROOT}")
        print(f"  Expected ICSI at: {ICSI_ROOT}")
        return

    # ── Resume: load existing transcripts.json ────────────────
    done = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, encoding="utf-8") as f:
                data = json.load(f)
            done = {item["meeting_id"]: item for item in data}
            print(f"Resuming    : {len(done)} meetings already done (loaded from {OUTPUT_FILE})")
        except Exception as e:
            print(f"Warning: could not parse existing output ({e}). Starting fresh.")

    todo = [m for m in meetings if m["id"] not in done]
    print(f"To process  : {len(todo)} meetings")
    print(f"Output file : {OUTPUT_FILE}\n")

    if not todo:
        print("Nothing left to do — all meetings already transcribed!")
        return

    # ── Progress tracking (outer bar, no GPU overhead) ────────
    meeting_bar = tqdm(
        todo,
        desc="Meetings",
        unit="mtg",
        file=sys.stdout,
        dynamic_ncols=True,
    )

    processed  = 0
    errors     = 0
    t_global   = time.time()
    total_audio_s = 0.0
    total_wall_s  = 0.0

    for m in meeting_bar:
        mid, path = m["id"], m["path"]

        # Update outer bar description with current meeting
        meeting_bar.set_description(f"[{m['source']}] {mid}")

        print(f"\n{'─'*55}")
        print(f"  [{processed+1+errors}/{len(todo)}] {mid}  ({m['source']})")
        print(f"  Path: {path}")

        try:
            t0    = time.time()
            audio = load_audio(path)
            dur_s = len(audio) / TARGET_SR
            n_ch  = math.ceil(max(1, len(audio) - OVERLAP_SEC * TARGET_SR) / (STEP_SEC * TARGET_SR))
            n_bat = math.ceil(n_ch / BATCH_SIZE)
            print(f"  Audio : {dur_s/60:.1f} min  →  {n_ch} chunks  →  {n_bat} batch(es) of ≤{BATCH_SIZE}")

            transcript = transcribe_meeting(audio, mid)
            t1 = time.time()

            wall_s = t1 - t0
            rtf    = wall_s / dur_s if dur_s > 0 else 0
            total_audio_s += dur_s
            total_wall_s  += wall_s

            print(f"  Done  : {wall_s:.1f}s wall  RTF={rtf:.3f}  {len(transcript)} chars")
            if total_wall_s > 0:
                avg_rtf = total_wall_s / total_audio_s if total_audio_s > 0 else 0
                remaining_audio = sum(
                    60 * 45  # rough estimate 45 min/meeting for remaining
                    for mm in todo
                    if mm["id"] not in done and mm["id"] != mid
                )
                print(f"  Avg RTF so far: {avg_rtf:.3f}  (lower is faster)")

            done[mid] = {
                "meeting_id":       mid,
                "source":           m["source"],
                "wav_path":         path,
                "transcript":       transcript,
                "duration_seconds": round(wall_s, 2),
            }

            # Save after EVERY meeting — resume survives session timeout
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(list(done.values()), f, indent=2, ensure_ascii=False)

            processed += 1
            meeting_bar.set_postfix(done=processed, errors=errors, rtf=f"{rtf:.3f}")

        except Exception as e:
            errors += 1
            msg = f"[ERROR] {mid}: {e}"
            print(f"  ✗ {msg}")
            log_error(msg)
            meeting_bar.set_postfix(done=processed, errors=errors)

    # ── Final summary ─────────────────────────────────────────
    elapsed = time.time() - t_global
    print(f"\n{'='*55}")
    print(f"  Session complete!")
    print(f"  Processed   : {processed}")
    print(f"  Errors      : {errors}")
    print(f"  Audio total : {total_audio_s/3600:.2f} hours")
    print(f"  Wall time   : {elapsed/3600:.2f} hours")
    if total_audio_s > 0 and elapsed > 0:
        print(f"  Avg RTF     : {elapsed/total_audio_s:.3f}  (real-time factor)")
        print(f"  Throughput  : {total_audio_s/elapsed:.1f}x faster than real-time")
    print(f"  Output      : {OUTPUT_FILE}")
    if errors:
        print(f"  Error log   : {ERROR_LOG}")
    print("=" * 55)


if __name__ == "__main__":
    main()
