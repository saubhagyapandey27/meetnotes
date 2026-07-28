import os
import wave
import numpy as np

# Try importing moonshine_voice, providing a helpful error if missing
try:
    from moonshine_voice import Transcriber, TranscriptEventListener, get_model_for_language
except ImportError:
    # Fallback placeholders for testing/compilation
    Transcriber = None
    TranscriptEventListener = object
    get_model_for_language = None

class MoonshineASR:
    """
    ASR Wrapper for moonshine-tiny model using moonshine-voice.
    Loads the model once and transcribes local WAV audio files.
    """
    def __init__(self):
        if get_model_for_language is None or Transcriber is None:
            raise ImportError(
                "moonshine-voice package is not installed. Please run: pip install moonshine-voice"
            )
        
        from moonshine_voice import ModelArch
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        model_path = os.path.abspath(os.path.join(models_dir, "moonshine_tiny_ort"))
        self.transcriber = Transcriber(model_path=model_path, model_arch=ModelArch.TINY)

    def load_and_resample_wav(self, wav_path: str) -> tuple[np.ndarray, int]:
        """
        Loads a WAV file, converts it to float32 mono, and resamples to 16kHz using NumPy interpolation.
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file not found at: {wav_path}")
            
        with wave.open(wav_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            
            if num_frames == 0:
                raise ValueError(f"WAV file {wav_path} is empty (0 frames)")
                
            raw_data = wf.readframes(num_frames)
            
            # Map based on sample width
            if sample_width == 2:
                audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 1:
                audio_data = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sample_width == 4:
                audio_data = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
                
            # Mono conversion
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            elif channels > 2:
                # Keep first channel only
                audio_data = audio_data.reshape(-1, channels)[:, 0]
                
            # Resampling to 16000 Hz if needed
            target_rate = 16000
            if sample_rate != target_rate:
                num_samples = int(len(audio_data) * target_rate / sample_rate)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data) - 1, num_samples),
                    np.arange(len(audio_data)),
                    audio_data
                )
                sample_rate = target_rate
                
            return audio_data, sample_rate

    def transcribe(self, wav_path: str, progress_callback=None) -> str:
        """
        Transcribes a WAV file in 30-second chunks using transcribe_without_streaming
        to avoid large memory usage and provide fine-grained progress callbacks.
        """
        audio_data, sample_rate = self.load_and_resample_wav(wav_path)
        
        # 30-second chunks (480,000 samples at 16kHz)
        chunk_sec = 30
        chunk_size = int(sample_rate * chunk_sec)
        total_samples = len(audio_data)
        
        transcribed_lines = []
        
        for i in range(0, total_samples, chunk_size):
            chunk = audio_data[i:i+chunk_size]
            # Convert NumPy array to standard Python list of float values (required by binding)
            chunk_list = chunk.tolist()
            
            # Transcribe chunk
            transcript = self.transcriber.transcribe_without_streaming(chunk_list, sample_rate)
            
            for line in transcript.lines:
                text = line.text.strip()
                if text:
                    transcribed_lines.append(text)
                    
            if progress_callback:
                progress = min(1.0, (i + len(chunk)) / total_samples)
                progress_callback(progress)
                
        return " ".join(transcribed_lines)


if __name__ == "__main__":
    # Test script if executed directly
    import sys
    if len(sys.argv) < 2:
        print("Usage: python asr.py <path_to_wav>")
        sys.exit(1)
        
    wav_file = sys.argv[1]
    print(f"Loading MoonshineASR...")
    try:
        asr = MoonshineASR()
        print(f"Transcribing {wav_file}...")
        def progress(p):
            print(f"Progress: {p:.1%}", end="\r")
        text = asr.transcribe(wav_file, progress_callback=progress)
        print("\nTranscription Result:")
        print(text)
    except Exception as e:
        print(f"Error during transcription: {e}")
