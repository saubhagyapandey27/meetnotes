import os
import wave
import threading
import time
import numpy as np
import soundcard as sc

class AudioRecorder:
    """
    Background audio recorder that can capture Microphone input,
    System Audio (via loopback), or both simultaneously, mixing them to mono 16kHz WAV.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.mic_thread = None
        self.sys_thread = None
        
        self.mic_buffer = []
        self.sys_buffer = []
        self.lock = threading.Lock()
        
        self.start_time = 0
        self.stop_time = None
        self.record_mic = False
        self.record_sys = False

    def _record_device_loop(self, device, buffer_list, native_rate: int):
        """
        Records from a soundcard device, converts to mono, resamples to target rate,
        and appends to the buffer list.
        """
        import os
        import ctypes
        if os.name == 'nt':
            ctypes.windll.ole32.CoInitialize(None)
            
        try:
            with device.recorder(samplerate=native_rate) as recorder:
                chunk_frames = int(native_rate * 0.1)  # 100ms chunks
                
                while self.is_recording:
                    # Record a chunk
                    data = recorder.record(numframes=chunk_frames)
                    
                    # Convert to mono
                    if len(data.shape) > 1 and data.shape[1] > 1:
                        data = data.mean(axis=1)
                    elif len(data.shape) > 1:
                        data = data[:, 0]
                        
                    # Resample to target rate (16kHz) if native rate is different
                    if native_rate != self.sample_rate:
                        num_samples = int(len(data) * self.sample_rate / native_rate)
                        data = np.interp(
                            np.linspace(0, len(data) - 1, num_samples),
                            np.arange(len(data)),
                            data
                        )
                        
                    with self.lock:
                        buffer_list.append(data)
                        
        except Exception as e:
            print(f"Error in recording thread for {device.name}: {e}")

    def start(self, record_mic: bool = True, record_sys: bool = False):
        """
        Starts the background recording threads for the selected sources.
        """
        if self.is_recording:
            return
            
        self.record_mic = record_mic
        self.record_sys = record_sys
        
        if not (self.record_mic or self.record_sys):
            raise ValueError("At least one recording source (Mic or System Audio) must be selected.")
            
        self.is_recording = True
        self.mic_buffer = []
        self.sys_buffer = []
        self.start_time = time.time()
        self.stop_time = None
        
        # 1. Microphone Thread
        if self.record_mic:
            try:
                mic_device = sc.default_microphone()
                # Determine native rate (default to 48000 to be safe)
                native_rate = 48000
                self.mic_thread = threading.Thread(
                    target=self._record_device_loop,
                    args=(mic_device, self.mic_buffer, native_rate),
                    daemon=True
                )
                self.mic_thread.start()
                print(f"Started microphone recording: {mic_device.name}")
            except Exception as e:
                self.is_recording = False
                raise RuntimeError(f"Failed to access default microphone: {e}")
                
        # 2. System Audio (Loopback) Thread
        if self.record_sys:
            try:
                default_speaker = sc.default_speaker()
                sys_device = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)
                native_rate = 48000
                self.sys_thread = threading.Thread(
                    target=self._record_device_loop,
                    args=(sys_device, self.sys_buffer, native_rate),
                    daemon=True
                )
                self.sys_thread.start()
                print(f"Started system audio loopback recording: {sys_device.name}")
            except Exception as e:
                self.is_recording = False
                raise RuntimeError(f"Failed to access system loopback loop: {e}")

    def get_elapsed_seconds(self) -> float:
        if self.start_time == 0:
            return 0.0
        if not self.is_recording and self.stop_time is not None:
            return self.stop_time - self.start_time
        return time.time() - self.start_time

    def get_current_volume(self) -> float:
        """
        Returns the peak volume (0.0 to 1.0) of the most recent audio chunk.
        """
        with self.lock:
            chunks = []
            if self.record_mic and self.mic_buffer:
                chunks.append(self.mic_buffer[-1])
            if self.record_sys and self.sys_buffer:
                chunks.append(self.sys_buffer[-1])
                
        if not chunks:
            return 0.0
            
        if len(chunks) == 2:
            min_len = min(len(chunks[0]), len(chunks[1]))
            if min_len == 0: return 0.0
            data = (chunks[0][:min_len] + chunks[1][:min_len]) / 2.0
        else:
            data = chunks[0]
            
        if len(data) == 0:
            return 0.0
            
        return float(np.max(np.abs(data)))

    def get_current_audio_data(self) -> np.ndarray:
        """
        Mixes and returns the accumulated buffer data up to the current moment.
        """
        with self.lock:
            mic_data = np.concatenate(self.mic_buffer) if (self.record_mic and self.mic_buffer) else np.array([], dtype=np.float32)
            sys_data = np.concatenate(self.sys_buffer) if (self.record_sys and self.sys_buffer) else np.array([], dtype=np.float32)
            
        # Mix the channels based on configuration
        if self.record_mic and self.record_sys:
            min_len = min(len(mic_data), len(sys_data))
            if min_len == 0:
                return np.array([], dtype=np.float32)
            # Average the two signals
            mixed_data = (mic_data[:min_len] + sys_data[:min_len]) / 2.0
            return mixed_data
        elif self.record_mic:
            return mic_data
        elif self.record_sys:
            return sys_data
        
        return np.array([], dtype=np.float32)

    def save_wav(self, path: str, audio_data: np.ndarray):
        """
        Saves float32 audio data to a 16-bit PCM mono WAV file at target sample rate.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            # Convert float32 [-1.0, 1.0] to int16
            int_data = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
            wf.writeframes(int_data.tobytes())

    def get_current_wav_snapshot(self, temp_wav_path: str) -> bool:
        """
        Saves a snapshot of the audio recorded so far to a temporary file
        for the 'Make Notes Up To Now' feature.
        """
        data = self.get_current_audio_data()
        if len(data) == 0:
            return False
        self.save_wav(temp_wav_path, data)
        return True

    def stop(self, save_path: str = None) -> float:
        """
        Stops the recording loops and optionally saves the full mixed audio to the WAV path.
        Returns the duration of the recording in seconds.
        """
        if not self.is_recording:
            return 0.0
            
        self.stop_time = time.time()
        self.is_recording = False
        duration = self.get_elapsed_seconds()
        
        # Wait for threads to finish
        if self.mic_thread:
            self.mic_thread.join(timeout=1.0)
            self.mic_thread = None
        if self.sys_thread:
            self.sys_thread.join(timeout=1.0)
            self.sys_thread = None
            
        if save_path:
            full_data = self.get_current_audio_data()
            if len(full_data) > 0:
                self.save_wav(save_path, full_data)
                
        return duration
