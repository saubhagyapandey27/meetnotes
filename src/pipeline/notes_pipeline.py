import os
from src.pipeline.asr import MoonshineASR
from src.pipeline.chunker import SemanticChunker
from src.pipeline.llm import LlamaServerManager

class NotesPipeline:
    """
    Main orchestrator that coordinates the ASR, Chunker, and LLM layers
    to transcribe audio and compile structured notes.
    Supports GUI progress updates via callback functions.
    """
    def __init__(self, model_gguf_path: str, port: int = 8082):
        self.model_gguf_path = model_gguf_path
        self.port = port
        
        # Loaded lazily on first request to speed up startup
        self.asr = None
        self.chunker = None
        self.llm_manager = None

    def initialize_asr(self):
        if self.asr is None:
            self.asr = MoonshineASR()

    def initialize_chunker(self):
        if self.chunker is None:
            self.chunker = SemanticChunker()

    def initialize_llm(self):
        if self.llm_manager is None:
            self.llm_manager = LlamaServerManager(self.model_gguf_path, port=self.port)
            self.llm_manager.start()

    def process_audio(self, wav_path: str, progress_callback=None) -> str:
        """
        Runs the full pipeline:
        1. Transcribe audio to text (ASR)
        2. Divide transcript into semantic chunks (Chunker)
        3. Convert each chunk into structured notes (LLM)
        4. Join all chunk notes together
        
        progress_callback receives (stage_text: str, percentage: float)
        """
        if not os.path.exists(wav_path):
            return f"Error: Audio file not found at {wav_path}"

        # -------------------------------------------------------------
        # Stage 1: Transcription (takes approx 45% of total progress)
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("Initializing transcription engine...", 0.02)
        try:
            self.initialize_asr()
        except Exception as e:
            return f"Error initializing ASR engine: {e}"

        if progress_callback:
            progress_callback("Transcribing audio...", 0.05)
            
        def asr_prog(p):
            if progress_callback:
                # Map 0.0-1.0 to 0.05-0.45
                progress_callback("Transcribing audio...", 0.05 + (p * 0.40))
                
        try:
            transcript = self.asr.transcribe(wav_path, progress_callback=asr_prog)
        except Exception as e:
            return f"Error transcribing audio: {e}"

        if not transcript.strip():
            return "No speech detected in the audio file. Notes cannot be made."

        # -------------------------------------------------------------
        # Stage 2: Chunking (takes approx 5% of total progress)
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("Segmenting transcription...", 0.46)
        try:
            self.initialize_chunker()
            chunks = self.chunker.chunk(transcript)
        except Exception as e:
            return f"Error splitting transcription: {e}"

        if not chunks:
            return "Transcription segmentation failed. Notes cannot be made."
        
        if progress_callback:
            progress_callback("Segments created. Initializing LLM server...", 0.50)

        # -------------------------------------------------------------
        # Stage 3: LLM Inference (takes approx 50% of total progress)
        # -------------------------------------------------------------
        try:
            self.initialize_llm()
        except Exception as e:
            return f"Error starting LLM server: {e}"
            
        total_chunks = len(chunks)
        notes_list = []
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                chunk_num = i + 1
                progress_val = 0.50 + ((i / total_chunks) * 0.48)
                progress_callback(f"Generating notes (chunk {chunk_num}/{total_chunks})...", progress_val)
                
            try:
                notes = self.llm_manager.generate(chunk)
                if notes.strip():
                    notes_list.append(notes)
            except Exception as e:
                # Log error and append fallback note
                notes_list.append(f"[Error making notes for this section: {e}]")

        if progress_callback:
            progress_callback("Compiling final report...", 0.99)
            
        if not notes_list:
            return "No structured notes could be generated from the transcription."
            
        # Compile final output
        final_notes = "\n\n---\n\n".join(notes_list)
        
        if progress_callback:
            progress_callback("Done!", 1.0)
            
        return final_notes

    def shutdown(self):
        """
        Stops the local LLM server process.
        """
        if self.llm_manager:
            self.llm_manager.stop()
            self.llm_manager = None
