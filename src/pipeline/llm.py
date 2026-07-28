import os
import sys
import time
import subprocess
import requests

class LlamaServerManager:
    """
    Manages the lifecycle of a llama-server.exe process for local GGUF model inference.
    Communicates via OpenAI-compatible REST API.
    """
    def __init__(self, model_gguf_path: str, port: int = 8082, threads: int = None, ngl: int = 0):
        self.model_gguf_path = os.path.abspath(model_gguf_path)
        self.port = port
        self.threads = threads if threads is not None else max(1, os.cpu_count() - 1)
        self.ngl = ngl
        self.process = None
        self.server_url = f"http://127.0.0.1:{self.port}"
        
        # Locate binary
        self.bin_path = self.find_llama_server()
        
    def find_llama_server(self) -> str:
        """
        Locates the llama-server.exe binary on the system.
        Supports both packaged PyInstaller environment and development paths.
        """
        # 1. PyInstaller bundled path
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundled_path = os.path.join(sys._MEIPASS, 'bin', 'llama-server.exe')
            if os.path.exists(bundled_path):
                return bundled_path
                
        # 2. Environment variable path
        env_path = os.environ.get("LLAMA_SERVER_PATH")
        if env_path and os.path.exists(env_path):
            return env_path
            
        # 3. Development/known locations relative to workspace
        workspace_candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "bin", "llama-server.exe"),
            os.path.join(os.path.dirname(__file__), "..", "bin", "llama-server.exe"),
            # Place llama-server.exe in a 'bin/' folder at the repo root, or
            # set the LLAMA_SERVER_PATH environment variable to its full path.
            os.path.join("bin", "llama-server.exe"),
        ]
        
        for path in workspace_candidates:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
                
        raise FileNotFoundError(
            "Could not find llama-server.exe. Please place it in bin/ folder or set LLAMA_SERVER_PATH env variable."
        )

    def start(self, timeout_sec: int = 60) -> bool:
        """
        Starts the llama-server.exe subprocess with the specified GGUF model.
        Hides the terminal window on Windows to keep application fully headless.
        """
        if self.process and self.process.poll() is None:
            # Server is already running
            return True
            
        if not os.path.exists(self.model_gguf_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_gguf_path}")
            
        args = [
            self.bin_path,
            "-m", self.model_gguf_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", "1536",      # Context window.
                               # 600-word chunk ≈ 780 tokens + ~130 prompt tokens + 600 output
                               # = ~1510 tokens needed. 2048 gives comfortable headroom.
            "-ngl", str(self.ngl), # Offload layers to GPU
            "-b", "512",        # Batch size
            "-ub", "512",       # Ubatch size
            "-t", str(self.threads),
            "--no-log-prefix",
            "-lv", "0"          # Quiet output
        ]
        
        # Hide command prompt window on Windows
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
        print(f"Starting llama-server on port {self.port} with threads={self.threads}...")
        self.process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            text=True
        )
        
        # Wait for server to load model and respond to /health
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if self.process.poll() is not None:
                # Process died
                stderr = self.process.stderr.read()
                raise RuntimeError(f"llama-server failed to start. Exit code: {self.process.returncode}. Stderr: {stderr}")
                
            try:
                response = requests.get(f"{self.server_url}/health", timeout=1)
                if response.status_code == 200:
                    print(f"llama-server is up and running on port {self.port}!")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)
            
        self.stop()
        raise TimeoutError("llama-server startup timed out.")

    def stop(self):
        """
        Gracefully terminates the llama-server.exe process.
        """
        if self.process:
            if self.process.poll() is None:
                print(f"Stopping llama-server on port {self.port}...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None

    def generate(self, chunk_text: str) -> str:
        """
        Calls the llama-server completion API to generate notes for a transcript chunk.
        """
        system_prompt = (
            "You are a meeting notes assistant. Convert the transcript to structured notes. "
            "Organize the notes only under these headings: TECHNICAL & RESEARCH, DESIGN & STRATEGY, BUSINESS & PROJECT MANAGEMENT, PEOPLE & LOGISTICS, TASKS TO DO, DECISIONS & CONCLUSIONS, QUESTIONS & ISSUES, THOUGHTS & IDEAS. Include a heading only if the transcript has relevant content for it. "
            "Use ALL-CAPS headings. Start each bullet with *. No markdown, no extra text."
        )
        user_prompt = f"Transcript:\n{chunk_text}\n\nWrite the meeting notes:"
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "top_p": 0.8,
            "repeat_penalty": 1.15,
            "max_tokens": 600   # 600 output tokens — consistent with QwenServerManager
        }
        
        # Retry up to 3 times on connection failure
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    raise RuntimeError(f"Error from llama-server API: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise e
                time.sleep(1)
        return ""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()




if __name__ == "__main__":
    # Test script if executed directly
    if len(sys.argv) < 3:
        print("Usage: python llm.py <path_to_gguf> <port>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    server_port = int(sys.argv[2])
    
    print(f"Starting test server for GGUF model...")
    try:
        with LlamaServerManager(model_path, port=server_port) as server:
            test_prompt = "Hello! I am testing the local inference pipeline. Confirm that you can hear me."
            print("Sending completion request...")
            result = server.generate(test_prompt)
            print("\nResponse:")
            print(result)
    except Exception as e:
        print(f"Error during test execution: {e}")
