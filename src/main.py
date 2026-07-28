import os
import sys
import glob
import atexit
import tkinter as tk
from tkinter import messagebox

# Configure sys.path so we can import src correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gui.app import MeetNotesApp

def get_model_path() -> str:
    """
    Searches the models/ directory for any GGUF model files.
    If multiple, prioritizes smollm2 and UD_Q5_K_XL.
    """
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    gguf_files = glob.glob(os.path.join(models_dir, "*.gguf"))
    
    if not gguf_files:
        # Check current directory as fallback
        gguf_files = glob.glob("*.gguf")
        
    if not gguf_files:
        return ""
        
    # Prioritization logic: look for UD_Q5_K_XL or smollm2
    for file in gguf_files:
        filename = os.path.basename(file).lower()
        if "ud_q5_k_xl" in filename:
            return os.path.abspath(file)
            
    for file in gguf_files:
        filename = os.path.basename(file).lower()
        if "smollm2" in filename:
            return os.path.abspath(file)
            
    # Return first GGUF file if no specific prioritized file is found
    return os.path.abspath(gguf_files[0])

def check_llama_server() -> bool:
    """
    Checks if llama-server.exe is present in the standard locations.
    """
    # 1. Bundled PyInstaller location
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        path = os.path.join(sys._MEIPASS, 'bin', 'llama-server.exe')
        if os.path.exists(path):
            return True
            
    # 2. Env variable
    if os.environ.get("LLAMA_SERVER_PATH") and os.path.exists(os.environ.get("LLAMA_SERVER_PATH")):
        return True
        
    # 3. Dev paths
    workspace_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "bin", "llama-server.exe"),
        # Place llama-server.exe in a 'bin/' folder at the repo root, or
        # set the LLAMA_SERVER_PATH environment variable to its full path.
        os.path.join("bin", "llama-server.exe"),
    ]
    for p in workspace_candidates:
        if os.path.exists(os.path.abspath(p)):
            return True
            
    return False

def main():
    # 1. Check for llama-server.exe
    if not check_llama_server():
        # Setup temporary Tk instance to show dialog
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "System Requirement Missing",
            "Error: Could not locate 'llama-server.exe'.\n\n"
            "If you are in development mode, please build llama.cpp or place llama-server.exe "
            "in the 'bin/' folder."
        )
        sys.exit(1)
        
    # 2. Check for GGUF model
    model_path = get_model_path()
    if not model_path:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Model File Missing",
            "Error: No GGUF language model found in the 'models/' folder.\n\n"
            "Please drop your finetuned 'SmolLM2-135M-Meetnotes-gguf-Q8.gguf' model file "
            "into the 'models/' directory and restart the application."
        )
        sys.exit(1)
        
    print(f"MeetNotes started successfully!")
    print(f"Loading GGUF Model: {os.path.basename(model_path)}")
    
    # 3. Launch App
    root = tk.Tk()
    app = MeetNotesApp(root, model_path)
    
    # Setup graceful cleanup on close
    def on_closing():
        app.cleanup()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Register global exit handler just in case
    atexit.register(app.cleanup)
    
    root.mainloop()

if __name__ == "__main__":
    main()
