"""
setup.py — MeetNotes First-Run Model Setup
==========================================
Run this script once after cloning the repo to download and cache the two
on-device models needed by MeetNotes:

  1. Moonshine Tiny ORT  — the ASR model (speech-to-text, <200ms latency)
  2. potion-mxbai-256d-v2 — the embedding model (semantic transcript chunking)

Usage:
    python setup.py

The models are saved into the  models/  folder at the repo root.
After setup completes, follow the README for the remaining step:
  placing your fine-tuned SmolLM2 GGUF into  models/  and
  placing  llama-server.exe  into  bin/ .
"""

import os
import shutil
import sys

# ── Resolve paths relative to this script (works from any working directory) ──
REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: Download / copy Moonshine Tiny ORT (ASR model)
# ─────────────────────────────────────────────────────────────────────────────
def setup_moonshine():
    target_dir = os.path.join(MODELS_DIR, "moonshine_tiny_ort")
    
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print("[Moonshine] Already present — skipping.")
        return
    
    print("[Moonshine] Locating cached moonshine-tiny model from moonshine-voice package...")
    try:
        from moonshine_voice import get_model_for_language, ModelArch
    except ImportError:
        print("  ERROR: moonshine-voice is not installed. Run: pip install moonshine-voice")
        sys.exit(1)
    
    source_path, _ = get_model_for_language('en', ModelArch.TINY)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"  Copying files from package cache to {target_dir} ...")
    copied = 0
    for f in os.listdir(source_path):
        shutil.copy(os.path.join(source_path, f), os.path.join(target_dir, f))
        copied += 1
    print(f"  Done — {copied} file(s) copied.")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Download potion-mxbai-256d-v2 (embedding model for chunking)
# ─────────────────────────────────────────────────────────────────────────────
def setup_embedding_model():
    target_dir = os.path.join(MODELS_DIR, "potion-mxbai-256d-v2")
    
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print("[Embeddings] Already present — skipping.")
        return
    
    print("[Embeddings] Downloading blobbybob/potion-mxbai-256d-v2 from Hugging Face...")
    try:
        from model2vec import StaticModel
    except ImportError:
        print("  ERROR: model2vec is not installed. Run: pip install model2vec")
        sys.exit(1)
    
    model = StaticModel.from_pretrained("blobbybob/potion-mxbai-256d-v2")
    os.makedirs(target_dir, exist_ok=True)
    model.save_pretrained(target_dir)
    print(f"  Done — model saved to {target_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MeetNotes — First-Run Model Setup")
    print("=" * 60)
    print()
    
    setup_moonshine()
    print()
    setup_embedding_model()
    
    print()
    print("=" * 60)
    print("  Setup complete!")
    print()
    print("  Remaining manual steps:")
    print("  1. Download your fine-tuned SmolLM2 GGUF from Hugging Face")
    print("     and place it in:  models/")
    print("  2. Download llama-server.exe from the llama.cpp releases page")
    print("     and place it in:  bin/")
    print("  3. Run the app:  python src/main.py")
    print("=" * 60)
