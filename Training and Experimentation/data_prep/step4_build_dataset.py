import os
import sys
import json
import random

# Paths config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_FILE = os.path.join(BASE_DIR, "data_prep", "output", "training_pairs.json")
TRAIN_OUTPUT = os.path.join(BASE_DIR, "data_prep", "output", "train.jsonl")
VAL_OUTPUT = os.path.join(BASE_DIR, "data_prep", "output", "val.jsonl")

def format_chatml(chunk_text: str, notes: str) -> str:
    """
    Formats the prompt and response into the exact SmolLM2 ChatML template syntax.
    """
    system_content = (
        "You are a meeting notes assistant. Convert the transcript to structured notes. "
        "Use ALL-CAPS headings. Start each bullet with *. No markdown, no extra text."
    )
    user_content = f"Transcript:\n{chunk_text}\n\nWrite the meeting notes:"
    
    text = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{notes}<|im_end|>"
    )
    return text

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Please run step3 first.")
        sys.exit(1)
        
    print("Loading training pairs...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pairs = json.load(f)
        
    print(f"Loaded {len(pairs)} training pairs.")
    
    if not pairs:
        print("Error: No training pairs available to build the dataset.")
        sys.exit(1)
        
    # Format and verify sizes
    formatted_data = []
    long_count = 0
    
    for p in pairs:
        text = format_chatml(p["chunk_text"], p["notes"])
        
        # Approximate token count (words * 1.35)
        approx_tokens = int(len(text.split()) * 1.35)
        
        if approx_tokens > 1024:
            long_count += 1
            
        formatted_data.append({
            "text": text,
            "approx_tokens": approx_tokens,
            "meeting_id": p["meeting_id"],
            "chunk_idx": p["chunk_idx"]
        })
        
    print(f"Approximated token size checking:")
    print(f" - Chunks exceeding 1024 tokens: {long_count} / {len(formatted_data)}")
    
    # Shuffle and Split 90/10
    random.seed(42)
    random.shuffle(formatted_data)
    
    split_idx = int(len(formatted_data) * 0.9)
    train_set = formatted_data[:split_idx]
    val_set = formatted_data[split_idx:]
    
    print(f"Dataset split:")
    print(f" - Train set size: {len(train_set)}")
    print(f" - Validation set size: {len(val_set)}")
    
    # Write train.jsonl
    print(f"Writing train set to {TRAIN_OUTPUT}...")
    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for item in train_set:
            # We only write the 'text' field to keep the JSONL clean for SFTTrainer
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
            
    # Write val.jsonl
    print(f"Writing validation set to {VAL_OUTPUT}...")
    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for item in val_set:
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
            
    print("\nDataset preparation successful!")
    print("\n=== Sample Formatted Entry ===")
    sample = train_set[0]["text"]
    print(sample[:600] + "\n... [TRUNCATED] ...\n" + sample[-200:])
    print("==============================")

if __name__ == "__main__":
    main()
