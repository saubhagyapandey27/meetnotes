import os
import sys
import json
import time
import random
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Paths config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_FILE = os.path.join(BASE_DIR, "data_prep", "output", "chunks.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data_prep", "output", "training_pairs.json")
REJECTED_LOG = os.path.join(BASE_DIR, "data_prep", "output", "rejected_chunks.log")

# Google GenAI Settings
MODEL_NAME = "gemma-4-31b-it"
BATCH_SIZE = 10

class ChunkNote(BaseModel):
    batch_index: int = Field(description="The simple 0-9 index of the chunk provided in the prompt")
    notes: str = Field(description="The structured notes for this transcript excerpt. Must use ALL-CAPS headings and '*' bullet points.")

class BatchNotesResponse(BaseModel):
    results: List[ChunkNote]

SYSTEM_PROMPT = """You are a specialized meeting transcription analyst.
Your task is to convert a raw meeting transcript excerpts into structured meeting notes. 
This is a precision task — every important point must be captured.

STRICT OUTPUT RULES FOR 'notes' FIELD:
- Start immediately with the first section heading. No preamble.
- Use ALL-CAPS section headings followed by a colon and newline.
- Every bullet point must begin with '* ' (asterisk + space).
- No nested bullets. No sub-bullets. No numbered lists. No markdown formatting (no **, no #, no -).
- No conclusion statements.
- If a topic has no notable content, skip that section entirely.
- CRITICAL: Each transcript excerpt is completely independent and from random, unrelated meetings. Do NOT blend information across excerpts. The notes for a chunk must ONLY use information found within that specific chunk's text.

FAITHFULNESS REQUIREMENTS:
- Preserve the speaker's level of certainty.
- Do not present suggestions, proposals, or speculation as finalized decisions.
- Express tentative ideas as proposals, recommendations, or options rather than confirmed outcomes.
- Only describe something as a decision when the transcript clearly indicates agreement, confirmation, or commitment.
- Do not infer conclusions that are not explicitly supported by the transcript.

You will be provided with a batch of transcript chunks. Process each one independently and return the results as a structured JSON object."""

def log_rejection(reason: str, chunk_idx: int, meeting_id: str, transcript: str, output: str):
    os.makedirs(os.path.dirname(REJECTED_LOG), exist_ok=True)
    with open(REJECTED_LOG, "a", encoding="utf-8") as f:
        f.write(f"=== REJECTION ===\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Meeting ID: {meeting_id}, Chunk ID: {chunk_idx}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Chunk Text:\n{transcript}\n")
        f.write(f"Model Output:\n{output}\n")
        f.write(f"=================\n\n")

def is_output_clean(text: str) -> tuple[bool, str]:
    text_stripped = text.strip()
    if len(text_stripped) < 50:
        return False, "Output is too short (< 50 chars)"
    if "*" not in text_stripped:
        return False, "Output contains no '*' bullet points"
    
    import re
    has_caps_heading = False
    for line in text_stripped.split("\n"):
        line = line.strip()
        if line and not line.startswith("*"):
            if re.search(r'\b[A-Z]{3,}\b', line):
                has_caps_heading = True
                break
    if not has_caps_heading:
        return False, "Output contains no ALL-CAPS subheadings"
        
    preamble_junk = ["Here is", "Here are", "Sure", "Certainly", "I can help", "According to", "In the transcript"]
    for junk in preamble_junk:
        if text_stripped.lower().startswith(junk.lower()):
            return False, f"Output starts with preamble junk: '{junk}'"
            
    if "**" in text_stripped:
        return False, "Output contains disallowed bold markdown '**'"
    if "#" in text_stripped:
        return False, "Output contains disallowed markdown headers '#'"
        
    for line in text_stripped.split("\n"):
        line = line.strip()
        if line.startswith("-"):
            return False, "Output contains '-' bullets instead of '*'"
            
    return True, ""

def generate_batch(client: genai.Client, chunks: List[dict], max_retries=5) -> Optional[BatchNotesResponse]:
    prompt = "Here is the batch of transcript excerpts to process:\n\n"
    for idx, c in enumerate(chunks):
        prompt += f"--- START CHUNK {idx} ---\n"
        prompt += f"TRANSCRIPT EXCERPT:\n{c['chunk_text']}\n"
        prompt += f"--- END CHUNK {idx} ---\n\n"
        
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=BatchNotesResponse,
                    temperature=0.3,
                ),
            )
            if resp.parsed is not None:
                return resp.parsed
            
            # Manual parse fallback
            if resp.text:
                cleaned_text = resp.text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                elif cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                return BatchNotesResponse.model_validate_json(cleaned_text.strip())
            
            raise ValueError("LLM returned empty response")
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["429", "resource_exhausted", "503", "unavailable", "500", "timeout", "temporary"]):
                wait = 2 ** attempt * 5
                print(f"    Transient error: {e}. Retrying in {wait}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"    Non-retryable error during generation: {e}")
                return None
    return None

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
        
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Please run step2 first.")
        sys.exit(1)
        
    client = genai.Client(api_key=api_key)
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    print(f"Loaded {len(chunks_data)} chunks.")
    
    progress_data = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                pairs = json.load(f)
                progress_data = {f"{p['meeting_id']}_{p['chunk_idx']}": p for p in pairs}
            print(f"Loaded existing progress: {len(progress_data)} pairs. Resuming...")
        except Exception as e:
            print(f"Warning: Failed to load {OUTPUT_FILE} ({e}). Overwriting...")

    # Filter out already processed chunks
    pending_chunks = []
    for c in chunks_data:
        key = f"{c['meeting_id']}_{c['chunk_idx']}"
        if key not in progress_data:
            pending_chunks.append(c)

    print(f"{len(chunks_data) - len(pending_chunks)} chunks already processed. {len(pending_chunks)} pending.")
    
    # Shuffle pending chunks to ensure consecutive chunks aren't sent in the same batch, preventing context blending
    random.shuffle(pending_chunks)
    
    processed_count = 0
    rejected_count = 0
    t0 = time.time()
    
    for i in range(0, len(pending_chunks), BATCH_SIZE):
        batch = pending_chunks[i:i + BATCH_SIZE]
        print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(len(pending_chunks) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} chunks)...")
        
        parsed_resp = generate_batch(client, batch)
        
        if parsed_resp:
            # Map generated notes to original chunks by simple array index
            for result in parsed_resp.results:
                if result.batch_index < 0 or result.batch_index >= len(batch):
                    print(f"    Warning: LLM returned invalid batch_index {result.batch_index}")
                    continue
                    
                orig_chunk = batch[result.batch_index]
                
                is_clean, reason = is_output_clean(result.notes)
                if is_clean:
                    key = f"{orig_chunk['meeting_id']}_{orig_chunk['chunk_idx']}"
                    progress_data[key] = {
                        "meeting_id": orig_chunk['meeting_id'],
                        "chunk_idx": orig_chunk['chunk_idx'],
                        "chunk_text": orig_chunk["chunk_text"],
                        "notes": result.notes
                    }
                    processed_count += 1
                else:
                    rejected_count += 1
                    print(f"    Failed: {orig_chunk['meeting_id']}_{orig_chunk['chunk_idx']}. Reason: {reason}.")
                    log_rejection(reason, orig_chunk['chunk_idx'], orig_chunk['meeting_id'], orig_chunk["chunk_text"], result.notes)
            
            # Save progressively after each batch
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(list(progress_data.values()), f, indent=2, ensure_ascii=False)
        else:
            print("    Batch generation failed.")
            
        print("    Waiting 2 seconds to respect rate limits...")
        time.sleep(2)
        
    t_total = time.time() - t0
    print(f"\n=== Target Generation Completed ===")
    print(f"Total processed in this session: {processed_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Total time elapsed: {t_total/60:.2f} minutes")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
