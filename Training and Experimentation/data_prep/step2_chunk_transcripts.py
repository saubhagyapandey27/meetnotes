import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pipeline.chunker import SemanticChunker, DISCARD_WORDS

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_FILE  = os.path.join(BASE_DIR, "data_prep", "output", "transcripts.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data_prep", "output", "chunks.json")

# Discard entire meetings whose transcript is shorter than this.
# A meeting this short has too little content to train the notes model on.
MEETING_DISCARD_WORDS = DISCARD_WORDS   # reuse the per-chunk threshold


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run step1 first.")
        sys.exit(1)

    print("Loading transcripts...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        transcripts_data = json.load(f)
    print(f"Loaded {len(transcripts_data)} meetings.\n")

    # Resume: load existing progress
    existing_chunks      = []
    processed_meetings   = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_chunks = json.load(f)
            for chunk in existing_chunks:
                processed_meetings.add(chunk["meeting_id"])
            print(f"Resuming: {len(existing_chunks)} chunks for "
                  f"{len(processed_meetings)} meetings already done.\n")
        except Exception as e:
            print(f"Warning: could not load existing output ({e}). Starting fresh.\n")
            existing_chunks    = []
            processed_meetings = set()

    print("Loading Semantic Chunker model...")
    try:
        chunker = SemanticChunker()
    except Exception as e:
        print(f"Fatal: could not load chunker: {e}")
        sys.exit(1)
    print("Chunker ready.\n")

    all_chunks       = list(existing_chunks)
    processed_count  = 0
    skipped_count    = 0
    discarded_count  = 0   # meetings too short to use
    error_count      = 0
    t0 = time.time()

    for idx, meeting in enumerate(transcripts_data):
        meeting_id = meeting["meeting_id"]
        transcript = meeting.get("transcript", "").strip()

        if meeting_id in processed_meetings:
            skipped_count += 1
            continue

        word_count = len(transcript.split())

        # Discard entire meeting if transcript is too short
        if word_count < MEETING_DISCARD_WORDS:
            print(f"[{idx+1}/{len(transcripts_data)}] DISCARDED {meeting_id} "
                  f"({word_count} words — below minimum of {MEETING_DISCARD_WORDS})")
            discarded_count += 1
            processed_meetings.add(meeting_id)   # mark done so it isn't retried
            continue

        if not transcript:
            print(f"[{idx+1}/{len(transcripts_data)}] Skipping {meeting_id} (empty)")
            continue

        print(f"[{idx+1}/{len(transcripts_data)}] Chunking {meeting_id} "
              f"({word_count:,} words)...", end="  ", flush=True)

        try:
            chunks = chunker.chunk(transcript)

            if not chunks:
                # chunker returned empty (entire transcript below DISCARD_WORDS)
                print("→ discarded (too short after chunking)")
                discarded_count += 1
                processed_meetings.add(meeting_id)
                continue

            meeting_chunks = [
                {
                    "meeting_id": meeting_id,
                    "chunk_idx":  i,
                    "chunk_text": c,
                    "word_count": len(c.split()),
                }
                for i, c in enumerate(chunks)
            ]

            wcs = [mc["word_count"] for mc in meeting_chunks]
            print(f"→ {len(chunks)} chunks  "
                  f"(min={min(wcs)} max={max(wcs)} avg={sum(wcs)//len(wcs)} words)")

            all_chunks.extend(meeting_chunks)
            processed_meetings.add(meeting_id)
            processed_count += 1

            # Save after every meeting for resume support
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"→ ERROR: {e}")
            error_count += 1

    t_total    = time.time() - t0
    word_counts = [c["word_count"] for c in all_chunks]

    brackets = [
        ("< 100 words   (discarded — noise)",          lambda w: w < 100),
        ("100–199 words (below merge floor — rare)",   lambda w: 100 <= w < 200),
        ("200–299 words (below merge floor — rare)",   lambda w: 200 <= w < 300),
        ("300–399 words (merged short-topic chunks)",  lambda w: 300 <= w < 400),
        ("400–499 words",                              lambda w: 400 <= w < 500),
        ("500–600 words",                              lambda w: 500 <= w <= 600),
        ("> 600 words   (above hard ceiling)",         lambda w: w > 600),
    ]

    print(f"\n{'='*55}")
    print(f"=== Chunking Completed ===")
    print(f"  Meetings processed  : {processed_count}")
    print(f"  Meetings skipped    : {skipped_count}  (already done)")
    print(f"  Meetings discarded  : {discarded_count}  (too short to use)")
    print(f"  Errors              : {error_count}")
    print(f"  Total chunks        : {len(all_chunks)}")
    if word_counts:
        print(f"  Avg words / chunk   : {sum(word_counts)/len(word_counts):.1f}")
        print(f"  Min words / chunk   : {min(word_counts)}")
        print(f"  Max words / chunk   : {max(word_counts)}")
    print(f"\n  Word-count distribution:")
    for label, fn in brackets:
        count = sum(1 for w in word_counts if fn(w))
        pct   = 100 * count / len(word_counts) if word_counts else 0
        print(f"    {label:<42} {count:>5} ({pct:.1f}%)")
    print(f"\n  Time elapsed        : {t_total:.1f}s")
    print(f"  Output              : {OUTPUT_FILE}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
