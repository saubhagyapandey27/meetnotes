import numpy as np

try:
    from model2vec import StaticModel
except ImportError:
    StaticModel = None

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
#
#  WINDOW_WORDS  –  words in each context block used for embedding.
#                   50 words gives strong semantic signal without blurring
#                   across multiple topic transitions.
#
#  STEP          –  evaluate a boundary every STEP words.
#                   5 words = high resolution boundary detection.
#                   Must divide WINDOW_WORDS evenly.
#
#  PEAK_K        –  topic boundary sensitivity.
#                   A boundary is declared where the smoothed boundary score
#                   exceeds  mean + PEAK_K * std.
#                   0.5 → top ~30% of scores → moderately sensitive.
#
#  MIN_WORDS     –  soft lower bound.  Chunks below this are merged into
#                   their neighbour AFTER topic detection is complete.
#
#  MAX_WORDS     –  hard ceiling.  Chunks above this are re-split at the
#                   strongest interior boundary score.
#
#  DISCARD_WORDS –  chunks (or whole meetings) below this are discarded —
#                   too little content to generate meaningful notes from.
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_WORDS  = 150   # 150 words ≈ ~1 min of speech (macro-topic scale)
STEP          = 15    # evaluate boundary every 15 words
PEAK_K        = 0.5
MIN_WORDS     = 300   # merge chunks below this into their neighbour
                      # 300 words ensures larger chunks to reduce total chunk count
MAX_WORDS     = 600   # split chunks above this
DISCARD_WORDS = 100   # hard discard: truly empty/noise fragments only
                      # (a chunk that is still < 100 words after merging has
                      # no neighbour to absorb it, so it gets dropped)
SNAP_RADIUS   = 25    # search radius to snap cuts to natural punctuation

assert WINDOW_WORDS % STEP == 0, "WINDOW_WORDS must be divisible by STEP"
_OFFSET = WINDOW_WORDS // STEP   # = 10  (right window is OFFSET slots ahead)


class SemanticChunker:
    """
    High-resolution topic-boundary chunker for raw ASR transcripts.

    Topic detection
    ───────────────
    For every candidate split position p (every STEP words), we measure how
    semantically different the WINDOW_WORDS words BEFORE p are from the
    WINDOW_WORDS words AFTER p.  This detects macro-topic shifts.

    Boundary Snapping
    ─────────────────
    Once a mathematical peak is found, the chunker scans within SNAP_RADIUS
    to find the nearest sentence boundary (., ?, !) or speaker dash (-).
    This prevents slicing sentences or noun phrases in half. If no punctuation
    is found (e.g. poor ASR for a long stretch), it safely falls back to
    the mathematical peak.

    Post-processing philosophy
    ──────────────────────────
    After detecting topic boundaries, sizes are adjusted in this order:

    1. Merge  — any chunk below MIN_WORDS (300) is merged into its smallest
                neighbour.  In real meetings, short topics ("quick update",
                "action item recap") genuinely exist.  Merging them teaches the
                LLM to handle those cases in production, where the same chunker
                runs and may produce the same short segments.

    2. Split  — any chunk above MAX_WORDS (600) is re-split at the strongest
                interior semantic boundary, not a naive midpoint cut.

    3. Discard — only truly unabsorbable fragments (< DISCARD_WORDS = 100)
                 are dropped.  These are edge-of-meeting noise with no content.

    Pipeline
    ────────
    1. Embed all context windows in one batch.
    2. Compute boundary scores at every STEP-word position.
    3. Smooth the score curve to suppress noise.
    4. Find peak clusters above  mean + PEAK_K × std  → topic boundaries.
    5. Merge fragments below MIN_WORDS into smallest neighbour.
    6. Re-split anything above MAX_WORDS using the same boundary score method.
    7. Discard anything still below DISCARD_WORDS (truly unabsorbable noise).
    """

    def __init__(self, model_path_or_id: str = None):
        if StaticModel is None:
            raise ImportError("Run: pip install model2vec")
        
        import os
        if model_path_or_id is None:
            models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
            model_path_or_id = os.path.abspath(os.path.join(models_dir, "potion-mxbai-256d-v2"))
            
        self.model = StaticModel.from_pretrained(model_path_or_id)

    # ──────────────────────────────────────────────────────────────────────
    #  Core: compute boundary scores
    # ──────────────────────────────────────────────────────────────────────
    def _boundary_scores(self, words: list) -> tuple:
        """
        Embed all context windows in one batch, then compute a boundary score
        at every potential split position.

        boundary_score[i]  =  1 − cosine( left_block_i, right_block_i )

        where
          left_block_i  = WINDOW_WORDS words BEFORE split position p_i
          right_block_i = WINDOW_WORDS words AFTER  split position p_i

        A high score means the text changes dramatically at p_i → likely topic shift.
        A low score means the same topic continues → don't split here.

        Because left_block_i starts at p_i − W  and  right_block_i starts at p_i,
        and windows are spaced by STEP words, we just need the dot product between
        embedding[k] and embedding[k + _OFFSET] for each window index k.

        Returns
        -------
        scores        : np.ndarray of shape (N,)  — boundary scores
        split_positions : list of int              — corresponding word indices
        """
        n = len(words)

        # All window starting positions (step = STEP words)
        win_starts = list(range(0, n - WINDOW_WORDS + 1, STEP))
        if len(win_starts) <= _OFFSET:
            return np.array([]), []

        # ── Single batch embed call ───────────────────────────────────────
        win_texts = [" ".join(words[s : s + WINDOW_WORDS]) for s in win_starts]
        embs = self.model.encode(win_texts)   # shape: (n_wins, dim)

        # L2 normalise (vectorised)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs /= norms

        # ── Boundary scores: dot product between window k and window k+_OFFSET ─
        # left context  = embs[k]           (ends at split point)
        # right context = embs[k + _OFFSET] (starts at split point)
        left  = embs[: -_OFFSET]       # shape: (n_wins - _OFFSET, dim)
        right = embs[_OFFSET :]        # shape: (n_wins - _OFFSET, dim)

        scores = 1.0 - np.sum(left * right, axis=1)   # boundary dissimilarity

        # Split word position: exactly where the right window begins
        #   = win_starts[k] + WINDOW_WORDS = win_starts[k + _OFFSET]
        split_positions = [win_starts[k] + WINDOW_WORDS
                           for k in range(len(scores))]

        return scores, split_positions

    # ──────────────────────────────────────────────────────────────────────
    #  Smoothing
    # ──────────────────────────────────────────────────────────────────────
    def _smooth(self, scores: np.ndarray) -> np.ndarray:
        """
        Box-filter smooth over a width of 2*_OFFSET+1 score positions
        (≈ one full WINDOW_WORDS region on each side).  This suppresses
        noise from individual window variation without blurring broad topic
        transitions.
        """
        if len(scores) == 0:
            return scores
        width  = 2 * _OFFSET + 1
        kernel = np.ones(min(width, len(scores))) / min(width, len(scores))
        return np.convolve(scores, kernel, mode="same")

    # ──────────────────────────────────────────────────────────────────────
    #  Peak detection & Snapping
    # ──────────────────────────────────────────────────────────────────────
    def _snap_to_boundary(self, pos: int, words: list) -> int:
        """
        Pass 1: Scan within SNAP_RADIUS for natural punctuation.
        Pass 2: If no punctuation, scan for linguistic clause-starters (pronouns, conjunctions).
        Pass 3: Fall back to mathematical pos.
        """
        start = max(0, pos - SNAP_RADIUS)
        end = min(len(words), pos + SNAP_RADIUS)
        
        # --- Pass 1: Punctuation (highest confidence) ---
        best_pos = pos
        min_dist = float('inf')
        found_punct = False
        
        for i in range(start, end):
            w = words[i]
            if w.endswith('.') or w.endswith('?') or w.endswith('!') or w == '-' or w == '--':
                # The cut should happen AFTER the punctuation word
                cut_point = i + 1
                dist = abs(cut_point - pos)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = cut_point
                    found_punct = True
                    
        if found_punct:
            return best_pos

        # --- Pass 2: Clause Starters (Lightweight Linguistic Fallback) ---
        # If ASR stripped punctuation, we look for words that naturally start clauses.
        # We avoid "and", "or", "it", "you" because they often connect mid-phrase or act as objects.
        clause_starters = {
            # Strong conjunctions
            "but", "so", "because", "if", "when", "although", "unless", "while",
            # Discourse markers / Transition words
            "well", "okay", "right", "anyway", "anyways", "now", "basically", "actually",
            # Subjective pronouns (almost always start a clause, unlike 'it' or 'you')
            "i", "we", "they", "he", "she",
            # Question words
            "what", "why", "how", "where", "who"
        }
        
        for i in range(start, end):
            w_clean = words[i].lower().strip("',\"()")
            if w_clean in clause_starters:
                # The cut should happen BEFORE the clause starter
                cut_point = i
                dist = abs(cut_point - pos)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = cut_point
                    
        return best_pos

    def _find_boundaries(self, smoothed: np.ndarray,
                          split_positions: list, words: list) -> list:
        """
        Find word positions where smoothed boundary score peaks above threshold.
        Then snap those peaks to the nearest natural punctuation.
        """
        if len(smoothed) == 0:
            return []

        threshold = float(np.mean(smoothed) + PEAK_K * np.std(smoothed))

        raw_boundaries = []
        in_peak    = False
        peak_score = -1.0
        peak_pos   = -1

        for i, score in enumerate(smoothed):
            if score >= threshold:
                if not in_peak:
                    in_peak    = True
                    peak_score = score
                    peak_pos   = split_positions[i]
                elif score > peak_score:
                    peak_score = score
                    peak_pos   = split_positions[i]
            else:
                if in_peak:
                    raw_boundaries.append(peak_pos)
                    in_peak = False

        if in_peak:
            raw_boundaries.append(peak_pos)

        # Apply boundary snapping
        return [self._snap_to_boundary(p, words) for p in raw_boundaries]

    # ──────────────────────────────────────────────────────────────────────
    #  Post-processing: merge undersized chunks
    # ──────────────────────────────────────────────────────────────────────
    def _merge_small(self, chunks: list) -> list:
        """
        Merge any chunk below MIN_WORDS into its smallest neighbour.

        This runs AFTER topic detection.  Short chunks are a natural part of
        real meetings (quick updates, action-item recaps) and the LLM needs to
        be trained on those cases, so they are merged rather than discarded.
        Uses has_prev/has_next guards to avoid IndexError on empty result lists.
        """
        merged  = [c for c in chunks if c.strip()]
        changed = True
        while changed:
            changed = False
            result, i = [], 0
            while i < len(merged):
                wc       = len(merged[i].split())
                has_prev = len(result) > 0
                has_next = i + 1 < len(merged)

                if wc < MIN_WORDS and (has_prev or has_next):
                    if has_prev and has_next:
                        if len(result[-1].split()) <= len(merged[i + 1].split()):
                            result[-1] = result[-1] + " " + merged[i]
                        else:
                            merged[i + 1] = merged[i] + " " + merged[i + 1]
                    elif has_prev:
                        result[-1] = result[-1] + " " + merged[i]
                    else:
                        merged[i + 1] = merged[i] + " " + merged[i + 1]
                    changed = True
                else:
                    result.append(merged[i])
                i += 1
            merged = result
        return merged

    # ──────────────────────────────────────────────────────────────────────
    #  Post-processing: re-split oversized chunks
    # ──────────────────────────────────────────────────────────────────────
    def _split_large(self, chunks: list) -> list:
        """
        Re-split any chunk above MAX_WORDS using the same boundary score
        method (not a naive midpoint cut).  Finds the split position with
        the highest boundary score that also leaves MIN_WORDS on each side.
        Falls back to midpoint if no such boundary exists.
        Recurses until all chunks are within MAX_WORDS.
        """
        final = []
        for chunk in chunks:
            words = chunk.split()
            if len(words) <= MAX_WORDS:
                final.append(chunk)
                continue

            # Compute boundary scores inside this oversized chunk
            scores, positions = self._boundary_scores(words)
            best_pos = len(words) // 2   # default: midpoint

            if len(scores) > 0:
                smoothed = self._smooth(scores)
                # Among positions that give MIN_WORDS on each side,
                # pick the one with the highest boundary score
                best_score = -1.0
                for score, pos in zip(smoothed, positions):
                    if MIN_WORDS <= pos <= len(words) - MIN_WORDS:
                        if score > best_score:
                            best_score = score
                            best_pos   = pos
                            
                best_pos = self._snap_to_boundary(best_pos, words)
                
                # Failsafe: if snapping pushed it too far to an edge, revert to midpoint
                if best_pos < 50 or best_pos > len(words) - 50:
                    best_pos = len(words) // 2

            left  = " ".join(words[:best_pos])
            right = " ".join(words[best_pos:])
            final.extend(self._split_large([left, right]))   # recurse

        return final

    # ──────────────────────────────────────────────────────────────────────
    #  Post-processing: discard anything too small to be useful
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _discard_tiny(chunks: list) -> list:
        return [c for c in chunks if len(c.split()) >= DISCARD_WORDS]

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────
    def chunk(self, text: str) -> list:
        """
        Chunk a raw ASR meeting transcript by topic.

        Returns [] if the transcript is below DISCARD_WORDS.
        """
        words   = text.split()
        n_words = len(words)

        if n_words < DISCARD_WORDS:
            return []

        # Too short for the boundary detector → single chunk
        min_viable = WINDOW_WORDS * 2 + WINDOW_WORDS
        if n_words < min_viable:
            return [text]

        # 1. One batch encode → boundary scores at every STEP words
        scores, split_positions = self._boundary_scores(words)
        if len(scores) == 0:
            return [text]

        # 2. Smooth
        smoothed = self._smooth(scores)

        # 3. Topic boundaries (pure semantics + snapping)
        boundaries = self._find_boundaries(smoothed, split_positions, words)

        # 4. Assemble raw chunks
        raw_chunks = []
        last = 0
        for bp in sorted(set(boundaries)):
            if last < bp < n_words:
                raw_chunks.append(" ".join(words[last:bp]))
                last = bp
        raw_chunks.append(" ".join(words[last:]))

        # 5. Merge fragments below MIN_WORDS into smallest neighbour
        merged = self._merge_small(raw_chunks)

        # 6. Re-split anything above MAX_WORDS using boundary scores
        sized = self._split_large(merged)

        # 7. Discard unabsorbable fragments below DISCARD_WORDS
        return self._discard_tiny(sized)


# ─────────────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading SemanticChunker (high-resolution boundary detection)...")
    chunker = SemanticChunker()

    # Simulate 3 distinct topics in raw ASR (no punctuation)
    finance = (
        "good morning everyone let us start with the financials our revenue for q2 "
        "came in at twelve point four million dollars that is up eighteen percent year "
        "over year which is ahead of our guidance operating expenses were tightly "
        "controlled at eight point one million net income therefore landed at four "
        "point three million the cfo has proposed increasing the marketing budget by "
        "twenty percent in q3 she believes this will drive another fifteen percent "
        "revenue lift before year end the board has approved the proposal contingent "
        "on hitting the july sales target we need to close at least two enterprise "
        "deals this month to unlock that budget sales has three warm leads in final "
        "negotiation stages right now we are cautiously optimistic about closing "
        "at least two of the three this week the pipeline looks very healthy "
    ) * 2

    engineering = (
        "shifting now to engineering the team completed sprint twenty two the new "
        "search feature is in internal qa and looking very solid performance benchmarks "
        "show a sixty percent improvement over the legacy implementation the mobile "
        "redesign is about forty percent complete we expect to wrap the core screens "
        "by end of july and ship to beta users in august the ai recommendation engine "
        "is still in research phase the team is evaluating three open source models "
        "infrastructure costs have been a concern devops is migrating to spot instances "
        "which should cut costs by thirty percent that migration completes july fifteenth "
    ) * 2

    hr = (
        "switching to team growth hr has approved six new headcount positions for q3 "
        "two senior engineers one data scientist one product manager and two sales reps "
        "job descriptions are live on linkedin we have received strong early applications "
        "the goal is to fill all six roles before september first the new structured "
        "four week onboarding program replaces the old ad hoc approach that feedback "
        "surveys consistently rated poorly all new hires starting august go through it "
        "the company off site is in lisbon over three days in september travel bookings "
        "must be done by august first to get the negotiated rates please get manager "
        "approval for your attendance by next friday that covers all items for today "
    ) * 2

    sample = finance + " " + engineering + " " + hr
    words = sample.split()
    print(f"Input: {len(words)} words across 3 distinct topics (no punctuation)\n")

    chunks = chunker.chunk(sample)
    print(f"Produced {len(chunks)} chunk(s):\n")
    for i, c in enumerate(chunks):
        wc = len(c.split())
        flag = ""
        if wc < DISCARD_WORDS: flag = "  ⚠ BELOW DISCARD"
        elif wc < MIN_WORDS:   flag = "  ⚠ BELOW MIN"
        elif wc > MAX_WORDS:   flag = "  ⚠ ABOVE MAX"
        print(f"  Chunk {i+1}: {wc} words{flag}")
        print(f"  Starts: {' '.join(c.split()[:15])}...")
        print()
