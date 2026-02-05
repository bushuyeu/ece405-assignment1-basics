from typing import List, Dict, Tuple, Iterable      # Typing helpers for annotations
import os                                           # PathLike typing support
import regex as re                                  # Regex module with Unicode property support
from collections import defaultdict                 # lets increment counts without checking for missing keys
import multiprocessing                              # For parallel pre-tokenization

# GPT-2 pre-tokenization pattern - splits text into word-like chunks
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Internal helper to pre-tokenize text into byte chunks, respecting special tokens as boundaries
# 1. Split on special tokens first, so those markers act as document boundaries and don’t get mixed into surrounding text. 
# 2. Run GPT‑2 regex pattern over each segment, producing word‑like chunks 
# 3. Each chunk is encoded to UTF‑8 bytes 

# It enforces the pre‑tokenization scheme (GPT‑2 style).
# It cleanly handles special tokens as hard boundaries.
# It converts to bytes, which is required for byte‑level BPE training.

def _iter_pretokens(text: str, special_tokens: List[str], pat: re.Pattern) -> Iterable[bytes]:
    if special_tokens:                              # Split on special tokens first
        delimiter = "|".join(re.escape(tok) for tok in special_tokens) # Build escaped split pattern
        segments = re.split(delimiter, text)        # Split text around special tokens
    else:                                           # No special tokens provided
        segments = [text]                           # Use full text as one segment

    for segment in segments:                        # Process each segment separately
        if not segment:                             # Skip empty segments
            continue                                # Nothing to tokenize
        for match in pat.finditer(segment):         # Find regex matches
            token = match.group(0)                  # Extract matched substring
            if token:                               # Ensure token is non-empty
                yield token.encode("utf-8")         # Return UTF-8 bytes

# Apply a BPE merge to a single token, return new sequence and whether it changed
def merge_key(ids: Tuple[int, ...], pair: Tuple[int, int], idx: int) -> Tuple[Tuple[int, ...], bool]:
    new_ids: List[int] = []                         # Output list of token ids after merge
    i: int = 0                                      # Position pointer over input
    changed: bool = False                           # Track if any merge happened
    while i < len(ids):                             # Continue until end of sequence
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:  # Check if next two tokens match pair
            new_ids.append(idx)                     # Replace pair with new merged token
            i += 2                                  # Skip both merged tokens
            changed = True                          # Mark that we made a change
        else:                                       # No match at this position
            new_ids.append(ids[i])                  # Keep current token as-is
            i += 1                                  # Advance by one
    return tuple(new_ids), changed                  # Return new sequence and change flag

# Helper for multiprocessing: lets each worker pre-tokenize its own file chunk and return counts.
def _pretokenize_chunk(args: Tuple) -> Dict[Tuple[int, ...], int]:  
    """Pre-tokenize a byte range of a file and return token counts."""  
    input_path, start, end, special_tokens = args   # Unpack file path, byte range, and special tokens
    pat = re.compile(GPT2_SPLIT_PATTERN)            # Compile GPT-2 regex for pre-tokenization
    counts: Dict[Tuple[int, ...], int] = {}         # Map from token byte tuples to counts
    with open(input_path, "rb") as f:               # Open file in binary mode for byte-accurate slicing
        f.seek(start)                               # Seek to the start byte offset
        raw = f.read(end - start)                   # Read the specified byte range
    # Trim trailing incomplete UTF-8 bytes so we don't silently drop characters at chunk edges.
    # A UTF-8 continuation byte has the form 10xxxxxx (0x80-0xBF). Walk backwards from the end
    # to find the last leading byte; if it starts a multi-byte sequence that extends past our
    # slice, strip it (the next chunk will pick it up from its start).
    if raw:
        i = len(raw) - 1                            # Start at last byte
        while i >= 0 and (raw[i] & 0xC0) == 0x80:  # Skip continuation bytes (10xxxxxx)
            i -= 1
        if i >= 0:                                  # Found a leading byte
            lead = raw[i]
            if lead >= 0xF0:                        # 4-byte sequence start
                expected = 4
            elif lead >= 0xE0:                      # 3-byte sequence start
                expected = 3
            elif lead >= 0xC0:                      # 2-byte sequence start
                expected = 2
            else:                                   # ASCII byte (0xxxxxxx)
                expected = 1
            if len(raw) - i < expected:             # Sequence extends past our slice
                raw = raw[:i]                       # Trim incomplete sequence
    text = raw.decode("utf-8", errors="ignore")     # Decode bytes to text (now safe at boundaries)
    for token_bytes in _iter_pretokens(text, special_tokens, pat):  # Iterate GPT-2 pre-tokens
        key = tuple(token_bytes)                    # Convert token bytes to a hashable tuple of ints
        counts[key] = counts.get(key, 0) + 1        # Increment count for this token
    return counts                                   # Return per-chunk token frequency map


# Helper for multiprocessing: find safe chunk boundaries so workers don't split documents mid-special-token.
def _find_chunk_boundaries(
    file_path: str | os.PathLike,                   # Path to the input text file
    desired_num_chunks: int,                        # Target number of chunks (usually = num_workers)
    split_special_token: bytes,                     # Byte sequence to align chunk cuts (e.g., <|endoftext|>)
) -> List[int]:
    """Split file into chunks on special token boundaries."""  # Keep chunks aligned to document separators
    with open(file_path, "rb") as f:                # Open file in binary mode for byte-accurate offsets
        f.seek(0, os.SEEK_END)                      # Seek to end to measure file size
        file_size = f.tell()                        # Total file size in bytes
        f.seek(0)                                   # Reset to start of file

        chunk_size = file_size // desired_num_chunks  # Initial equal-sized chunk guess
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]  # Byte offsets for chunks
        boundaries[-1] = file_size                  # Ensure final boundary is end of file

        mini_chunk_size = 4096                      # Read in small blocks to search for token boundary
        overlap = len(split_special_token) - 1      # Overlap between windows to catch tokens spanning reads
        for bi in range(1, len(boundaries) - 1):    # Skip first and last boundaries
            initial_position = boundaries[bi]       # Proposed boundary position
            f.seek(initial_position)                # Seek to proposed boundary
            while True:                             # Scan forward until we find the split token
                mini_chunk = f.read(mini_chunk_size) # Read a small window of bytes
                if mini_chunk == b"":               # Reached EOF without finding token
                    boundaries[bi] = file_size      # Clamp boundary to EOF
                    break                           # Stop scanning this boundary
                found_at = mini_chunk.find(split_special_token)  # Look for the split token
                if found_at != -1:                  # Found a split token in this window
                    boundaries[bi] = initial_position + found_at  # Move boundary to token start
                    break                           # Stop scanning this boundary
                initial_position += mini_chunk_size - overlap  # Advance with overlap to catch spanning tokens

    return sorted(set(boundaries))                  # Return unique, sorted boundary offsets


# Train a byte-level BPE tokenizer from a text file
def train_bpe(
    input_path: str | os.PathLike,                  # Path to training text file
    vocab_size: int,                                # Maximum final vocabulary size (bytes + merges + specials)
    special_tokens: List[str],                      # Special tokens to append to the vocabulary
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:  # Return vocab and merges

    assert vocab_size > 0, "vocab_size must be positive"  # Validate vocab_size is positive

    num_special: int = len(special_tokens)          # Count how many special tokens will be added
    num_merges: int = vocab_size - 256 - num_special # Number of merges allowed after reserving space
    assert num_merges >= 0, f"vocab_size={vocab_size} is too small for 256 byte tokens + {num_special} special tokens"

    # --- Phase 1: Pre-tokenize with multiprocessing ---
    input_path = str(input_path)
    file_size = os.path.getsize(input_path)

    # Use multiprocessing for files larger than 1MB
    num_workers = max(1, os.cpu_count() or 1)
    if file_size > 1_000_000 and num_workers > 1:
        # Determine split token for chunk boundaries
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        boundaries = _find_chunk_boundaries(input_path, num_workers, split_token)

        chunk_args = [
            (input_path, boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
        ]

        with multiprocessing.Pool(num_workers) as pool:
            chunk_results = pool.map(_pretokenize_chunk, chunk_args)

        # Merge all chunk counts
        pre_token_counts: Dict[Tuple[int, ...], int] = {}
        for chunk_counts in chunk_results:
            for key, count in chunk_counts.items():
                pre_token_counts[key] = pre_token_counts.get(key, 0) + count
    else:
        # Small file: single-threaded (avoids multiprocessing overhead)
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        pre_token_counts = {}
        pat = re.compile(GPT2_SPLIT_PATTERN)
        for token_bytes in _iter_pretokens(text, special_tokens, pat):
            key = tuple(token_bytes)
            pre_token_counts[key] = pre_token_counts.get(key, 0) + 1

    # --- Phase 2: Build data structures for incremental BPE ---
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Convert to indexed mutable pre-token lists with weights
    pt_tokens: List[List[int]] = []       # mutable token sequences
    pt_weights: List[int] = []            # frequency weights
    for token_tuple, weight in pre_token_counts.items():
        pt_tokens.append(list(token_tuple))
        pt_weights.append(weight)
    del pre_token_counts

    # Build initial pair counts and reverse index
    pair_counts: Dict[Tuple[int, int], int] = {}
    pair_to_pts: Dict[Tuple[int, int], set] = defaultdict(set)

    for idx in range(len(pt_tokens)):
        tokens = pt_tokens[idx]
        w = pt_weights[idx]
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + w
            pair_to_pts[pair].add(idx)

    merges_order: List[Tuple[int, int]] = []

    # --- Phase 3: Incremental BPE merge loop ---
    for i in range(num_merges):
        if not pair_counts:
            break

        # Find most frequent pair with tie-breaking by largest byte representation
        best: Tuple[int, int] = None
        best_count: int = -1
        best_bytes: Tuple[bytes, bytes] = None
        for p, count in pair_counts.items():
            if count > best_count:
                best = p
                best_count = count
                best_bytes = (vocab[p[0]], vocab[p[1]])
            elif count == best_count:
                p_bytes = (vocab[p[0]], vocab[p[1]])
                if p_bytes > best_bytes:
                    best = p
                    best_bytes = p_bytes

        new_id: int = 256 + i
        a, b = best

        # Update vocab
        vocab[new_id] = vocab[a] + vocab[b]
        merges_order.append(best)

        # Get affected pre-tokens and remove the merged pair from tracking
        affected = pair_to_pts.pop((a, b), set())
        pair_counts.pop((a, b), None)

        # Apply merge incrementally to each affected pre-token
        for pt_idx in affected:
            tokens = pt_tokens[pt_idx]
            w = pt_weights[pt_idx]

            j = 0
            while j < len(tokens) - 1:
                if tokens[j] == a and tokens[j + 1] == b:
                    # Remove old left neighbor pair count
                    if j > 0:
                        old_left = (tokens[j - 1], a)
                        pair_counts[old_left] = pair_counts.get(old_left, 0) - w
                        if pair_counts[old_left] <= 0:
                            pair_counts.pop(old_left, None)

                    # Remove old right neighbor pair count
                    if j + 2 < len(tokens):
                        old_right = (b, tokens[j + 2])
                        pair_counts[old_right] = pair_counts.get(old_right, 0) - w
                        if pair_counts[old_right] <= 0:
                            pair_counts.pop(old_right, None)

                    # Apply merge in-place
                    tokens[j] = new_id
                    del tokens[j + 1]

                    # Add new left neighbor pair
                    if j > 0:
                        new_left = (tokens[j - 1], new_id)
                        pair_counts[new_left] = pair_counts.get(new_left, 0) + w
                        pair_to_pts.setdefault(new_left, set()).add(pt_idx)

                    # Add new right neighbor pair
                    if j + 1 < len(tokens):
                        new_right = (new_id, tokens[j + 1])
                        pair_counts[new_right] = pair_counts.get(new_right, 0) + w
                        pair_to_pts.setdefault(new_right, set()).add(pt_idx)

                    # Don't increment j: after merge, tokens[j]=new_id.
                    # Since new_id != a (new_id >= 256+i, a < 256+i for previously
                    # existing tokens), the next check will fall to else and increment.
                else:
                    j += 1

    # --- Phase 4: Add special tokens ---
    next_id: int = 256 + len(merges_order)
    for sp in special_tokens:
        vocab[next_id] = sp.encode("utf-8")
        next_id += 1

    merges_bytes: List[Tuple[bytes, bytes]] = [
        (vocab[a_id], vocab[b_id])
        for (a_id, b_id) in merges_order
    ]

    return vocab, merges_bytes
