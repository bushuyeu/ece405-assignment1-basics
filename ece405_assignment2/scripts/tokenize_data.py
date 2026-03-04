"""
Tokenize filtered data with GPT-2 tokenizer and save as uint16 numpy array.

Usage:
    python scripts/tokenize_data.py --input data/filtered/ --output data/tokenized/train.bin

Each line in the input files is one document. The GPT-2 end-of-sequence token
<|endoftext|> is appended after each document.

Streaming implementation: processes one file at a time to avoid loading all data
into memory.
"""

from __future__ import annotations

import argparse
import multiprocessing
import pathlib
import struct
import time

import tiktoken
from tqdm import tqdm


def tokenize_line(line: str) -> list[int]:
    """Tokenize a single line and append EOS token."""
    text = line.strip()
    if not text:
        return []
    tokens = enc.encode(text, disallowed_special=())
    tokens.append(EOT)
    return tokens


def init_worker():
    """Initialize tokenizer in each worker process."""
    global enc, EOT
    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.eot_token


def main():
    parser = argparse.ArgumentParser(description="Tokenize filtered data with GPT-2 tokenizer")
    parser.add_argument("--input", required=True, help="Input file or directory of filtered text files")
    parser.add_argument("--output", required=True, help="Output .bin file (numpy uint16)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: cpu_count)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if input_path.is_dir():
        files = sorted(
            fp for fp in input_path.glob("*")
            if fp.is_file() and fp.suffix not in (".json", ".bin")
        )
    else:
        files = [input_path]

    print(f"Found {len(files)} input files")

    # Initialize tokenizer for token count
    enc = tiktoken.get_encoding("gpt2")
    print(f"EOT token ID: {enc.eot_token}")

    num_workers = args.workers or multiprocessing.cpu_count()
    print(f"Workers: {num_workers}")

    start_time = time.time()
    total_tokens = 0
    total_docs = 0

    # Stream: process one file at a time, write tokens to output incrementally
    with open(output_path, "wb") as out_f:
        pool = multiprocessing.Pool(num_workers, initializer=init_worker)

        for file_idx, fp in enumerate(files):
            with open(fp) as f:
                lines = [l for l in f if l.strip()]

            if not lines:
                continue

            file_tokens = 0
            for token_list in pool.imap(tokenize_line, lines, chunksize=200):
                if token_list:
                    # Write as uint16 directly
                    out_f.write(struct.pack(f"<{len(token_list)}H", *token_list))
                    file_tokens += len(token_list)
                    total_docs += 1

            total_tokens += file_tokens

            if (file_idx + 1) % 100 == 0 or file_idx == len(files) - 1:
                elapsed = time.time() - start_time
                print(
                    f"  [{file_idx+1}/{len(files)}] "
                    f"{total_docs:,} docs, {total_tokens:,} tokens, "
                    f"{elapsed:.0f}s elapsed"
                )

        pool.close()
        pool.join()

    elapsed = time.time() - start_time
    size_mb = output_path.stat().st_size / 1e6

    print()
    print("=" * 60)
    print("TOKENIZATION SUMMARY")
    print("=" * 60)
    print(f"Files processed:  {len(files)}")
    print(f"Documents:        {total_docs:,}")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Output file:      {output_path} ({size_mb:.1f} MB)")
    print(f"Time:             {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
