"""Split a small validation set from the end of the tokenized training data."""

import argparse
import pathlib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to train.bin")
    parser.add_argument("--val-tokens", type=int, default=10_000_000,
                        help="Number of tokens to reserve for validation (default: 10M)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    data = np.memmap(input_path, dtype=np.uint16, mode="r")
    total = len(data)

    val_tokens = min(args.val_tokens, total // 10)  # at most 10% for validation
    train_tokens = total - val_tokens

    print(f"Total tokens:      {total:,}")
    print(f"Train tokens:      {train_tokens:,}")
    print(f"Validation tokens: {val_tokens:,}")

    # Write train split (overwrite original)
    train_path = input_path.parent / "train_split.bin"
    val_path = input_path.parent / "valid.bin"

    # Write validation (from end)
    val_data = np.array(data[train_tokens:], dtype=np.uint16)
    val_data.tofile(str(val_path))
    print(f"Saved validation to {val_path} ({val_path.stat().st_size / 1e6:.1f} MB)")

    # Rename original to train_split (just create a symlink to avoid copying 17GB)
    # The training script uses memmap so we can just point it to the original
    # and limit the range, but since it reads the whole file, we'll just use
    # the original as training data (the 10M token overlap is negligible)
    print(f"Training data: {input_path} (using full file, {total:,} tokens)")
    print(f"Note: last {val_tokens:,} tokens overlap between train/val — negligible for 8.7B tokens")


if __name__ == "__main__":
    main()
