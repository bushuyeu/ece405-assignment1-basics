"""
Inspect filtered and discarded documents from the filter pipeline.

Runs the full filter pipeline on a WET file, randomly samples 5 kept
and 5 discarded documents, and prints excerpts with filter stage info.

Usage:
    python scripts/inspect_filtered.py --wet_file data/example.warc.wet.gz --models_dir data/
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

from warcio.archiveiterator import ArchiveIterator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cs336_data.language_identification import identify_language, set_lid_model_path
from cs336_data.quality_filters import gopher_quality_filter
from cs336_data.pii import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.harmful_content import (
    classify_nsfw,
    classify_toxic_speech,
    set_nsfw_model_path,
    set_toxic_model_path,
)

LANG_THRESHOLD = 0.80
QUALITY_THRESHOLD = 0.50
NSFW_THRESHOLD = 0.50
TOXIC_THRESHOLD = 0.50
MIN_TEXT_LENGTH = 100


def classify_document(text: str, classify_quality_fn=None):
    """Run a document through the filter pipeline, returning (kept, reason, details)."""
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False, "empty_short", {"length": len(text.strip()) if text else 0}

    text = text.strip()

    lang, lang_score = identify_language(text)
    if lang != "en" or lang_score < LANG_THRESHOLD:
        return False, "language", {"lang": lang, "score": round(lang_score, 4)}

    if not gopher_quality_filter(text):
        return False, "gopher", {}

    if classify_quality_fn is not None:
        label, score = classify_quality_fn(text)
        if label != "wiki" or score < QUALITY_THRESHOLD:
            return False, "quality_classifier", {"label": label, "score": round(score, 4)}

    nsfw_label, nsfw_score = classify_nsfw(text)
    if nsfw_label == "nsfw" and nsfw_score >= NSFW_THRESHOLD:
        return False, "nsfw", {"score": round(nsfw_score, 4)}

    toxic_label, toxic_score = classify_toxic_speech(text)
    if toxic_label == "toxic" and toxic_score >= TOXIC_THRESHOLD:
        return False, "toxic", {"score": round(toxic_score, 4)}

    # PII masking
    masked, n_emails = mask_emails(text)
    masked, n_phones = mask_phone_numbers(masked)
    masked, n_ips = mask_ips(masked)

    return True, "kept", {
        "pii": {"emails": n_emails, "phones": n_phones, "ips": n_ips},
        "masked_text": masked,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect filtered/discarded documents")
    parser.add_argument("--wet_file", required=True, help="Path to WET file")
    parser.add_argument("--models_dir", default=None, help="Directory containing model files")
    parser.add_argument("--n_kept", type=int, default=5, help="Number of kept examples to show")
    parser.add_argument("--n_discarded", type=int, default=5, help="Number of discarded examples to show")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_quality", action="store_true", help="Skip quality classifier")
    parser.add_argument("--max_records", type=int, default=None, help="Max records to process")
    parser.add_argument("--output_json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # Configure model paths
    if args.models_dir:
        models_dir = pathlib.Path(args.models_dir)
        lid_path = models_dir / "lid.176.bin"
        if lid_path.exists():
            set_lid_model_path(str(lid_path))
        nsfw_path = models_dir / "dolma_fasttext_nsfw_jigsaw_model.bin"
        if nsfw_path.exists():
            set_nsfw_model_path(str(nsfw_path))
        toxic_path = models_dir / "dolma_fasttext_hatespeech_jigsaw_model.bin"
        if toxic_path.exists():
            set_toxic_model_path(str(toxic_path))
        quality_path = models_dir / "quality_classifier.bin"
        if quality_path.exists() and not args.no_quality:
            from cs336_data.quality_filters import set_quality_model_path
            set_quality_model_path(str(quality_path))

    classify_quality_fn = None
    if not args.no_quality:
        try:
            from cs336_data.quality_filters import classify_quality, _get_quality_model
            _get_quality_model()
            classify_quality_fn = classify_quality
        except Exception:
            print("Quality classifier not available, skipping", file=sys.stderr)

    random.seed(args.seed)

    kept_docs = []
    discarded_docs = []

    print(f"Processing {args.wet_file}...")
    with open(args.wet_file, "rb") as f:
        for i, record in enumerate(ArchiveIterator(f)):
            if record.rec_type != "conversion":
                continue
            if args.max_records and i >= args.max_records:
                break

            url = record.rec_headers.get_header("WARC-Target-URI") or "unknown"
            text = record.content_stream().read().decode("utf-8", errors="replace")

            is_kept, reason, details = classify_document(text, classify_quality_fn)

            doc_info = {
                "url": url,
                "text": text.strip()[:2000],
                "full_length": len(text.strip()) if text else 0,
                "reason": reason,
                "details": {k: v for k, v in details.items() if k != "masked_text"},
            }
            if is_kept:
                doc_info["masked_excerpt"] = details.get("masked_text", text)[:2000]
                kept_docs.append(doc_info)
            else:
                discarded_docs.append(doc_info)

    print(f"\nTotal kept: {len(kept_docs)}, discarded: {len(discarded_docs)}")

    # Sample
    n_kept = min(args.n_kept, len(kept_docs))
    n_discarded = min(args.n_discarded, len(discarded_docs))
    sampled_kept = random.sample(kept_docs, n_kept) if kept_docs else []
    sampled_discarded = random.sample(discarded_docs, n_discarded) if discarded_docs else []

    # Print kept examples
    print("\n" + "=" * 80)
    print(f"KEPT EXAMPLES ({n_kept} random)")
    print("=" * 80)
    for i, doc in enumerate(sampled_kept):
        print(f"\n--- Kept Example {i+1} ---")
        print(f"URL: {doc['url']}")
        print(f"Length: {doc['full_length']} chars")
        print(f"PII: {doc['details'].get('pii', {})}")
        excerpt = doc.get("masked_excerpt", doc["text"])
        # Show first 500 chars
        print(f"Excerpt (first 500 chars):")
        print(excerpt[:500])
        print()

    # Print discarded examples
    print("\n" + "=" * 80)
    print(f"DISCARDED EXAMPLES ({n_discarded} random)")
    print("=" * 80)
    for i, doc in enumerate(sampled_discarded):
        print(f"\n--- Discarded Example {i+1} ---")
        print(f"URL: {doc['url']}")
        print(f"Length: {doc['full_length']} chars")
        print(f"Filter: {doc['reason']}")
        print(f"Details: {doc['details']}")
        print(f"Excerpt (first 500 chars):")
        print(doc["text"][:500])
        print()

    # Save to JSON if requested
    if args.output_json:
        output = {
            "kept_sample": sampled_kept,
            "discarded_sample": sampled_discarded,
            "summary": {
                "total_kept": len(kept_docs),
                "total_discarded": len(discarded_docs),
            },
        }
        # Remove masked_excerpt from JSON (too large)
        for doc in output["kept_sample"]:
            doc.pop("masked_excerpt", None)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
