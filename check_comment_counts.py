#!/usr/bin/env python3
"""Compare retrieved YouTube comment counts stored in `comments/*.json` against
`scraping_progress.txt`.

The script produces a CSV-style report to STDOUT with the following columns:

index, title, artist, expected_count, retrieved_count, delta, json_file

Usage
-----
$ python check_comment_counts.py \
    --comments-dir ./comments \
    --progress-file scraping_progress.txt

Notes
-----
1. The matching between a song in the progress file and a JSON file is done via
   a sanitised key of *<title>_<artist>* with whitespace collapsed to
   underscores and non-alphanumeric characters stripped.
2. If no matching JSON file is found the retrieved_count is reported as 0 and
   delta equals -expected_count.
3. Any extra JSON files not referenced in `scraping_progress.txt` are reported
   at the end with `index` set to "N/A".
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

def sanitise(text: str) -> str:
    """Return a normalised identifier for a *title* or *artist*.

    Rules:
    1. Lower-case.
    2. Replace any sequence of whitespace with a single underscore.
    3. Strip characters that are not alphanumeric, underscore or hyphen.
    """
    # Lower-case and replace whitespace with single underscore
    cleaned = re.sub(r"\s+", "_", text.lower().strip())
    # Remove all characters that are not word char or hyphen
    cleaned = re.sub(r"[^\w-]", "", cleaned)
    return cleaned


def build_json_index(comments_dir: Path) -> Dict[str, Tuple[Path, int]]:
    """Return mapping *key* -> (filepath, count).*key* is *sanitised(title)_sanitised(artist).*"""
    index: Dict[str, Tuple[Path, int]] = {}

    for json_path in comments_dir.glob("*.json"):
        stem = json_path.stem  # e.g. "A_Sky_Full_of_Stars_Coldplay"
        # Split on last underscore to separate title and artist heuristically.
        if "_" not in stem:
            continue  # skip malformed

        # Attempt to split title and artist by last underscore.
        title_part, artist_part = stem.rsplit("_", 1)
        key = f"{sanitise(title_part.replace('_', ' '))}_{sanitise(artist_part)}"

        # Count comments in file – minimal memory usage: stream lines.
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            count = len(data)
        except Exception as e:
            print(f"WARNING: Failed to read {json_path}: {e}", file=sys.stderr)
            count = 0

        index[key] = (json_path, count)

    return index

# ----------------------------------------------------------------------------


def compare_counts(comments_dir: Path, progress_file: Path) -> None:
    json_index = build_json_index(comments_dir)

    writer = csv.writer(sys.stdout, delimiter=",", lineterminator="\n")
    writer.writerow([
        "index",
        "title",
        "artist",
        "expected_count",
        "retrieved_count",
        "delta",
        "json_file",
    ])

    seen_keys = set()

    with open(progress_file, "r", encoding="utf-8") as pf:
        for line in pf:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                # Unexpected format – skip
                continue

            idx, _url, title, artist, *_rest = parts
            expected_count_str = parts[6] if len(parts) > 6 else "0"
            try:
                expected_count = int(expected_count_str)
            except ValueError:
                expected_count = 0

            key = f"{sanitise(title)}_{sanitise(artist)}"

            json_path: str | Path | None = None
            retrieved_count = 0
            if key in json_index:
                json_path, retrieved_count = json_index[key]
                seen_keys.add(key)

            delta = retrieved_count - expected_count

            writer.writerow([
                idx,
                title,
                artist,
                expected_count,
                retrieved_count,
                delta,
                str(json_path) if json_path else "(missing)",
            ])

    # Report any extra JSON files that were not referenced in progress file.
    for key, (json_path, retrieved_count) in json_index.items():
        if key in seen_keys:
            continue
        writer.writerow([
            "N/A",
            json_path.stem.rsplit("_", 1)[0].replace("_", " "),
            json_path.stem.rsplit("_", 1)[1],
            0,
            retrieved_count,
            retrieved_count,
            str(json_path),
        ])

# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YouTube comment counts with scraping progress log.")
    parser.add_argument(
        "--comments-dir",
        type=Path,
        default=Path("./comments"),
        help="Directory containing *.json comment files.",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("scraping_progress.txt"),
        help="Path to scraping_progress.txt",
    )

    args = parser.parse_args()

    if not args.comments_dir.is_dir():
        parser.error(f"comments directory '{args.comments_dir}' does not exist.")

    if not args.progress_file.is_file():
        parser.error(f"progress file '{args.progress_file}' does not exist.")

    compare_counts(args.comments_dir, args.progress_file)


if __name__ == "__main__":
    main() 