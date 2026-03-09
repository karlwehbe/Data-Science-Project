#!/usr/bin/env python3
"""
split_by_category.py

Reads an annotated CSV (expects header with 'title,description,...,categories,...')
Creates up to 100-row CSV files per category in ./categories/
"""

import os
import sys
import re
import pandas as pd

def safe_filename(name: str) -> str:
    # remove problematic characters and shorten
    name = name.strip().lower()
    name = re.sub(r'[/\\:*?"<>|]', '_', name)
    name = re.sub(r'\s+', '_', name)
    if len(name) > 80:
        name = name[:80]
    return name or "unknown"

def split_csv_in_categories(input_csv: str, out_dir: str = "categories", max_per_cat: int = 100):
    os.makedirs(out_dir, exist_ok=True)

    # read CSV robustly
    df = pd.read_csv(input_csv, engine="python", encoding="utf-8", on_bad_lines="skip")
    df = df.fillna("")  # avoid NaNs

    if "categories" not in df.columns:
        raise ValueError("Input CSV must have a 'categories' column")

    # If some rows have multiple categories separated (rare for your use), take the first
    # But we'll keep full string if it's single category.

    grouped = df.groupby("categories", sort=False)

    written = {}
    for cat, group in grouped:
        if cat is None or str(cat).strip() == "":
            cat = "Unknown"
        safe = safe_filename(str(cat))
        out_path = os.path.join(out_dir, f"{safe}.csv")
        # take up to max_per_cat rows, preserving original order
        subset = group.head(max_per_cat)
        subset.to_csv(out_path, index=False)
        written[cat] = {"file": out_path, "n_rows": len(subset)}

    print(f"Wrote {len(written)} category files to {out_dir}")
    for k,v in written.items():
        print(f" - {k}: {v['n_rows']} rows -> {v['file']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_by_category.py articles_annotated.csv")
        sys.exit(1)
    input_csv = sys.argv[1]
    split_csv_in_categories(input_csv)

