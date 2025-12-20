#!/usr/bin/env python3
import os
import argparse
import shutil
from pathlib import Path

import pandas as pd


def aggregate_results(source_dirs, master_dir):
    """
    Copy all files from source_dirs into master_dir, preserving
    a top-level folder per source to avoid name collisions.
    """
    master_dir = Path(master_dir)
    master_dir.mkdir(parents=True, exist_ok=True)

    for src in source_dirs:
        src_path = Path(src)
        if not src_path.exists():
            continue

        for root, dirs, files in os.walk(src_path):
            rel = Path(root).relative_to(src_path)
            dest_root = master_dir / src_path.name / rel
            dest_root.mkdir(parents=True, exist_ok=True)

            for f in files:
                src_file = Path(root) / f
                dest_file = dest_root / f
                if dest_file.exists():
                    dest_file = dest_root / f"{src_path.name}__{f}"
                shutil.copy2(src_file, dest_file)

    print(f"✅ Aggregated all results into: {master_dir}")


def build_master_index(master_dir, output_excel):
    """
    Walk master_dir and build a simple Excel index:
    columns = [source_root, relative_path, filename, full_path].
    """
    master_dir = Path(master_dir)
    rows = []

    for root, dirs, files in os.walk(master_dir):
        root_path = Path(root)
        # first element under master is treated as "source_root"
        try:
            source_root = root_path.relative_to(master_dir).parts[0]
        except IndexError:
            source_root = ""

        for f in files:
            full_path = root_path / f
            rel_path = full_path.relative_to(master_dir)
            rows.append(
                {
                    "source_root": source_root,
                    "relative_path": str(rel_path),
                    "filename": f,
                    "full_path": str(full_path),
                }
            )

    if not rows:
        print(f"⚠️ No files found under {master_dir}, index not created.")
        return

    df = pd.DataFrame(rows)
    output_excel = Path(output_excel)
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)
    print(f"✅ Master index written to: {output_excel}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate TITAN results and build a master Excel index."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=False,
        default="/Users/robin/Desktop/Titan_Final_Protocol/testdata",
        help="Root directory containing TITAN result folders.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=False,
        default="/Users/robin/Desktop/Titan_Final_Protocol/testdata/MASTER_TITAN_RESULTS/TITAN_Master_Index_all.xlsx",
        help="Path to output Excel file (will create parent dirs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Treat --input as the root and use its immediate subfolders as sources
    input_root = Path(args.input)
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    source_dirs = []
    for child in input_root.iterdir():
        if child.is_dir():
            source_dirs.append(str(child))
        else:
            # allow single CSV / file in root as well
            source_dirs.append(str(child))

    master_dir = Path(args.output).parent
    aggregate_results(source_dirs, master_dir)
    build_master_index(master_dir, args.output)


if __name__ == "__main__":
    main()
