#!/usr/bin/env python3
<<<<<<< HEAD
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
=======
"""
TITAN System 3 – Master Index Builder

- Asks (or accepts --input) a single test folder.
- Scans that folder (and subfolders) for TITAN charts and metadata.
- Creates TITAN_Master_Index_<foldername>.xlsx in the SAME folder.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
META_FILENAMES = {
    "stats_summary.txt",
    "stats_summary.csv",
    "metadata.json",
    "omni_metadata.json",
    "OMNI_SUMMARY_REPORT.txt",
    "MANIFEST.txt",
}


def find_charts_and_meta(root: Path) -> Dict[str, List[Path]]:
    charts: List[Path] = []
    meta: List[Path] = []

    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        for fn in filenames:
            p = d / fn
            ext = p.suffix.lower()

            if ext in IMAGE_EXTENSIONS:
                if fn.lower().startswith(
                    (
                        "bar_",
                        "box_",
                        "dist_",
                        "violin_",
                        "hist_",
                        "correlation_",
                        "pairplot",
                        "calibration",
                        "feature_",
                    )
                ):
                    charts.append(p)

            if fn in META_FILENAMES:
                meta.append(p)

    return {"charts": charts, "meta": meta}


def build_index_dataframe(charts: List[Path], root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(charts):
        rel = p.relative_to(root)
        sheet_name = p.stem

        rows.append(
            {
                "Chart Name": p.name,
                "Type": p.suffix.upper(),
                "Size KB": round(p.stat().st_size / 1024, 1),
                "Relative Path": str(rel),
                "Sheet Tab": sheet_name,
                "Notes": "Click to jump or embed",
            }
        )

    return pd.DataFrame(rows)


def build_metadata_sheet(charts: List[Path], meta_files: List[Path], root: Path) -> pd.DataFrame:
    testname = root.name
    timestamp = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    chart_sources = ", ".join([p.name for p in sorted(charts)])

    rows = [
        {"KEY": "testname", "VALUE": testname},
        {"KEY": "timestamp", "VALUE": timestamp},
        {"KEY": "totalcharts", "VALUE": len(charts)},
        {"KEY": "totalsheets", "VALUE": len(charts) + 1},
        {"KEY": "chartsources", "VALUE": chart_sources},
    ]

    for m in sorted(meta_files):
        rel = m.relative_to(root)
        rows.append({"KEY": f"meta:{m.name}", "VALUE": str(rel)})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build TITAN Master Index for a single test folder.")
    parser.add_argument(
        "--input",
        type=str,
        help="Test folder (or *_MASTER folder) containing TITAN charts. "
             "If omitted, you will be prompted.",
    )
    args = parser.parse_args()

    if args.input:
        root = Path(args.input).expanduser()
    else:
        raw = input("Enter TITAN test folder for master index (e.g., ./Titan_Evidence_Suite_MASTER): ").strip()
        if not raw:
            raise SystemExit("No directory provided; exiting.")
        root = Path(raw).expanduser()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {root}")

    found = find_charts_and_meta(root)
    charts = found["charts"]
    meta_files = found["meta"]

    print(f"Found {len(charts)} chart images and {len(meta_files)} metadata files under: {root}")

    df_index = build_index_dataframe(charts, root)
    df_meta = build_metadata_sheet(charts, meta_files, root)

    out_path = root / f"TITAN_Master_Index_{root.name}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_index.to_excel(writer, sheet_name="INDEX", index=False)
        df_meta.to_excel(writer, sheet_name="METADATA", index=False)

    print(f"✅ Master index written to: {out_path}")
>>>>>>> e6e2781d57008629045fc7343dc6a934a2ba6880


if __name__ == "__main__":
    main()
