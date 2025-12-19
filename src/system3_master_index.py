#!/usr/bin/env python3
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


if __name__ == "__main__":
    main()
