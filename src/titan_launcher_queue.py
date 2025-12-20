#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import multiprocessing

from titan_orchestrator_v3 import run_titan_pipeline  # this is your full pipeline[file:4ea99728-0ccf-4bdc-ad1b-acda7d34efeb]

def is_csv(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}

def gather_inputs(user_input: str):
    parts = [x.strip() for x in user_input.split(",") if x.strip()]
    valid = []
    for item in parts:
        p = Path(item).expanduser()
        if p.exists():
            valid.append(p)
        else:
            print(f"Invalid path: {item}")
    return valid

def expand_to_csvs(path: Path):
    if path.is_file():
        return [path] if is_csv(path) else []
    csvs = []
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            if is_csv(fp):
                csvs.append(fp)
    return sorted(csvs)

def main():
    print("=" * 70)
    print("TITAN OMNI QUEUE – FULL PIPELINE")
    print("RS + Research + Evidence + Results + Master Index")
    print("=" * 70)
    print("Paste or drag‑drop file/folder paths (comma‑separated).")
    print("Type START when your queue is ready, or EXIT to quit.\n")

    queue = []

    # 1) Build queue interactively
    while True:
        entry = input("QUEUE > Add path(s) or START/EXIT: ").strip()
        if not entry:
            continue
        if entry.upper() == "EXIT":
            return
        if entry.upper() == "START":
            if not queue:
                print("Queue is empty!")
                continue
            break

        new_paths = gather_inputs(entry)
        if not new_paths:
            continue
        queue.extend(new_paths)
        print("Current queue:")
        for p in queue:
            print(" -", p)

    # 2) Output root + optional config
    output_root = input(
        "\nOUTPUT ROOT (e.g. /Users/robin/titan-suite/titan_results) > "
    ).strip() or "titan_results"
    output_root = Path(output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = input("CONFIG (optional, press Enter to skip) > ").strip() or None

    # 3) Sequential full-pipeline runs for all queued CSVs
    print("\nSTARTING BATCH...")
    run_index = 1
    for item in queue:
        csv_files = expand_to_csvs(item)
        if not csv_files:
            print(f"Skipping {item} – no CSV/TSV/TXT files found.")
            continue

        for csv_path in csv_files:
            run_name = f"TITAN_RUN_{csv_path.stem}_{run_index:03d}"
            run_dir = (output_root / run_name).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "-" * 70)
            print(f"🧪 DATASET: {csv_path}")
            print(f"📂 RUN DIR: {run_dir}")
            print("-" * 70)

            try:
                run_titan_pipeline(
                    input_source=str(csv_path),
                    run_dir=str(run_dir),
                    config_path=config_path,
                )
            except Exception as e:
                print(f"✗ ERROR on {csv_path}: {e}")
            run_index += 1

    print("\n✓ BATCH COMPLETE. All TITAN runs are under:", output_root)

if __name__ == "__main__":
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn", force=True)
        multiprocessing.freeze_support()
    main()
