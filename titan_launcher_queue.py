#!/usr/bin/env python3
"""
TITAN RS - Interactive Queue Launcher

Mode:
- Lets you interactively build a queue of files/folders.
- On START, runs titan_orchestrator_v3.py once for each queued path.

Usage:
    python titan_launcher_queue.py
"""

import os
import sys
import subprocess
from pathlib import Path

# CONFIG
ORCHESTRATOR = "titan_orchestrator_v3.py"
DEFAULT_ENGINES = "omni,results,research,evidence"
DEFAULT_OUTPUT_DIR = "~/titan_results"


def resolve(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def main():
    print("=" * 78)
    print("TITAN RS INTERACTIVE QUEUE LAUNCHER")
    print("Mode: Queue files/folders, then auto-run TITAN Universal Orchestrator v3")
    print("=" * 78)

    queue = []

    print("\nINSTRUCTIONS")
    print("- Paste full path to a DATA FILE (.csv, .xlsx, etc.) or FOLDER per line.")
    print("- Press Enter on an empty line to finish adding items.")
    print("- Type START anytime to stop adding and launch all jobs.")
    print("- Type EXIT to cancel.\n")

    while True:
        entry = input("QUEUE > ").strip().replace("'", "").replace('"', "")
        if not entry:
            # Empty line: stop adding, proceed if we have items
            if queue:
                break
            else:
                print("Queue is empty. Add at least one item or type EXIT.")
                continue

        upper = entry.upper()
        if upper == "EXIT":
            print("Exiting without running anything.")
            return
        if upper == "START":
            if not queue:
                print("Queue is empty. Add at least one item first.")
                continue
            break

        path = Path(entry).expanduser()
        if path.is_file():
            queue.append(str(path.resolve()))
            print(f"  Added FILE : {path.name}")
        elif path.is_dir():
            queue.append(str(path.resolve()))
            print(f"  Added FOLDER: {path}")
        else:
            print("  Invalid path. Please provide an existing file or folder.")

    print(f"\nFINAL QUEUE ({len(queue)} items):")
    for i, p in enumerate(queue, 1):
        print(f"  {i}. {p}")

    # Confirm engines and output dir
    engines = input(
        f"\nEngines to run [default {DEFAULT_ENGINES}]: "
    ).strip() or DEFAULT_ENGINES
    output_dir = input(
        f"Base output dir [default {DEFAULT_OUTPUT_DIR}]: "
    ).strip() or DEFAULT_OUTPUT_DIR

    engines = engines.replace(" ", "")
    output_dir = resolve(output_dir)

    print("\nLAUNCHING JOBS...")
    print(f"  Orchestrator : {ORCHESTRATOR}")
    print(f"  Engines      : {engines}")
    print(f"  Output base  : {output_dir}\n")

    for path in queue:
        # For folders, titan_orchestrator_v3 will recurse if you pass --recursive
        is_dir = Path(path).is_dir()
        test_name = Path(path).stem if not is_dir else Path(path).name

        cmd = [
            sys.executable,
            ORCHESTRATOR,
            "--input",
            path,
            "--engine",
            engines,
            "--test-name",
            test_name,
            "--output-dir",
            output_dir,
        ]
        # Optional: auto-recursive for folders
        if is_dir:
            cmd.append("--recursive")

        print("=" * 78)
        print(f"[RUN] {test_name}")
        print("CMD:", " ".join(cmd))
        print("=" * 78)

        try:
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=False,
                check=False,
            )
            if result.returncode == 0:
                print(f"[OK] {test_name} completed.\n")
            else:
                print(f"[WARN] {test_name} finished with code {result.returncode}.\n")
        except KeyboardInterrupt:
            print("\n[ABORTED] User interrupted.")
            break
        except Exception as e:
            print(f"[ERROR] Failed to run {test_name}: {e}\n")

    print("All queued jobs processed.")


if __name__ == "__main__":
    main()
