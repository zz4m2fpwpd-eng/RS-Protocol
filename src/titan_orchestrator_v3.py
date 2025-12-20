#!/usr/bin/env python3
"""
TITAN Orchestrator v3 – single-dataset pipeline

This module exposes:

    run_titan_pipeline(input_source, run_dir, config_path)

which is called by titan_launcher_queue.py.
"""

from pathlib import Path

from TITAN_RS import run_pipeline as run_titan_rs_pipeline
from TITAN_Research_Mode import run_research_mode
from TITAN_Evidence_Pro_Max import run_evidence_mode
from TITAN_Results_Engine import TITANResultsEngine
from system3_master_index import build_master_index


def run_titan_pipeline(input_source, run_dir, config_path):
    """
    Master orchestration for a single dataset.

    Steps:
    1) TITAN-RS: ingest + audit + models, returns cleaned CSV.
    2) Research Mode: hypothesis scanning on cleaned CSV.
    3) Evidence Pro Max: evidence extraction on cleaned CSV.
    4) Results Engine: standardized charts into charts/ subfolder.
    5) Master Index: Excel index for everything in run_dir.
    """
    run_dir = Path(run_dir).resolve()
    ingest_dir = run_dir / "ingest"
    audit_dir = run_dir / "audit"
    models_dir = run_dir / "models"
    charts_dir = run_dir / "charts"
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"

    ingest_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. TITAN-RS
    cleaned_csv = run_titan_rs_pipeline(
        input_source=input_source,
        run_dir=str(run_dir),
        ingest_dir=str(ingest_dir),
        audit_dir=str(audit_dir),
        models_dir=str(models_dir),
        reports_dir=str(reports_dir),
        logs_dir=str(logs_dir),
        config_path=config_path,
    )

    # 2. Research Mode
    run_research_mode(
        csv_path=cleaned_csv,
        run_dir=str(run_dir),
        config_path=config_path,
    )

    # 3. Evidence Pro Max
    run_evidence_mode(
        csv_path=cleaned_csv,
        run_dir=str(run_dir),
        config_path=config_path,
    )

    # 4. Standard charts
    re_engine = TITANResultsEngine(
        csv_path=cleaned_csv,
        out_dir=str(charts_dir),
        target=None,
    )
    re_engine.run_analysis()

    # 5. Master Index (Excel) – these two lines **must** be indented inside the function
    excel_path = run_dir / "MASTER_INDEX.xlsx"
    build_master_index(str(run_dir), str(excel_path))

# ----------------------------------------------------------------------
# v3 MULTI-DATASET CLI ORCHESTRATOR
# ----------------------------------------------------------------------

import sys
import json
import logging
import argparse
import subprocess
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


def _setup_logging(logdir: Path) -> logging.Logger:
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"titan_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("TITANv3")


def _discover_files(path_or_pattern: str, recursive: bool, logger: logging.Logger) -> list[Path]:
    root = Path(path_or_pattern).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    files: list[Path] = []
    if root.is_file():
        files.append(root)
    else:
        pattern = "**/*" if recursive else "*"
        for f in root.glob(pattern):
            if f.is_file() and f.suffix.lower() in {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pkl"}:
                files.append(f)

    if not files:
        logger.warning(f"No input files discovered under {root}")
    else:
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            logger.info(f"✓ Found: {f.name} ({size_mb:.2f} MB)")
    return sorted(files)


def _run_single_dataset(
    input_file: Path,
    base_output: Path,
    test_name: str,
    config_path: str | None,
    logger: logging.Logger,
) -> tuple[Path, bool]:
    """
    Worker: run full v3 pipeline on a single dataset using run_titan_pipeline.
    """
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output / f"{test_name}_{input_file.stem}_{run_timestamp}"
    logger.info(f"🚀 DATASET: {input_file}")
    logger.info(f"📂 RUN DIR: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_titan_pipeline(
            input_source=str(input_file),
            run_dir=str(run_dir),
            config_path=config_path or "",
        )
        return run_dir, True
    except Exception as e:
        logger.error(f"❌ Error in pipeline for {input_file.name}: {e}", exc_info=True)
        return run_dir, False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TITAN RS Universal Orchestrator v3.0 – Full Engine Pipeline (RS + Research + Evidence + Results + Master Index)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input file or folder path",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="titan_results",
        help="Base output directory (default: titan_results)",
    )
    parser.add_argument(
        "--test-name",
        "-t",
        default=f"titantest_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Base test name for run folders (per-dataset suffix is added)",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="",
        help="Optional path to JSON/YAML config file",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input directory for data files",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help="Maximum parallel workers (default: CPU count - 1)",
    )

    args = parser.parse_args()

    base_output = Path(args.output_dir).expanduser()
    base_output.mkdir(parents=True, exist_ok=True)

    logdir = base_output / "logs"
    logger = _setup_logging(logdir)

    logger.info("=" * 70)
    logger.info("TITAN RS UNIVERSAL ORCHESTRATOR v3.0 – FULL PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Input       : {args.input}")
    logger.info(f"Output root : {base_output}")
    logger.info(f"Test name   : {args.test_name}")
    logger.info(f"Recursive   : {args.recursive}")
    logger.info(f"Max workers : {args.max_workers}")

    files = _discover_files(args.input, args.recursive, logger)
    if not files:
        logger.error("No input files found; exiting.")
        return 1

    success = 0
    failure = 0
    run_dirs: list[Path] = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {
            executor.submit(
                _run_single_dataset,
                f,
                base_output,
                args.test_name,
                args.config,
                logger,
            ): f
            for f in files
        }
        for fut in as_completed(future_to_file):
            f = future_to_file[fut]
            try:
                run_dir, ok = fut.result()
                run_dirs.append(run_dir)
                if ok:
                    success += 1
                    logger.info(f"✅ Completed: {f.name}")
                else:
                    failure += 1
            except Exception as e:
                failure += 1
                logger.error(f"FATAL worker error on {f.name}: {e}", exc_info=True)

    logger.info("-" * 70)
    logger.info(f"SUMMARY: {success} success, {failure} failed")
    logger.info(f"Runs are under: {base_output}")
    logger.info("-" * 70)

    # Optional: write a simple top-level manifest of all run dirs
    manifest = {
        "test_name": args.test_name,
        "timestamp": datetime.now().isoformat(),
        "input_root": args.input,
        "output_root": str(base_output),
        "runs": [str(rd) for rd in run_dirs],
        "success": success,
        "failure": failure,
    }
    with open(base_output / f"{args.test_name}_v3_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return 0 if failure == 0 else 1


if __name__ == "__main__":
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.freeze_support()
    sys.exit(main())

