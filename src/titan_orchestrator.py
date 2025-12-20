#!/usr/bin/env python3
"""
TITAN RS UNIVERSAL ORCHESTRATOR v2.0 - MULTICORE PARALLEL
Parallel execution of multiple engines across all CPU cores
Dramatically faster processing of multiple datasets
"""

import sys
import json
import logging
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


@dataclass
class Config:
    """Central configuration for TITAN orchestrator"""
    PROJECT_NAME = "TITAN-RS"
    VERSION = "2.0"
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    ENGINES = {
        "omni": "TITAN_Omni_Protocol.py",
        "research": "TITAN_Research_Mode.py",
        "results": "TITAN_Results_Engine.py",
        "fork": "TITAN_RS_Fork.py",
        "evidence": "TITAN_Evidence_Pro_Max.py",
        "rstitan": "RSTITAN.py",
    }
    
    RESULT_SUBDIRS = ["charts", "reports", "data", "logs", "metadata", "xlsxoutput"]


def setup_logging(logdir):
    """Setup console + file logging"""
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"titan_run_{Config.TIMESTAMP}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


class InputDiscovery:
    """Discover and validate input files/folders"""
    ALLOWED_FORMATS = [".csv", ".xlsx", ".xls", ".data", ".json", ".parquet", ".pkl"]
    
    @staticmethod
    def discover_files(path_or_pattern, recursive=False):
        """Find all data files in given path/pattern"""
        path = Path(path_or_pattern).expanduser()
        
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        files_found = []
        
        if path.is_file():
            if path.suffix in InputDiscovery.ALLOWED_FORMATS:
                files_found.append(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        elif path.is_dir():
            glob_pattern = "**/*" if recursive else "*"
            for f in path.glob(glob_pattern):
                if f.is_file() and f.suffix in InputDiscovery.ALLOWED_FORMATS:
                    files_found.append(f)
        
        return sorted(files_found)
    
    @staticmethod
    def validate_files(files, logger):
        """Validate each file exists and is readable"""
        valid = []
        for f in files:
            try:
                size_mb = f.stat().st_size / 1024 / 1024
                logger.info(f"✓ Found: {f.name} ({size_mb:.2f} MB)")
                valid.append(f)
            except Exception as e:
                logger.warning(f"Skipped {f.name} - {str(e)}")
        return valid


class ResultOrganizer:
    """Organize and manage output files"""
    
    def __init__(self, base_output_dir, test_name):
        """Initialize result directory structure"""
        self.base = Path(base_output_dir).expanduser()
        self.testname = test_name
        self.run_timestamp = Config.TIMESTAMP
        
        self.root = self.base / f"{test_name}_{self.run_timestamp}"
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.subdirs = {}
        for subdir in Config.RESULT_SUBDIRS:
            subpath = self.root / subdir
            subpath.mkdir(exist_ok=True)
            self.subdirs[subdir] = subpath
        
        # Metadata tracking
        self.metadata = {
            "test_name": test_name,
            "timestamp": self.run_timestamp,
            "engines_used": [],
            "input_files": [],
            "status": "initialized",
            "errors": [],
            "charts_generated": 0,
            "data_files_generated": 0,
            "xlsx_files_generated": 0,
        }
    
    def register_input(self, filepath):
        """Register an input file"""
        self.metadata["input_files"].append(str(filepath))
    
    def move_charts(self, source_dir, logger):
        """Find and move all chart files"""
        chart_exts = [".png", ".jpg", ".jpeg", ".pdf", ".svg"]
        count = 0
        source = Path(source_dir)
        
        if not source.exists():
            logger.warning(f"Chart source dir not found: {source}")
            return count
        
        for chart_file in source.glob("*"):
            if chart_file.suffix.lower() in chart_exts:
                try:
                    dest = self.subdirs["charts"] / chart_file.name
                    chart_file.rename(dest)
                    count += 1
                except Exception as e:
                    logger.warning(f"Could not move chart {chart_file.name}: {e}")
        
        self.metadata["charts_generated"] += count
        return count
    
    def move_xlsx(self, source_dir, logger):
        """Find and move all Excel files"""
        xlsx_exts = [".xlsx", ".xls"]
        count = 0
        source = Path(source_dir)
        
        if not source.exists():
            logger.warning(f"XLSX source dir not found: {source}")
            return count
        
        for xlsx_file in source.glob("*"):
            if xlsx_file.suffix.lower() in xlsx_exts:
                try:
                    dest = self.subdirs["xlsxoutput"] / xlsx_file.name
                    xlsx_file.rename(dest)
                    count += 1
                except Exception as e:
                    logger.warning(f"Could not move Excel {xlsx_file.name}: {e}")
        
        self.metadata["xlsx_files_generated"] += count
        return count
    
    def move_data_files(self, source_dir, logger):
        """Find and move all result data files"""
        data_exts = [".csv", ".json", ".parquet"]
        count = 0
        source = Path(source_dir)
        
        if not source.exists():
            logger.warning(f"Data source dir not found: {source}")
            return count
        
        for data_file in source.glob("*"):
            if data_file.suffix.lower() in data_exts:
                try:
                    dest = self.subdirs["data"] / data_file.name
                    data_file.rename(dest)
                    count += 1
                except Exception as e:
                    logger.warning(f"Could not move data {data_file.name}: {e}")
        
        self.metadata["data_files_generated"] += count
        return count
    
    def save_metadata(self):
        """Save run metadata as JSON"""
        metafile = self.root / "metadata.json"
        with open(metafile, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_manifest(self):
        """Generate file manifest"""
        manifest_file = self.root / "MANIFEST.txt"
        with open(manifest_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("TITAN RS Test Results Manifest\n")
            f.write(f"Test Name: {self.metadata['test_name']}\n")
            f.write(f"Timestamp: {self.metadata['timestamp']}\n")
            f.write(f"Status: {self.metadata['status']}\n")
            f.write("=" * 80 + "\n")
            
            f.write("\nINPUT FILES:\n")
            for inp in self.metadata["input_files"]:
                f.write(f"  - {inp}\n")
            f.write(f"  ({len(self.metadata['input_files'])} inputs)\n")
            
            f.write("\nOUTPUT STRUCTURE:\n")
            for name, subdir in self.subdirs.items():
                files = list(subdir.glob("*"))
                f.write(f"  {name}: {len(files)} files\n")
                for fpath in sorted(files)[:10]:
                    f.write(f"    - {fpath.name}\n")
                if len(files) > 10:
                    f.write(f"    ... and {len(files) - 10} more\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write(f"  Charts: {self.metadata['charts_generated']}\n")
            f.write(f"  XLSX files: {self.metadata['xlsx_files_generated']}\n")
            f.write(f"  Data files: {self.metadata['data_files_generated']}\n")
            f.write("=" * 80 + "\n")


class EngineRunner:
    """Execute chosen TITAN engine with error handling"""
    
    def __init__(self, engine_name, logger):
        self.engine_name = engine_name.lower()
        self.logger = logger
        self.engine_file = Config.ENGINES.get(self.engine_name)
        
        if not self.engine_file:
            raise ValueError(
                f"Unknown engine: {self.engine_name}. "
                f"Available: {list(Config.ENGINES.keys())}"
            )
    
    def validate_engine(self):
        """Check if engine file exists"""
        engine_path = Path(self.engine_file)
        if not engine_path.exists():
            self.logger.error(f"Engine file not found: {self.engine_file}")
            return False
        self.logger.info(f"✓ Engine found: {self.engine_name} ({self.engine_file})")
        return True
    
    def run(self, input_file, output_dir):
        """Execute engine on input file (terminal‑compatible, non‑interactive)"""
        self.logger.info(f"Running engine: {self.engine_name}")
        self.logger.info(f"Input: {input_file.name}")
        self.logger.info(f"Output dir: {output_dir}")
        
        try:
            cmd = [
                sys.executable,
                self.engine_file,
                "--input", str(input_file),
                "--output", str(output_dir),
            ]
            self.logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,   # prevent blocking on input()
                capture_output=True,
                text=True,
                timeout=3600,               # 1 hour max
            )
            
            if result.stdout.strip():
                self.logger.info(f"[{self.engine_name}] stdout:\n{result.stdout}")
            if result.stderr.strip():
                self.logger.error(f"[{self.engine_name}] stderr:\n{result.stderr}")
            
            if result.returncode == 0:
                self.logger.info("✓ Completed successfully")
                return True
            else:
                self.logger.error(f"Engine failed with code {result.returncode}")
                return False
        
        except subprocess.TimeoutExpired:
            self.logger.error("Engine execution timed out (1 hour)")
            return False
        except Exception as e:
            self.logger.error(f"Execution error: {str(e)}\n{traceback.format_exc()}")
            return False


def run_engine_on_file(engine_name, input_file, output_dir, test_name, logger):
    """Worker function: run single engine on single file (for multiprocessing)"""
    try:
        runner = EngineRunner(engine_name, logger)
        if not runner.validate_engine():
            return False
        return runner.run(input_file, output_dir)
    except Exception as e:
        logger.error(f"Worker error ({engine_name}): {str(e)}")
        return False


class TitanOrchestratorMulticore:
    """Main orchestrator: parallel execution across cores"""
    
    def __init__(self):
        self.logger = None
        self.organizer = None
        self.files = []
    
    def run(self, args):
        """Main execution flow with multicore parallelism"""
        try:
            print("=" * 80)
            print("TITAN RS UNIVERSAL ORCHESTRATOR v2.0 - MULTICORE")
            print("=" * 80)
            
            self.organizer = ResultOrganizer(args.output_dir, args.test_name)
            self.logger = setup_logging(self.organizer.subdirs["logs"])
            
            self.logger.info(f"Initialized test: {args.test_name}")
            self.logger.info(f"Result root: {self.organizer.root}")
            
            # --- STEP 1: INPUT DISCOVERY ---
            self.logger.info("\n--- STEP 1: INPUT DISCOVERY ---")
            self.files = InputDiscovery.discover_files(args.input, recursive=args.recursive)
            self.logger.info(f"Found {len(self.files)} files")
            
            if not self.files:
                raise ValueError("No input files found!")
            
            valid_files = InputDiscovery.validate_files(self.files, self.logger)
            self.logger.info(f"Validated {len(valid_files)} files")
            
            # --- STEP 2: ENGINE SELECTION ---
            self.logger.info("\n--- STEP 2: ENGINE SELECTION ---")
            
            if args.engine.lower() == "auto":
                print("\nAvailable TITAN RS Engines:")
                engines = list(Config.ENGINES.keys())
                for i, eng in enumerate(engines, 1):
                    print(f"  {i}. {eng}")
                while True:
                    try:
                        choice = input(f"Select engine(s) (1-{len(engines)}, comma-separated, or 'all'): ").strip()
                        if choice.lower() == "all":
                            engines_to_run = engines
                            break
                        choices = [int(c.strip()) - 1 for c in choice.split(",")]
                        if all(0 <= idx < len(engines) for idx in choices):
                            engines_to_run = [engines[idx] for idx in choices]
                            break
                    except Exception:
                        pass
                    print(f"Invalid choice, try again.")
            else:
                engines_to_run = args.engine.lower().split(",")
            
            self.logger.info(f"Will run {len(engines_to_run)} engine(s): {', '.join(engines_to_run)}")
            
            # --- STEP 3: PARALLEL PROCESSING ---
            self.logger.info(f"\n--- STEP 3: PARALLEL PROCESSING ---")
            self.logger.info(f"Running {len(engines_to_run)} engines × {len(valid_files)} files in parallel")
            
            # Determine max workers (use all cores, reserve 1 for system)
            max_workers = max(1, multiprocessing.cpu_count() - 1)
            self.logger.info(f"Using {max_workers} CPU cores")
            
            # Build task list: (engine, file) tuples
            tasks = []
            for engine_name in engines_to_run:
                for input_file in valid_files:
                    self.organizer.register_input(input_file)
                    tasks.append((engine_name, input_file))
            
            # Run all tasks in parallel
            success_count = 0
            error_count = 0
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(run_engine_on_file, engine, f, self.organizer.root, args.test_name, self.logger): (engine, f)
                    for engine, f in tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    engine, input_file = future_to_task[future]
                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                            self.logger.info(f"✓ {engine} on {input_file.name} completed")
                        else:
                            error_count += 1
                            self.organizer.metadata["errors"].append(
                                f"{engine} on {input_file.name} failed"
                            )
                    except Exception as e:
                        error_count += 1
                        self.organizer.metadata["errors"].append(
                            f"{engine} on {input_file.name}: {str(e)}"
                        )
                        self.logger.error(f"✗ {engine} on {input_file.name}: {str(e)}")
            
            # --- STEP 4: RESULT ORGANIZATION ---
            self.logger.info(f"\n--- STEP 4: RESULT ORGANIZATION ---")
            charts_moved = self.organizer.move_charts(self.organizer.root, self.logger)
            xlsx_moved = self.organizer.move_xlsx(self.organizer.root, self.logger)
            data_moved = self.organizer.move_data_files(self.organizer.root, self.logger)
            self.logger.info(
                f"Organized {charts_moved} charts, {xlsx_moved} xlsx, {data_moved} data files"
            )
            
            # --- STEP 5: FINALIZATION ---
            self.logger.info(f"\n--- STEP 5: FINALIZATION ---")
            self.organizer.metadata["status"] = "completed" if error_count == 0 else "completed_with_errors"
            self.organizer.save_metadata()
            self.organizer.save_manifest()
            
            self.logger.info("=" * 80)
            self.logger.info("SUMMARY")
            self.logger.info(f"Tasks executed: {success_count + error_count} ({success_count} success, {error_count} errors)")
            self.logger.info(f"Results saved to: {self.organizer.root}")
            self.logger.info("=" * 80)
            
            print(f"\n✓ All complete! Results in {self.organizer.root}")
            return 0 if error_count == 0 else 1
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"FATAL: {str(e)}\n{traceback.format_exc()}")
            else:
                print(f"FATAL ERROR: {str(e)}")
                traceback.print_exc()
            return 2


def main():
    parser = argparse.ArgumentParser(
        description="TITAN RS Universal Orchestrator v2.0 - Multicore Parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 titan_orchestrator_multicore.py --input data.csv --engine omni --test-name test_001
  python3 titan_orchestrator_multicore.py --input data/ --engine "omni,rstitan,research"
  python3 titan_orchestrator_multicore.py --input data/ --engine all --recursive
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file or folder path",
    )
    parser.add_argument(
        "--engine", "-e",
        default="auto",
        help="TITAN engine(s) to use: omni, rstitan, research, results, fork, evidence, or comma-separated list. Default: auto (interactive)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="titan_results",
        help="Base output directory (default: titan_results)",
    )
    parser.add_argument(
        "--test-name", "-t",
        default=f"titan_test_{datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Test name for result folder naming",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input directory",
    )
    
    args = parser.parse_args()
    
    orch = TitanOrchestratorMulticore()
    return orch.run(args)


if __name__ == "__main__":
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.freeze_support()
    
    sys.exit(main())
