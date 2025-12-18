import os
import shutil
from pathlib import Path

SOURCE_DIRS = [
    "/Users/robin/Desktop/Titan_Final_Protocol/testdata/Titan_Evidence_Suite",
    "/Users/robin/Desktop/Titan_Final_Protocol/testdata/TITAN_RESULTS_heart_2022_no_nans.csv",
    "/Users/robin/titan_sprint/TITAN_RESEARCH_RESULTS",
    "/Users/robin/titan_sprint/TITAN_EVIDENCE_BENCHMARK",
    "/Users/robin/titan_results/heart_omni_test_20251216_190056",
]

MASTER = Path("/Users/robin/Desktop/Titan_Final_Protocol/testdata/MASTER_TITAN_RESULTS")
MASTER.mkdir(parents=True, exist_ok=True)

for src in SOURCE_DIRS:
    src_path = Path(src)
    if not src_path.exists():
        continue
    for root, dirs, files in os.walk(src_path):
        rel = Path(root).relative_to(src_path)
        # keep a top-level folder per source to avoid collisions
        dest_root = MASTER / src_path.name / rel
        dest_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            src_file = Path(root) / f
            dest_file = dest_root / f
            if dest_file.exists():
                dest_file = dest_root / f"{src_path.name}__{f}"
            shutil.copy2(src_file, dest_file)

print(f"✅ Aggregated all results into: {MASTER}")
