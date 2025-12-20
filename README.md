TITAN RS(RS-Protocol): Transparent Integration of Training, Audit and Normalization – Research System
TITAN RS( RS-Protocol) is an open‑source framework for automated auditing and analysis of biomedical and survey datasets.
You point it at a CSV file; it performs data quality checks, detects leakage, trains a calibrated model, and produces a full visual audit trail in a single run.

Key Features
Smart data loading

Auto‑detects separators (,, ;, tab)

Handles large files via chunking and sampling

Works with typical CSV exports from surveys, EHRs, and registries

Automatic target detection

Recognises common medical outcome codes
(e.g. heart disease, stroke, diabetes indicators)

Fallback rules for generic ML datasets (classification targets)

Data quality & cleaning

Missing‑value handling (numeric: median; categorical: “Unknown”)

Outlier detection and removal using Isolation Forest

Removal of artifact columns (index, ID, unnamed columns)

Leakage detection (two‑stage)

Scalar leakage: flags features highly correlated with the target

Non‑linear leakage: flags features with extreme Random Forest importance

Model training & calibration

Random Forest classifier with train/calibration/test split

Three‑tier calibration fallback (prefit → CV=3 → uncalibrated)

AUC‑ROC and calibration error reporting

Automated visual reports

Distribution plots (histograms & violin plots) for top predictors

Feature importance network graph

Calibration / reliability curve

Data quality summaries and logs

Reproducible outputs

Deterministic seeds for all random operations

Structured output folders with charts, reports, metrics, and logs

Repository Structure (High Level)
RSTITAN.py – Core robust batch engine (data audit + modelling + charts)

TITAN_Omni_Protocol.py – Omni‑protocol engine with evidence suite

TITAN_Results_Engine.py – Aggregates metrics and exports Excel summaries

TITAN_RS_Fork.py – Safe dataset fusion and parallel processing

TITAN_Evidence_Pro_Max.py – Additional evidence and superiority charts

TITAN_RS_GUI.py – (Optional) graphical interface

titan_orchestrator*.py – Entry‑point scripts to run one or more engines

sample_data/ – Example datasets (if provided)

Titan_Synergy_Results/ (generated) – Output reports and charts

You do not need to understand every file to use TITAN RS; see the quickstart below.

Quickstart
1. Requirements
Python 3.8+

Recommended:

RAM: ≥ 8 GB

CPU: ≥ 4 cores

2. Installation
bash
git clone https://github.com/yourusername/TITAN-RS.git
cd TITAN-RS

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Run a demo on a CSV file
Place a CSV (e.g. heart.csv) in a folder, or use your own dataset.

bash
# Basic one‑file audit using the RSTITAN engine
python RSTITAN.py
The script will:

Ask you to drag & drop / paste a file or folder path (or URL to a CSV).

Process each dataset (with quality checks, modelling, charts).

Create an output directory, typically:

text
Titan_Synergy_Results/
└── <your_file>_Audit/
    ├── REPORT.md
    ├── <many> .png charts
    ├── *_FULL_REPORT.pdf   (if PDF dependency installed)
    ├── metrics / logs
    └── ...
You can open the generated PNG/PDF files to inspect distributions, feature importance, calibration, and more.

Typical Workflow
Prepare your data

Export your dataset as CSV.

Ensure the target/outcome column is present (e.g. death, disease, class).

Run TITAN RS

Launch RSTITAN.py and point it to:

a single CSV file, or

a folder containing multiple CSVs.

Review outputs

Check the generated REPORT.md and PDF report.

Inspect:

AUC and calibration curves

Outlier counts and leakage warnings

Top predictive features and their distributions

Iterate

Adjust your dataset/columns based on detected leakage or data quality issues.

Re‑run to confirm improvements.

For Biostatisticians / Methodologists
If you are reviewing the methods, key components are:

Target detection & medical decoder – how outcome columns are identified and normalised

Leakage detection – correlation‑based and Random Forest importance thresholds

Outlier handling – Isolation Forest parameters and contamination fraction

Model & calibration – Random Forest configuration and three‑tier calibration scheme

Reproducibility – fixed seeds, logging, and deterministic pipelines

Feedback on any of these design choices is very welcome.

Contributing & Feedback
Feedback, issues, and contributions are strongly encouraged.

Open an Issue on GitHub for:

Bugs / crashes

Unexpected behaviour on a dataset

Suggestions for better statistical defaults

Open a Pull Request for:

New checks (e.g. additional leakage rules)

Better visualisations

Performance or stability improvements

If you are a biostatistician or data scientist and do a methodological review, you may (with consent) be acknowledged in the associated manuscript.

Citation
If you use TITAN RS in academic work, please cite the associated manuscript / preprint (replace with actual citation):

TITAN RS/RS-Protocol: A universal data audit and modelling framework for biomedical datasets. Computer Methods and Programs in Biomedicine, in submission pipeline, 2025.

License:
"RS Protocol © 2025 by Robin Sandhu is licensed under Creative Commons Attribution-NonCommercial 4.0 International"

Code: CC BY-NC

Text / documentation: CC BY‑NC 4.0 
