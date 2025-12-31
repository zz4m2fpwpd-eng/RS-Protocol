# TITAN RS (RS-Protocol)
**Transparent Integration of Training, Audit and Normalization – Research System**

TITAN RS (RS-Protocol) is an open‑source framework for the automated auditing and analysis of biomedical and survey datasets. 

The system bridges the gap between raw data collection and reproducible evidence. You point it at a CSV file, and it executes a full pipeline in a single run: performing data quality checks, detecting leakage, training a calibrated model, and producing a visual audit trail.

## Key Features

**Smart Data Loading**
* **Auto‑detects separators:** Handles comma, semi-colon, and tab delimiters automatically.
* **Scalable:** Handles large files via chunking and sampling.
* **Broad Compatibility:** Works with typical CSV exports from surveys, EHRs, and registries.

**Automatic Target Detection**
* **Medical Decoder:** Recognises common outcome codes (e.g., heart disease, stroke, diabetes indicators).
* **Fallback:** Applies standard rules for generic ML datasets (classification targets).

**Data Quality & Cleaning**
* **Imputation:** Handles missing values (Median for numeric; “Unknown” for categorical).
* **Sanitization:** Removes artifact columns (index, ID, unnamed columns).
* **Outlier Removal:** Uses Isolation Forest to detect and exclude statistical anomalies.

**Leakage Detection (Two‑Stage)**
* **Scalar Leakage:** Flags features highly correlated with the target.
* **Non‑linear Leakage:** Flags features with extreme Random Forest importance.

**Model Training & Calibration**
* **Algorithm:** Random Forest classifier with strict train/calibration/test split.
* **Reliability:** Three‑tier calibration fallback (Prefit → CV=3 → Uncalibrated).
* **Reporting:** Comprehensive AUC‑ROC and calibration error metrics.

**Automated Visual Reports**
* Distribution plots (histograms & violin plots) for top predictors.
* Feature importance network graphs.
* Calibration / reliability curves.
* Data quality summaries and logs.

**Reproducibility**
* Deterministic seeds for all random operations.
* Structured output folders with charts, reports, metrics, and logs.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `RSTITAN.py` | **Core Engine:** Robust batch engine (data audit + modelling + charts). |
| `TITAN_Omni_Protocol.py` | Omni‑protocol engine with evidence suite. |
| `TITAN_Results_Engine.py` | Aggregates metrics and exports Excel summaries. |
| `TITAN_RS_Fork.py` | Safe dataset fusion and parallel processing. |
| `TITAN_Evidence_Pro_Max.py` | Additional evidence and superiority charts. |
| `TITAN_RS_GUI.py` | (Optional) Graphical user interface. |
| `titan_orchestrator*.py` | Entry‑point scripts to run one or more engines. |
| `sample_data/` | Example datasets. |
| `Titan_Synergy_Results/` | **Output:** Generated reports and charts appear here. |

*Note: You do not need to understand every file to use TITAN RS; see the quickstart below.*

---

## Quickstart

### 1. Requirements
* **Python 3.8+**
* **RAM:** ≥ 8 GB (Recommended)
* **CPU:** ≥ 4 cores (Recommended)

### 2. Installation

```bash
# Clone the repository
git clone [https://github.com/zz4m2fpwpd-eng/RS-Protocol.git](https://github.com/zz4m2fpwpd-eng/RS-Protocol.git)

# Enter the directory
cd RS-Protocol

# Create and activate a virtual environment (Recommended)
# Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt):
# python -m venv venv
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Run a demo on a CSV file
Place a CSV (e.g. heart.csv) in a folder, or use your own dataset.

Basic one‑file audit using the RSTITAN engine:

Bash

python RSTITAN.py
The script will:

Ask you to drag & drop / paste a file or folder path (or URL to a CSV).

Process each dataset (performing quality checks, modelling, and charting).

Create an output directory, typically:

Plaintext

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

Ensure the target/outcome column is present (e.g., death, disease, class).

Run TITAN RS

Launch RSTITAN.py.

Point it to a single CSV file, or a folder containing multiple CSVs.

Review outputs

Check the generated REPORT.md and PDF report.

Inspect AUC and calibration curves.

Review outlier counts and leakage warnings.

Analyze top predictive features and their distributions.

Iterate

Adjust your dataset/columns based on detected leakage or data quality issues.

Re‑run to confirm improvements.

For Biostatisticians / Methodologists
If you are reviewing the methods, the key components are:

Target detection & medical decoder: How outcome columns are identified and normalised.

Leakage detection: Correlation‑based thresholds and Random Forest importance thresholds.

Outlier handling: Isolation Forest parameters and contamination fraction.

Model & calibration: Random Forest configuration and the three‑tier calibration scheme.

Reproducibility: Fixed seeds, logging, and deterministic pipelines.

Feedback on any of these design choices is very welcome.

Contributing & Feedback
Feedback, issues, and contributions are strongly encouraged.

Open an Issue on GitHub for:

Bugs / crashes.

Unexpected behaviour on a dataset.

Suggestions for better statistical defaults.

Open a Pull Request for:

New checks (e.g. additional leakage rules).

Better visualisations.

Performance or stability improvements.

If you are a biostatistician or data scientist and do a methodological review, you may (with consent) be acknowledged in the associated manuscript.

Citation
If you use TITAN RS in academic work, please cite the Sandhu, R. (2025). RS-Protocol (Version 1.0.0) [Computer software]. https://github.com/zz4m2fpwpd-eng/RS-Protocol

License
RS Protocol © 2025 by Robin Sandhu is licensed under Creative Commons Attribution-NonCommercial 4.0 International
