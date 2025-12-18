# ==============================================================================
# PROJECT: TITAN RESEARCH MODE (SCIENTIFIC DISCOVERY ENGINE)
# BASED ON: TITAN-RS V1.0 (Audit Core)
# PURPOSE: Generate "Results Section" using strictly audited features.
# ==============================================================================

import os
import sys
import time
import warnings
import multiprocessing
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from fpdf import FPDF

# ML Libraries (Shared with TITAN-RS)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# CONFIGURATION: VOGUE VISUALS (SCIENTIFIC CONTEXT)
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300

SAFE_CORES = max(1, multiprocessing.cpu_count() - 2)

# ==============================================================================
# 1. CORE LOGIC (IMPORTED FROM TITAN-RS FOR CONSISTENCY)
# ==============================================================================

def smart_load(filepath):
    try:
        # Exact same loader as Audit Mode
        if os.path.getsize(filepath) > 200 * 1024 * 1024:
            return pd.read_csv(
                filepath,
                nrows=50000,
                encoding="latin1",
                on_bad_lines="skip",
            )
        return pd.read_csv(filepath, encoding="latin1", on_bad_lines="skip")
    except Exception:
        return None


def analyze_structure_and_fix(df):
    # Exact same decoder as Audit Mode
    codebook = {
        r"^_MICHD": "TARGET_HEART_DISEASE",
        r"^CVDSTRK": "TARGET_STROKE",
        r"^DIABETE": "TARGET_DIABETES",
        r"^CVDINFR": "TARGET_HEART_ATTACK",
        r"^CVDCRHD": "TARGET_ANGINA",
        r"^_RFHLTH": "TARGET_GOOD_HEALTH",
        r"^HadHeartAttack": "TARGET_HEART_ATTACK",
        r"^Stroke": "TARGET_STROKE",
        r"^class$": "TARGET_CLASS",
    }

    for col in df.columns:
        for p, n in codebook.items():
            if re.search(p, col, re.IGNORECASE) and n not in df.columns:
                df.rename(columns={col: n}, inplace=True)

    # Exact same Target Logic
    best_target = None
    priority = [
        "DIED",
        "DEATH",
        "TARGET_HEART_DISEASE",
        "TARGET_STROKE",
        "TARGET_DIABETES",
        "TARGET",
        "CLASS",
    ]
    for p in priority:
        matches = [c for c in df.columns if c.upper() == p]
        if matches:
            best_target = matches[0]
            break

    if not best_target:
        # Fallback heuristic
        if 2 <= df[df.columns[-1]].nunique() <= 10:
            best_target = df.columns[-1]
        elif 2 <= df[df.columns[0]].nunique() <= 10:
            best_target = df.columns[0]

    # Exact same leakage filter
    leak = ["ID", "DATE", "TEXT", "DESC"]
    preds = [
        c
        for c in df.columns
        if c != best_target
        and df[c].nunique() < 100
        and not any(l in c.upper() for l in leak)
    ]

    return best_target, preds, df

# ==============================================================================
# 2. RESEARCH ENGINE (THE NEW "RESULTS" LAYER)
# ==============================================================================

def generate_result_chart(task):
    type_, data, save_path, meta = task
    out_file = f"{save_path}/{meta['fname']}"
    try:
        fig = plt.figure(figsize=(10, 6))
        pretty_col = meta["title"].replace("_", " ")
        pretty_target = meta["target"].replace("_", " ")

        # LOGIC CHANGE: Charts are now styled for "Clinical Results", not "Audit Checks"
        if type_ == "association":
            # Bar Chart with Counts (for Papers)
            sns.countplot(
                x=meta["col"],
                hue=meta["target"],
                data=data,
                palette="viridis",
            )
            plt.title(
                f"Clinical association: {pretty_col} vs {pretty_target}",
                fontweight="bold",
            )
            plt.xlabel(pretty_col)
            plt.ylabel("Patient count")
            plt.legend(title=pretty_target)

        elif type_ == "distribution":
            # Violin Plot with Stat Annotation
            sns.violinplot(
                x=meta["target"],
                y=meta["col"],
                data=data,
                palette="mako",
                split=True,
                inner="quartile",
            )
            plt.title(f"{pretty_col} by {pretty_target}", fontweight="bold")
            plt.xlabel(f"Outcome ({pretty_target})")
            plt.ylabel(pretty_col)

        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close(fig)
        return out_file
    except Exception:
        plt.close("all")
        return None


def worker_research(filepath):
    try:
        fname = os.path.basename(filepath)
        print(f"\n[RESEARCH MODE] Analyzing {fname}...")

        # 1. LOAD (Using Audit Logic)
        df = smart_load(filepath)
        if df is None:
            return

        # 2. DETECT (Using Audit Logic)
        target, preds, df = analyze_structure_and_fix(df)
        print(f" > Target Identified: {target}")

        # 3. IDENTIFY KEY FEATURES (Using Audit Logic: Random Forest)
        # We use the Audit engine to find what matters, so we don't analyze useless noise.
        df_clean = df.dropna(axis=1, how="all").fillna(0)
        X = pd.get_dummies(df_clean[preds], drop_first=True)
        y = LabelEncoder().fit_transform(df_clean[target].astype(str))

        model = RandomForestClassifier(
            n_estimators=50, max_depth=8, n_jobs=SAFE_CORES
        )
        model.fit(X, y)
        auc = (
            roc_auc_score(y, model.predict_proba(X)[:, 1])
            if len(np.unique(y)) == 2
            else 0.0
        )

        # Extract TOP 8 Features (The "Results")
        importances = (
            pd.Series(model.feature_importances_, index=X.columns)
            .sort_values(ascending=False)
        )
        top_features = importances.head(8).index.tolist()
        print(f" > Top Clinical Predictors: {top_features}")

        # 4. GENERATE RESULTS (The New Output Layer)
        res_dir = f"TITAN_RESEARCH_RESULTS/{fname}_Findings"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        chart_paths = []
        stats_text = []

        stats_text.append(f"STUDY: {fname}")
        stats_text.append(f"PRIMARY OUTCOME: {target}")
        stats_text.append(f"PREDICTIVE POWER (AUC): {auc:.3f}")
        stats_text.append("-" * 50)

        for feat in top_features:
            # Map back to original column name if dummy-encoded
            orig_col = feat.split("_")[0]
            if orig_col in df.columns:
                # A. Statistical Test
                p_val = "N/A"
                if np.issubdtype(df[orig_col].dtype, np.number):
                    # T-Test for Numerical
                    g1 = df[df[target] == 0][orig_col]
                    g2 = df[df[target] == 1][orig_col]
                    t, p = stats.ttest_ind(g1, g2, nan_policy="omit")
                    p_val = f"{p:.4e}"
                    chart_type = "distribution"
                else:
                    # Chi-Square for Categorical
                    contingency = pd.crosstab(df[orig_col], df[target])
                    c, p, d, e = stats.chi2_contingency(contingency)
                    p_val = f"{p:.4e}"
                    chart_type = "association"

                try:
                    p_float = float(p_val)
                    sig = "***" if p_float < 0.001 else "ns"
                except Exception:
                    sig = "ns"

                stats_text.append(
                    f"PREDICTOR: {orig_col} | P-Value: {p_val} {sig}"
                )

                # B. Generate Chart
                c_path = generate_result_chart(
                    (
                        chart_type,
                        df,
                        res_dir,
                        {
                            "col": orig_col,
                            "target": target,
                            "fname": f"Fig_{orig_col}.png",
                            "title": orig_col,
                        },
                    )
                )
                if c_path:
                    chart_paths.append((c_path, orig_col))

        # 5. COMPILE MANUSCRIPT PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"RESEARCH FINDINGS: {fname}", ln=1, align="C")
        pdf.set_font("Courier", size=10)
        pdf.ln(10)
        for line in stats_text:
            pdf.cell(0, 5, line, ln=1)

        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "KEY CLINICAL FIGURES", ln=1)

        for path, name in chart_paths:
            if pdf.get_y() > 200:
                pdf.add_page()
            pretty_name = name.replace("_", " ")
            pretty_target = target.replace("_", " ")
            pdf.set_font("Arial", "I", 10)
            pdf.cell(
                0,
                8,
                f"Figure: {pretty_name} by {pretty_target}",
                ln=1,
            )
            pdf.image(path, x=20, w=170)
            pdf.ln(4)

        pdf.output(f"{res_dir}/FINAL_RESULTS_MANUSCRIPT.pdf")
        print(f" ✅ Research Paper Generated: {res_dir}")

    except Exception as e:
        print(f" ❌ Error: {e}")

# ==============================================================================
# MAIN ENTRYPOINTS (INTERACTIVE + ORCHESTRATOR-SAFE)
# ==============================================================================

def run_research_core(path):
    """
    Core Research pipeline without any input() calls.
    Accepts a file or folder path.
    """
    if os.path.isfile(path):
        worker_research(path)
    elif os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".csv"):
                worker_research(os.path.join(path, f))


def cli_main():
    """
    CLI entrypoint for orchestrator: --input, --output (output is unused but accepted).
    """
    import argparse

    parser = argparse.ArgumentParser(description="TITAN Research Mode")
    parser.add_argument("--input", "-i", required=True,
                        help="Input CSV file or folder")
    parser.add_argument("--output", "-o", required=False,
                        help="Output directory (not required; uses TITAN_RESEARCH_RESULTS)")
    args = parser.parse_args()

    path = args.input
    run_research_core(path)


if __name__ == "__main__":
    # Orchestrator / CLI mode: no interactive input
    if len(sys.argv) > 1:
        if sys.platform == "darwin":
            multiprocessing.set_start_method("spawn", force=True)
        cli_main()
    else:
        # Original interactive behavior
        if sys.platform == "darwin":
            multiprocessing.set_start_method("spawn", force=True)

        print("\n[INPUT] Drag & Drop the CSV file/folder:")
        path = (
            input(" > ")
            .strip()
            .replace("'", "")
            .replace('"', "")
            .replace("\\ ", " ")
        )

        if not path:
            print("❌ No path provided. Exiting.")
        else:
            run_research_core(path)
