#!/usr/bin/env python3
import os
import sys
import json
import argparse
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(style="whitegrid")

PALETTE = "viridis"


class TITANResultsEngine:
    """
    TITAN Results Engine

    - Loads cleaned CSV
    - Identifies target and predictors
    - Runs basic stats:
        * T-tests for numeric vs binary target
        * Chi-square for categorical vs binary target
    - Generates charts:
        * Violin plots for numeric vs target
        * Bar plots for categorical vs target
        * Core visual suite (exactly like FINAL_STUDY_RESULTS):
            - Dist_PhysicalHealthDays, Dist_MentalHealthDays,
              Dist_SleepHours, Dist_HeightInMeters,
              Dist_WeightInKilograms, Dist_BMI
            - Box_*_vs_Target for the same six vs target
            - Bar_Sex, Bar_State, Bar_GeneralHealth,
              Bar_PhysicalActivities, Bar_LastCheckupTime
            - Pairplot and Correlation_Heatmap
    """

    def __init__(self, csv_path, out_dir, target=None):
        self.csv_path = csv_path
        self.out_dir = out_dir
        self.df = pd.read_csv(csv_path)

        # Auto-detect binary target if not provided
        if target is None:
            self.target = self._auto_target()
        else:
            self.target = target

        # Basic type splits
        self.nums = [
            c for c in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[c]) and c != self.target
        ]
        self.cats = [
            c for c in self.df.columns
            if not pd.api.types.is_numeric_dtype(self.df[c]) and c != self.target
        ]

        os.makedirs(self.out_dir, exist_ok=True)
        self.stats_log = []
        self.charts = []

    def _auto_target(self):
        # Prefer common outcome names
        candidates = [
            "HadDiabetes",
            "Outcome",
            "target",
            "Target",
            "label",
            "Label",
        ]
        for c in candidates:
            if c in self.df.columns:
                return c

        # Otherwise, choose first binary column
        for c in self.df.columns:
            if self.df[c].dropna().nunique() == 2:
                return c

        # Fallback: last column
        return self.df.columns[-1]

    # ------------------------------------------------------------------
    # CORE ANALYSIS
    # ------------------------------------------------------------------
    def run_analysis(self):
        # 1. Numerical Predictors (T-Test + Violin-like view)
        for col in self.nums[:15]:
            try:
                g1 = self.df[self.df[self.target] == 0][col]
                g2 = self.df[self.df[self.target] == 1][col]
                t, p = stats.ttest_ind(g1, g2, nan_policy="omit")
                sig = (
                    "***"
                    if p < 0.001
                    else "**"
                    if p < 0.01
                    else "*"
                    if p < 0.05
                    else "ns"
                )
                self.stats_log.append(
                    f"{col[:28]:<30} {'T-Test':<10} {p:.4f}     {sig}"
                )

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.violinplot(
                    x=self.target,
                    y=col,
                    data=self.df,
                    palette=PALETTE,
                    split=True,
                    ax=ax,
                )
                ax.set_title(
                    f"{col.replace('_', ' ')} by {self.target.replace('_', ' ')} (p={p:.4f})",
                    fontsize=12,
                    fontweight="bold",
                    pad=15,
                )
                ax.set_xlabel(self.target.replace("_", " "), fontsize=11, fontweight="bold")
                ax.set_ylabel(col.replace("_", " "), fontsize=11, fontweight="bold")

                fname = f"{self.out_dir}/Violin_{col}.png"
                plt.tight_layout()
                plt.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.charts.append((fname, f"Distribution of {col} by {self.target}"))
            except Exception:
                pass

        # 2. Categorical Predictors (Chi-Square + Bar)
        for col in self.cats[:10]:
            try:
                ct = pd.crosstab(self.df[col], self.df[self.target])
                c, p, d, e = stats.chi2_contingency(ct)
                sig = (
                    "***"
                    if p < 0.001
                    else "**"
                    if p < 0.01
                    else "*"
                    if p < 0.05
                    else "ns"
                )
                self.stats_log.append(
                    f"{col[:28]:<30} {'Chi2':<10} {p:.4f}     {sig}"
                )

                df_plot = self.df.copy()

                fig, ax = plt.subplots(figsize=(12, 7))
                sns.countplot(
                    x=col,
                    hue=self.target,
                    data=df_plot,
                    palette="rocket",
                    ax=ax,
                )

                pretty_col = col.replace("_", " ")
                pretty_target = self.target.replace("_", " ")

                ax.set_title(
                    f"{pretty_col} distribution by {pretty_target} (p={p:.4f})",
                    pad=18,
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_xlabel(pretty_col, fontsize=11, fontweight="bold")
                ax.set_ylabel("Count", fontsize=11, fontweight="bold")

                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha("right")
                    tick.set_fontsize(8)

                ax.legend(title=pretty_target, loc="best", fontsize=9)
                plt.subplots_adjust(bottom=0.23)

                fname = f"{self.out_dir}/Bar_{col}.png"
                plt.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)

                self.charts.append(
                    (fname, f"Prevalence of {pretty_col} by {pretty_target}")
                )
            except Exception:
                pass

        # 3. Core visual suite (matches FINAL_STUDY_RESULTS figures)
        self.generate_core_visuals()

        # 4. Save stats log
        stats_path = os.path.join(self.out_dir, "stats_summary.txt")
        with open(stats_path, "w") as f:
            f.write("\n".join(self.stats_log))

    # ------------------------------------------------------------------
    # CORE VISUALS: Dist_*, Box_*_vs_Target, Bar_*, Pairplot, Heatmap
    # ------------------------------------------------------------------
    def generate_core_visuals(self):
        # Continuous variables used in your previous figures
        cont_vars = [
            c
            for c in [
                "PhysicalHealthDays",
                "MentalHealthDays",
                "SleepHours",
                "HeightInMeters",
                "WeightInKilograms",
                "BMI",
            ]
            if c in self.df.columns
        ]

        # --- 1. Distributions (hist + KDE, Dist_*.png) ---
        for col in cont_vars:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.histplot(
                    self.df[col].dropna(),
                    bins=80,
                    kde=True,
                    color="steelblue",
                    ax=ax,
                )
                ax.set_title(f"Distribution of {col}", fontsize=14)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
                fname = f"{self.out_dir}/Dist_{col}.png"
                plt.tight_layout()
                plt.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.charts.append((fname, f"Distribution of {col}"))
            except Exception:
                pass

        # --- 2. Boxplots vs outcome (Box_*_vs_Target.png) ---
        for col in cont_vars:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(
                    x=self.target,
                    y=col,
                    data=self.df,
                    palette="deep",
                    ax=ax,
                )
                ax.set_title(f"{col} by {self.target}", fontsize=14)
                ax.set_xlabel(self.target, fontsize=12)
                ax.set_ylabel(col, fontsize=12)

                for t in ax.get_xticklabels():
                    t.set_rotation(0)
                    t.set_ha("center")
                    t.set_fontsize(10)

                fname = f"{self.out_dir}/Box_{col}_vs_Target.png"
                plt.tight_layout()
                plt.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.charts.append((fname, f"{col} by {self.target}"))
            except Exception:
                pass

        # --- 3. Key categorical countplots (Bar_*.png) ---
        cat_targets = [
            c
            for c in [
                "Sex",
                "GeneralHealth",
                "PhysicalActivities",
                "LastCheckupTime",
                "State",
            ]
            if c in self.df.columns
        ]

        for col in cat_targets:
            try:
                fig, ax = plt.subplots(figsize=(14, 7))
                sns.countplot(
                    x=col,
                    hue=self.target,
                    data=self.df,
                    palette="mako",
                    ax=ax,
                )
                ax.set_title(f"{col} Distribution by Outcome", fontsize=14)
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel("count", fontsize=12)

                rot = 60 if col in ["State", "LastCheckupTime"] else 45
                bottom = 0.28 if col in ["State", "LastCheckupTime"] else 0.20
                for t in ax.get_xticklabels():
                    t.set_rotation(rot)
                    t.set_ha("right")
                    t.set_fontsize(9)

                ax.legend(title=self.target, fontsize=9)
                plt.subplots_adjust(bottom=bottom)

                fname = f"{self.out_dir}/Bar_{col}.png"
                plt.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.charts.append((fname, f"{col} distribution by {self.target}"))
            except Exception:
                pass

        # --- 4. Pairplot & Correlation Heatmap ---
        if len(cont_vars) >= 2:
            try:
                pair_df = self.df[cont_vars + [self.target]].dropna()

                # Pairplot (Pairplot.png)
                g = sns.pairplot(
                    pair_df,
                    vars=cont_vars,
                    hue=self.target,
                    diag_kind="kde",
                    height=2.5,
                    plot_kws={"s": 8, "alpha": 0.7},
                )
                g.fig.suptitle("Feature Relationships by Outcome", y=1.02)
                pair_fname = f"{self.out_dir}/Pairplot.png"
                g.savefig(pair_fname, dpi=300, bbox_inches="tight")
                plt.close("all")
                self.charts.append((pair_fname, "Pairwise feature relationships"))

                # Correlation heatmap (Correlation_Heatmap.png)
                corr = pair_df[cont_vars].corr()
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(
                    corr,
                    annot=False,
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    square=True,
                    cbar_kws={"shrink": 0.8},
                    ax=ax,
                )
                ax.set_title("Feature Correlation Matrix", fontsize=14)
                heat_fname = f"{self.out_dir}/Correlation_Heatmap.png"
                plt.tight_layout()
                plt.savefig(heat_fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                self.charts.append((heat_fname, "Feature correlation matrix"))
            except Exception:
                pass


def parse_args():
    p = argparse.ArgumentParser(
        description="TITAN Results Engine: stats + charts for cleaned CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Cleaned CSV file")
    p.add_argument("--output-dir", required=True, help="Directory for charts + stats")
    p.add_argument("--target", default=None, help="Target column (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    engine = TITANResultsEngine(args.input, args.output_dir, target=args.target)
    engine.run_analysis()


if __name__ == "__main__":
    main()
