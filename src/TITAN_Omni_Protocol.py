#!/usr/bin/env python3
"""
TITAN RS - Omni Protocol v2.0
Universal statistical analysis engine with comprehensive data profiling,
descriptive statistics, distribution analysis, and correlation matrices.

Handles missing data, outliers, and generates publication-ready visualizations.
"""

import sys
import os
import json
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class OmniProtocol:
    """Universal statistical analysis engine."""
    
    def __init__(self, input_file, output_dir=None):
        """Initialize Omni Protocol."""
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir) if output_dir else self.input_file.parent / "omni_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.metadata = {}
        
    def load_data(self):
        """Load CSV data."""
        try:
            logger.info(f"Loading data from {self.input_file}")
            self.df = pd.read_csv(self.input_file, low_memory=False)
            logger.info(f"✓ Loaded {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            
            self.metadata['sample_size'] = self.df.shape[0]
            self.metadata['num_features'] = self.df.shape[1]
            self.metadata['columns'] = list(self.df.columns)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def profile_data(self):
        """Generate data profile."""
        logger.info("\n--- DATA PROFILING ---")
        
        profile = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': self.df.duplicated().sum(),
            'columns_with_nulls': self.df.isnull().sum().sum(),
            'null_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        }
        
        logger.info(f"  Rows: {profile['total_rows']}")
        logger.info(f"  Columns: {profile['total_columns']}")
        logger.info(f"  Memory: {profile['memory_mb']:.2f} MB")
        logger.info(f"  Duplicates: {profile['duplicate_rows']}")
        logger.info(f"  Missing values: {profile['columns_with_nulls']} ({profile['null_percentage']:.2f}%)")
        
        self.metadata.update(profile)
        return profile
    
    def describe_statistics(self):
        """Generate descriptive statistics."""
        logger.info("\n--- DESCRIPTIVE STATISTICS ---")
        
        # Numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats_dict = {}
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                stats_dict[col] = {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'skewness': float(stats.skew(col_data.dropna())),
                    'kurtosis': float(stats.kurtosis(col_data.dropna())),
                    'null_count': int(self.df[col].isnull().sum())
                }
                logger.info(f"  {col}: μ={stats_dict[col]['mean']:.2f}, σ={stats_dict[col]['std']:.2f}")
        
        self.metadata['descriptive_stats'] = stats_dict
        return stats_dict
    
    def analyze_distributions(self):
        """Analyze distributions of numeric columns."""
        logger.info("\n--- DISTRIBUTION ANALYSIS ---")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        fig, axes = plt.subplots(nrows=(len(numeric_cols)+2)//3, ncols=3, figsize=(15, 4*(len(numeric_cols)+2)//3))
        if len(numeric_cols) < 4:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            data = self.df[col].dropna()
            
            # Histogram with KDE
            ax.hist(data, bins=30, alpha=0.7, edgecolor='black', density=True)
            data.plot(kind='kde', ax=ax, secondary_y=False, color='red', linewidth=2)
            ax.set_title(f'{col}\n(n={len(data)}, μ={data.mean():.2f})', fontsize=10, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            
            logger.info(f"  {col}: Analyzed {len(data)} non-null values")
        
        # Hide empty subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        chart_path = self.output_dir / "01_Distributions.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {chart_path.name}")
        plt.close()
    
    def correlation_analysis(self):
        """Analyze correlations between numeric features."""
        logger.info("\n--- CORRELATION ANALYSIS ---")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Compute correlation
            corr_matrix = self.df[numeric_cols].corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                       square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('Correlation Matrix (Pearson r)', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            chart_path = self.output_dir / "02_Correlation_Matrix.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            logger.info(f"  Saved: {chart_path.name}")
            plt.close()
            
            # Find high correlations
            high_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corrs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_matrix.iloc[i, j])
                        })
            
            logger.info(f"  High correlations (|r| > 0.7): {len(high_corrs)}")
            self.metadata['high_correlations'] = high_corrs
        else:
            logger.info("  Insufficient numeric columns for correlation analysis")
    
    def categorical_analysis(self):
        """Analyze categorical features."""
        logger.info("\n--- CATEGORICAL ANALYSIS ---")
        
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        cat_stats = {}
        for col in cat_cols:
            value_counts = self.df[col].value_counts()
            cat_stats[col] = {
                'unique_values': int(self.df[col].nunique()),
                'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'null_count': int(self.df[col].isnull().sum())
            }
            logger.info(f"  {col}: {cat_stats[col]['unique_values']} unique values")
        
        self.metadata['categorical_stats'] = cat_stats
    
    def outlier_detection(self):
        """Detect outliers using IQR method."""
        logger.info("\n--- OUTLIER DETECTION ---")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        outliers_dict = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_mask = (self.df[col] < (Q1 - 1.5*IQR)) | (self.df[col] > (Q3 + 1.5*IQR))
            n_outliers = outlier_mask.sum()
            
            outliers_dict[col] = {
                'count': int(n_outliers),
                'percentage': float(n_outliers / len(self.df) * 100)
            }
            
            if n_outliers > 0:
                logger.info(f"  {col}: {n_outliers} outliers ({outliers_dict[col]['percentage']:.2f}%)")
        
        self.metadata['outliers'] = outliers_dict
    
    def generate_summary_report(self):
        """Generate text summary report."""
        logger.info("\n--- GENERATING SUMMARY REPORT ---")
        
        report_path = self.output_dir / "OMNI_SUMMARY_REPORT.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TITAN RS - OMNI PROTOCOL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file.name}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            # Data Overview
            f.write("--- DATA OVERVIEW ---\n")
            f.write(f"Rows: {self.metadata.get('sample_size', 'N/A')}\n")
            f.write(f"Columns: {self.metadata.get('num_features', 'N/A')}\n")
            f.write(f"Memory: {self.metadata.get('memory_mb', 'N/A'):.2f} MB\n")
            f.write(f"Duplicates: {self.metadata.get('duplicate_rows', 'N/A')}\n")
            f.write(f"Missing Values: {self.metadata.get('null_percentage', 'N/A'):.2f}%\n\n")
            
            # Statistics Summary
            if 'descriptive_stats' in self.metadata:
                f.write("--- DESCRIPTIVE STATISTICS ---\n")
                for col, stats_info in self.metadata['descriptive_stats'].items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  Mean: {stats_info['mean']:.4f}\n")
                    f.write(f"  Median: {stats_info['median']:.4f}\n")
                    f.write(f"  Std Dev: {stats_info['std']:.4f}\n")
                    f.write(f"  Range: [{stats_info['min']:.4f}, {stats_info['max']:.4f}]\n")
                    f.write(f"  Skewness: {stats_info['skewness']:.4f}\n")
                    f.write(f"  Kurtosis: {stats_info['kurtosis']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"  Saved: {report_path.name}")
    
    def save_metadata(self):
        """Save metadata to JSON."""
        logger.info("\n--- SAVING METADATA ---")
        
        metadata_path = self.output_dir / "omni_metadata.json"
        
        # Convert non-serializable types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_metadata = convert_to_serializable(self.metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        logger.info(f"  Saved: {metadata_path.name}")
    
    def run(self):
        """Execute full Omni Protocol pipeline."""
        try:
            logger.info("="*80)
            logger.info("TITAN RS - OMNI PROTOCOL v2.0")
            logger.info("="*80 + "\n")
            
            if not self.load_data():
                return False
            
            self.profile_data()
            self.describe_statistics()
            self.analyze_distributions()
            self.correlation_analysis()
            self.categorical_analysis()
            self.outlier_detection()
            self.generate_summary_report()
            self.save_metadata()
            
            logger.info("\n" + "="*80)
            logger.info("✅ OMNI PROTOCOL COMPLETE")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("="*80 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            logger.error(traceback.format_exc())
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='TITAN RS - Omni Protocol: Universal Statistical Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory (default: input_dir/omni_results)')
    
    args = parser.parse_args()
    
    # Initialize and run Omni Protocol
    omni = OmniProtocol(input_file=args.input, output_dir=args.output_dir)
    success = omni.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
