#!/usr/bin/env python3
"""
TITAN RS Test Suite & Error Detector
Test on sample data, identify faults, suggest fixes
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SampleDataGenerator:
    """Generate synthetic test datasets"""
    
    @staticmethod
    def heart_disease_sample(n_rows=100):
        """Synthetic heart disease data"""
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(30, 80, n_rows),
            'sex': np.random.choice([0, 1], n_rows),
            'cp': np.random.randint(0, 4, n_rows),
            'trestbps': np.random.randint(90, 200, n_rows),
            'chol': np.random.randint(120, 400, n_rows),
            'fbs': np.random.choice([0, 1], n_rows),
            'restecg': np.random.randint(0, 2, n_rows),
            'thalach': np.random.randint(60, 200, n_rows),
            'exang': np.random.choice([0, 1], n_rows),
            'oldpeak': np.random.uniform(0, 7, n_rows),
            'slope': np.random.randint(0, 3, n_rows),
            'ca': np.random.randint(0, 4, n_rows),
            'thal': np.random.randint(1, 4, n_rows),
            'target': np.random.choice([0, 1], n_rows)
        })
        return df
    
    @staticmethod
    def fraud_detection_sample(n_rows=200):
        """Synthetic credit card fraud data"""
        np.random.seed(42)
        # Mostly legitimate, few frauds
        fraud_ratio = 0.02
        n_fraud = int(n_rows * fraud_ratio)
        
        df = pd.DataFrame({
            'amount': np.random.gamma(50, 2, n_rows),
            'merchant_type': np.random.randint(0, 10, n_rows),
            'timestamp_hour': np.random.randint(0, 24, n_rows),
            'days_since_last': np.random.randint(0, 30, n_rows),
            'user_age': np.random.randint(18, 80, n_rows),
            'is_fraud': np.concatenate([
                np.ones(n_fraud),
                np.zeros(n_rows - n_fraud)
            ])
        })
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
        return df
    
    @staticmethod
    def diabetes_sample(n_rows=150):
        """Synthetic diabetes data"""
        np.random.seed(42)
        df = pd.DataFrame({
            'pregnancies': np.random.randint(0, 10, n_rows),
            'glucose': np.random.randint(60, 200, n_rows),
            'blood_pressure': np.random.randint(60, 120, n_rows),
            'skin_thickness': np.random.randint(0, 100, n_rows),
            'insulin': np.random.randint(0, 900, n_rows),
            'bmi': np.random.uniform(18, 50, n_rows),
            'dpf': np.random.uniform(0.1, 2.5, n_rows),
            'age': np.random.randint(21, 80, n_rows),
            'outcome': np.random.choice([0, 1], n_rows, p=[0.67, 0.33])
        })
        return df
    
    @staticmethod
    def problematic_data(n_rows=100):
        """Data with known issues"""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_a': np.random.randn(n_rows),
            'feature_b': np.random.randn(n_rows),
            'target': np.random.choice([0, 1], n_rows)
        })
        
        # Issue 1: Add leakage (target duplicated as feature)
        df['feature_c'] = df['target'].copy()
        
        # Issue 2: Add missing values
        df.loc[::10, 'feature_a'] = np.nan
        
        # Issue 3: Add outliers
        df.loc[::15, 'feature_b'] = 999
        
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR DETECTOR & CRITIC
# ═══════════════════════════════════════════════════════════════════════════════

class DataQualityCritic:
    """Critique data quality and detect issues"""
    
    def __init__(self, df, name="dataset"):
        self.df = df
        self.name = name
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def analyze(self):
        """Run full analysis"""
        self._check_shape()
        self._check_missing()
        self._check_duplicates()
        self._check_outliers()
        self._check_leakage()
        self._check_types()
        return self
    
    def _check_shape(self):
        """Basic shape info"""
        self.stats['rows'] = len(self.df)
        self.stats['cols'] = len(self.df.columns)
        
        if len(self.df) < 50:
            self.warnings.append(f"Small dataset ({len(self.df)} rows) - may have unstable statistics")
        if len(self.df.columns) > 100:
            self.warnings.append(f"High dimensionality ({len(self.df.columns)} features)")
    
    def _check_missing(self):
        """Check for missing values"""
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        cols_with_missing = missing_pct[missing_pct > 0]
        
        if len(cols_with_missing) > 0:
            for col, pct in cols_with_missing.items():
                if pct > 50:
                    self.issues.append(f"Column '{col}': {pct:.1f}% missing (>50%)")
                elif pct > 20:
                    self.warnings.append(f"Column '{col}': {pct:.1f}% missing")
    
    def _check_duplicates(self):
        """Check for duplicate rows"""
        n_dupes = len(self.df) - len(self.df.drop_duplicates())
        if n_dupes > 0:
            pct = n_dupes / len(self.df) * 100
            if pct > 10:
                self.issues.append(f"{n_dupes} duplicate rows ({pct:.1f}%)")
            else:
                self.warnings.append(f"{n_dupes} duplicate rows ({pct:.1f}%)")
    
    def _check_outliers(self):
        """Detect outliers using IQR"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((self.df[col] < Q1 - 1.5*IQR) | 
                       (self.df[col] > Q3 + 1.5*IQR)).sum()
            
            if outliers > 0:
                pct = outliers / len(self.df) * 100
                if pct > 10:
                    self.issues.append(f"Column '{col}': {outliers} outliers ({pct:.1f}%)")
                elif pct > 2:
                    self.warnings.append(f"Column '{col}': {outliers} outliers ({pct:.1f}%)")
    
    def _check_leakage(self):
        """Detect potential data leakage"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if 'target' in self.df.columns:
            target = self.df['target']
            for col in numeric_cols:
                if col == 'target':
                    continue
                
                # Check if feature is suspiciously similar to target
                corr = abs(self.df[col].corr(target))
                if corr > 0.99:
                    self.issues.append(f"LEAKAGE: Column '{col}' highly correlated with target (r={corr:.4f})")
                elif corr > 0.95:
                    self.warnings.append(f"Potential leakage: Column '{col}' corr with target = {corr:.4f}")
    
    def _check_types(self):
        """Check for type issues"""
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                n_unique = self.df[col].nunique()
                if n_unique > len(self.df) * 0.5:
                    self.warnings.append(f"Column '{col}' (object): {n_unique} unique values (>50% of rows)")
    
    def report(self):
        """Generate human-readable report"""
        report = f"\n{'=' * 80}\n"
        report += f"DATA QUALITY REPORT: {self.name}\n"
        report += f"{'=' * 80}\n"
        report += f"\nShape: {self.stats['rows']} rows × {self.stats['cols']} columns\n"
        
        if self.issues:
            report += f"\n❌ CRITICAL ISSUES ({len(self.issues)}):\n"
            for issue in self.issues:
                report += f"   - {issue}\n"
        else:
            report += f"\n✓ No critical issues detected\n"
        
        if self.warnings:
            report += f"\n⚠️  WARNINGS ({len(self.warnings)}):\n"
            for warn in self.warnings:
                report += f"   - {warn}\n"
        else:
            report += f"\n✓ No warnings\n"
        
        report += f"\n{'=' * 80}\n"
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# CODE FAULT DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CodeFaultDetector:
    """Detect common faults in TITAN engines"""
    
    COMMON_FAULTS = {
        'hardcoded_path': {
            'pattern': r'["\']\/[a-zA-Z0-9_/\-\.]+["\']',
            'description': 'Hard-coded file path - use config/args instead',
            'severity': 'HIGH',
            'fix': 'Replace with argument or config variable'
        },
        'bare_except': {
            'pattern': r'except\s*:',
            'description': 'Bare except clause - catches all errors including KeyboardInterrupt',
            'severity': 'MEDIUM',
            'fix': 'Use except Exception as e: instead'
        },
        'magic_number': {
            'pattern': r'if\s+\w+\s*[<>=]{1,2}\s*\d{2,}',
            'description': 'Magic number threshold - move to config',
            'severity': 'LOW',
            'fix': 'Define as CONFIG constant with comment'
        },
        'no_logging': {
            'pattern': r'print\(',
            'description': 'Using print() instead of logging',
            'severity': 'LOW',
            'fix': 'Use logger.info() / logger.error() instead'
        }
    }
    
    @staticmethod
    def scan_file(file_path):
        """Scan Python file for common faults"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            import re
            faults = []
            
            for fault_name, fault_info in CodeFaultDetector.COMMON_FAULTS.items():
                matches = re.finditer(fault_info['pattern'], content)
                for match in matches:
                    # Count line number
                    line_num = content[:match.start()].count('\n') + 1
                    faults.append({
                        'type': fault_name,
                        'line': line_num,
                        'description': fault_info['description'],
                        'severity': fault_info['severity'],
                        'fix': fault_info['fix'],
                        'code': match.group(0)[:50]
                    })
            
            return faults
        except Exception as e:
            return [{'error': str(e)}]


# ═══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class TitanTestSuite:
    """Run full test suite on TITAN engines"""
    
    def __init__(self, output_dir='~/titan_test_results'):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'summary': {}
        }
    
    def test_data_quality(self):
        """Test on synthetic data with known issues"""
        print("\n[TEST 1] Data Quality Analysis")
        print("-" * 80)
        
        datasets = {
            'heart': SampleDataGenerator.heart_disease_sample(),
            'fraud': SampleDataGenerator.fraud_detection_sample(),
            'diabetes': SampleDataGenerator.diabetes_sample(),
            'problematic': SampleDataGenerator.problematic_data()
        }
        
        for name, df in datasets.items():
            print(f"\n  Testing: {name}")
            critic = DataQualityCritic(df, name).analyze()
            report = critic.report()
            print(report)
            
            # Save report
            report_file = self.output_dir / f"critique_{name}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            # Save sample data
            data_file = self.output_dir / f"sample_{name}.csv"
            df.to_csv(data_file, index=False)
            
            self.results['tests'].append({
                'test': 'data_quality',
                'dataset': name,
                'issues': len(critic.issues),
                'warnings': len(critic.warnings),
                'output_file': str(report_file)
            })
        
        print(f"\n✓ Data quality tests complete. Files saved to {self.output_dir}")
    
    def test_code_quality(self, engine_files=None):
        """Scan TITAN engines for code faults"""
        print("\n[TEST 2] Code Quality Scan")
        print("-" * 80)
        
        if engine_files is None:
            engine_files = list(Path('.').glob('TITAN*.py'))
        
        all_faults = []
        
        for engine_file in engine_files:
            print(f"\n  Scanning: {engine_file.name}")
            faults = CodeFaultDetector.scan_file(engine_file)
            
            if faults:
                for fault in faults:
                    if 'error' not in fault:
                        print(f"    Line {fault['line']}: [{fault['severity']}] {fault['description']}")
                        all_faults.append(fault)
            else:
                print(f"    ✓ No major faults detected")
            
            self.results['tests'].append({
                'test': 'code_quality',
                'file': engine_file.name,
                'faults_found': len(faults),
                'faults': faults[:5]  # Top 5
            })
        
        # Save fault report
        fault_report = self.output_dir / "code_faults_report.json"
        with open(fault_report, 'w') as f:
            json.dump(all_faults, f, indent=2)
        
        print(f"\n✓ Code quality scan complete. Report: {fault_report}")
    
    def save_summary(self):
        """Save test summary"""
        summary_file = self.output_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Summary saved: {summary_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 80)
    print("TITAN RS TEST SUITE & ERROR DETECTOR")
    print("=" * 80)
    
    suite = TitanTestSuite()
    
    try:
        # Run data quality tests
        suite.test_data_quality()
        
        # Run code quality scan
        suite.test_code_quality()
        
        # Save summary
        suite.save_summary()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS COMPLETE")
        print(f"Results saved to: {suite.output_dir}")
        print("=" * 80 + "\n")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
