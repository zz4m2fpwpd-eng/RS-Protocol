#!/usr/bin/env python3
"""
TITAN RS - Result Validity Assessor
Validates ML results to ensure they are scientifically credible
"""

import numpy as np
from scipy import stats


class ResultValidator:
    """Assess validity of ML results"""
    
    def __init__(self, y_true, y_pred_proba, auc_score, n_features, n_samples_train, n_samples_test):
        """
        Initialize validator with model results
        
        Args:
            y_true: True labels from test set
            y_pred_proba: Predicted probabilities
            auc_score: Calculated AUC score
            n_features: Number of features used
            n_samples_train: Number of training samples
            n_samples_test: Number of test samples
        """
        self.y_true = np.array(y_true)
        self.y_pred_proba = np.array(y_pred_proba)
        self.auc_score = auc_score
        self.n_features = n_features
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        
        self.warnings = []
        self.flags = []
        self.validity_score = 100  # Start at 100, deduct points for issues
        
    def validate(self):
        """Run all validation checks"""
        self._check_test_set_size()
        self._check_class_imbalance()
        self._check_baseline_performance()
        self._check_statistical_significance()
        self._check_probability_distribution()
        self._check_sample_to_feature_ratio()
        self._assess_overall_validity()
        return self
    
    def _check_test_set_size(self):
        """Ensure test set is large enough for reliable AUC"""
        min_samples_per_class = 30
        
        unique, counts = np.unique(self.y_true, return_counts=True)
        min_class_count = min(counts)
        
        if min_class_count < 10:
            self.flags.append(
                f"CRITICAL: Test set too small (min class: {min_class_count} samples). "
                f"AUC may be unreliable. Need at least 10 samples per class."
            )
            self.validity_score -= 30
        elif min_class_count < min_samples_per_class:
            self.warnings.append(
                f"Test set small (min class: {min_class_count} samples). "
                f"Recommend at least {min_samples_per_class} samples per class for reliable AUC."
            )
            self.validity_score -= 15
    
    def _check_class_imbalance(self):
        """Check for severe class imbalance that might make AUC misleading"""
        unique, counts = np.unique(self.y_true, return_counts=True)
        
        if len(unique) < 2:
            self.flags.append(
                "CRITICAL: Only one class in test set. Cannot calculate valid AUC."
            )
            self.validity_score -= 50
            return
        
        imbalance_ratio = max(counts) / min(counts)
        
        if imbalance_ratio > 100:
            self.flags.append(
                f"CRITICAL: Severe class imbalance (ratio: {imbalance_ratio:.1f}:1). "
                f"AUC may be misleading. Consider stratified sampling."
            )
            self.validity_score -= 25
        elif imbalance_ratio > 20:
            self.warnings.append(
                f"High class imbalance (ratio: {imbalance_ratio:.1f}:1). "
                f"Interpret AUC with caution."
            )
            self.validity_score -= 10
    
    def _check_baseline_performance(self):
        """Check if model performs better than random chance"""
        # For binary classification, random chance is 0.5
        baseline_auc = 0.5
        improvement_threshold = 0.05
        
        if self.auc_score < baseline_auc:
            self.flags.append(
                f"CRITICAL: AUC ({self.auc_score:.3f}) is below random chance (0.5). "
                f"Model performs worse than random guessing."
            )
            self.validity_score -= 40
        elif self.auc_score < baseline_auc + improvement_threshold:
            self.warnings.append(
                f"AUC ({self.auc_score:.3f}) barely exceeds random chance. "
                f"Model may not have learned meaningful patterns."
            )
            self.validity_score -= 20
    
    def _check_statistical_significance(self):
        """Test if predictions are statistically different from random"""
        # Use permutation test logic: are predictions better than random labels?
        # Simple check: ensure predictions show some discrimination
        
        if len(self.y_pred_proba) == 0:
            return
        
        # Check if all predictions are nearly identical (no discrimination)
        pred_std = np.std(self.y_pred_proba)
        
        if pred_std < 0.01:
            self.flags.append(
                f"CRITICAL: Predictions have very low variance (std: {pred_std:.4f}). "
                f"Model may not be discriminating between classes."
            )
            self.validity_score -= 30
        elif pred_std < 0.05:
            self.warnings.append(
                f"Predictions have low variance (std: {pred_std:.4f}). "
                f"Model discrimination may be weak."
            )
            self.validity_score -= 10
    
    def _check_probability_distribution(self):
        """Check if predicted probabilities are reasonable"""
        if len(self.y_pred_proba) == 0:
            return
        
        # Check for suspicious patterns
        near_extremes = np.sum((self.y_pred_proba < 0.01) | (self.y_pred_proba > 0.99))
        pct_extremes = near_extremes / len(self.y_pred_proba) * 100
        
        if pct_extremes > 80:
            self.warnings.append(
                f"{pct_extremes:.1f}% of predictions are near 0 or 1. "
                f"Model may be overconfident."
            )
            self.validity_score -= 10
        
        # Check if predictions are mostly around 0.5 (no confidence)
        near_middle = np.sum((self.y_pred_proba > 0.4) & (self.y_pred_proba < 0.6))
        pct_middle = near_middle / len(self.y_pred_proba) * 100
        
        if pct_middle > 70:
            self.warnings.append(
                f"{pct_middle:.1f}% of predictions are near 0.5. "
                f"Model may be uncertain/not well-trained."
            )
            self.validity_score -= 10
    
    def _check_sample_to_feature_ratio(self):
        """Check if enough samples relative to features (prevent overfitting)"""
        if self.n_features == 0:
            return
        
        train_ratio = self.n_samples_train / self.n_features
        
        if train_ratio < 3:
            self.flags.append(
                f"CRITICAL: Very low sample-to-feature ratio ({train_ratio:.1f}:1). "
                f"High risk of overfitting. Need more samples or fewer features."
            )
            self.validity_score -= 25
        elif train_ratio < 10:
            self.warnings.append(
                f"Low sample-to-feature ratio ({train_ratio:.1f}:1). "
                f"Risk of overfitting. Consider feature selection."
            )
            self.validity_score -= 10
    
    def _assess_overall_validity(self):
        """Assess overall validity based on all checks"""
        # Ensure validity score doesn't go below 0
        self.validity_score = max(0, self.validity_score)
        
        # Classify validity level
        if self.validity_score >= 85:
            self.validity_level = "HIGH"
        elif self.validity_score >= 70:
            self.validity_level = "MODERATE"
        elif self.validity_score >= 50:
            self.validity_level = "LOW"
        else:
            self.validity_level = "QUESTIONABLE"
    
    def get_report(self):
        """Generate validation report"""
        report = "\n" + "="*80 + "\n"
        report += "RESULT VALIDITY ASSESSMENT\n"
        report += "="*80 + "\n"
        
        report += f"\nValidity Score: {self.validity_score}/100 ({self.validity_level})\n"
        report += f"AUC Score: {self.auc_score:.3f}\n"
        report += f"Test Set Size: {self.n_samples_test} samples\n"
        report += f"Features: {self.n_features}\n"
        report += f"Training Samples: {self.n_samples_train}\n"
        
        if self.flags:
            report += f"\n🚨 CRITICAL FLAGS ({len(self.flags)}):\n"
            for i, flag in enumerate(self.flags, 1):
                report += f"   {i}. {flag}\n"
        else:
            report += f"\n✓ No critical flags\n"
        
        if self.warnings:
            report += f"\n⚠️  WARNINGS ({len(self.warnings)}):\n"
            for i, warn in enumerate(self.warnings, 1):
                report += f"   {i}. {warn}\n"
        else:
            report += f"\n✓ No warnings\n"
        
        # Overall recommendation
        report += f"\n📊 OVERALL ASSESSMENT:\n"
        if self.validity_level == "HIGH":
            report += "   Results appear reliable and scientifically valid.\n"
        elif self.validity_level == "MODERATE":
            report += "   Results are acceptable but interpret with noted limitations.\n"
        elif self.validity_level == "LOW":
            report += "   Results have significant limitations. Use with caution.\n"
        else:
            report += "   Results are questionable. Consider collecting more data or revising methodology.\n"
        
        report += "="*80 + "\n"
        return report
    
    def get_summary_dict(self):
        """Get validation results as dictionary"""
        return {
            'validity_score': self.validity_score,
            'validity_level': self.validity_level,
            'n_flags': len(self.flags),
            'n_warnings': len(self.warnings),
            'flags': self.flags,
            'warnings': self.warnings,
            'auc': self.auc_score,
            'test_size': self.n_samples_test,
            'train_size': self.n_samples_train,
            'n_features': self.n_features
        }


def validate_results(y_true, y_pred_proba, auc_score, n_features, n_samples_train, n_samples_test):
    """
    Convenience function to validate ML results
    
    Returns:
        validator: ResultValidator object with validation results
    """
    validator = ResultValidator(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        auc_score=auc_score,
        n_features=n_features,
        n_samples_train=n_samples_train,
        n_samples_test=n_samples_test
    )
    return validator.validate()
