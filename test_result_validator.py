#!/usr/bin/env python3
"""
Test Suite for TITAN RS Result Validator
Tests validation thresholds, scoring logic, and edge cases
"""

import unittest
import numpy as np
from result_validator import ResultValidator, validate_results


class TestResultValidatorBasics(unittest.TestCase):
    """Test basic functionality and initialization"""
    
    def test_initialization(self):
        """Test validator initializes correctly with valid inputs"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9])
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.85,
            n_features=10,
            n_samples_train=100,
            n_samples_test=50
        )
        
        self.assertEqual(len(validator.y_true), 6)
        self.assertEqual(len(validator.y_pred_proba), 6)
        self.assertEqual(validator.auc_score, 0.85)
        self.assertEqual(validator.validity_score, 100)
        self.assertEqual(len(validator.warnings), 0)
        self.assertEqual(len(validator.flags), 0)
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays"""
        validator = ResultValidator(
            y_true=np.array([]),
            y_pred_proba=np.array([]),
            auc_score=0.5,
            n_features=10,
            n_samples_train=100,
            n_samples_test=0
        )
        
        # Empty arrays should cause an error during validation
        # This is expected behavior - you can't validate with no data
        with self.assertRaises(ValueError):
            validator.validate()


class TestTestSetSizeValidation(unittest.TestCase):
    """Test validation of test set size"""
    
    def test_adequate_test_set(self):
        """Test with adequate samples per class"""
        np.random.seed(42)  # For reproducible tests
        # 50 samples per class - should be fine
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have no flags or warnings about test set size
        size_issues = [f for f in validator.flags if 'Test set too small' in f]
        self.assertEqual(len(size_issues), 0)
    
    def test_critical_small_test_set(self):
        """Test with critically small test set (<10 samples per class)"""
        np.random.seed(42)  # For reproducible tests
        # 5 samples per class - critical
        y_true = np.concatenate([np.zeros(5), np.ones(5)])
        y_pred_proba = np.random.rand(10)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=100,
            n_samples_test=10
        )
        validator.validate()
        
        # Should have critical flag
        critical_flags = [f for f in validator.flags if 'CRITICAL: Test set too small' in f]
        self.assertGreater(len(critical_flags), 0)
        self.assertLess(validator.validity_score, 100)
    
    def test_warning_small_test_set(self):
        """Test with small test set (10-30 samples per class)"""
        np.random.seed(42)  # For reproducible tests
        # 15 samples per class - warning level
        y_true = np.concatenate([np.zeros(15), np.ones(15)])
        y_pred_proba = np.random.rand(30)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=100,
            n_samples_test=30
        )
        validator.validate()
        
        # Should have warning but no critical flag
        warnings = [w for w in validator.warnings if 'Test set small' in w]
        self.assertGreater(len(warnings), 0)
        self.assertLess(validator.validity_score, 100)


class TestClassImbalanceValidation(unittest.TestCase):
    """Test detection of class imbalance"""
    
    def test_balanced_classes(self):
        """Test with balanced classes"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have no imbalance warnings or flags
        imbalance_issues = [f for f in validator.flags + validator.warnings 
                           if 'imbalance' in f.lower()]
        self.assertEqual(len(imbalance_issues), 0)
    
    def test_single_class_dataset(self):
        """Test with only one class (edge case)"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.ones(100)  # All same class
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.5,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have critical flag for single class
        single_class_flags = [f for f in validator.flags 
                             if 'Only one class' in f]
        self.assertGreater(len(single_class_flags), 0)
        self.assertLess(validator.validity_score, 50)
    
    def test_extreme_imbalance(self):
        """Test with extreme imbalance ratio (>100:1)"""
        np.random.seed(42)  # For reproducible tests
        # 200:1 ratio
        y_true = np.concatenate([np.zeros(200), np.ones(1)])
        y_pred_proba = np.random.rand(201)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=400,
            n_samples_test=201
        )
        validator.validate()
        
        # Should have critical flag for severe imbalance
        imbalance_flags = [f for f in validator.flags 
                          if 'Severe class imbalance' in f]
        self.assertGreater(len(imbalance_flags), 0)
    
    def test_moderate_imbalance(self):
        """Test with moderate imbalance ratio (20-100:1)"""
        np.random.seed(42)  # For reproducible tests
        # 30:1 ratio
        y_true = np.concatenate([np.zeros(90), np.ones(3)])
        y_pred_proba = np.random.rand(93)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=200,
            n_samples_test=93
        )
        validator.validate()
        
        # Should have warning for high imbalance
        imbalance_warnings = [w for w in validator.warnings 
                             if 'High class imbalance' in w]
        self.assertGreater(len(imbalance_warnings), 0)


class TestBaselinePerformance(unittest.TestCase):
    """Test baseline performance validation"""
    
    def test_good_performance(self):
        """Test with good AUC (>0.7)"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.85,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have no baseline performance issues
        baseline_issues = [f for f in validator.flags + validator.warnings 
                          if 'random chance' in f.lower()]
        self.assertEqual(len(baseline_issues), 0)
    
    def test_below_random_chance(self):
        """Test with AUC below random chance (<0.5)"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.45,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have critical flag
        baseline_flags = [f for f in validator.flags 
                         if 'below random chance' in f]
        self.assertGreater(len(baseline_flags), 0)
    
    def test_barely_above_baseline(self):
        """Test with AUC barely above random (0.5-0.55)"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.52,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have warning
        baseline_warnings = [w for w in validator.warnings 
                            if 'barely exceeds random chance' in w]
        self.assertGreater(len(baseline_warnings), 0)


class TestStatisticalSignificance(unittest.TestCase):
    """Test statistical significance checks"""
    
    def test_good_variance(self):
        """Test with good prediction variance"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        # Good variance in predictions
        y_pred_proba = np.concatenate([np.random.uniform(0.1, 0.4, 50),
                                       np.random.uniform(0.6, 0.9, 50)])
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.85,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have no variance issues
        variance_issues = [f for f in validator.flags + validator.warnings 
                          if 'variance' in f.lower()]
        self.assertEqual(len(variance_issues), 0)
    
    def test_critical_low_variance(self):
        """Test with critically low prediction variance"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        # All predictions very similar (std < PRED_STD_CRITICAL)
        # Use a std well below the critical threshold to ensure test reliability
        target_std = ResultValidator.PRED_STD_CRITICAL * 0.1
        y_pred_proba = np.full(100, 0.5) + np.random.normal(0, target_std, 100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.51,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have critical flag
        variance_flags = [f for f in validator.flags 
                         if 'very low variance' in f]
        self.assertGreater(len(variance_flags), 0)
    
    def test_warning_low_variance(self):
        """Test with low but not critical variance"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        # Low variance (PRED_STD_CRITICAL < std < PRED_STD_WARNING)
        # Use the midpoint between critical and warning thresholds
        target_std = (ResultValidator.PRED_STD_CRITICAL + ResultValidator.PRED_STD_WARNING) / 2
        y_pred_proba = np.full(100, 0.5) + np.random.normal(0, target_std, 100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.55,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have warning
        variance_warnings = [w for w in validator.warnings 
                            if 'low variance' in w]
        self.assertGreater(len(variance_warnings), 0)


class TestProbabilityDistribution(unittest.TestCase):
    """Test probability distribution validation"""
    
    def test_reasonable_distribution(self):
        """Test with reasonable probability distribution"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        # Mixed distribution
        y_pred_proba = np.random.beta(2, 2, 100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have no distribution issues
        dist_issues = [w for w in validator.warnings 
                      if 'near 0 or 1' in w or 'near 0.5' in w]
        self.assertEqual(len(dist_issues), 0)
    
    def test_overconfident_predictions(self):
        """Test with mostly extreme predictions (near 0 or 1)"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        # Most predictions near extremes (>80% to trigger warning)
        # Use validator's PROB_EXTREME constants
        y_pred_proba = np.concatenate([
            np.random.uniform(0, ResultValidator.PROB_EXTREME_LOW / 2, 45),
            np.random.uniform((1 + ResultValidator.PROB_EXTREME_HIGH) / 2, 1, 45),
            np.random.uniform(ResultValidator.PROB_MIDDLE_LOW, ResultValidator.PROB_MIDDLE_HIGH, 10)
        ])
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.85,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have warning about overconfidence
        overconf_warnings = [w for w in validator.warnings 
                            if 'near 0 or 1' in w]
        self.assertGreater(len(overconf_warnings), 0)
    
    def test_uncertain_predictions(self):
        """Test with mostly middle predictions (near 0.5)"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        # Most predictions near 0.5 - use validator's PROB_MIDDLE constants
        y_pred_proba = np.random.uniform(
            ResultValidator.PROB_MIDDLE_LOW, 
            ResultValidator.PROB_MIDDLE_HIGH, 
            100
        )
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.52,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        # Should have warning about uncertainty
        uncertain_warnings = [w for w in validator.warnings 
                             if 'near 0.5' in w]
        self.assertGreater(len(uncertain_warnings), 0)


class TestSampleFeatureRatio(unittest.TestCase):
    """Test sample-to-feature ratio validation"""
    
    def test_adequate_ratio(self):
        """Test with adequate sample-to-feature ratio (>10:1)"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=200,  # 200:10 = 20:1 ratio
            n_samples_test=100
        )
        validator.validate()
        
        # Should have no ratio issues
        ratio_issues = [f for f in validator.flags + validator.warnings 
                       if 'sample-to-feature ratio' in f]
        self.assertEqual(len(ratio_issues), 0)
    
    def test_critical_low_ratio(self):
        """Test with critically low ratio (<3:1)"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=100,
            n_samples_train=200,  # 200:100 = 2:1 ratio
            n_samples_test=100
        )
        validator.validate()
        
        # Should have critical flag
        ratio_flags = [f for f in validator.flags 
                      if 'Very low sample-to-feature ratio' in f]
        self.assertGreater(len(ratio_flags), 0)
    
    def test_warning_low_ratio(self):
        """Test with low ratio (3-10:1)"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=30,
            n_samples_train=200,  # 200:30 = 6.7:1 ratio
            n_samples_test=100
        )
        validator.validate()
        
        # Should have warning
        ratio_warnings = [w for w in validator.warnings 
                         if 'Low sample-to-feature ratio' in w]
        self.assertGreater(len(ratio_warnings), 0)


class TestValidityScoring(unittest.TestCase):
    """Test overall validity scoring logic"""
    
    def test_perfect_scenario(self):
        """Test with ideal data - should have high validity score"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(100), np.ones(100)])
        y_pred_proba = np.concatenate([np.random.uniform(0.1, 0.4, 100),
                                       np.random.uniform(0.6, 0.9, 100)])
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.85,
            n_features=10,
            n_samples_train=500,
            n_samples_test=200
        )
        validator.validate()
        
        # Should have high validity
        self.assertEqual(validator.validity_level, "HIGH")
        self.assertGreaterEqual(validator.validity_score, 85)
    
    def test_questionable_scenario(self):
        """Test with many issues - should have low validity score"""
        np.random.seed(42)  # For reproducible tests
        # Small, imbalanced, low performance
        y_true = np.concatenate([np.zeros(5), np.ones(1)])
        # Very low variance using a fraction of the critical threshold
        target_std = ResultValidator.PRED_STD_CRITICAL * 0.1
        y_pred_proba = np.full(6, 0.5) + np.random.normal(0, target_std, 6)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.48,
            n_features=50,
            n_samples_train=100,
            n_samples_test=6
        )
        validator.validate()
        
        # Should have low or questionable validity
        self.assertIn(validator.validity_level, ["LOW", "QUESTIONABLE"])
        self.assertLess(validator.validity_score, 50)
    
    def test_score_never_negative(self):
        """Test that validity score never goes below 0"""
        # Worst case scenario with multiple critical issues
        y_true = np.ones(3)  # Single class
        y_pred_proba = np.full(3, 0.5)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.3,
            n_features=100,
            n_samples_train=50,
            n_samples_test=3
        )
        validator.validate()
        
        # Score should be at least 0
        self.assertGreaterEqual(validator.validity_score, 0)


class TestReportGeneration(unittest.TestCase):
    """Test report generation functionality"""
    
    def test_get_report(self):
        """Test that get_report generates valid report string"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        report = validator.get_report()
        
        # Report should contain key sections
        self.assertIn("RESULT VALIDITY ASSESSMENT", report)
        self.assertIn("Validity Score:", report)
        self.assertIn("AUC Score:", report)
        self.assertIn("OVERALL ASSESSMENT:", report)
    
    def test_get_summary_dict(self):
        """Test that get_summary_dict returns valid dictionary"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = ResultValidator(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        validator.validate()
        
        summary = validator.get_summary_dict()
        
        # Should contain all expected keys
        expected_keys = ['validity_score', 'validity_level', 'n_flags', 
                        'n_warnings', 'flags', 'warnings', 'auc', 
                        'test_size', 'train_size', 'n_features']
        for key in expected_keys:
            self.assertIn(key, summary)


class TestConvenienceFunction(unittest.TestCase):
    """Test the validate_results convenience function"""
    
    def test_validate_results_function(self):
        """Test that convenience function works correctly"""
        np.random.seed(42)  # For reproducible tests
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_pred_proba = np.random.rand(100)
        
        validator = validate_results(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            auc_score=0.75,
            n_features=10,
            n_samples_train=200,
            n_samples_test=100
        )
        
        # Should return validated ResultValidator instance
        self.assertIsInstance(validator, ResultValidator)
        self.assertIsNotNone(validator.validity_level)
        self.assertIsNotNone(validator.validity_score)


class TestConfigurableThresholds(unittest.TestCase):
    """Test that validation thresholds are properly defined as class constants"""
    
    def test_all_thresholds_defined(self):
        """Test that all threshold constants are defined"""
        expected_constants = [
            'MIN_CLASS_SAMPLES_CRITICAL',
            'MIN_CLASS_SAMPLES_WARNING',
            'IMBALANCE_RATIO_CRITICAL',
            'IMBALANCE_RATIO_WARNING',
            'BASELINE_AUC',
            'BASELINE_IMPROVEMENT',
            'PRED_STD_CRITICAL',
            'PRED_STD_WARNING',
            'PROB_EXTREME_LOW',
            'PROB_EXTREME_HIGH',
            'PROB_MIDDLE_LOW',
            'PROB_MIDDLE_HIGH',
            'SAMPLE_FEATURE_RATIO_CRITICAL',
            'SAMPLE_FEATURE_RATIO_WARNING'
        ]
        
        for const in expected_constants:
            self.assertTrue(hasattr(ResultValidator, const),
                          f"Missing constant: {const}")
    
    def test_thresholds_are_reasonable(self):
        """Test that threshold values are reasonable"""
        self.assertGreater(ResultValidator.MIN_CLASS_SAMPLES_WARNING,
                          ResultValidator.MIN_CLASS_SAMPLES_CRITICAL)
        self.assertGreater(ResultValidator.IMBALANCE_RATIO_CRITICAL,
                          ResultValidator.IMBALANCE_RATIO_WARNING)
        self.assertEqual(ResultValidator.BASELINE_AUC, 0.5)
        self.assertLess(ResultValidator.PRED_STD_CRITICAL,
                       ResultValidator.PRED_STD_WARNING)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
