# Result Validity Assessment

## Overview

The TITAN RS Protocol now includes automatic result validity assessment to ensure machine learning results are scientifically credible and reliable. This feature helps identify potentially misleading or unreliable results before they are reported.

## Features

The result validator performs comprehensive checks on ML model outputs:

### 1. Test Set Size Validation
- **Check**: Ensures test set has sufficient samples per class
- **Critical Threshold**: < 10 samples per class
- **Warning Threshold**: < 30 samples per class
- **Why**: Small test sets lead to unreliable AUC estimates with high variance

### 2. Class Imbalance Detection
- **Check**: Identifies severe imbalances in test set
- **Critical Threshold**: > 100:1 ratio
- **Warning Threshold**: > 20:1 ratio
- **Why**: Extreme imbalance can make AUC misleading; model may just predict majority class

### 3. Baseline Performance Check
- **Check**: Ensures model performs better than random guessing
- **Critical Threshold**: AUC < 0.5 (worse than random)
- **Warning Threshold**: AUC < 0.55 (barely better than random)
- **Why**: Models worse than random indicate fundamental issues

### 4. Statistical Significance
- **Check**: Verifies predictions show meaningful discrimination
- **Critical Threshold**: Prediction std < 0.01
- **Warning Threshold**: Prediction std < 0.05
- **Why**: Low variance means model isn't differentiating between classes

### 5. Probability Distribution Analysis
- **Check**: Examines distribution of predicted probabilities
- **Warning**: > 80% predictions near extremes (overconfidence)
- **Warning**: > 70% predictions near 0.5 (under-confidence)
- **Why**: Unrealistic probability distributions indicate calibration issues

### 6. Sample-to-Feature Ratio
- **Check**: Ensures enough training data relative to features
- **Critical Threshold**: < 3:1 ratio
- **Warning Threshold**: < 10:1 ratio
- **Why**: Low ratio leads to overfitting and unreliable generalization

## Validity Scores

Results receive a validity score from 0-100 and a classification:

- **HIGH (85-100)**: Results appear reliable and scientifically valid
- **MODERATE (70-84)**: Results acceptable but interpret with noted limitations
- **LOW (50-69)**: Results have significant limitations; use with caution
- **QUESTIONABLE (0-49)**: Results questionable; consider collecting more data or revising methodology

## Output

### In REPORT.md
The validation assessment is automatically included in the audit report:

```markdown
# TITAN AUDIT: dataset.csv
Target: outcome
AUC: 0.946
Threats Removed: 50
Validity Score: 100/100 (HIGH)

## Statistical Analysis
- > Normality Check (age): p=0.0000 -> Protocol: Kruskal-Wallis

## Validity Assessment
================================================================================
RESULT VALIDITY ASSESSMENT
================================================================================

Validity Score: 100/100 (HIGH)
AUC Score: 0.946
Test Set Size: 285 samples
Features: 5
Training Samples: 570

✓ No critical flags
✓ No warnings

📊 OVERALL ASSESSMENT:
   Results appear reliable and scientifically valid.
================================================================================
```

### In Console Output
```
[VALIDATOR] Assessing result validity...
  ✓ Validity Score: 100/100 (HIGH)
```

### In Log Files
```
[2025-12-20 12:00:00] dataset.csv | SUCCESS | AUC 0.946 | 31 Charts | Validity: HIGH
```

## Usage

### Automatic (Default)
Validation is automatically applied in:
- `RSTITAN.py`
- `TITAN_RS_Fork.py`

No additional configuration needed. If `result_validator.py` is present, validation runs automatically.

### Programmatic Usage
```python
from result_validator import validate_results

# After training and testing your model
validator = validate_results(
    y_true=y_test,
    y_pred_proba=predicted_probabilities,
    auc_score=calculated_auc,
    n_features=X_train.shape[1],
    n_samples_train=len(X_train),
    n_samples_test=len(X_test)
)

# Get detailed report
print(validator.get_report())

# Get summary dictionary
summary = validator.get_summary_dict()
print(f"Validity: {summary['validity_level']}")
print(f"Score: {summary['validity_score']}/100")
print(f"Flags: {summary['n_flags']}")
print(f"Warnings: {summary['n_warnings']}")
```

## Example Scenarios

### Scenario 1: Small Test Set
```
Input: 5 samples per class
Result: CRITICAL FLAG - Test set too small
Validity: LOW (30/100)
Recommendation: Collect more data
```

### Scenario 2: Below Baseline
```
Input: AUC = 0.47
Result: CRITICAL FLAG - Performs worse than random
Validity: LOW (60/100)
Recommendation: Review feature engineering and model selection
```

### Scenario 3: Class Imbalance
```
Input: 950 class 0, 10 class 1
Result: CRITICAL FLAG - Severe imbalance (95:1)
Validity: MODERATE (75/100)
Recommendation: Use stratified sampling or resampling techniques
```

### Scenario 4: Good Results
```
Input: Balanced classes, AUC=0.85, sufficient samples
Result: ✓ No flags or warnings
Validity: HIGH (100/100)
Recommendation: Results are scientifically credible
```

## Benefits

1. **Credibility**: Automatically flags unreliable results before publication
2. **Transparency**: Clear documentation of result quality
3. **Education**: Helps users understand ML result limitations
4. **Publication Ready**: Provides evidence of rigorous validation for papers
5. **Early Detection**: Catches data quality issues early in the pipeline

## Implementation Details

- **Module**: `result_validator.py`
- **Dependencies**: `numpy`, `scipy`
- **Performance**: Negligible overhead (< 1 second per validation)
- **Compatibility**: Works with binary classification tasks (primary use case)

## Future Enhancements

Potential additions:
- Multi-class classification support
- Regression task validation
- Cross-validation stability checks
- Bootstrap confidence intervals
- Permutation test p-values
- Custom validation rules via configuration

## References

This validation framework follows best practices from:
- Saito & Rehmsmeier (2015) - "The Precision-Recall Plot Is More Informative than the ROC Plot"
- Provost et al. (1998) - "The Case Against Accuracy Estimation for Comparing Induction Algorithms"
- Japkowicz & Shah (2011) - "Evaluating Learning Algorithms: A Classification Perspective"

## Support

For issues or questions about result validation:
1. Check the validation report in REPORT.md
2. Review flags and warnings to understand issues
3. Consult this documentation for recommendations
4. Open an issue on GitHub if you believe validation is incorrect
