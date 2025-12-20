# TITAN AUDIT: test_heart.csv
Target: target
AUC: 0.493
Threats Removed: 0
Validity Score: 50/100 (LOW)

## Statistical Analysis
-      > Normality Check (age): p=0.0000 -> Protocol: Kruskal-Wallis

## Validity Assessment

================================================================================
RESULT VALIDITY ASSESSMENT
================================================================================

Validity Score: 50/100 (LOW)
AUC Score: 0.493
Test Set Size: 150 samples
Features: 13
Training Samples: 300

🚨 CRITICAL FLAGS (1):
   1. CRITICAL: AUC (0.493) is below random chance (0.5). Model performs worse than random guessing.

⚠️  WARNINGS (1):
   1. 86.7% of predictions are near 0.5. Model may be uncertain/not well-trained.

📊 OVERALL ASSESSMENT:
   Results have significant limitations. Use with caution.
================================================================================
