=================================================================
Ensembles Evaluation And Uncertainty Quantification
=================================================================

============================================================
DUAL-EXPERT LABELING — CERTAIN vs UNCERTAIN (HASH MATCHING)
============================================================
  Images in Expert-I:            1633
  Images in Expert-II:           1633
  Images matched:                1633

  Certain Labels (Experts agree): 1622  (99.3%)
  Uncertain Labels:                 11  (0.7%)

  Confusion Matrix (Expert-I vs Expert-II):
  (shows only disagreements )
  Expert-I \ II |  KL0  KL1  KL2  KL3  KL4
  ----------------------------------------
  KL0          |    0   11    0    0    0

============================================================
 DATA SPLIT
============================================================
  K-Fold CV data (85%): 1388 images
  Hold-out data (15%): 245 images

Hold-out: 245 samples
  Expert-certain:   244
  Expert-uncertain: 1
 Experts labels saved to: results/expert_agreement_labels.npy

 Building Homogeneous Ensemble for : resnet50

 Buliding model: resnet50

 Buliding model: resnet50
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

 Buliding model: resnet50

 Buliding model: resnet50

 Buliding model: resnet50
Evaluation of resnet50_Homogeneous
  Uncertainty per sample saved to: results/resnet50_Homogeneous_uncertainty.npz

 Building Homogeneous Ensemble for : efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3
Evaluation of efficientnet_b3_Homogeneous
  Uncertainty per sample saved to: results/efficientnet_b3_Homogeneous_uncertainty.npz

 Building Homogeneous Ensemble for : densenet121

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: densenet121
Evaluation of densenet121_Homogeneous
  Uncertainty per sample saved to: results/densenet121_Homogeneous_uncertainty.npz

 Building Homogeneous Ensemble for : mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large
Evaluation of mobilenetv3_large_Homogeneous
  Uncertainty per sample saved to: results/mobilenetv3_large_Homogeneous_uncertainty.npz

 Building Homogeneous Ensemble for : convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny
Evaluation of convnext_tiny_Homogeneous
  Uncertainty per sample saved to: results/convnext_tiny_Homogeneous_uncertainty.npz

 Building Heterogeneous Ensemble (Best fold of each architecture)

 Buliding model: resnet50

 Buliding model: efficientnet_b3

 Buliding model: densenet121

 Buliding model: mobilenetv3_large

 Buliding model: convnext_tiny
Evaluation of Heterogeneous_Avg
  Uncertainty per sample saved to: results/Heterogeneous_Avg_uncertainty.npz

 Building Weighted Ensemble (Mixture of Experts on F1 basis)

 Buliding model: resnet50

 Buliding model: efficientnet_b3

 Buliding model: densenet121

 Buliding model: mobilenetv3_large

 Buliding model: convnext_tiny
Evaluation of Heterogeneous_Weighted
  Uncertainty per sample saved to: results/Heterogeneous_Weighted_uncertainty.npz

Building MEGA ENSEMBLE (Type D: 25 models)

 Buliding model: resnet50

 Buliding model: resnet50

 Buliding model: resnet50

 Buliding model: resnet50

 Buliding model: resnet50

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny
Evaluation of Mega_Ensemble_TypD
  Uncertainty per sample saved to: results/Mega_Ensemble_TypD_uncertainty.npz

Building CLASS-SPECIFIC ENSEMBLES (Version 1 and 2)

 Buliding model: resnet50

 Buliding model: resnet50

 Buliding model: resnet50

 Buliding model: resnet50

 Buliding model: resnet50

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: efficientnet_b3

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: densenet121

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: mobilenetv3_large

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

 Buliding model: convnext_tiny

  Version 1: Best of the best with allowed repetitions:
    KL0: convnext_tiny_f2       (F1: 0.9326)
    KL1: densenet121_f4         (F1: 0.8242)
    KL2: densenet121_f2         (F1: 0.7619)
    KL3: densenet121_f5         (F1: 0.9114)
    KL4: densenet121_f5         (F1: 0.9296)

  Version 2: Diverse ensemble, unique specialists:
    KL0: convnext_tiny_f2       (F1: 0.9326)
    KL1: densenet121_f4         (F1: 0.8242)
    KL2: densenet121_f2         (F1: 0.7619)
    KL3: densenet121_f5         (F1: 0.9114)
    KL4: efficientnet_b3_f1     (F1: 0.9167)
Evaluation of Class_Specific_With_Rep
  Uncertainty per sample saved to: results/Class_Specific_With_Rep_uncertainty.npz
Evaluation of Class_Specific_Unique
  Uncertainty per sample saved to: results/Class_Specific_Unique_uncertainty.npz


=================================================================
UQ Analysis — 3σ Threshold and Comparison with Experts
=================================================================

  Uncertainty threshold (3σ):
    mean(unc) = 0.0531
    std(unc)  = 0.0315
    threshold = 0.1477
    Uncertain samples (unc > threshold): 1 / 245

=================================================================
UQ VALIDATION: resnet50_Homogeneous
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   1 (0.4%)

  AUROC (uncertain detection):     0.6025
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0531
    Uncertain (experts disagree):    0.0595

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      243        1
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           0.99       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      0.99      0.99       245


  Uncertainty threshold (3σ):
    mean(unc) = 0.0925
    std(unc)  = 0.0626
    threshold = 0.2802
    Uncertain samples (unc > threshold): 0 / 245

=================================================================
UQ VALIDATION: efficientnet_b3_Homogeneous
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   0 (0.0%)

  AUROC (uncertain detection):     0.6475
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0923
    Uncertain (experts disagree):    0.1308

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      244        0
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           1.00       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      1.00      0.99       245


  Uncertainty threshold (3σ):
    mean(unc) = 0.0447
    std(unc)  = 0.0414
    threshold = 0.1689
    Uncertain samples (unc > threshold): 0 / 245

=================================================================
UQ VALIDATION: densenet121_Homogeneous
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   0 (0.0%)

  AUROC (uncertain detection):     0.7500
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0446
    Uncertain (experts disagree):    0.0725

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      244        0
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           1.00       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      1.00      0.99       245


  Uncertainty threshold (3σ):
    mean(unc) = 0.0617
    std(unc)  = 0.0557
    threshold = 0.2287
    Uncertain samples (unc > threshold): 0 / 245

=================================================================
UQ VALIDATION: mobilenetv3_large_Homogeneous
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   0 (0.0%)

  AUROC (uncertain detection):     0.8074
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0615
    Uncertain (experts disagree):    0.1192

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      244        0
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           1.00       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      1.00      0.99       245


  Uncertainty threshold (3σ):
    mean(unc) = 0.0584
    std(unc)  = 0.0391
    threshold = 0.1759
    Uncertain samples (unc > threshold): 1 / 245

=================================================================
UQ VALIDATION: convnext_tiny_Homogeneous
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   1 (0.4%)

  AUROC (uncertain detection):     0.6967
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0584
    Uncertain (experts disagree):    0.0779

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      243        1
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           0.99       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      0.99      0.99       245


  Uncertainty threshold (3σ):
    mean(unc) = 0.0745
    std(unc)  = 0.0500
    threshold = 0.2245
    Uncertain samples (unc > threshold): 0 / 245

=================================================================
UQ VALIDATION: Heterogeneous_Avg
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   0 (0.0%)

  AUROC (uncertain detection):     0.9139
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0742
    Uncertain (experts disagree):    0.1510

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      244        0
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           1.00       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      1.00      0.99       245


  Uncertainty threshold (3σ):
    mean(unc) = 0.0745
    std(unc)  = 0.0500
    threshold = 0.2245
    Uncertain samples (unc > threshold): 0 / 245

=================================================================
UQ VALIDATION: Heterogeneous_Weighted
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   0 (0.0%)

  AUROC (uncertain detection):     0.9139
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0742
    Uncertain (experts disagree):    0.1510

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      244        0
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           1.00       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      1.00      0.99       245


  Uncertainty threshold (3σ):
    mean(unc) = 0.0959
    std(unc)  = 0.0473
    threshold = 0.2377
    Uncertain samples (unc > threshold): 0 / 245

=================================================================
UQ VALIDATION: Mega_Ensemble_TypD
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   0 (0.0%)

  AUROC (uncertain detection):     0.7295
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0958
    Uncertain (experts disagree):    0.1278

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      244        0
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           1.00       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      1.00      0.99       245


  Uncertainty threshold (3σ):
    mean(unc) = 0.0447
    std(unc)  = 0.0461
    threshold = 0.1831
    Uncertain samples (unc > threshold): 0 / 245

=================================================================
UQ VALIDATION: Class_Specific_With_Rep
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   0 (0.0%)

  AUROC (uncertain detection):     0.6926
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0446
    Uncertain (experts disagree):    0.0567

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      244        0
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           1.00       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      1.00      0.99       245


  Uncertainty threshold (3σ):
    mean(unc) = 0.0536
    std(unc)  = 0.0514
    threshold = 0.2080
    Uncertain samples (unc > threshold): 1 / 245

=================================================================
UQ VALIDATION: Class_Specific_Unique
=================================================================
  Total Samples:                  245
  Expert-uncertain (ground truth): 1 (0.4%)
  Ensemble-flagged (>threshold):   1 (0.4%)

  AUROC (uncertain detection):     0.8115
  F1 (uncertain class):            0.0000

  average unc(x):
    Certain   (experts agree):   0.0534
    Uncertain (experts disagree):    0.1108

  Confusion Matrix [Certain/Uncertain]:
    Predicted →     Certain  Uncertain
    True Certain:      243        1
    True Uncertain:      1        0

  Classification report:
              precision    recall  f1-score   support

     Certain       1.00      1.00      1.00       244
   Uncertain       0.00      0.00      0.00         1

    accuracy                           0.99       245
   macro avg       0.50      0.50      0.50       245
weighted avg       0.99      0.99      0.99       245



=================================================================
 MANN-WHITNEY U Statistical test — Mega_Ensemble_TypD
=================================================================
H0: Ensemble std is identical for certain and uncertain.
H1: Ensemble std is higher for uncertain group (p < 0.05).

  Test Mann-Whitney U (certain vs uncertain):
    n_certain   = 244
    n_uncertain = 1
    U           = 66.00
    p-value     = 0.216305  Insignificant
    effect r    = 0.4590  (average efect)

=================================================================
 SAVING RESULTS TO EXCEL (PANDAS)
=================================================================
 All tables compiled successfully to:
  results/MASTER_RESULTS_SUMMARY.xlsx


=========================================================================================================
 KL Classification Results — Ensembles tested on HOLD-OUT
=========================================================================================================
Model                           Kappa   F1-Mac |  UQ-Mean |    KL0    KL1    KL2    KL3    KL4
---------------------------------------------------------------------------------------------------------
Heterogeneous_Avg              0.9665   0.9136 |   0.0745 | 0.9202 0.8217 0.8571 0.9846 0.9841
Heterogeneous_Weighted         0.9665   0.9136 |   0.0745 | 0.9202 0.8217 0.8571 0.9846 0.9841
Mega_Ensemble_TypD             0.9644   0.9201 |   0.0959 | 0.9125 0.8333 0.8857 1.0000 0.9688
densenet121_Homogeneous        0.9639   0.9206 |   0.0447 | 0.9091 0.8429 0.8824 1.0000 0.9688
Class_Specific_Unique          0.9620   0.8965 |   0.0536 | 0.8974 0.8261 0.8696 0.9355 0.9538
Class_Specific_With_Rep        0.9592   0.8962 |   0.0447 | 0.8974 0.8261 0.8657 0.9524 0.9394
convnext_tiny_Homogeneous      0.9587   0.8892 |   0.0584 | 0.9290 0.8271 0.7838 0.9375 0.9688
mobilenetv3_large_Homogeneous   0.9478   0.8877 |   0.0617 | 0.8957 0.7717 0.8493 0.9841 0.9375
efficientnet_b3_Homogeneous    0.9455   0.8838 |   0.0925 | 0.8696 0.7727 0.8529 0.9394 0.9841
resnet50_Homogeneous           0.9344   0.8098 |   0.0531 | 0.8452 0.6393 0.7042 0.9062 0.9538
=========================================================================================================


===============================================================================================
 UNCERTAIN Detection — Validation by experts agreement
===============================================================================================
Model                          AUROC  F1-Unc  Flagged   E-Unc |  μ-unc(C)  μ-unc(U)
                                               (pred)  (true) |   certain uncertain
-----------------------------------------------------------------------------------------------
Heterogeneous_Avg             0.9139  0.0000       0       1  |    0.0742    0.1510
Heterogeneous_Weighted        0.9139  0.0000       0       1  |    0.0742    0.1510
Class_Specific_Unique         0.8115  0.0000       1       1  |    0.0534    0.1109
mobilenetv3_large_Homogeneous  0.8074  0.0000       0       1  |    0.0615    0.1192
densenet121_Homogeneous       0.7500  0.0000       0       1  |    0.0446    0.0725
Mega_Ensemble_TypD            0.7295  0.0000       0       1  |    0.0958    0.1278
convnext_tiny_Homogeneous     0.6967  0.0000       1       1  |    0.0584    0.0779
Class_Specific_With_Rep       0.6926  0.0000       0       1  |    0.0446    0.0567
efficientnet_b3_Homogeneous   0.6475  0.0000       0       1  |    0.0923    0.1308
resnet50_Homogeneous          0.6025  0.0000       1       1  |    0.0531    0.0595
===============================================================================================
AUROC: Separation ability certain/uncertain | μ-unc(C/U): Mean unc(x) in each group
