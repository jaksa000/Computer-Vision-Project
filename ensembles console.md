=================================================================
Ensemble Evaluation and Uncertainty Quantification
=================================================================

============================================================
DUAL-EXPERT LABELLING — Hungarian image matching (128×128 MSE)
============================================================
  Loading Expert-I  (MedicalExpert-I.
  Loading Expert-II (MedicalExpert-II)

  Expert-I images:  1650
  Expert-II images: 1650
  Match cache saved: results/expert_matches_cache.csv

  Images matched:               1650
  Certain  (experts agree):     1639  (99.3%)
  Uncertain (experts disagree):   11  (0.7%)
  [OK] Split integrity verified — no overlap between CV (1402) and holdout (248) sets.
  [OK] Holdout matches split manifest exactly (248 images).

============================================================
 DATA SPLIT — loaded from persistent manifest
============================================================
  K-Fold CV data: 1402 images
  Hold-out data:  248 images
  Manifest:       results/split_manifest.json
  [OK] Split integrity verified — no overlap between CV (1402) and holdout (248) sets.

  CV set (threshold):   1402 samples
  Holdout (KL):         248 samples
  Full dataset (UQ):    1650 samples
    Expert-certain:     1639
    Expert-uncertain:   11

=================================================================
 STEP 1 & 2: Evaluate Ensembles (with CV thresholds)
=================================================================

 Building Homogeneous Ensemble: resnet50

 Building model: resnet50
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
model.safetensors: 100% 102M/102M [00:01<00:00, 74.0MB/s]
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable
  Computing CV uncertainty scores for resnet50_Homogeneous...
    CV samples: 1402
  [resnet50_Homogeneous] Thresholds (95th pct) -> unc_mean: 0.1210,  unc_max: 0.2865,  entropy: 1.0463

Evaluating: resnet50_Homogeneous
  [Holdout]  Kappa: 0.9341  F1: 0.8212  MAE: 0.2097  Off-1: 0.9839  ECE: 0.0354  Brier: 0.0520
  [Full dataset] Saved: results/ensembles/resnet50_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: resnet50_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.6788  AUPRC=0.0112  F1=0.0000  flagged=81/1650  μ_certain=0.0464  μ_uncertain=0.0655
    [unc_max  ]  AUROC=0.6769  AUPRC=0.0109  F1=0.0000  flagged=78/1650  μ_certain=0.1074  μ_uncertain=0.1509
    [entropy  ]  AUROC=0.7550  AUPRC=0.0241  F1=0.0400  flagged=89/1650  μ_certain=0.5125  μ_uncertain=0.8064

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 4 : 43 : 22 : 8 : 12
              precision    recall  f1-score   support

     Certain       0.99      0.95      0.97      1639
   Uncertain       0.02      0.18      0.04        11

    accuracy                           0.94      1650
   macro avg       0.51      0.56      0.51      1650
weighted avg       0.99      0.94      0.96      1650


 Building Homogeneous Ensemble: efficientnet_b3

 Building model: efficientnet_b3
model.safetensors: 100% 49.3M/49.3M [00:00<00:00, 64.5MB/s]
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable
  Computing CV uncertainty scores for efficientnet_b3_Homogeneous...
    CV samples: 1402
  [efficientnet_b3_Homogeneous] Thresholds (95th pct) -> unc_mean: 0.1810,  unc_max: 0.3911,  entropy: 1.0339

Evaluating: efficientnet_b3_Homogeneous
  [Holdout]  Kappa: 0.9320  F1: 0.8507  MAE: 0.1895  Off-1: 0.9718  ECE: 0.0748  Brier: 0.0523
  [Full dataset] Saved: results/ensembles/efficientnet_b3_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: efficientnet_b3_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.5035  AUPRC=0.0077  F1=0.0000  flagged=86/1650  μ_certain=0.0847  μ_uncertain=0.0850
    [unc_max  ]  AUROC=0.4711  AUPRC=0.0066  F1=0.0000  flagged=86/1650  μ_certain=0.1933  μ_uncertain=0.1844
    [entropy  ]  AUROC=0.5536  AUPRC=0.0111  F1=0.0183  flagged=98/1650  μ_certain=0.4996  μ_uncertain=0.5703

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 10 : 35 : 31 : 6 : 16
              precision    recall  f1-score   support

     Certain       0.99      0.94      0.97      1639
   Uncertain       0.01      0.09      0.02        11

    accuracy                           0.94      1650
   macro avg       0.50      0.52      0.49      1650
weighted avg       0.99      0.94      0.96      1650


 Building Homogeneous Ensemble: densenet121

 Building model: densenet121
model.safetensors: 100% 32.3M/32.3M [00:00<00:00, 36.0MB/s]
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable
  Computing CV uncertainty scores for densenet121_Homogeneous...
    CV samples: 1402
  [densenet121_Homogeneous] Thresholds (95th pct) -> unc_mean: 0.1339,  unc_max: 0.3260,  entropy: 0.8576

Evaluating: densenet121_Homogeneous
  [Holdout]  Kappa: 0.9316  F1: 0.8738  MAE: 0.1734  Off-1: 0.9718  ECE: 0.0525  Brier: 0.0401
  [Full dataset] Saved: results/ensembles/densenet121_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: densenet121_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.7017  AUPRC=0.0134  F1=0.0000  flagged=91/1650  μ_certain=0.0412  μ_uncertain=0.0692
    [unc_max  ]  AUROC=0.7006  AUPRC=0.0129  F1=0.0000  flagged=85/1650  μ_certain=0.0977  μ_uncertain=0.1617
    [entropy  ]  AUROC=0.7310  AUPRC=0.0650  F1=0.0696  flagged=104/1650  μ_certain=0.3673  μ_uncertain=0.6338

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 22 : 50 : 14 : 10 : 8
              precision    recall  f1-score   support

     Certain       1.00      0.94      0.97      1639
   Uncertain       0.04      0.36      0.07        11

    accuracy                           0.94      1650
   macro avg       0.52      0.65      0.52      1650
weighted avg       0.99      0.94      0.96      1650


 Building Homogeneous Ensemble: mobilenetv3_large

 Building model: mobilenetv3_large
model.safetensors: 100% 22.1M/22.1M [00:00<00:00, 30.0MB/s]
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable
  Computing CV uncertainty scores for mobilenetv3_large_Homogeneous...
    CV samples: 1402
  [mobilenetv3_large_Homogeneous] Thresholds (95th pct) -> unc_mean: 0.1502,  unc_max: 0.3551,  entropy: 0.8319

Evaluating: mobilenetv3_large_Homogeneous
  [Holdout]  Kappa: 0.9192  F1: 0.8431  MAE: 0.1976  Off-1: 0.9597  ECE: 0.0334  Brier: 0.0479
  [Full dataset] Saved: results/ensembles/mobilenetv3_large_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: mobilenetv3_large_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.6535  AUPRC=0.0116  F1=0.0194  flagged=92/1650  μ_certain=0.0510  μ_uncertain=0.0713
    [unc_max  ]  AUROC=0.6337  AUPRC=0.0100  F1=0.0000  flagged=86/1650  μ_certain=0.1201  μ_uncertain=0.1601
    [entropy  ]  AUROC=0.6933  AUPRC=0.0181  F1=0.0500  flagged=109/1650  μ_certain=0.3351  μ_uncertain=0.5358

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 14 : 48 : 25 : 6 : 16
              precision    recall  f1-score   support

     Certain       0.99      0.94      0.96      1639
   Uncertain       0.03      0.27      0.05        11

    accuracy                           0.93      1650
   macro avg       0.51      0.60      0.51      1650
weighted avg       0.99      0.93      0.96      1650


 Building Homogeneous Ensemble: convnext_tiny

 Building model: convnext_tiny
model.safetensors: 100% 114M/114M [00:01<00:00, 105MB/s]  
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable
  Computing CV uncertainty scores for convnext_tiny_Homogeneous...
    CV samples: 1402
  [convnext_tiny_Homogeneous] Thresholds (95th pct) -> unc_mean: 0.1370,  unc_max: 0.3267,  entropy: 0.8068

Evaluating: convnext_tiny_Homogeneous
  [Holdout]  Kappa: 0.9382  F1: 0.8542  MAE: 0.1815  Off-1: 0.9718  ECE: 0.0203  Brier: 0.0455
  [Full dataset] Saved: results/ensembles/convnext_tiny_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: convnext_tiny_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.6402  AUPRC=0.0133  F1=0.0404  flagged=88/1650  μ_certain=0.0453  μ_uncertain=0.0652
    [unc_max  ]  AUROC=0.6385  AUPRC=0.0114  F1=0.0200  flagged=89/1650  μ_certain=0.1076  μ_uncertain=0.1510
    [entropy  ]  AUROC=0.6615  AUPRC=0.0174  F1=0.0381  flagged=94/1650  μ_certain=0.3495  μ_uncertain=0.5141

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 13 : 55 : 13 : 5 : 8
              precision    recall  f1-score   support

     Certain       0.99      0.94      0.97      1639
   Uncertain       0.02      0.18      0.04        11

    accuracy                           0.94      1650
   macro avg       0.51      0.56      0.50      1650
weighted avg       0.99      0.94      0.96      1650


 Building Heterogeneous Ensemble (best CV-fold per architecture)

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable
  Computing CV uncertainty scores for Heterogeneous_Avg...
    CV samples: 1402
  [Heterogeneous_Avg] Thresholds (95th pct) -> unc_mean: 0.1399,  unc_max: 0.3272,  entropy: 0.8804

Evaluating: Heterogeneous_Avg
  [Holdout]  Kappa: 0.9199  F1: 0.8508  MAE: 0.1935  Off-1: 0.9677  ECE: 0.0775  Brier: 0.0415
  [Full dataset] Saved: results/ensembles/Heterogeneous_Avg_uncertainty.npz

=================================================================
UQ VALIDATION: Heterogeneous_Avg
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.6921  AUPRC=0.0117  F1=0.0000  flagged=100/1650  μ_certain=0.0544  μ_uncertain=0.0806
    [unc_max  ]  AUROC=0.6942  AUPRC=0.0118  F1=0.0000  flagged=95/1650  μ_certain=0.1254  μ_uncertain=0.1879
    [entropy  ]  AUROC=0.7214  AUPRC=0.0198  F1=0.0354  flagged=102/1650  μ_certain=0.4094  μ_uncertain=0.6361

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 8 : 46 : 25 : 8 : 15
              precision    recall  f1-score   support

     Certain       0.99      0.94      0.97      1639
   Uncertain       0.02      0.18      0.04        11

    accuracy                           0.93      1650
   macro avg       0.51      0.56      0.50      1650
weighted avg       0.99      0.93      0.96      1650


 Building Weighted Ensemble (per-class F1 weights, best CV-fold)

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable
  Computing CV uncertainty scores for Heterogeneous_Weighted...
    CV samples: 1402
  [Heterogeneous_Weighted] Thresholds (95th pct) -> unc_mean: 0.1399,  unc_max: 0.3272,  entropy: 0.8742

Evaluating: Heterogeneous_Weighted
  [Holdout]  Kappa: 0.9199  F1: 0.8508  MAE: 0.1935  Off-1: 0.9677  ECE: 0.0837  Brier: 0.0414
  [Full dataset] Saved: results/ensembles/Heterogeneous_Weighted_uncertainty.npz

=================================================================
UQ VALIDATION: Heterogeneous_Weighted
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.6921  AUPRC=0.0117  F1=0.0000  flagged=100/1650  μ_certain=0.0544  μ_uncertain=0.0806
    [unc_max  ]  AUROC=0.6942  AUPRC=0.0118  F1=0.0000  flagged=95/1650  μ_certain=0.1254  μ_uncertain=0.1879
    [entropy  ]  AUROC=0.7239  AUPRC=0.0201  F1=0.0354  flagged=102/1650  μ_certain=0.4104  μ_uncertain=0.6407

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 9 : 47 : 24 : 7 : 15
              precision    recall  f1-score   support

     Certain       0.99      0.94      0.97      1639
   Uncertain       0.02      0.18      0.04        11

    accuracy                           0.93      1650
   macro avg       0.51      0.56      0.50      1650
weighted avg       0.99      0.93      0.96      1650


 Building Mega Ensemble (Type D — 25 models)

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable
  Loaded 25 models.
  Computing CV uncertainty scores for Mega_Ensemble...
    CV samples: 1402
  [Mega_Ensemble] Thresholds (95th pct) -> unc_mean: 0.1615,  unc_max: 0.3345,  entropy: 0.9140

Evaluating: Mega_Ensemble
  [Holdout]  Kappa: 0.9481  F1: 0.8779  MAE: 0.1532  Off-1: 0.9758  ECE: 0.0792  Brier: 0.0413
  [Full dataset] Saved: results/ensembles/Mega_Ensemble_uncertainty.npz

=================================================================
UQ VALIDATION: Mega_Ensemble
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.6628  AUPRC=0.0129  F1=0.0174  flagged=104/1650  μ_certain=0.0800  μ_uncertain=0.1077
    [unc_max  ]  AUROC=0.6591  AUPRC=0.0104  F1=0.0000  flagged=90/1650  μ_certain=0.1725  μ_uncertain=0.2257
    [entropy  ]  AUROC=0.7138  AUPRC=0.1049  F1=0.0339  flagged=107/1650  μ_certain=0.4702  μ_uncertain=0.6993

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 10 : 48 : 26 : 9 : 14
              precision    recall  f1-score   support

     Certain       0.99      0.94      0.96      1639
   Uncertain       0.02      0.18      0.03        11

    accuracy                           0.93      1650
   macro avg       0.51      0.56      0.50      1650
weighted avg       0.99      0.93      0.96      1650


=================================================================
 Weighted Uncertainty Experiment (WeightedEnsemble)
=================================================================

 Building Weighted Ensemble (per-class F1 weights, best CV-fold)

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

=================================================================
  WEIGHTED UNCERTAINTY EXPERIMENT: Heterogeneous_Weighted
=================================================================
    [unc_mean_weighted   ]  thr(CV 95pct)=0.137555  AUROC=0.6927  AUPRC=0.0118  F1=0.0000  flagged=101/1650  μ_certain=0.0541  μ_uncertain=0.0801
    [unc_max_weighted    ]  thr(CV 95pct)=0.322030  AUROC=0.6936  AUPRC=0.0118  F1=0.0000  flagged=98/1650  μ_certain=0.1247  μ_uncertain=0.1870

=================================================================
 STEP 3: Threshold Sensitivity Analysis (percentile sweep + 3σ)
=================================================================

  Threshold sensitivity sweep: resnet50_Homogeneous
    303 threshold combinations evaluated.

  Threshold sensitivity sweep: efficientnet_b3_Homogeneous
    303 threshold combinations evaluated.

  Threshold sensitivity sweep: densenet121_Homogeneous
    303 threshold combinations evaluated.

  Threshold sensitivity sweep: mobilenetv3_large_Homogeneous
    303 threshold combinations evaluated.

  Threshold sensitivity sweep: convnext_tiny_Homogeneous
    303 threshold combinations evaluated.

  Threshold sensitivity sweep: Heterogeneous_Avg
    303 threshold combinations evaluated.

  Threshold sensitivity sweep: Heterogeneous_Weighted
    303 threshold combinations evaluated.

  Threshold sensitivity sweep: Mega_Ensemble
    303 threshold combinations evaluated.
  Sensitivity CSVs saved to results/ensembles

=================================================================
 STEP 4: Building Consensus Matrices
=================================================================
  Consensus CSVs saved to results/ensembles

=================================================================
 Mann-Whitney U — Mega_Ensemble, all signals, Bonferroni
=================================================================

  Mann-Whitney U — Bonferroni-corrected (n_tests=3, α=0.05)
  n_certain=1639,  n_uncertain=11
  Signal                U      p_raw     p_bonf   Sig        r  Effect
  --------------------------------------------------------------------
  unc_mean        6078.50   0.031171   0.093512         0.3257  medium
  unc_max         6146.50   0.034327   0.102982         0.3182  medium
  entropy         5160.50   0.007208   0.021624     *   0.4275  medium

  Master results saved: results/ensembles/MASTER_RESULTS_SUMMARY.xlsx

========================================================================================================================
 KL Classification — all ensembles on HOLD-OUT
========================================================================================================================
Model                           Kappa   F1-Mac     MAE   Off-1     ECE   Brier |    KL0    KL1    KL2    KL3    KL4
------------------------------------------------------------------------------------------------------------------------
Mega_Ensemble                  0.9481   0.8779  0.1532  0.9758  0.0792  0.0413 | 0.8970 0.8060 0.7647 0.9697 0.9524
convnext_tiny_Homogeneous      0.9382   0.8542  0.1815  0.9718  0.0203  0.0455 | 0.8834 0.7727 0.6944 0.9851 0.9355
resnet50_Homogeneous           0.9341   0.8212  0.2097  0.9839  0.0354  0.0520 | 0.8521 0.6970 0.6970 0.9394 0.9206
efficientnet_b3_Homogeneous    0.9320   0.8507  0.1895  0.9718  0.0748  0.0523 | 0.8721 0.7704 0.7576 0.9355 0.9180
densenet121_Homogeneous        0.9316   0.8738  0.1734  0.9718  0.0525  0.0401 | 0.8780 0.7971 0.7879 0.9552 0.9508
Heterogeneous_Avg              0.9199   0.8508  0.1935  0.9677  0.0775  0.0415 | 0.9012 0.7862 0.7097 0.9538 0.9032
Heterogeneous_Weighted         0.9199   0.8508  0.1935  0.9677  0.0837  0.0414 | 0.9012 0.7862 0.7097 0.9538 0.9032
mobilenetv3_large_Homogeneous   0.9192   0.8431  0.1976  0.9597  0.0334  0.0479 | 0.9114 0.8082 0.6984 0.9538 0.8438
========================================================================================================================

========================================================================================================================
 UQ Detection — full dataset (Independent CV thresholds)
========================================================================================================================
Model                         AUROC_mn  AUPRC_mn  AUROC_mx  AUPRC_mx  AUROC_ent  AUPRC_ent |  E-Unc      Best
------------------------------------------------------------------------------------------------------------------------
resnet50_Homogeneous            0.6788    0.0112    0.6769    0.0109     0.7550     0.0241 |     11   entropy
densenet121_Homogeneous         0.7017    0.0134    0.7006    0.0129     0.7310     0.0650 |     11   entropy
Heterogeneous_Weighted          0.6921    0.0117    0.6942    0.0118     0.7239     0.0201 |     11   entropy
Heterogeneous_Avg               0.6921    0.0117    0.6942    0.0118     0.7214     0.0198 |     11   entropy
Mega_Ensemble                   0.6628    0.0129    0.6591    0.0104     0.7138     0.1049 |     11   entropy
mobilenetv3_large_Homogeneous    0.6535    0.0116    0.6337    0.0100     0.6933     0.0181 |     11   entropy
convnext_tiny_Homogeneous       0.6402    0.0133    0.6385    0.0114     0.6615     0.0174 |     11   entropy
efficientnet_b3_Homogeneous     0.5035    0.0077    0.4711    0.0066     0.5536     0.0111 |     11   entropy
========================================================================================================================
mn=unc_mean  mx=unc_max  ent=entropy  |  Best = signal with highest AUROC

====================================================================================================
 Threshold Sensitivity Summary — Mega_Ensemble
====================================================================================================
Method                 Signal        Percentile   Threshold  n_flag   %flag    Prec     Rec      F1
----------------------------------------------------------------------------------------------------
default_95pct          unc_mean            95.0      0.1615     104     6.3  0.0096  0.0909  0.0174
default_95pct          unc_max             95.0      0.3345      90     5.5  0.0000  0.0000  0.0000
default_95pct          entropy             95.0      0.9141     107     6.5  0.0187  0.1818  0.0339
sigma3                 unc_mean            μ+3σ      0.2265       4     0.2  0.0000  0.0000  0.0000
sigma3                 unc_max             μ+3σ      0.4751       0     0.0  0.0000  0.0000  0.0000
sigma3                 entropy             μ+3σ      1.2818       8     0.5  0.1250  0.0909  0.1053
best_f1_percentile     entropy             99.9      1.3327       3     0.2  0.3333  0.0909  0.1428
best_f1_percentile     unc_max             90.0      0.3079     178    10.8  0.0000  0.0000  0.0000
best_f1_percentile     unc_mean            98.8      0.1901      32     1.9  0.0312  0.0909  0.0465
====================================================================================================
NOTE: best_f1_percentile is an exploratory upper bound (selected on full eval set).

==================================================================================================================================
 BASELINE COMPARISON: Individual Models (best fold, holdout) vs Ensembles (holdout)
==================================================================================================================================
Model                            Type              Kappa   F1-Mac     MAE   Off-1     ECE   Brier
----------------------------------------------------------------------------------------------------------------------------------
Mega_Ensemble                    Ensemble         0.9481   0.8779  0.1532  0.9758  0.0792  0.0413
convnext_tiny_Homogeneous        Ensemble         0.9382   0.8542  0.1815  0.9718  0.0203  0.0455
resnet50_Homogeneous             Ensemble         0.9341   0.8212  0.2097  0.9839  0.0354  0.0520
efficientnet_b3_Homogeneous      Ensemble         0.9320   0.8507  0.1895  0.9718  0.0748  0.0523
densenet121_Homogeneous          Ensemble         0.9316   0.8738  0.1734  0.9718  0.0525  0.0401
convnext_tiny                    Individual       0.9288   0.8420  0.1895  0.9758  0.0693  0.0471
densenet121                      Individual       0.9238   0.8610  0.1895  0.9677  0.0662  0.0492
Heterogeneous_Avg                Ensemble         0.9199   0.8508  0.1935  0.9677  0.0775  0.0415
Heterogeneous_Weighted           Ensemble         0.9199   0.8508  0.1935  0.9677  0.0837  0.0414
mobilenetv3_large_Homogeneous    Ensemble         0.9192   0.8431  0.1976  0.9597  0.0334  0.0479
resnet50                         Individual       0.8979   0.7791  0.2742  0.9677  0.0475  0.0578
efficientnet_b3                  Individual       0.8911   0.8140  0.2500  0.9597  0.1000  0.0581
mobilenetv3_large                Individual       0.8724   0.7670  0.3105  0.9355  0.1097  0.0701
==================================================================================================================================
Sorted by Cohen's Kappa (Quadratic).

Results saved to: results
