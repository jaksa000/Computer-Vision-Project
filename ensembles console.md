=================================================================
Ensemble Evaluation and Uncertainty Quantification
=================================================================

  Loaded 1650 matched pairs from cache: results/expert_matches_cache.csv
  Certain: 1639  Uncertain: 11

============================================================
 DATA SPLIT
============================================================
  K-Fold CV data (85%): 1402 images
  Hold-out data  (15%): 248 images

  CV set (threshold):   1402 samples
  Holdout (KL):         248 samples
  Full dataset (UQ):    1650 samples
    Expert-certain:     1639
    Expert-uncertain:   11

=================================================================
 STEP 1 & 2: Evaluate Ensembles (with specific CV thresholds)
=================================================================

 Building Homogeneous Ensemble: resnet50

 Building model: resnet50
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
model.safetensors: 100% 102M/102M [00:03<00:00, 31.0MB/s]
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable
  Computing uncertainty thresholds from CV set (95th percentile) for resnet50_Homogeneous...
    Thresholds -> unc_mean: 0.1184, unc_max: 0.2646, entropy: 1.2401

Evaluating: resnet50_Homogeneous
  [Holdout]  Kappa: 0.9334  F1: 0.8344  ECE: 0.1245  Brier: 0.0519
  [Full dataset] Saved: results/ensembles/resnet50_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: resnet50_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.6100  AUPRC=0.0114  F1=0.0208  flagged=85/1650  μ_certain=0.0578  μ_uncertain=0.0707
    [unc_max  ]  AUROC=0.6198  AUPRC=0.0112  F1=0.0211  flagged=84/1650  μ_certain=0.1293  μ_uncertain=0.1591
    [entropy  ]  AUROC=0.6744  AUPRC=0.0156  F1=0.0217  flagged=81/1650  μ_certain=0.7817  μ_uncertain=0.9717

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 2 : 35 : 22 : 8 : 14
              precision    recall  f1-score   support

     Certain       0.99      0.95      0.97      1639
   Uncertain       0.01      0.09      0.02        11

    accuracy                           0.95      1650
   macro avg       0.50      0.52      0.50      1650
weighted avg       0.99      0.95      0.97      1650


 Building Homogeneous Ensemble: efficientnet_b3

 Building model: efficientnet_b3
model.safetensors: 100% 49.3M/49.3M [00:01<00:00, 30.5MB/s]
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable
  Computing uncertainty thresholds from CV set (95th percentile) for efficientnet_b3_Homogeneous...
    Thresholds -> unc_mean: 0.1799, unc_max: 0.3804, entropy: 1.1139

Evaluating: efficientnet_b3_Homogeneous
  [Holdout]  Kappa: 0.9658  F1: 0.9271  ECE: 0.1302  Brier: 0.0328
  [Full dataset] Saved: results/ensembles/efficientnet_b3_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: efficientnet_b3_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.6026  AUPRC=0.0128  F1=0.0440  flagged=80/1650  μ_certain=0.0826  μ_uncertain=0.1026
    [unc_max  ]  AUROC=0.6172  AUPRC=0.0259  F1=0.0220  flagged=80/1650  μ_certain=0.1846  μ_uncertain=0.2350
    [entropy  ]  AUROC=0.5936  AUPRC=0.0096  F1=0.0000  flagged=86/1650  μ_certain=0.5353  μ_uncertain=0.6402

  Best signal by AUROC: unc_max
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 13 : 31 : 17 : 13 : 6
              precision    recall  f1-score   support

     Certain       0.99      0.95      0.97      1639
   Uncertain       0.01      0.09      0.02        11

    accuracy                           0.95      1650
   macro avg       0.50      0.52      0.50      1650
weighted avg       0.99      0.95      0.97      1650


 Building Homogeneous Ensemble: densenet121

 Building model: densenet121
model.safetensors: 100% 32.3M/32.3M [00:01<00:00, 20.0MB/s]
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable
  Computing uncertainty thresholds from CV set (95th percentile) for densenet121_Homogeneous...
    Thresholds -> unc_mean: 0.1273, unc_max: 0.2957, entropy: 1.0791

Evaluating: densenet121_Homogeneous
  [Holdout]  Kappa: 0.9809  F1: 0.9515  ECE: 0.1155  Brier: 0.0222
  [Full dataset] Saved: results/ensembles/densenet121_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: densenet121_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.5626  AUPRC=0.0080  F1=0.0000  flagged=78/1650  μ_certain=0.0483  μ_uncertain=0.0510
    [unc_max  ]  AUROC=0.5626  AUPRC=0.0083  F1=0.0000  flagged=77/1650  μ_certain=0.1125  μ_uncertain=0.1222
    [entropy  ]  AUROC=0.6607  AUPRC=0.0134  F1=0.0217  flagged=81/1650  μ_certain=0.5153  μ_uncertain=0.6793

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 5 : 42 : 15 : 9 : 10
              precision    recall  f1-score   support

     Certain       0.99      0.95      0.97      1639
   Uncertain       0.01      0.09      0.02        11

    accuracy                           0.95      1650
   macro avg       0.50      0.52      0.50      1650
weighted avg       0.99      0.95      0.97      1650


 Building Homogeneous Ensemble: mobilenetv3_large

 Building model: mobilenetv3_large
model.safetensors: 100% 22.1M/22.1M [00:01<00:00, 13.7MB/s]
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable
  Computing uncertainty thresholds from CV set (95th percentile) for mobilenetv3_large_Homogeneous...
    Thresholds -> unc_mean: 0.1518, unc_max: 0.3537, entropy: 0.9673

Evaluating: mobilenetv3_large_Homogeneous
  [Holdout]  Kappa: 0.9808  F1: 0.9479  ECE: 0.0894  Brier: 0.0203
  [Full dataset] Saved: results/ensembles/mobilenetv3_large_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: mobilenetv3_large_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.7597  AUPRC=0.0193  F1=0.0202  flagged=88/1650  μ_certain=0.0552  μ_uncertain=0.1044
    [unc_max  ]  AUROC=0.7377  AUPRC=0.0185  F1=0.0396  flagged=90/1650  μ_certain=0.1280  μ_uncertain=0.2283
    [entropy  ]  AUROC=0.7722  AUPRC=0.0183  F1=0.0213  flagged=83/1650  μ_certain=0.3779  μ_uncertain=0.6752

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 9 : 33 : 23 : 10 : 8
              precision    recall  f1-score   support

     Certain       0.99      0.95      0.97      1639
   Uncertain       0.01      0.09      0.02        11

    accuracy                           0.94      1650
   macro avg       0.50      0.52      0.50      1650
weighted avg       0.99      0.94      0.96      1650


 Building Homogeneous Ensemble: convnext_tiny

 Building model: convnext_tiny
model.safetensors: 100% 114M/114M [00:02<00:00, 43.7MB/s]
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable
  Computing uncertainty thresholds from CV set (95th percentile) for convnext_tiny_Homogeneous...
    Thresholds -> unc_mean: 0.1367, unc_max: 0.3239, entropy: 0.8787

Evaluating: convnext_tiny_Homogeneous
  [Holdout]  Kappa: 0.9725  F1: 0.9108  ECE: 0.0894  Brier: 0.0294
  [Full dataset] Saved: results/ensembles/convnext_tiny_Homogeneous_uncertainty.npz

=================================================================
UQ VALIDATION: convnext_tiny_Homogeneous
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.5693  AUPRC=0.0108  F1=0.0208  flagged=85/1650  μ_certain=0.0524  μ_uncertain=0.0611
    [unc_max  ]  AUROC=0.5763  AUPRC=0.0101  F1=0.0213  flagged=83/1650  μ_certain=0.1229  μ_uncertain=0.1448
    [entropy  ]  AUROC=0.6448  AUPRC=0.0138  F1=0.0196  flagged=91/1650  μ_certain=0.4462  μ_uncertain=0.5904

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 5 : 32 : 21 : 7 : 26
              precision    recall  f1-score   support

     Certain       0.99      0.95      0.97      1639
   Uncertain       0.01      0.09      0.02        11

    accuracy                           0.94      1650
   macro avg       0.50      0.52      0.49      1650
weighted avg       0.99      0.94      0.96      1650


 Building Heterogeneous Ensemble (best fold per architecture)

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
  Computing uncertainty thresholds from CV set (95th percentile) for Heterogeneous_Avg...
    Thresholds -> unc_mean: 0.1515, unc_max: 0.3460, entropy: 0.9982

Evaluating: Heterogeneous_Avg
  [Holdout]  Kappa: 0.9819  F1: 0.9505  ECE: 0.1359  Brier: 0.0242
  [Full dataset] Saved: results/ensembles/Heterogeneous_Avg_uncertainty.npz

=================================================================
UQ VALIDATION: Heterogeneous_Avg
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.5094  AUPRC=0.0072  F1=0.0000  flagged=84/1650  μ_certain=0.0669  μ_uncertain=0.0634
    [unc_max  ]  AUROC=0.5144  AUPRC=0.0074  F1=0.0000  flagged=84/1650  μ_certain=0.1537  μ_uncertain=0.1490
    [entropy  ]  AUROC=0.6157  AUPRC=0.0108  F1=0.0215  flagged=82/1650  μ_certain=0.5069  μ_uncertain=0.6115

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 9 : 30 : 17 : 12 : 14
              precision    recall  f1-score   support

     Certain       0.99      0.95      0.97      1639
   Uncertain       0.01      0.09      0.02        11

    accuracy                           0.94      1650
   macro avg       0.50      0.52      0.50      1650
weighted avg       0.99      0.94      0.97      1650


 Building Weighted Ensemble (per-class F1 weights)

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
  Computing uncertainty thresholds from CV set (95th percentile) for Heterogeneous_Weighted...
    Thresholds -> unc_mean: 0.1515, unc_max: 0.3460, entropy: 0.9931

Evaluating: Heterogeneous_Weighted
  [Holdout]  Kappa: 0.9819  F1: 0.9505  ECE: 0.1335  Brier: 0.0238
  [Full dataset] Saved: results/ensembles/Heterogeneous_Weighted_uncertainty.npz

=================================================================
UQ VALIDATION: Heterogeneous_Weighted
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.5094  AUPRC=0.0072  F1=0.0000  flagged=84/1650  μ_certain=0.0669  μ_uncertain=0.0634
    [unc_max  ]  AUROC=0.5144  AUPRC=0.0074  F1=0.0000  flagged=84/1650  μ_certain=0.1537  μ_uncertain=0.1490
    [entropy  ]  AUROC=0.6148  AUPRC=0.0107  F1=0.0213  flagged=83/1650  μ_certain=0.4999  μ_uncertain=0.6028

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 8 : 32 : 17 : 12 : 14
              precision    recall  f1-score   support

     Certain       0.99      0.95      0.97      1639
   Uncertain       0.01      0.09      0.02        11

    accuracy                           0.94      1650
   macro avg       0.50      0.52      0.50      1650
weighted avg       0.99      0.94      0.96      1650


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
  Computing uncertainty thresholds from CV set (95th percentile) for Mega_Ensemble...
    Thresholds -> unc_mean: 0.1649, unc_max: 0.3235, entropy: 1.1025

Evaluating: Mega_Ensemble
  [Holdout]  Kappa: 0.9830  F1: 0.9585  ECE: 0.1584  Brier: 0.0262
  [Full dataset] Saved: results/ensembles/Mega_Ensemble_uncertainty.npz

=================================================================
UQ VALIDATION: Mega_Ensemble
=================================================================
  Total samples (full dataset):     1650
  Expert-uncertain (ground truth):  11  (0.7%)
    [unc_mean ]  AUROC=0.6614  AUPRC=0.0119  F1=0.0000  flagged=79/1650  μ_certain=0.0882  μ_uncertain=0.1126
    [unc_max  ]  AUROC=0.6767  AUPRC=0.0144  F1=0.0000  flagged=77/1650  μ_certain=0.1871  μ_uncertain=0.2406
    [entropy  ]  AUROC=0.6921  AUPRC=0.0133  F1=0.0208  flagged=85/1650  μ_certain=0.5978  μ_uncertain=0.7903

  Best signal by AUROC: entropy
  KL Breakdown for Flagged (KL0:KL1:KL2:KL3:KL4) = 4 : 35 : 21 : 14 : 11
              precision    recall  f1-score   support

     Certain       0.99      0.95      0.97      1639
   Uncertain       0.01      0.09      0.02        11

    accuracy                           0.94      1650
   macro avg       0.50      0.52      0.50      1650
weighted avg       0.99      0.94      0.96      1650


  Mann-Whitney U test — Mega Ensemble  [entropy]
    n_certain=1639,  n_uncertain=11

  Mann-Whitney U test (certain vs uncertain):
    n_certain   = 1639
    n_uncertain = 11
    U           = 5551.50
    p-value     = 0.013958  Significant (p < 0.05)
    effect r    = 0.3842  (medium effect)

  Master results saved: results/ensembles/MASTER_RESULTS_SUMMARY.xlsx

==============================================================================================================
 KL Classification — all ensembles on HOLD-OUT (clean)
==============================================================================================================
Model                           Kappa   F1-Mac     ECE   Brier |    KL0    KL1    KL2    KL3    KL4
--------------------------------------------------------------------------------------------------------------
Mega_Ensemble                  0.9830   0.9585  0.1584  0.0262 | 0.9434 0.9078 0.9565 0.9846 1.0000
Heterogeneous_Avg              0.9819   0.9505  0.1359  0.0242 | 0.9494 0.9103 0.9394 0.9697 0.9836
Heterogeneous_Weighted         0.9819   0.9505  0.1335  0.0238 | 0.9494 0.9103 0.9394 0.9697 0.9836
densenet121_Homogeneous        0.9809   0.9515  0.1155  0.0222 | 0.9367 0.8966 0.9394 0.9846 1.0000
mobilenetv3_large_Homogeneous   0.9808   0.9479  0.0894  0.0203 | 0.9434 0.9014 0.9412 0.9697 0.9836
convnext_tiny_Homogeneous      0.9725   0.9108  0.0894  0.0294 | 0.9068 0.8175 0.8451 0.9846 1.0000
efficientnet_b3_Homogeneous    0.9658   0.9271  0.1302  0.0328 | 0.9308 0.8794 0.8571 0.9846 0.9836
resnet50_Homogeneous           0.9334   0.8344  0.1245  0.0519 | 0.8861 0.7465 0.6769 0.9394 0.9231
==============================================================================================================

========================================================================================================================
 UQ Detection — full dataset (Independent CV thresholds)
========================================================================================================================
Model                         AUROC_mn  AUPRC_mn  AUROC_mx  AUPRC_mx  AUROC_ent  AUPRC_ent |  E-Unc      Best
------------------------------------------------------------------------------------------------------------------------
mobilenetv3_large_Homogeneous    0.7597    0.0193    0.7377    0.0185     0.7722     0.0183 |     11   entropy
Mega_Ensemble                   0.6614    0.0119    0.6767    0.0144     0.6921     0.0133 |     11   entropy
resnet50_Homogeneous            0.6100    0.0114    0.6198    0.0112     0.6744     0.0156 |     11   entropy
densenet121_Homogeneous         0.5626    0.0080    0.5626    0.0083     0.6607     0.0134 |     11   entropy
convnext_tiny_Homogeneous       0.5693    0.0108    0.5763    0.0101     0.6448     0.0138 |     11   entropy
efficientnet_b3_Homogeneous     0.6026    0.0128    0.6172    0.0259     0.5936     0.0096 |     11   unc_max
Heterogeneous_Avg               0.5094    0.0072    0.5144    0.0074     0.6157     0.0108 |     11   entropy
Heterogeneous_Weighted          0.5094    0.0072    0.5144    0.0074     0.6148     0.0107 |     11   entropy
========================================================================================================================
mn=unc_mean  mx=unc_max  ent=entropy  |  Best = signal with highest AUROC