Device: cuda
Models to run: ['resnet50', 'efficientnet_b3', 'densenet121', 'mobilenetv3_large', 'convnext_tiny']
Folds number: 5
Loading data: data/MedicalExpert-I
  Class 0 (0Normal): 514 images
  Class 1 (1Doubtful): 477 images
  Class 2 (2Mild): 232 images
  Class 3 (3Moderate): 221 images
  Class 4 (4Severe): 206 images

  Together: 1650 images
  Together: 1650 images, 5  folds CV

============================================================
 DATA SPLIT
============================================================
  K-Fold CV data (85%): 1402 images
  Hold-out data (15%): 248 images

================================================================================
MODEL TRAINING START: resnet50
================================================================================

--- resnet50 | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: resnet50

============================================================
TRAINING: resnet50_fold1
============================================================
Epoch [  1/20]  Train Loss: 1.6082  Bal.Acc: 20.5%  F1: 0.1620  |  Val Loss: 1.6010  Bal.Acc: 22.2%  F1: 0.2147  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 1.6010)
Epoch [  2/20]  Train Loss: 1.5921  Bal.Acc: 27.0%  F1: 0.2602  |  Val Loss: 1.5821  Bal.Acc: 37.6%  F1: 0.3750  |  LR: 1.00e-04  (21.9s)
 Best checkpoint saved (val_loss: 1.5821)
Epoch [  3/20]  Train Loss: 1.5748  Bal.Acc: 36.5%  F1: 0.3416  |  Val Loss: 1.5750  Bal.Acc: 33.9%  F1: 0.3522  |  LR: 1.00e-04  (28.6s)
 Best checkpoint saved (val_loss: 1.5750)
Epoch [  4/20]  Train Loss: 1.5520  Bal.Acc: 41.0%  F1: 0.3965  |  Val Loss: 1.5539  Bal.Acc: 44.2%  F1: 0.4267  |  LR: 1.00e-04  (26.6s)
 Best checkpoint saved (val_loss: 1.5539)
Epoch [  5/20]  Train Loss: 1.5265  Bal.Acc: 42.9%  F1: 0.3798  |  Val Loss: 1.5033  Bal.Acc: 49.0%  F1: 0.4291  |  LR: 1.00e-04  (28.9s)
 Best checkpoint saved (val_loss: 1.5033)
Epoch [  6/20]  Train Loss: 1.4857  Bal.Acc: 48.3%  F1: 0.4238  |  Val Loss: 1.4744  Bal.Acc: 51.6%  F1: 0.4401  |  LR: 1.00e-04  (28.2s)
 Best checkpoint saved (val_loss: 1.4744)
Epoch [  7/20]  Train Loss: 1.4315  Bal.Acc: 50.7%  F1: 0.4430  |  Val Loss: 1.3815  Bal.Acc: 49.3%  F1: 0.4480  |  LR: 1.00e-04  (28.2s)
 Best checkpoint saved (val_loss: 1.3815)
Epoch [  8/20]  Train Loss: 1.3502  Bal.Acc: 54.9%  F1: 0.4944  |  Val Loss: 1.3225  Bal.Acc: 50.5%  F1: 0.4806  |  LR: 1.00e-04  (27.7s)
 Best checkpoint saved (val_loss: 1.3225)
Epoch [  9/20]  Train Loss: 1.2452  Bal.Acc: 58.7%  F1: 0.5396  |  Val Loss: 1.1757  Bal.Acc: 56.5%  F1: 0.5330  |  LR: 1.00e-04  (27.9s)
 Best checkpoint saved (val_loss: 1.1757)
Epoch [ 10/20]  Train Loss: 1.1271  Bal.Acc: 59.1%  F1: 0.5417  |  Val Loss: 1.1276  Bal.Acc: 52.0%  F1: 0.5030  |  LR: 1.00e-04  (26.9s)
 Best checkpoint saved (val_loss: 1.1276)
Epoch [ 11/20]  Train Loss: 1.0289  Bal.Acc: 63.3%  F1: 0.5986  |  Val Loss: 1.0068  Bal.Acc: 57.7%  F1: 0.5563  |  LR: 1.00e-04  (27.6s)
 Best checkpoint saved (val_loss: 1.0068)
Epoch [ 12/20]  Train Loss: 0.9393  Bal.Acc: 66.2%  F1: 0.6373  |  Val Loss: 0.9261  Bal.Acc: 61.9%  F1: 0.6078  |  LR: 1.00e-04  (28.3s)
 Best checkpoint saved (val_loss: 0.9261)
Epoch [ 13/20]  Train Loss: 0.8550  Bal.Acc: 69.4%  F1: 0.6773  |  Val Loss: 0.8970  Bal.Acc: 65.8%  F1: 0.6576  |  LR: 1.00e-04  (28.6s)
 Best checkpoint saved (val_loss: 0.8970)
Epoch [ 14/20]  Train Loss: 0.7872  Bal.Acc: 71.5%  F1: 0.7039  |  Val Loss: 0.8146  Bal.Acc: 67.9%  F1: 0.6679  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 0.8146)
Epoch [ 15/20]  Train Loss: 0.7520  Bal.Acc: 72.1%  F1: 0.7130  |  Val Loss: 0.9103  Bal.Acc: 62.0%  F1: 0.6378  |  LR: 1.00e-04  (27.7s)
Epoch [ 16/20]  Train Loss: 0.6769  Bal.Acc: 75.6%  F1: 0.7469  |  Val Loss: 0.7200  Bal.Acc: 73.0%  F1: 0.7155  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 0.7200)
Epoch [ 17/20]  Train Loss: 0.6423  Bal.Acc: 75.9%  F1: 0.7515  |  Val Loss: 0.6872  Bal.Acc: 72.5%  F1: 0.7220  |  LR: 1.00e-04  (22.9s)
 Best checkpoint saved (val_loss: 0.6872)
Epoch [ 18/20]  Train Loss: 0.6078  Bal.Acc: 77.0%  F1: 0.7637  |  Val Loss: 0.7128  Bal.Acc: 74.3%  F1: 0.7320  |  LR: 1.00e-04  (20.7s)
Epoch [ 19/20]  Train Loss: 0.5546  Bal.Acc: 79.0%  F1: 0.7792  |  Val Loss: 0.7857  Bal.Acc: 63.7%  F1: 0.6359  |  LR: 1.00e-04  (22.0s)
Epoch [ 20/20]  Train Loss: 0.5649  Bal.Acc: 79.5%  F1: 0.7908  |  Val Loss: 0.7500  Bal.Acc: 68.6%  F1: 0.7023  |  LR: 1.00e-04  (16.4s)

 Training finished. Checkpoint: checkpoints/resnet50_fold1_best.pt
Log CSV: results/resnet50_fold1_training_log.csv
Best weights loaded from  17

Model evaluation: resnet50_fold1
----------------------------------------
  Balanced Accuracy:       72.46%
  F1 (macro):              0.7220
  Quadratic Cohen's Kappa: 0.9008
  ECE:                     0.0596
  Brier Score (mean):      0.0728

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.76      0.93      0.84        87
    Doubtful       0.76      0.62      0.68        81
        Mild       0.50      0.45      0.47        40
    Moderate       0.77      0.71      0.74        38
      Severe       0.84      0.91      0.88        35

    accuracy                           0.74       281
   macro avg       0.73      0.72      0.72       281
weighted avg       0.74      0.74      0.73       281

  Metrics saved to: results/resnet50_fold1_metrics.json
  Probability saved to: results/resnet50_fold1_test_probs.npz

--- resnet50 | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: resnet50

============================================================
TRAINING: resnet50_fold2
============================================================
Epoch [  1/20]  Train Loss: 1.6085  Bal.Acc: 22.4%  F1: 0.1321  |  Val Loss: 1.6063  Bal.Acc: 22.5%  F1: 0.2033  |  LR: 1.00e-04  (15.5s)
 Best checkpoint saved (val_loss: 1.6063)
Epoch [  2/20]  Train Loss: 1.5929  Bal.Acc: 27.0%  F1: 0.2569  |  Val Loss: 1.5963  Bal.Acc: 25.8%  F1: 0.2459  |  LR: 1.00e-04  (21.2s)
 Best checkpoint saved (val_loss: 1.5963)
Epoch [  3/20]  Train Loss: 1.5737  Bal.Acc: 35.4%  F1: 0.3574  |  Val Loss: 1.5686  Bal.Acc: 29.6%  F1: 0.2851  |  LR: 1.00e-04  (28.2s)
 Best checkpoint saved (val_loss: 1.5686)
Epoch [  4/20]  Train Loss: 1.5600  Bal.Acc: 41.4%  F1: 0.4083  |  Val Loss: 1.5347  Bal.Acc: 35.7%  F1: 0.3395  |  LR: 1.00e-04  (28.2s)
 Best checkpoint saved (val_loss: 1.5347)
Epoch [  5/20]  Train Loss: 1.5291  Bal.Acc: 48.2%  F1: 0.4733  |  Val Loss: 1.4980  Bal.Acc: 43.6%  F1: 0.4379  |  LR: 1.00e-04  (26.7s)
 Best checkpoint saved (val_loss: 1.4980)
Epoch [  6/20]  Train Loss: 1.4899  Bal.Acc: 50.1%  F1: 0.4867  |  Val Loss: 1.4422  Bal.Acc: 44.0%  F1: 0.4251  |  LR: 1.00e-04  (22.0s)
 Best checkpoint saved (val_loss: 1.4422)
Epoch [  7/20]  Train Loss: 1.4392  Bal.Acc: 53.2%  F1: 0.5005  |  Val Loss: 1.3888  Bal.Acc: 45.3%  F1: 0.4358  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 1.3888)
Epoch [  8/20]  Train Loss: 1.3549  Bal.Acc: 57.1%  F1: 0.5356  |  Val Loss: 1.3751  Bal.Acc: 40.4%  F1: 0.3970  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 1.3751)
Epoch [  9/20]  Train Loss: 1.2634  Bal.Acc: 57.9%  F1: 0.5327  |  Val Loss: 1.1899  Bal.Acc: 50.7%  F1: 0.4810  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 1.1899)
Epoch [ 10/20]  Train Loss: 1.1400  Bal.Acc: 61.1%  F1: 0.5606  |  Val Loss: 1.0584  Bal.Acc: 54.9%  F1: 0.5115  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 1.0584)
Epoch [ 11/20]  Train Loss: 1.0381  Bal.Acc: 62.7%  F1: 0.5886  |  Val Loss: 0.9822  Bal.Acc: 60.9%  F1: 0.5950  |  LR: 1.00e-04  (28.1s)
 Best checkpoint saved (val_loss: 0.9822)
Epoch [ 12/20]  Train Loss: 0.9374  Bal.Acc: 65.1%  F1: 0.6190  |  Val Loss: 0.9231  Bal.Acc: 60.1%  F1: 0.5978  |  LR: 1.00e-04  (27.6s)
 Best checkpoint saved (val_loss: 0.9231)
Epoch [ 13/20]  Train Loss: 0.8481  Bal.Acc: 67.6%  F1: 0.6565  |  Val Loss: 0.8284  Bal.Acc: 65.9%  F1: 0.6580  |  LR: 1.00e-04  (28.2s)
 Best checkpoint saved (val_loss: 0.8284)
Epoch [ 14/20]  Train Loss: 0.7729  Bal.Acc: 72.7%  F1: 0.7137  |  Val Loss: 0.7902  Bal.Acc: 66.9%  F1: 0.6726  |  LR: 1.00e-04  (28.0s)
 Best checkpoint saved (val_loss: 0.7902)
Epoch [ 15/20]  Train Loss: 0.7314  Bal.Acc: 72.9%  F1: 0.7199  |  Val Loss: 0.7093  Bal.Acc: 72.1%  F1: 0.7059  |  LR: 1.00e-04  (26.4s)
 Best checkpoint saved (val_loss: 0.7093)
Epoch [ 16/20]  Train Loss: 0.6675  Bal.Acc: 74.8%  F1: 0.7399  |  Val Loss: 0.7098  Bal.Acc: 72.3%  F1: 0.7176  |  LR: 1.00e-04  (28.0s)
Epoch [ 17/20]  Train Loss: 0.6376  Bal.Acc: 75.0%  F1: 0.7441  |  Val Loss: 0.6310  Bal.Acc: 73.5%  F1: 0.7250  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 0.6310)
Epoch [ 18/20]  Train Loss: 0.5935  Bal.Acc: 78.2%  F1: 0.7695  |  Val Loss: 0.7504  Bal.Acc: 67.9%  F1: 0.6922  |  LR: 1.00e-04  (22.7s)
Epoch [ 19/20]  Train Loss: 0.5359  Bal.Acc: 80.1%  F1: 0.7954  |  Val Loss: 0.6097  Bal.Acc: 74.2%  F1: 0.7320  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 0.6097)
Epoch [ 20/20]  Train Loss: 0.5236  Bal.Acc: 79.4%  F1: 0.7885  |  Val Loss: 0.6976  Bal.Acc: 65.7%  F1: 0.6615  |  LR: 1.00e-04  (21.7s)

 Training finished. Checkpoint: checkpoints/resnet50_fold2_best.pt
Log CSV: results/resnet50_fold2_training_log.csv
Best weights loaded from  19

Model evaluation: resnet50_fold2
----------------------------------------
  Balanced Accuracy:       74.20%
  F1 (macro):              0.7320
  Quadratic Cohen's Kappa: 0.8987
  ECE:                     0.0502
  Brier Score (mean):      0.0712

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.73      0.92      0.82        87
    Doubtful       0.70      0.47      0.56        81
        Mild       0.60      0.72      0.66        40
    Moderate       0.82      0.71      0.76        38
      Severe       0.84      0.89      0.86        35

    accuracy                           0.73       281
   macro avg       0.74      0.74      0.73       281
weighted avg       0.73      0.73      0.72       281

  Metrics saved to: results/resnet50_fold2_metrics.json
  Probability saved to: results/resnet50_fold2_test_probs.npz

--- resnet50 | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: resnet50
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

============================================================
TRAINING: resnet50_fold3
============================================================
Epoch [  1/20]  Train Loss: 1.6067  Bal.Acc: 18.7%  F1: 0.1514  |  Val Loss: 1.6031  Bal.Acc: 17.9%  F1: 0.1304  |  LR: 1.00e-04  (17.2s)
 Best checkpoint saved (val_loss: 1.6031)
Epoch [  2/20]  Train Loss: 1.5875  Bal.Acc: 28.6%  F1: 0.2628  |  Val Loss: 1.5780  Bal.Acc: 23.9%  F1: 0.2070  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 1.5780)
Epoch [  3/20]  Train Loss: 1.5686  Bal.Acc: 36.7%  F1: 0.3567  |  Val Loss: 1.5625  Bal.Acc: 29.3%  F1: 0.2724  |  LR: 1.00e-04  (21.6s)
 Best checkpoint saved (val_loss: 1.5625)
Epoch [  4/20]  Train Loss: 1.5449  Bal.Acc: 39.6%  F1: 0.3967  |  Val Loss: 1.5309  Bal.Acc: 35.0%  F1: 0.3511  |  LR: 1.00e-04  (28.3s)
 Best checkpoint saved (val_loss: 1.5309)
Epoch [  5/20]  Train Loss: 1.5148  Bal.Acc: 45.4%  F1: 0.4407  |  Val Loss: 1.4956  Bal.Acc: 42.4%  F1: 0.4144  |  LR: 1.00e-04  (26.5s)
 Best checkpoint saved (val_loss: 1.4956)
Epoch [  6/20]  Train Loss: 1.4787  Bal.Acc: 48.0%  F1: 0.4496  |  Val Loss: 1.4715  Bal.Acc: 39.3%  F1: 0.3910  |  LR: 1.00e-04  (27.9s)
 Best checkpoint saved (val_loss: 1.4715)
Epoch [  7/20]  Train Loss: 1.4184  Bal.Acc: 51.7%  F1: 0.4948  |  Val Loss: 1.4421  Bal.Acc: 49.0%  F1: 0.4656  |  LR: 1.00e-04  (27.5s)
 Best checkpoint saved (val_loss: 1.4421)
Epoch [  8/20]  Train Loss: 1.3295  Bal.Acc: 55.6%  F1: 0.5045  |  Val Loss: 1.3719  Bal.Acc: 42.8%  F1: 0.4134  |  LR: 1.00e-04  (28.5s)
 Best checkpoint saved (val_loss: 1.3719)
Epoch [  9/20]  Train Loss: 1.2386  Bal.Acc: 58.3%  F1: 0.5256  |  Val Loss: 1.2395  Bal.Acc: 51.2%  F1: 0.4802  |  LR: 1.00e-04  (27.0s)
 Best checkpoint saved (val_loss: 1.2395)
Epoch [ 10/20]  Train Loss: 1.1185  Bal.Acc: 60.5%  F1: 0.5621  |  Val Loss: 1.1703  Bal.Acc: 52.9%  F1: 0.5106  |  LR: 1.00e-04  (28.6s)
 Best checkpoint saved (val_loss: 1.1703)
Epoch [ 11/20]  Train Loss: 1.0215  Bal.Acc: 62.5%  F1: 0.5825  |  Val Loss: 1.1103  Bal.Acc: 55.2%  F1: 0.5344  |  LR: 1.00e-04  (27.7s)
 Best checkpoint saved (val_loss: 1.1103)
Epoch [ 12/20]  Train Loss: 0.9228  Bal.Acc: 64.7%  F1: 0.6183  |  Val Loss: 0.9882  Bal.Acc: 60.9%  F1: 0.5859  |  LR: 1.00e-04  (26.6s)
 Best checkpoint saved (val_loss: 0.9882)
Epoch [ 13/20]  Train Loss: 0.8392  Bal.Acc: 68.5%  F1: 0.6619  |  Val Loss: 1.1407  Bal.Acc: 49.5%  F1: 0.4965  |  LR: 1.00e-04  (27.3s)
Epoch [ 14/20]  Train Loss: 0.7524  Bal.Acc: 73.7%  F1: 0.7249  |  Val Loss: 0.9137  Bal.Acc: 59.8%  F1: 0.5985  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 0.9137)
Epoch [ 15/20]  Train Loss: 0.7089  Bal.Acc: 74.4%  F1: 0.7345  |  Val Loss: 1.0240  Bal.Acc: 56.0%  F1: 0.5716  |  LR: 1.00e-04  (27.5s)
Epoch [ 16/20]  Train Loss: 0.6386  Bal.Acc: 76.8%  F1: 0.7607  |  Val Loss: 0.9008  Bal.Acc: 63.9%  F1: 0.6367  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 0.9008)
Epoch [ 17/20]  Train Loss: 0.6132  Bal.Acc: 77.5%  F1: 0.7664  |  Val Loss: 0.8356  Bal.Acc: 64.6%  F1: 0.6472  |  LR: 1.00e-04  (27.5s)
 Best checkpoint saved (val_loss: 0.8356)
Epoch [ 18/20]  Train Loss: 0.5911  Bal.Acc: 75.9%  F1: 0.7536  |  Val Loss: 0.8461  Bal.Acc: 65.7%  F1: 0.6602  |  LR: 1.00e-04  (28.4s)
Epoch [ 19/20]  Train Loss: 0.5561  Bal.Acc: 79.1%  F1: 0.7875  |  Val Loss: 0.8393  Bal.Acc: 67.0%  F1: 0.6738  |  LR: 1.00e-04  (16.7s)
Epoch [ 20/20]  Train Loss: 0.5042  Bal.Acc: 81.6%  F1: 0.8081  |  Val Loss: 0.9139  Bal.Acc: 65.0%  F1: 0.6586  |  LR: 1.00e-04  (16.8s)

 Training finished. Checkpoint: checkpoints/resnet50_fold3_best.pt
Log CSV: results/resnet50_fold3_training_log.csv
Best weights loaded from  17

Model evaluation: resnet50_fold3
----------------------------------------
  Balanced Accuracy:       64.59%
  F1 (macro):              0.6472
  Quadratic Cohen's Kappa: 0.8537
  ECE:                     0.0756
  Brier Score (mean):      0.0917

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.67      0.94      0.78        88
    Doubtful       0.64      0.43      0.51        81
        Mild       0.49      0.44      0.46        39
    Moderate       0.76      0.68      0.71        37
      Severe       0.79      0.74      0.76        35

    accuracy                           0.66       280
   macro avg       0.67      0.65      0.65       280
weighted avg       0.66      0.66      0.65       280

  Metrics saved to: results/resnet50_fold3_metrics.json
  Probability saved to: results/resnet50_fold3_test_probs.npz

--- resnet50 | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: resnet50

============================================================
TRAINING: resnet50_fold4
============================================================
Epoch [  1/20]  Train Loss: 1.6058  Bal.Acc: 22.9%  F1: 0.2021  |  Val Loss: 1.6057  Bal.Acc: 24.4%  F1: 0.1733  |  LR: 1.00e-04  (15.9s)
 Best checkpoint saved (val_loss: 1.6057)
Epoch [  2/20]  Train Loss: 1.5914  Bal.Acc: 30.0%  F1: 0.2971  |  Val Loss: 1.5927  Bal.Acc: 27.2%  F1: 0.2592  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 1.5927)
Epoch [  3/20]  Train Loss: 1.5766  Bal.Acc: 33.0%  F1: 0.3266  |  Val Loss: 1.5643  Bal.Acc: 26.2%  F1: 0.2114  |  LR: 1.00e-04  (22.6s)
 Best checkpoint saved (val_loss: 1.5643)
Epoch [  4/20]  Train Loss: 1.5497  Bal.Acc: 38.6%  F1: 0.3758  |  Val Loss: 1.5346  Bal.Acc: 37.2%  F1: 0.3655  |  LR: 1.00e-04  (26.8s)
 Best checkpoint saved (val_loss: 1.5346)
Epoch [  5/20]  Train Loss: 1.5270  Bal.Acc: 44.4%  F1: 0.4150  |  Val Loss: 1.5075  Bal.Acc: 35.9%  F1: 0.3403  |  LR: 1.00e-04  (28.4s)
 Best checkpoint saved (val_loss: 1.5075)
Epoch [  6/20]  Train Loss: 1.4914  Bal.Acc: 46.0%  F1: 0.4257  |  Val Loss: 1.4616  Bal.Acc: 34.9%  F1: 0.3210  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 1.4616)
Epoch [  7/20]  Train Loss: 1.4478  Bal.Acc: 49.8%  F1: 0.4671  |  Val Loss: 1.3975  Bal.Acc: 44.4%  F1: 0.4294  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 1.3975)
Epoch [  8/20]  Train Loss: 1.3793  Bal.Acc: 52.2%  F1: 0.4823  |  Val Loss: 1.3836  Bal.Acc: 49.4%  F1: 0.4774  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 1.3836)
Epoch [  9/20]  Train Loss: 1.2809  Bal.Acc: 57.1%  F1: 0.5289  |  Val Loss: 1.3001  Bal.Acc: 52.7%  F1: 0.4846  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 1.3001)
Epoch [ 10/20]  Train Loss: 1.1816  Bal.Acc: 59.4%  F1: 0.5443  |  Val Loss: 1.2159  Bal.Acc: 48.5%  F1: 0.4661  |  LR: 1.00e-04  (17.4s)
 Best checkpoint saved (val_loss: 1.2159)
Epoch [ 11/20]  Train Loss: 1.0796  Bal.Acc: 60.6%  F1: 0.5431  |  Val Loss: 1.1656  Bal.Acc: 54.8%  F1: 0.5057  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 1.1656)
Epoch [ 12/20]  Train Loss: 0.9994  Bal.Acc: 63.2%  F1: 0.5943  |  Val Loss: 1.1094  Bal.Acc: 55.2%  F1: 0.5524  |  LR: 1.00e-04  (27.3s)
 Best checkpoint saved (val_loss: 1.1094)
Epoch [ 13/20]  Train Loss: 0.8948  Bal.Acc: 69.1%  F1: 0.6625  |  Val Loss: 0.9583  Bal.Acc: 59.8%  F1: 0.5878  |  LR: 1.00e-04  (23.4s)
 Best checkpoint saved (val_loss: 0.9583)
Epoch [ 14/20]  Train Loss: 0.8180  Bal.Acc: 68.4%  F1: 0.6619  |  Val Loss: 0.9964  Bal.Acc: 61.8%  F1: 0.6385  |  LR: 1.00e-04  (16.9s)
Epoch [ 15/20]  Train Loss: 0.7799  Bal.Acc: 70.3%  F1: 0.6941  |  Val Loss: 0.8292  Bal.Acc: 70.3%  F1: 0.7084  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 0.8292)
Epoch [ 16/20]  Train Loss: 0.7095  Bal.Acc: 74.8%  F1: 0.7373  |  Val Loss: 0.9402  Bal.Acc: 63.8%  F1: 0.6546  |  LR: 1.00e-04  (23.1s)
Epoch [ 17/20]  Train Loss: 0.6682  Bal.Acc: 74.6%  F1: 0.7411  |  Val Loss: 0.8784  Bal.Acc: 65.4%  F1: 0.6704  |  LR: 1.00e-04  (16.3s)
Epoch [ 18/20]  Train Loss: 0.6150  Bal.Acc: 76.6%  F1: 0.7586  |  Val Loss: 0.8579  Bal.Acc: 65.9%  F1: 0.6824  |  LR: 1.00e-04  (15.5s)
Epoch [ 19/20]  Train Loss: 0.5530  Bal.Acc: 79.3%  F1: 0.7878  |  Val Loss: 0.7464  Bal.Acc: 69.2%  F1: 0.6837  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 0.7464)
Epoch [ 20/20]  Train Loss: 0.5379  Bal.Acc: 78.9%  F1: 0.7853  |  Val Loss: 0.6822  Bal.Acc: 74.2%  F1: 0.7397  |  LR: 1.00e-04  (21.7s)
 Best checkpoint saved (val_loss: 0.6822)

 Training finished. Checkpoint: checkpoints/resnet50_fold4_best.pt
Log CSV: results/resnet50_fold4_training_log.csv
Best weights loaded from  20

Model evaluation: resnet50_fold4
----------------------------------------
  Balanced Accuracy:       74.21%
  F1 (macro):              0.7397
  Quadratic Cohen's Kappa: 0.8674
  ECE:                     0.0489
  Brier Score (mean):      0.0698

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.82      0.85        88
    Doubtful       0.70      0.72      0.71        81
        Mild       0.47      0.51      0.49        39
    Moderate       0.87      0.89      0.88        37
      Severe       0.77      0.77      0.77        35

    accuracy                           0.75       280
   macro avg       0.74      0.74      0.74       280
weighted avg       0.76      0.75      0.75       280

  Metrics saved to: results/resnet50_fold4_metrics.json
  Probability saved to: results/resnet50_fold4_test_probs.npz

--- resnet50 | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 5):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: resnet50

============================================================
TRAINING: resnet50_fold5
============================================================
Epoch [  1/20]  Train Loss: 1.6040  Bal.Acc: 22.7%  F1: 0.1491  |  Val Loss: 1.6041  Bal.Acc: 26.8%  F1: 0.1887  |  LR: 1.00e-04  (27.5s)
 Best checkpoint saved (val_loss: 1.6041)
Epoch [  2/20]  Train Loss: 1.5851  Bal.Acc: 32.0%  F1: 0.2995  |  Val Loss: 1.5945  Bal.Acc: 23.7%  F1: 0.2343  |  LR: 1.00e-04  (17.5s)
 Best checkpoint saved (val_loss: 1.5945)
Epoch [  3/20]  Train Loss: 1.5693  Bal.Acc: 37.8%  F1: 0.3729  |  Val Loss: 1.5812  Bal.Acc: 31.2%  F1: 0.3113  |  LR: 1.00e-04  (21.8s)
 Best checkpoint saved (val_loss: 1.5812)
Epoch [  4/20]  Train Loss: 1.5478  Bal.Acc: 43.2%  F1: 0.4026  |  Val Loss: 1.5457  Bal.Acc: 42.8%  F1: 0.4302  |  LR: 1.00e-04  (27.8s)
 Best checkpoint saved (val_loss: 1.5457)
Epoch [  5/20]  Train Loss: 1.5230  Bal.Acc: 45.0%  F1: 0.4357  |  Val Loss: 1.5263  Bal.Acc: 38.0%  F1: 0.3594  |  LR: 1.00e-04  (21.4s)
 Best checkpoint saved (val_loss: 1.5263)
Epoch [  6/20]  Train Loss: 1.4737  Bal.Acc: 50.4%  F1: 0.4665  |  Val Loss: 1.4906  Bal.Acc: 39.0%  F1: 0.3717  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 1.4906)
Epoch [  7/20]  Train Loss: 1.4123  Bal.Acc: 53.4%  F1: 0.4992  |  Val Loss: 1.4196  Bal.Acc: 46.8%  F1: 0.4457  |  LR: 1.00e-04  (17.3s)
 Best checkpoint saved (val_loss: 1.4196)
Epoch [  8/20]  Train Loss: 1.3239  Bal.Acc: 55.9%  F1: 0.5190  |  Val Loss: 1.3779  Bal.Acc: 44.7%  F1: 0.4382  |  LR: 1.00e-04  (15.5s)
 Best checkpoint saved (val_loss: 1.3779)
Epoch [  9/20]  Train Loss: 1.2095  Bal.Acc: 58.7%  F1: 0.5350  |  Val Loss: 1.2839  Bal.Acc: 53.7%  F1: 0.5219  |  LR: 1.00e-04  (26.8s)
 Best checkpoint saved (val_loss: 1.2839)
Epoch [ 10/20]  Train Loss: 1.0910  Bal.Acc: 59.7%  F1: 0.5496  |  Val Loss: 1.0850  Bal.Acc: 59.7%  F1: 0.5790  |  LR: 1.00e-04  (23.3s)
 Best checkpoint saved (val_loss: 1.0850)
Epoch [ 11/20]  Train Loss: 0.9686  Bal.Acc: 63.2%  F1: 0.6005  |  Val Loss: 1.0668  Bal.Acc: 61.0%  F1: 0.5993  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 1.0668)
Epoch [ 12/20]  Train Loss: 0.8729  Bal.Acc: 67.3%  F1: 0.6510  |  Val Loss: 1.0130  Bal.Acc: 56.9%  F1: 0.5655  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 1.0130)
Epoch [ 13/20]  Train Loss: 0.8022  Bal.Acc: 69.4%  F1: 0.6759  |  Val Loss: 1.1072  Bal.Acc: 53.6%  F1: 0.5604  |  LR: 1.00e-04  (16.7s)
Epoch [ 14/20]  Train Loss: 0.7607  Bal.Acc: 72.4%  F1: 0.7092  |  Val Loss: 0.9157  Bal.Acc: 64.3%  F1: 0.6476  |  LR: 1.00e-04  (19.1s)
 Best checkpoint saved (val_loss: 0.9157)
Epoch [ 15/20]  Train Loss: 0.6965  Bal.Acc: 74.6%  F1: 0.7422  |  Val Loss: 0.9287  Bal.Acc: 67.4%  F1: 0.6815  |  LR: 1.00e-04  (20.6s)
Epoch [ 16/20]  Train Loss: 0.6384  Bal.Acc: 76.0%  F1: 0.7509  |  Val Loss: 0.8050  Bal.Acc: 69.4%  F1: 0.6747  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 0.8050)
Epoch [ 17/20]  Train Loss: 0.6056  Bal.Acc: 77.8%  F1: 0.7648  |  Val Loss: 0.8756  Bal.Acc: 64.0%  F1: 0.6432  |  LR: 1.00e-04  (21.8s)
Epoch [ 18/20]  Train Loss: 0.5866  Bal.Acc: 77.2%  F1: 0.7655  |  Val Loss: 0.7313  Bal.Acc: 70.8%  F1: 0.7126  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 0.7313)
Epoch [ 19/20]  Train Loss: 0.5653  Bal.Acc: 78.0%  F1: 0.7720  |  Val Loss: 0.7030  Bal.Acc: 75.2%  F1: 0.7400  |  LR: 1.00e-04  (22.9s)
 Best checkpoint saved (val_loss: 0.7030)
Epoch [ 20/20]  Train Loss: 0.5067  Bal.Acc: 81.3%  F1: 0.8009  |  Val Loss: 0.8867  Bal.Acc: 63.8%  F1: 0.6307  |  LR: 1.00e-04  (27.0s)

 Training finished. Checkpoint: checkpoints/resnet50_fold5_best.pt
Log CSV: results/resnet50_fold5_training_log.csv
Best weights loaded from  19

Model evaluation: resnet50_fold5
----------------------------------------
  Balanced Accuracy:       75.23%
  F1 (macro):              0.7400
  Quadratic Cohen's Kappa: 0.8784
  ECE:                     0.0393
  Brier Score (mean):      0.0739

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.92      0.87        87
    Doubtful       0.80      0.60      0.69        81
        Mild       0.50      0.56      0.53        39
    Moderate       0.79      0.82      0.81        38
      Severe       0.75      0.86      0.80        35

    accuracy                           0.76       280
   macro avg       0.74      0.75      0.74       280
weighted avg       0.76      0.76      0.75       280

  Metrics saved to: results/resnet50_fold5_metrics.json
  Probability saved to: results/resnet50_fold5_test_probs.npz

 FINISHED: resnet50. Average kappa out of 5 folds: 0.8798 ±0.0181

================================================================================
MODEL TRAINING START: efficientnet_b3
================================================================================

--- efficientnet_b3 | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: efficientnet_b3
model.safetensors: 100% 49.3M/49.3M [00:01<00:00, 34.6MB/s]

============================================================
TRAINING: efficientnet_b3_fold1
============================================================
Epoch [  1/20]  Train Loss: 2.3076  Bal.Acc: 34.2%  F1: 0.3206  |  Val Loss: 1.8412  Bal.Acc: 39.7%  F1: 0.3652  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 1.8412)
Epoch [  2/20]  Train Loss: 1.6285  Bal.Acc: 47.6%  F1: 0.4645  |  Val Loss: 1.5331  Bal.Acc: 46.6%  F1: 0.4479  |  LR: 1.00e-04  (17.3s)
 Best checkpoint saved (val_loss: 1.5331)
Epoch [  3/20]  Train Loss: 1.1708  Bal.Acc: 58.6%  F1: 0.5678  |  Val Loss: 1.4043  Bal.Acc: 53.0%  F1: 0.5311  |  LR: 1.00e-04  (20.3s)
 Best checkpoint saved (val_loss: 1.4043)
Epoch [  4/20]  Train Loss: 1.0156  Bal.Acc: 63.4%  F1: 0.6167  |  Val Loss: 1.2568  Bal.Acc: 56.5%  F1: 0.5680  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 1.2568)
Epoch [  5/20]  Train Loss: 0.7930  Bal.Acc: 70.9%  F1: 0.6925  |  Val Loss: 158.1670  Bal.Acc: 63.3%  F1: 0.6197  |  LR: 1.00e-04  (22.0s)
Epoch [  6/20]  Train Loss: 0.7062  Bal.Acc: 75.1%  F1: 0.7391  |  Val Loss: 0.9038  Bal.Acc: 64.0%  F1: 0.6305  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 0.9038)
Epoch [  7/20]  Train Loss: 0.6387  Bal.Acc: 75.2%  F1: 0.7403  |  Val Loss: 55.7104  Bal.Acc: 63.1%  F1: 0.6243  |  LR: 1.00e-04  (19.4s)
Epoch [  8/20]  Train Loss: 0.5888  Bal.Acc: 78.6%  F1: 0.7775  |  Val Loss: 47.0899  Bal.Acc: 64.4%  F1: 0.6528  |  LR: 1.00e-04  (16.2s)
Epoch [  9/20]  Train Loss: 0.5045  Bal.Acc: 80.7%  F1: 0.7970  |  Val Loss: 496.6240  Bal.Acc: 70.3%  F1: 0.6942  |  LR: 1.00e-04  (15.6s)
Epoch [ 10/20]  Train Loss: 0.4785  Bal.Acc: 81.7%  F1: 0.8093  |  Val Loss: 0.7141  Bal.Acc: 71.9%  F1: 0.7112  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 0.7141)
Epoch [ 11/20]  Train Loss: 0.3789  Bal.Acc: 84.7%  F1: 0.8375  |  Val Loss: 1310.5426  Bal.Acc: 70.6%  F1: 0.7011  |  LR: 1.00e-04  (19.9s)
Epoch [ 12/20]  Train Loss: 0.3686  Bal.Acc: 86.6%  F1: 0.8598  |  Val Loss: 0.6930  Bal.Acc: 74.3%  F1: 0.7342  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 0.6930)
Epoch [ 13/20]  Train Loss: 0.3666  Bal.Acc: 86.1%  F1: 0.8578  |  Val Loss: 0.6993  Bal.Acc: 75.3%  F1: 0.7490  |  LR: 1.00e-04  (19.2s)
Epoch [ 14/20]  Train Loss: 0.3177  Bal.Acc: 88.1%  F1: 0.8705  |  Val Loss: 0.6678  Bal.Acc: 76.6%  F1: 0.7645  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 0.6678)
Epoch [ 15/20]  Train Loss: 0.2822  Bal.Acc: 89.0%  F1: 0.8844  |  Val Loss: 0.6684  Bal.Acc: 77.7%  F1: 0.7714  |  LR: 1.00e-04  (19.4s)
Epoch [ 16/20]  Train Loss: 0.2666  Bal.Acc: 90.7%  F1: 0.9015  |  Val Loss: 8180.9708  Bal.Acc: 77.3%  F1: 0.7689  |  LR: 1.00e-04  (16.1s)
Epoch [ 17/20]  Train Loss: 0.2088  Bal.Acc: 92.9%  F1: 0.9223  |  Val Loss: 18.5923  Bal.Acc: 76.4%  F1: 0.7608  |  LR: 1.00e-04  (16.4s)
Epoch [ 18/20]  Train Loss: 0.2126  Bal.Acc: 92.9%  F1: 0.9254  |  Val Loss: 45.0615  Bal.Acc: 77.5%  F1: 0.7697  |  LR: 5.00e-05  (16.7s)
Epoch [ 19/20]  Train Loss: 0.1957  Bal.Acc: 93.4%  F1: 0.9318  |  Val Loss: 0.6625  Bal.Acc: 77.6%  F1: 0.7688  |  LR: 5.00e-05  (16.6s)
 Best checkpoint saved (val_loss: 0.6625)
Epoch [ 20/20]  Train Loss: 0.1817  Bal.Acc: 94.0%  F1: 0.9351  |  Val Loss: 9.5453  Bal.Acc: 76.5%  F1: 0.7570  |  LR: 5.00e-05  (19.1s)

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold1_best.pt
Log CSV: results/efficientnet_b3_fold1_training_log.csv
Best weights loaded from  19

Model evaluation: efficientnet_b3_fold1
----------------------------------------
  Balanced Accuracy:       77.64%
  F1 (macro):              0.7688
  Quadratic Cohen's Kappa: 0.9019
  ECE:                     0.1242
  Brier Score (mean):      0.0687

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.78      0.78      0.78        87
    Doubtful       0.71      0.64      0.68        81
        Mild       0.60      0.70      0.64        40
    Moderate       0.84      0.82      0.83        38
      Severe       0.89      0.94      0.92        35

    accuracy                           0.75       281
   macro avg       0.76      0.78      0.77       281
weighted avg       0.76      0.75      0.75       281

  Metrics saved to: results/efficientnet_b3_fold1_metrics.json
  Probability saved to: results/efficientnet_b3_fold1_test_probs.npz

--- efficientnet_b3 | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: efficientnet_b3

============================================================
TRAINING: efficientnet_b3_fold2
============================================================
Epoch [  1/20]  Train Loss: 2.5985  Bal.Acc: 30.2%  F1: 0.2812  |  Val Loss: 2.3769  Bal.Acc: 36.2%  F1: 0.3190  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 2.3769)
Epoch [  2/20]  Train Loss: 1.5644  Bal.Acc: 49.9%  F1: 0.4753  |  Val Loss: 1.7777  Bal.Acc: 45.1%  F1: 0.4398  |  LR: 1.00e-04  (17.1s)
 Best checkpoint saved (val_loss: 1.7777)
Epoch [  3/20]  Train Loss: 1.2291  Bal.Acc: 57.1%  F1: 0.5526  |  Val Loss: 1.3978  Bal.Acc: 53.1%  F1: 0.5314  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 1.3978)
Epoch [  4/20]  Train Loss: 1.0064  Bal.Acc: 64.0%  F1: 0.6243  |  Val Loss: 1.1699  Bal.Acc: 60.4%  F1: 0.6181  |  LR: 1.00e-04  (18.9s)
 Best checkpoint saved (val_loss: 1.1699)
Epoch [  5/20]  Train Loss: 0.7798  Bal.Acc: 72.7%  F1: 0.7097  |  Val Loss: 1.0305  Bal.Acc: 64.1%  F1: 0.6513  |  LR: 1.00e-04  (23.4s)
 Best checkpoint saved (val_loss: 1.0305)
Epoch [  6/20]  Train Loss: 0.7016  Bal.Acc: 73.4%  F1: 0.7233  |  Val Loss: 0.8866  Bal.Acc: 65.7%  F1: 0.6696  |  LR: 1.00e-04  (18.8s)
 Best checkpoint saved (val_loss: 0.8866)
Epoch [  7/20]  Train Loss: 0.6142  Bal.Acc: 77.7%  F1: 0.7642  |  Val Loss: 0.8418  Bal.Acc: 68.2%  F1: 0.6863  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 0.8418)
Epoch [  8/20]  Train Loss: 0.5738  Bal.Acc: 77.4%  F1: 0.7653  |  Val Loss: 0.7541  Bal.Acc: 73.0%  F1: 0.7269  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 0.7541)
Epoch [  9/20]  Train Loss: 0.4978  Bal.Acc: 80.5%  F1: 0.7959  |  Val Loss: 0.6907  Bal.Acc: 75.9%  F1: 0.7559  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 0.6907)
Epoch [ 10/20]  Train Loss: 0.4065  Bal.Acc: 83.7%  F1: 0.8286  |  Val Loss: 0.7274  Bal.Acc: 74.1%  F1: 0.7407  |  LR: 1.00e-04  (22.8s)
Epoch [ 11/20]  Train Loss: 0.3760  Bal.Acc: 86.3%  F1: 0.8571  |  Val Loss: 0.7061  Bal.Acc: 77.7%  F1: 0.7841  |  LR: 1.00e-04  (15.8s)
Epoch [ 12/20]  Train Loss: 0.2971  Bal.Acc: 89.0%  F1: 0.8828  |  Val Loss: 0.6525  Bal.Acc: 78.7%  F1: 0.7930  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 0.6525)
Epoch [ 13/20]  Train Loss: 0.2837  Bal.Acc: 89.2%  F1: 0.8862  |  Val Loss: 0.6467  Bal.Acc: 79.0%  F1: 0.7920  |  LR: 1.00e-04  (18.7s)
 Best checkpoint saved (val_loss: 0.6467)
Epoch [ 14/20]  Train Loss: 0.2605  Bal.Acc: 90.7%  F1: 0.9007  |  Val Loss: 0.6473  Bal.Acc: 80.1%  F1: 0.7999  |  LR: 1.00e-04  (22.6s)
Epoch [ 15/20]  Train Loss: 0.2596  Bal.Acc: 90.5%  F1: 0.9003  |  Val Loss: 0.6368  Bal.Acc: 81.3%  F1: 0.8164  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 0.6368)
Epoch [ 16/20]  Train Loss: 0.2242  Bal.Acc: 92.2%  F1: 0.9202  |  Val Loss: 0.6288  Bal.Acc: 80.3%  F1: 0.8033  |  LR: 1.00e-04  (19.0s)
 Best checkpoint saved (val_loss: 0.6288)
Epoch [ 17/20]  Train Loss: 0.1825  Bal.Acc: 93.8%  F1: 0.9326  |  Val Loss: 0.6593  Bal.Acc: 81.8%  F1: 0.8200  |  LR: 1.00e-04  (22.5s)
Epoch [ 18/20]  Train Loss: 0.2146  Bal.Acc: 92.3%  F1: 0.9221  |  Val Loss: 0.5946  Bal.Acc: 82.2%  F1: 0.8213  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 0.5946)
Epoch [ 19/20]  Train Loss: 0.1911  Bal.Acc: 93.3%  F1: 0.9254  |  Val Loss: 0.5937  Bal.Acc: 82.6%  F1: 0.8313  |  LR: 1.00e-04  (19.3s)
 Best checkpoint saved (val_loss: 0.5937)
Epoch [ 20/20]  Train Loss: 0.1574  Bal.Acc: 94.0%  F1: 0.9367  |  Val Loss: 0.6235  Bal.Acc: 81.4%  F1: 0.8151  |  LR: 1.00e-04  (22.7s)

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold2_best.pt
Log CSV: results/efficientnet_b3_fold2_training_log.csv
Best weights loaded from  19

Model evaluation: efficientnet_b3_fold2
----------------------------------------
  Balanced Accuracy:       82.59%
  F1 (macro):              0.8313
  Quadratic Cohen's Kappa: 0.9139
  ECE:                     0.0797
  Brier Score (mean):      0.0528

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.86      0.92      0.89        87
    Doubtful       0.78      0.77      0.78        81
        Mild       0.77      0.75      0.76        40
    Moderate       0.85      0.89      0.87        38
      Severe       0.93      0.80      0.86        35

    accuracy                           0.83       281
   macro avg       0.84      0.83      0.83       281
weighted avg       0.83      0.83      0.83       281

  Metrics saved to: results/efficientnet_b3_fold2_metrics.json
  Probability saved to: results/efficientnet_b3_fold2_test_probs.npz

--- efficientnet_b3 | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: efficientnet_b3

============================================================
TRAINING: efficientnet_b3_fold3
============================================================
Epoch [  1/20]  Train Loss: 2.7528  Bal.Acc: 30.3%  F1: 0.2965  |  Val Loss: 1.9898  Bal.Acc: 38.5%  F1: 0.3572  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 1.9898)
Epoch [  2/20]  Train Loss: 1.6081  Bal.Acc: 48.7%  F1: 0.4578  |  Val Loss: 1.7515  Bal.Acc: 42.1%  F1: 0.4050  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 1.7515)
Epoch [  3/20]  Train Loss: 1.2822  Bal.Acc: 56.1%  F1: 0.5492  |  Val Loss: 1.5603  Bal.Acc: 49.6%  F1: 0.4996  |  LR: 1.00e-04  (19.6s)
 Best checkpoint saved (val_loss: 1.5603)
Epoch [  4/20]  Train Loss: 1.0400  Bal.Acc: 63.1%  F1: 0.6171  |  Val Loss: 1.4078  Bal.Acc: 52.9%  F1: 0.5328  |  LR: 1.00e-04  (20.1s)
 Best checkpoint saved (val_loss: 1.4078)
Epoch [  5/20]  Train Loss: 0.8325  Bal.Acc: 69.1%  F1: 0.6720  |  Val Loss: 1.1572  Bal.Acc: 59.7%  F1: 0.5906  |  LR: 1.00e-04  (19.0s)
 Best checkpoint saved (val_loss: 1.1572)
Epoch [  6/20]  Train Loss: 0.7248  Bal.Acc: 73.7%  F1: 0.7222  |  Val Loss: 1.1489  Bal.Acc: 63.1%  F1: 0.6355  |  LR: 1.00e-04  (21.6s)
 Best checkpoint saved (val_loss: 1.1489)
Epoch [  7/20]  Train Loss: 0.6939  Bal.Acc: 74.3%  F1: 0.7304  |  Val Loss: 0.9730  Bal.Acc: 65.6%  F1: 0.6636  |  LR: 1.00e-04  (22.2s)
 Best checkpoint saved (val_loss: 0.9730)
Epoch [  8/20]  Train Loss: 0.5686  Bal.Acc: 77.6%  F1: 0.7641  |  Val Loss: 0.9341  Bal.Acc: 69.1%  F1: 0.6992  |  LR: 1.00e-04  (22.1s)
 Best checkpoint saved (val_loss: 0.9341)
Epoch [  9/20]  Train Loss: 0.4504  Bal.Acc: 83.5%  F1: 0.8244  |  Val Loss: 0.9147  Bal.Acc: 69.5%  F1: 0.7028  |  LR: 1.00e-04  (22.8s)
 Best checkpoint saved (val_loss: 0.9147)
Epoch [ 10/20]  Train Loss: 0.4276  Bal.Acc: 83.4%  F1: 0.8253  |  Val Loss: 0.9491  Bal.Acc: 71.9%  F1: 0.7204  |  LR: 1.00e-04  (21.2s)
Epoch [ 11/20]  Train Loss: 0.3749  Bal.Acc: 86.5%  F1: 0.8545  |  Val Loss: 0.9203  Bal.Acc: 72.0%  F1: 0.7278  |  LR: 1.00e-04  (16.3s)
Epoch [ 12/20]  Train Loss: 0.3647  Bal.Acc: 86.9%  F1: 0.8595  |  Val Loss: 0.9792  Bal.Acc: 72.3%  F1: 0.7340  |  LR: 1.00e-04  (16.6s)
Epoch [ 13/20]  Train Loss: 0.3009  Bal.Acc: 88.0%  F1: 0.8763  |  Val Loss: 0.9109  Bal.Acc: 73.5%  F1: 0.7401  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 0.9109)
Epoch [ 14/20]  Train Loss: 0.3276  Bal.Acc: 87.2%  F1: 0.8667  |  Val Loss: 0.8507  Bal.Acc: 73.0%  F1: 0.7340  |  LR: 1.00e-04  (19.3s)
 Best checkpoint saved (val_loss: 0.8507)
Epoch [ 15/20]  Train Loss: 0.2668  Bal.Acc: 90.7%  F1: 0.8987  |  Val Loss: 0.8095  Bal.Acc: 76.2%  F1: 0.7626  |  LR: 1.00e-04  (21.5s)
 Best checkpoint saved (val_loss: 0.8095)
Epoch [ 16/20]  Train Loss: 0.2563  Bal.Acc: 90.2%  F1: 0.8949  |  Val Loss: 0.8705  Bal.Acc: 76.9%  F1: 0.7712  |  LR: 1.00e-04  (23.1s)
Epoch [ 17/20]  Train Loss: 0.2392  Bal.Acc: 92.1%  F1: 0.9163  |  Val Loss: 0.8936  Bal.Acc: 75.5%  F1: 0.7635  |  LR: 1.00e-04  (15.7s)
Epoch [ 18/20]  Train Loss: 0.1885  Bal.Acc: 93.1%  F1: 0.9286  |  Val Loss: 0.8370  Bal.Acc: 76.0%  F1: 0.7660  |  LR: 1.00e-04  (15.7s)
Epoch [ 19/20]  Train Loss: 0.1852  Bal.Acc: 93.1%  F1: 0.9264  |  Val Loss: 0.8041  Bal.Acc: 76.3%  F1: 0.7682  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 0.8041)
Epoch [ 20/20]  Train Loss: 0.1743  Bal.Acc: 94.5%  F1: 0.9408  |  Val Loss: 0.7792  Bal.Acc: 77.4%  F1: 0.7732  |  LR: 1.00e-04  (18.6s)
 Best checkpoint saved (val_loss: 0.7792)

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold3_best.pt
Log CSV: results/efficientnet_b3_fold3_training_log.csv
Best weights loaded from  20

Model evaluation: efficientnet_b3_fold3
----------------------------------------
  Balanced Accuracy:       77.41%
  F1 (macro):              0.7732
  Quadratic Cohen's Kappa: 0.8991
  ECE:                     0.1388
  Brier Score (mean):      0.0718

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.79      0.86      0.83        88
    Doubtful       0.73      0.67      0.70        81
        Mild       0.62      0.59      0.61        39
    Moderate       0.86      0.86      0.86        37
      Severe       0.86      0.89      0.87        35

    accuracy                           0.77       280
   macro avg       0.77      0.77      0.77       280
weighted avg       0.77      0.77      0.77       280

  Metrics saved to: results/efficientnet_b3_fold3_metrics.json
  Probability saved to: results/efficientnet_b3_fold3_test_probs.npz

--- efficientnet_b3 | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: efficientnet_b3

============================================================
TRAINING: efficientnet_b3_fold4
============================================================
Epoch [  1/20]  Train Loss: 2.7395  Bal.Acc: 29.3%  F1: 0.2757  |  Val Loss: 1.8908  Bal.Acc: 37.5%  F1: 0.3585  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 1.8908)
Epoch [  2/20]  Train Loss: 1.5347  Bal.Acc: 50.6%  F1: 0.4808  |  Val Loss: 1.5909  Bal.Acc: 42.0%  F1: 0.4003  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 1.5909)
Epoch [  3/20]  Train Loss: 1.2343  Bal.Acc: 57.4%  F1: 0.5622  |  Val Loss: 1.3069  Bal.Acc: 53.1%  F1: 0.5191  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 1.3069)
Epoch [  4/20]  Train Loss: 0.9869  Bal.Acc: 64.1%  F1: 0.6199  |  Val Loss: 1.1902  Bal.Acc: 58.9%  F1: 0.5888  |  LR: 1.00e-04  (18.1s)
 Best checkpoint saved (val_loss: 1.1902)
Epoch [  5/20]  Train Loss: 0.8506  Bal.Acc: 69.4%  F1: 0.6815  |  Val Loss: 1.0893  Bal.Acc: 60.6%  F1: 0.6215  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 1.0893)
Epoch [  6/20]  Train Loss: 0.7328  Bal.Acc: 71.4%  F1: 0.7013  |  Val Loss: 0.9774  Bal.Acc: 65.8%  F1: 0.6649  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 0.9774)
Epoch [  7/20]  Train Loss: 0.5952  Bal.Acc: 76.1%  F1: 0.7456  |  Val Loss: 0.9466  Bal.Acc: 65.2%  F1: 0.6563  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 0.9466)
Epoch [  8/20]  Train Loss: 0.5532  Bal.Acc: 79.2%  F1: 0.7790  |  Val Loss: 0.9136  Bal.Acc: 66.6%  F1: 0.6661  |  LR: 1.00e-04  (19.2s)
 Best checkpoint saved (val_loss: 0.9136)
Epoch [  9/20]  Train Loss: 0.5137  Bal.Acc: 80.7%  F1: 0.7969  |  Val Loss: 0.9192  Bal.Acc: 67.2%  F1: 0.6751  |  LR: 1.00e-04  (19.6s)
Epoch [ 10/20]  Train Loss: 0.4347  Bal.Acc: 84.0%  F1: 0.8281  |  Val Loss: 0.8533  Bal.Acc: 73.0%  F1: 0.7264  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 0.8533)
Epoch [ 11/20]  Train Loss: 0.3656  Bal.Acc: 86.7%  F1: 0.8606  |  Val Loss: 0.8813  Bal.Acc: 71.4%  F1: 0.7156  |  LR: 1.00e-04  (20.0s)
Epoch [ 12/20]  Train Loss: 0.3274  Bal.Acc: 87.0%  F1: 0.8629  |  Val Loss: 0.8429  Bal.Acc: 72.0%  F1: 0.7205  |  LR: 1.00e-04  (15.9s)
 Best checkpoint saved (val_loss: 0.8429)
Epoch [ 13/20]  Train Loss: 0.3335  Bal.Acc: 85.5%  F1: 0.8467  |  Val Loss: 0.8547  Bal.Acc: 72.8%  F1: 0.7328  |  LR: 1.00e-04  (19.2s)
Epoch [ 14/20]  Train Loss: 0.2945  Bal.Acc: 88.3%  F1: 0.8796  |  Val Loss: 0.7904  Bal.Acc: 73.7%  F1: 0.7399  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 0.7904)
Epoch [ 15/20]  Train Loss: 0.2761  Bal.Acc: 89.2%  F1: 0.8849  |  Val Loss: 0.8106  Bal.Acc: 76.0%  F1: 0.7580  |  LR: 1.00e-04  (19.4s)
Epoch [ 16/20]  Train Loss: 0.2714  Bal.Acc: 90.4%  F1: 0.8987  |  Val Loss: 0.8003  Bal.Acc: 74.6%  F1: 0.7464  |  LR: 1.00e-04  (16.7s)
Epoch [ 17/20]  Train Loss: 0.2407  Bal.Acc: 90.7%  F1: 0.9027  |  Val Loss: 0.7665  Bal.Acc: 76.4%  F1: 0.7623  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 0.7665)
Epoch [ 18/20]  Train Loss: 0.2429  Bal.Acc: 91.7%  F1: 0.9078  |  Val Loss: 0.8458  Bal.Acc: 72.4%  F1: 0.7232  |  LR: 1.00e-04  (19.0s)
Epoch [ 19/20]  Train Loss: 0.1924  Bal.Acc: 93.5%  F1: 0.9332  |  Val Loss: 0.8661  Bal.Acc: 72.2%  F1: 0.7259  |  LR: 1.00e-04  (16.0s)
Epoch [ 20/20]  Train Loss: 0.1907  Bal.Acc: 93.4%  F1: 0.9267  |  Val Loss: 0.8733  Bal.Acc: 72.6%  F1: 0.7291  |  LR: 1.00e-04  (15.9s)

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold4_best.pt
Log CSV: results/efficientnet_b3_fold4_training_log.csv
Best weights loaded from  17

Model evaluation: efficientnet_b3_fold4
----------------------------------------
  Balanced Accuracy:       76.44%
  F1 (macro):              0.7623
  Quadratic Cohen's Kappa: 0.9022
  ECE:                     0.1281
  Brier Score (mean):      0.0742

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.82      0.84      0.83        88
    Doubtful       0.71      0.67      0.69        81
        Mild       0.51      0.54      0.53        39
    Moderate       0.83      0.92      0.87        37
      Severe       0.94      0.86      0.90        35

    accuracy                           0.76       280
   macro avg       0.76      0.76      0.76       280
weighted avg       0.76      0.76      0.76       280

  Metrics saved to: results/efficientnet_b3_fold4_metrics.json
  Probability saved to: results/efficientnet_b3_fold4_test_probs.npz

--- efficientnet_b3 | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 5):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: efficientnet_b3

============================================================
TRAINING: efficientnet_b3_fold5
============================================================
Epoch [  1/20]  Train Loss: 2.7577  Bal.Acc: 27.4%  F1: 0.2671  |  Val Loss: 2.2234  Bal.Acc: 31.4%  F1: 0.2818  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 2.2234)
Epoch [  2/20]  Train Loss: 1.6090  Bal.Acc: 48.8%  F1: 0.4570  |  Val Loss: 1.9786  Bal.Acc: 40.0%  F1: 0.3600  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 1.9786)
Epoch [  3/20]  Train Loss: 1.2647  Bal.Acc: 57.0%  F1: 0.5531  |  Val Loss: 1.6396  Bal.Acc: 50.0%  F1: 0.4592  |  LR: 1.00e-04  (19.3s)
 Best checkpoint saved (val_loss: 1.6396)
Epoch [  4/20]  Train Loss: 1.0069  Bal.Acc: 64.2%  F1: 0.6218  |  Val Loss: 1.5597  Bal.Acc: 54.1%  F1: 0.5204  |  LR: 1.00e-04  (19.5s)
 Best checkpoint saved (val_loss: 1.5597)
Epoch [  5/20]  Train Loss: 0.8584  Bal.Acc: 68.6%  F1: 0.6740  |  Val Loss: 11.4732  Bal.Acc: 61.7%  F1: 0.6238  |  LR: 1.00e-04  (20.1s)
Epoch [  6/20]  Train Loss: 0.7282  Bal.Acc: 73.8%  F1: 0.7261  |  Val Loss: 14.7264  Bal.Acc: 67.4%  F1: 0.6750  |  LR: 1.00e-04  (16.0s)
Epoch [  7/20]  Train Loss: 0.6339  Bal.Acc: 76.7%  F1: 0.7551  |  Val Loss: 28.0535  Bal.Acc: 68.2%  F1: 0.6873  |  LR: 1.00e-04  (15.2s)
Epoch [  8/20]  Train Loss: 0.5772  Bal.Acc: 78.0%  F1: 0.7672  |  Val Loss: 214.6264  Bal.Acc: 68.2%  F1: 0.6788  |  LR: 5.00e-05  (15.3s)
Epoch [  9/20]  Train Loss: 0.4810  Bal.Acc: 82.8%  F1: 0.8190  |  Val Loss: 239.6594  Bal.Acc: 66.9%  F1: 0.6716  |  LR: 5.00e-05  (15.7s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 1.5597

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold5_best.pt
Log CSV: results/efficientnet_b3_fold5_training_log.csv
Best weights loaded from  4

Model evaluation: efficientnet_b3_fold5
----------------------------------------
  Balanced Accuracy:       54.06%
  F1 (macro):              0.5204
  Quadratic Cohen's Kappa: 0.7338
  ECE:                     0.1862
  Brier Score (mean):      0.1223

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.67      0.76      0.71        87
    Doubtful       0.54      0.42      0.47        81
        Mild       0.37      0.26      0.30        39
    Moderate       0.45      0.87      0.59        38
      Severe       0.74      0.40      0.52        35

    accuracy                           0.56       280
   macro avg       0.55      0.54      0.52       280
weighted avg       0.57      0.56      0.55       280

  Metrics saved to: results/efficientnet_b3_fold5_metrics.json
  Probability saved to: results/efficientnet_b3_fold5_test_probs.npz

 FINISHED: efficientnet_b3. Average kappa out of 5 folds: 0.8702 ±0.0684

================================================================================
MODEL TRAINING START: densenet121
================================================================================

--- densenet121 | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: densenet121
model.safetensors: 100% 32.3M/32.3M [00:02<00:00, 16.1MB/s]

============================================================
TRAINING: densenet121_fold1
============================================================
Epoch [  1/20]  Train Loss: 1.5373  Bal.Acc: 31.7%  F1: 0.2979  |  Val Loss: 1.4727  Bal.Acc: 30.9%  F1: 0.3112  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 1.4727)
Epoch [  2/20]  Train Loss: 1.2338  Bal.Acc: 53.7%  F1: 0.4937  |  Val Loss: 1.1990  Bal.Acc: 56.3%  F1: 0.5474  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 1.1990)
Epoch [  3/20]  Train Loss: 1.0111  Bal.Acc: 63.9%  F1: 0.6189  |  Val Loss: 0.9748  Bal.Acc: 62.7%  F1: 0.6251  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.9748)
Epoch [  4/20]  Train Loss: 0.8264  Bal.Acc: 70.5%  F1: 0.6903  |  Val Loss: 0.9074  Bal.Acc: 65.1%  F1: 0.6485  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.9074)
Epoch [  5/20]  Train Loss: 0.6733  Bal.Acc: 76.9%  F1: 0.7631  |  Val Loss: 0.7675  Bal.Acc: 72.7%  F1: 0.7192  |  LR: 1.00e-04  (18.8s)
 Best checkpoint saved (val_loss: 0.7675)
Epoch [  6/20]  Train Loss: 0.5506  Bal.Acc: 81.4%  F1: 0.8030  |  Val Loss: 0.7321  Bal.Acc: 70.8%  F1: 0.6999  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 0.7321)
Epoch [  7/20]  Train Loss: 0.4971  Bal.Acc: 82.5%  F1: 0.8162  |  Val Loss: 0.6268  Bal.Acc: 76.2%  F1: 0.7487  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 0.6268)
Epoch [  8/20]  Train Loss: 0.4537  Bal.Acc: 83.2%  F1: 0.8256  |  Val Loss: 0.6305  Bal.Acc: 77.2%  F1: 0.7611  |  LR: 1.00e-04  (19.9s)
Epoch [  9/20]  Train Loss: 0.3845  Bal.Acc: 86.9%  F1: 0.8620  |  Val Loss: 0.6113  Bal.Acc: 76.5%  F1: 0.7655  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 0.6113)
Epoch [ 10/20]  Train Loss: 0.3343  Bal.Acc: 88.6%  F1: 0.8822  |  Val Loss: 0.5162  Bal.Acc: 80.4%  F1: 0.7929  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.5162)
Epoch [ 11/20]  Train Loss: 0.3163  Bal.Acc: 89.5%  F1: 0.8904  |  Val Loss: 0.5297  Bal.Acc: 81.1%  F1: 0.8062  |  LR: 1.00e-04  (18.0s)
Epoch [ 12/20]  Train Loss: 0.2647  Bal.Acc: 91.6%  F1: 0.9118  |  Val Loss: 0.6087  Bal.Acc: 77.2%  F1: 0.7763  |  LR: 1.00e-04  (15.9s)
Epoch [ 13/20]  Train Loss: 0.2511  Bal.Acc: 92.1%  F1: 0.9150  |  Val Loss: 0.5515  Bal.Acc: 80.6%  F1: 0.8007  |  LR: 1.00e-04  (16.0s)
Epoch [ 14/20]  Train Loss: 0.2127  Bal.Acc: 93.4%  F1: 0.9277  |  Val Loss: 0.5555  Bal.Acc: 81.8%  F1: 0.8217  |  LR: 5.00e-05  (15.8s)
Epoch [ 15/20]  Train Loss: 0.2111  Bal.Acc: 93.0%  F1: 0.9290  |  Val Loss: 0.5014  Bal.Acc: 82.9%  F1: 0.8259  |  LR: 5.00e-05  (16.1s)
 Best checkpoint saved (val_loss: 0.5014)
Epoch [ 16/20]  Train Loss: 0.1845  Bal.Acc: 93.9%  F1: 0.9360  |  Val Loss: 0.5052  Bal.Acc: 81.6%  F1: 0.8091  |  LR: 5.00e-05  (18.8s)
Epoch [ 17/20]  Train Loss: 0.1931  Bal.Acc: 93.8%  F1: 0.9323  |  Val Loss: 0.5033  Bal.Acc: 81.8%  F1: 0.8177  |  LR: 5.00e-05  (15.4s)
Epoch [ 18/20]  Train Loss: 0.1577  Bal.Acc: 94.7%  F1: 0.9449  |  Val Loss: 0.5019  Bal.Acc: 81.5%  F1: 0.8136  |  LR: 5.00e-05  (15.4s)
Epoch [ 19/20]  Train Loss: 0.1575  Bal.Acc: 94.3%  F1: 0.9409  |  Val Loss: 0.4810  Bal.Acc: 83.7%  F1: 0.8329  |  LR: 5.00e-05  (16.3s)
 Best checkpoint saved (val_loss: 0.4810)
Epoch [ 20/20]  Train Loss: 0.1333  Bal.Acc: 95.2%  F1: 0.9504  |  Val Loss: 0.4810  Bal.Acc: 85.4%  F1: 0.8531  |  LR: 5.00e-05  (17.7s)

 Training finished. Checkpoint: checkpoints/densenet121_fold1_best.pt
Log CSV: results/densenet121_fold1_training_log.csv
Best weights loaded from  19

Model evaluation: densenet121_fold1
----------------------------------------
  Balanced Accuracy:       83.69%
  F1 (macro):              0.8329
  Quadratic Cohen's Kappa: 0.9316
  ECE:                     0.0631
  Brier Score (mean):      0.0502

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.91      0.85      0.88        87
    Doubtful       0.80      0.83      0.81        81
        Mild       0.72      0.70      0.71        40
    Moderate       0.81      0.92      0.86        38
      Severe       0.91      0.89      0.90        35

    accuracy                           0.84       281
   macro avg       0.83      0.84      0.83       281
weighted avg       0.84      0.84      0.84       281

  Metrics saved to: results/densenet121_fold1_metrics.json
  Probability saved to: results/densenet121_fold1_test_probs.npz

--- densenet121 | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: densenet121

============================================================
TRAINING: densenet121_fold2
============================================================
Epoch [  1/20]  Train Loss: 1.5017  Bal.Acc: 34.7%  F1: 0.3001  |  Val Loss: 1.3978  Bal.Acc: 42.5%  F1: 0.4252  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 1.3978)
Epoch [  2/20]  Train Loss: 1.2242  Bal.Acc: 52.2%  F1: 0.4775  |  Val Loss: 1.0894  Bal.Acc: 59.3%  F1: 0.5863  |  LR: 1.00e-04  (15.6s)
 Best checkpoint saved (val_loss: 1.0894)
Epoch [  3/20]  Train Loss: 0.9958  Bal.Acc: 62.5%  F1: 0.6034  |  Val Loss: 0.9216  Bal.Acc: 64.3%  F1: 0.6272  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 0.9216)
Epoch [  4/20]  Train Loss: 0.8167  Bal.Acc: 72.2%  F1: 0.7081  |  Val Loss: 0.8373  Bal.Acc: 65.4%  F1: 0.6431  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.8373)
Epoch [  5/20]  Train Loss: 0.6689  Bal.Acc: 76.8%  F1: 0.7648  |  Val Loss: 0.6779  Bal.Acc: 76.6%  F1: 0.7484  |  LR: 1.00e-04  (18.6s)
 Best checkpoint saved (val_loss: 0.6779)
Epoch [  6/20]  Train Loss: 0.5835  Bal.Acc: 79.2%  F1: 0.7853  |  Val Loss: 0.7642  Bal.Acc: 66.4%  F1: 0.6658  |  LR: 1.00e-04  (17.7s)
Epoch [  7/20]  Train Loss: 0.4975  Bal.Acc: 83.8%  F1: 0.8311  |  Val Loss: 0.5942  Bal.Acc: 78.6%  F1: 0.7709  |  LR: 1.00e-04  (15.5s)
 Best checkpoint saved (val_loss: 0.5942)
Epoch [  8/20]  Train Loss: 0.4419  Bal.Acc: 84.7%  F1: 0.8426  |  Val Loss: 0.5632  Bal.Acc: 77.4%  F1: 0.7806  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.5632)
Epoch [  9/20]  Train Loss: 0.4112  Bal.Acc: 85.7%  F1: 0.8524  |  Val Loss: 0.5701  Bal.Acc: 80.2%  F1: 0.8028  |  LR: 1.00e-04  (17.8s)
Epoch [ 10/20]  Train Loss: 0.3489  Bal.Acc: 88.2%  F1: 0.8784  |  Val Loss: 0.5036  Bal.Acc: 81.5%  F1: 0.8030  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 0.5036)
Epoch [ 11/20]  Train Loss: 0.3101  Bal.Acc: 89.0%  F1: 0.8828  |  Val Loss: 0.5309  Bal.Acc: 77.8%  F1: 0.7735  |  LR: 1.00e-04  (18.2s)
Epoch [ 12/20]  Train Loss: 0.2832  Bal.Acc: 89.9%  F1: 0.8950  |  Val Loss: 0.4923  Bal.Acc: 82.0%  F1: 0.8195  |  LR: 1.00e-04  (15.5s)
 Best checkpoint saved (val_loss: 0.4923)
Epoch [ 13/20]  Train Loss: 0.2578  Bal.Acc: 91.8%  F1: 0.9155  |  Val Loss: 0.4591  Bal.Acc: 82.0%  F1: 0.8116  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 0.4591)
Epoch [ 14/20]  Train Loss: 0.2008  Bal.Acc: 93.6%  F1: 0.9286  |  Val Loss: 0.4423  Bal.Acc: 83.2%  F1: 0.8223  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 0.4423)
Epoch [ 15/20]  Train Loss: 0.2062  Bal.Acc: 92.8%  F1: 0.9243  |  Val Loss: 0.4272  Bal.Acc: 84.1%  F1: 0.8431  |  LR: 1.00e-04  (18.6s)
 Best checkpoint saved (val_loss: 0.4272)
Epoch [ 16/20]  Train Loss: 0.1726  Bal.Acc: 94.4%  F1: 0.9394  |  Val Loss: 0.4863  Bal.Acc: 82.1%  F1: 0.8197  |  LR: 1.00e-04  (18.4s)
Epoch [ 17/20]  Train Loss: 0.1842  Bal.Acc: 93.7%  F1: 0.9339  |  Val Loss: 0.4790  Bal.Acc: 80.3%  F1: 0.7965  |  LR: 1.00e-04  (16.5s)
Epoch [ 18/20]  Train Loss: 0.1907  Bal.Acc: 94.0%  F1: 0.9351  |  Val Loss: 0.4869  Bal.Acc: 82.9%  F1: 0.8247  |  LR: 1.00e-04  (16.2s)
Epoch [ 19/20]  Train Loss: 0.1715  Bal.Acc: 95.0%  F1: 0.9487  |  Val Loss: 0.4903  Bal.Acc: 82.3%  F1: 0.8236  |  LR: 5.00e-05  (16.0s)
Epoch [ 20/20]  Train Loss: 0.1636  Bal.Acc: 94.7%  F1: 0.9447  |  Val Loss: 0.4345  Bal.Acc: 84.1%  F1: 0.8389  |  LR: 5.00e-05  (16.2s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.4272

 Training finished. Checkpoint: checkpoints/densenet121_fold2_best.pt
Log CSV: results/densenet121_fold2_training_log.csv
Best weights loaded from  15

Model evaluation: densenet121_fold2
----------------------------------------
  Balanced Accuracy:       84.06%
  F1 (macro):              0.8431
  Quadratic Cohen's Kappa: 0.9333
  ECE:                     0.0206
  Brier Score (mean):      0.0461

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.90      0.90      0.90        87
    Doubtful       0.79      0.80      0.80        81
        Mild       0.73      0.80      0.76        40
    Moderate       0.91      0.79      0.85        38
      Severe       0.91      0.91      0.91        35

    accuracy                           0.84       281
   macro avg       0.85      0.84      0.84       281
weighted avg       0.85      0.84      0.84       281

  Metrics saved to: results/densenet121_fold2_metrics.json
  Probability saved to: results/densenet121_fold2_test_probs.npz

--- densenet121 | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: densenet121

============================================================
TRAINING: densenet121_fold3
============================================================
Epoch [  1/20]  Train Loss: 1.4776  Bal.Acc: 36.0%  F1: 0.3236  |  Val Loss: 1.4132  Bal.Acc: 40.4%  F1: 0.3701  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 1.4132)
Epoch [  2/20]  Train Loss: 1.1775  Bal.Acc: 56.1%  F1: 0.5239  |  Val Loss: 1.1864  Bal.Acc: 55.9%  F1: 0.5400  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 1.1864)
Epoch [  3/20]  Train Loss: 0.9802  Bal.Acc: 64.7%  F1: 0.6265  |  Val Loss: 1.0312  Bal.Acc: 57.0%  F1: 0.5625  |  LR: 1.00e-04  (18.6s)
 Best checkpoint saved (val_loss: 1.0312)
Epoch [  4/20]  Train Loss: 0.7894  Bal.Acc: 73.4%  F1: 0.7173  |  Val Loss: 0.8916  Bal.Acc: 61.9%  F1: 0.6219  |  LR: 1.00e-04  (18.6s)
 Best checkpoint saved (val_loss: 0.8916)
Epoch [  5/20]  Train Loss: 0.6399  Bal.Acc: 78.4%  F1: 0.7787  |  Val Loss: 0.7739  Bal.Acc: 68.2%  F1: 0.6704  |  LR: 1.00e-04  (18.7s)
 Best checkpoint saved (val_loss: 0.7739)
Epoch [  6/20]  Train Loss: 0.5303  Bal.Acc: 82.4%  F1: 0.8177  |  Val Loss: 0.7108  Bal.Acc: 71.7%  F1: 0.7175  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.7108)
Epoch [  7/20]  Train Loss: 0.4523  Bal.Acc: 85.1%  F1: 0.8473  |  Val Loss: 0.6891  Bal.Acc: 73.8%  F1: 0.7282  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.6891)
Epoch [  8/20]  Train Loss: 0.3768  Bal.Acc: 87.3%  F1: 0.8644  |  Val Loss: 0.7465  Bal.Acc: 68.7%  F1: 0.6919  |  LR: 1.00e-04  (19.0s)
Epoch [  9/20]  Train Loss: 0.3465  Bal.Acc: 88.4%  F1: 0.8742  |  Val Loss: 0.7153  Bal.Acc: 72.6%  F1: 0.7316  |  LR: 1.00e-04  (16.5s)
Epoch [ 10/20]  Train Loss: 0.3087  Bal.Acc: 89.0%  F1: 0.8847  |  Val Loss: 0.6254  Bal.Acc: 76.0%  F1: 0.7518  |  LR: 1.00e-04  (15.3s)
 Best checkpoint saved (val_loss: 0.6254)
Epoch [ 11/20]  Train Loss: 0.2564  Bal.Acc: 91.7%  F1: 0.9107  |  Val Loss: 0.7425  Bal.Acc: 72.2%  F1: 0.7287  |  LR: 1.00e-04  (18.7s)
Epoch [ 12/20]  Train Loss: 0.2423  Bal.Acc: 92.3%  F1: 0.9185  |  Val Loss: 0.7257  Bal.Acc: 74.6%  F1: 0.7476  |  LR: 1.00e-04  (15.9s)
Epoch [ 13/20]  Train Loss: 0.2641  Bal.Acc: 90.7%  F1: 0.9064  |  Val Loss: 0.6695  Bal.Acc: 77.5%  F1: 0.7628  |  LR: 1.00e-04  (15.8s)
Epoch [ 14/20]  Train Loss: 0.2004  Bal.Acc: 94.0%  F1: 0.9322  |  Val Loss: 0.7027  Bal.Acc: 75.2%  F1: 0.7582  |  LR: 5.00e-05  (15.9s)
Epoch [ 15/20]  Train Loss: 0.1933  Bal.Acc: 93.8%  F1: 0.9369  |  Val Loss: 0.6855  Bal.Acc: 78.9%  F1: 0.7892  |  LR: 5.00e-05  (16.0s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.6254

 Training finished. Checkpoint: checkpoints/densenet121_fold3_best.pt
Log CSV: results/densenet121_fold3_training_log.csv
Best weights loaded from  10

Model evaluation: densenet121_fold3
----------------------------------------
  Balanced Accuracy:       76.03%
  F1 (macro):              0.7518
  Quadratic Cohen's Kappa: 0.8760
  ECE:                     0.0595
  Brier Score (mean):      0.0655

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.78      0.91      0.84        88
    Doubtful       0.73      0.60      0.66        81
        Mild       0.70      0.54      0.61        39
    Moderate       0.79      0.89      0.84        37
      Severe       0.77      0.86      0.81        35

    accuracy                           0.76       280
   macro avg       0.75      0.76      0.75       280
weighted avg       0.76      0.76      0.75       280

  Metrics saved to: results/densenet121_fold3_metrics.json
  Probability saved to: results/densenet121_fold3_test_probs.npz

--- densenet121 | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: densenet121

============================================================
TRAINING: densenet121_fold4
============================================================
Epoch [  1/20]  Train Loss: 1.4893  Bal.Acc: 34.8%  F1: 0.3178  |  Val Loss: 1.3859  Bal.Acc: 41.3%  F1: 0.4009  |  LR: 1.00e-04  (15.4s)
 Best checkpoint saved (val_loss: 1.3859)
Epoch [  2/20]  Train Loss: 1.1670  Bal.Acc: 58.5%  F1: 0.5452  |  Val Loss: 1.1818  Bal.Acc: 52.7%  F1: 0.5395  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 1.1818)
Epoch [  3/20]  Train Loss: 0.9405  Bal.Acc: 67.5%  F1: 0.6530  |  Val Loss: 0.9995  Bal.Acc: 67.3%  F1: 0.6798  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 0.9995)
Epoch [  4/20]  Train Loss: 0.7683  Bal.Acc: 73.3%  F1: 0.7240  |  Val Loss: 0.8632  Bal.Acc: 68.5%  F1: 0.6941  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.8632)
Epoch [  5/20]  Train Loss: 0.6363  Bal.Acc: 77.6%  F1: 0.7733  |  Val Loss: 0.8097  Bal.Acc: 68.5%  F1: 0.6915  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 0.8097)
Epoch [  6/20]  Train Loss: 0.5577  Bal.Acc: 80.5%  F1: 0.7978  |  Val Loss: 0.7117  Bal.Acc: 75.7%  F1: 0.7578  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 0.7117)
Epoch [  7/20]  Train Loss: 0.4607  Bal.Acc: 84.2%  F1: 0.8352  |  Val Loss: 0.6042  Bal.Acc: 78.3%  F1: 0.7779  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.6042)
Epoch [  8/20]  Train Loss: 0.4142  Bal.Acc: 85.3%  F1: 0.8485  |  Val Loss: 0.6637  Bal.Acc: 76.6%  F1: 0.7730  |  LR: 1.00e-04  (18.5s)
Epoch [  9/20]  Train Loss: 0.3729  Bal.Acc: 86.4%  F1: 0.8631  |  Val Loss: 0.5785  Bal.Acc: 79.5%  F1: 0.7876  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 0.5785)
Epoch [ 10/20]  Train Loss: 0.3118  Bal.Acc: 90.0%  F1: 0.8935  |  Val Loss: 0.5824  Bal.Acc: 79.7%  F1: 0.8082  |  LR: 1.00e-04  (18.2s)
Epoch [ 11/20]  Train Loss: 0.3036  Bal.Acc: 89.4%  F1: 0.8879  |  Val Loss: 0.5766  Bal.Acc: 80.6%  F1: 0.8124  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 0.5766)
Epoch [ 12/20]  Train Loss: 0.2886  Bal.Acc: 89.4%  F1: 0.8914  |  Val Loss: 0.6224  Bal.Acc: 78.3%  F1: 0.7627  |  LR: 1.00e-04  (17.7s)
Epoch [ 13/20]  Train Loss: 0.2650  Bal.Acc: 91.3%  F1: 0.9105  |  Val Loss: 0.5258  Bal.Acc: 82.0%  F1: 0.8177  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 0.5258)
Epoch [ 14/20]  Train Loss: 0.2170  Bal.Acc: 93.1%  F1: 0.9277  |  Val Loss: 0.5910  Bal.Acc: 81.0%  F1: 0.8231  |  LR: 1.00e-04  (19.0s)
Epoch [ 15/20]  Train Loss: 0.1991  Bal.Acc: 93.8%  F1: 0.9355  |  Val Loss: 0.6073  Bal.Acc: 79.1%  F1: 0.7791  |  LR: 1.00e-04  (16.0s)
Epoch [ 16/20]  Train Loss: 0.1724  Bal.Acc: 94.6%  F1: 0.9407  |  Val Loss: 0.5250  Bal.Acc: 84.7%  F1: 0.8484  |  LR: 1.00e-04  (15.6s)
 Best checkpoint saved (val_loss: 0.5250)
Epoch [ 17/20]  Train Loss: 0.1638  Bal.Acc: 94.5%  F1: 0.9441  |  Val Loss: 0.5884  Bal.Acc: 81.5%  F1: 0.8162  |  LR: 1.00e-04  (18.0s)
Epoch [ 18/20]  Train Loss: 0.1547  Bal.Acc: 95.4%  F1: 0.9522  |  Val Loss: 0.6154  Bal.Acc: 80.8%  F1: 0.8165  |  LR: 1.00e-04  (15.8s)
Epoch [ 19/20]  Train Loss: 0.1431  Bal.Acc: 95.8%  F1: 0.9540  |  Val Loss: 0.5557  Bal.Acc: 83.3%  F1: 0.8330  |  LR: 1.00e-04  (15.9s)
Epoch [ 20/20]  Train Loss: 0.1313  Bal.Acc: 95.5%  F1: 0.9520  |  Val Loss: 0.5933  Bal.Acc: 83.4%  F1: 0.8442  |  LR: 5.00e-05  (16.4s)

 Training finished. Checkpoint: checkpoints/densenet121_fold4_best.pt
Log CSV: results/densenet121_fold4_training_log.csv
Best weights loaded from  16

Model evaluation: densenet121_fold4
----------------------------------------
  Balanced Accuracy:       84.68%
  F1 (macro):              0.8484
  Quadratic Cohen's Kappa: 0.9386
  ECE:                     0.0472
  Brier Score (mean):      0.0470

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.92      0.90      0.91        88
    Doubtful       0.81      0.84      0.82        81
        Mild       0.71      0.69      0.70        39
    Moderate       0.85      0.92      0.88        37
      Severe       0.97      0.89      0.93        35

    accuracy                           0.85       280
   macro avg       0.85      0.85      0.85       280
weighted avg       0.86      0.85      0.85       280

  Metrics saved to: results/densenet121_fold4_metrics.json
  Probability saved to: results/densenet121_fold4_test_probs.npz

--- densenet121 | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 5):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: densenet121

============================================================
TRAINING: densenet121_fold5
============================================================
Epoch [  1/20]  Train Loss: 1.4938  Bal.Acc: 33.7%  F1: 0.3009  |  Val Loss: 1.4719  Bal.Acc: 37.4%  F1: 0.3498  |  LR: 1.00e-04  (15.6s)
 Best checkpoint saved (val_loss: 1.4719)
Epoch [  2/20]  Train Loss: 1.2169  Bal.Acc: 53.5%  F1: 0.4932  |  Val Loss: 1.2379  Bal.Acc: 49.9%  F1: 0.4677  |  LR: 1.00e-04  (15.9s)
 Best checkpoint saved (val_loss: 1.2379)
Epoch [  3/20]  Train Loss: 0.9993  Bal.Acc: 64.6%  F1: 0.6228  |  Val Loss: 1.1569  Bal.Acc: 57.0%  F1: 0.5726  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 1.1569)
Epoch [  4/20]  Train Loss: 0.8320  Bal.Acc: 69.9%  F1: 0.6798  |  Val Loss: 0.9395  Bal.Acc: 63.8%  F1: 0.6473  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 0.9395)
Epoch [  5/20]  Train Loss: 0.6839  Bal.Acc: 76.0%  F1: 0.7514  |  Val Loss: 0.7083  Bal.Acc: 77.3%  F1: 0.7654  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.7083)
Epoch [  6/20]  Train Loss: 0.5930  Bal.Acc: 78.4%  F1: 0.7801  |  Val Loss: 0.7976  Bal.Acc: 65.5%  F1: 0.6577  |  LR: 1.00e-04  (17.9s)
Epoch [  7/20]  Train Loss: 0.5025  Bal.Acc: 83.1%  F1: 0.8234  |  Val Loss: 0.6130  Bal.Acc: 76.0%  F1: 0.7411  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 0.6130)
Epoch [  8/20]  Train Loss: 0.4370  Bal.Acc: 85.8%  F1: 0.8524  |  Val Loss: 0.5782  Bal.Acc: 79.0%  F1: 0.7887  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 0.5782)
Epoch [  9/20]  Train Loss: 0.3800  Bal.Acc: 86.5%  F1: 0.8570  |  Val Loss: 0.5444  Bal.Acc: 79.0%  F1: 0.7831  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 0.5444)
Epoch [ 10/20]  Train Loss: 0.3258  Bal.Acc: 89.2%  F1: 0.8861  |  Val Loss: 0.5298  Bal.Acc: 81.3%  F1: 0.8104  |  LR: 1.00e-04  (19.0s)
 Best checkpoint saved (val_loss: 0.5298)
Epoch [ 11/20]  Train Loss: 0.2894  Bal.Acc: 90.5%  F1: 0.8975  |  Val Loss: 0.5420  Bal.Acc: 83.1%  F1: 0.8355  |  LR: 1.00e-04  (17.8s)
Epoch [ 12/20]  Train Loss: 0.2774  Bal.Acc: 89.8%  F1: 0.8979  |  Val Loss: 0.5361  Bal.Acc: 82.0%  F1: 0.8175  |  LR: 1.00e-04  (16.2s)
Epoch [ 13/20]  Train Loss: 0.2495  Bal.Acc: 92.4%  F1: 0.9147  |  Val Loss: 0.5005  Bal.Acc: 81.5%  F1: 0.8138  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 0.5005)
Epoch [ 14/20]  Train Loss: 0.2143  Bal.Acc: 93.0%  F1: 0.9247  |  Val Loss: 0.4983  Bal.Acc: 83.4%  F1: 0.8268  |  LR: 1.00e-04  (18.7s)
 Best checkpoint saved (val_loss: 0.4983)
Epoch [ 15/20]  Train Loss: 0.1797  Bal.Acc: 95.0%  F1: 0.9460  |  Val Loss: 0.5070  Bal.Acc: 83.3%  F1: 0.8410  |  LR: 1.00e-04  (18.0s)
Epoch [ 16/20]  Train Loss: 0.1782  Bal.Acc: 94.2%  F1: 0.9378  |  Val Loss: 0.5466  Bal.Acc: 82.5%  F1: 0.8212  |  LR: 1.00e-04  (15.2s)
Epoch [ 17/20]  Train Loss: 0.1761  Bal.Acc: 94.0%  F1: 0.9377  |  Val Loss: 0.4786  Bal.Acc: 84.4%  F1: 0.8433  |  LR: 1.00e-04  (15.5s)
 Best checkpoint saved (val_loss: 0.4786)
Epoch [ 18/20]  Train Loss: 0.1505  Bal.Acc: 95.7%  F1: 0.9549  |  Val Loss: 0.5248  Bal.Acc: 81.9%  F1: 0.8096  |  LR: 1.00e-04  (17.9s)
Epoch [ 19/20]  Train Loss: 0.1340  Bal.Acc: 96.0%  F1: 0.9558  |  Val Loss: 0.5102  Bal.Acc: 84.9%  F1: 0.8478  |  LR: 1.00e-04  (16.2s)
Epoch [ 20/20]  Train Loss: 0.1599  Bal.Acc: 95.0%  F1: 0.9472  |  Val Loss: 0.5505  Bal.Acc: 83.4%  F1: 0.8363  |  LR: 1.00e-04  (16.3s)

 Training finished. Checkpoint: checkpoints/densenet121_fold5_best.pt
Log CSV: results/densenet121_fold5_training_log.csv
Best weights loaded from  17

Model evaluation: densenet121_fold5
----------------------------------------
  Balanced Accuracy:       84.36%
  F1 (macro):              0.8433
  Quadratic Cohen's Kappa: 0.9434
  ECE:                     0.0512
  Brier Score (mean):      0.0467

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.90      0.88        87
    Doubtful       0.79      0.79      0.79        81
        Mild       0.78      0.64      0.70        39
    Moderate       0.88      0.95      0.91        38
      Severe       0.92      0.94      0.93        35

    accuracy                           0.84       280
   macro avg       0.85      0.84      0.84       280
weighted avg       0.84      0.84      0.84       280

  Metrics saved to: results/densenet121_fold5_metrics.json
  Probability saved to: results/densenet121_fold5_test_probs.npz

 FINISHED: densenet121. Average kappa out of 5 folds: 0.9246 ±0.0246

================================================================================
MODEL TRAINING START: mobilenetv3_large
================================================================================

--- mobilenetv3_large | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: mobilenetv3_large
model.safetensors: 100% 22.1M/22.1M [00:01<00:00, 12.1MB/s]

============================================================
TRAINING: mobilenetv3_large_fold1
============================================================
Epoch [  1/20]  Train Loss: 2.2822  Bal.Acc: 33.7%  F1: 0.3219  |  Val Loss: 2.5673  Bal.Acc: 30.1%  F1: 0.2452  |  LR: 1.00e-04  (12.6s)
 Best checkpoint saved (val_loss: 2.5673)
Epoch [  2/20]  Train Loss: 1.3036  Bal.Acc: 54.3%  F1: 0.5236  |  Val Loss: 2.0381  Bal.Acc: 40.0%  F1: 0.3781  |  LR: 1.00e-04  (12.4s)
 Best checkpoint saved (val_loss: 2.0381)
Epoch [  3/20]  Train Loss: 1.0026  Bal.Acc: 63.0%  F1: 0.6173  |  Val Loss: 1.7229  Bal.Acc: 49.9%  F1: 0.4887  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 1.7229)
Epoch [  4/20]  Train Loss: 0.8727  Bal.Acc: 67.0%  F1: 0.6576  |  Val Loss: 0.9884  Bal.Acc: 64.1%  F1: 0.6314  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 0.9884)
Epoch [  5/20]  Train Loss: 0.7276  Bal.Acc: 72.1%  F1: 0.7133  |  Val Loss: 0.8548  Bal.Acc: 67.2%  F1: 0.6689  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 0.8548)
Epoch [  6/20]  Train Loss: 0.6117  Bal.Acc: 76.0%  F1: 0.7484  |  Val Loss: 0.8128  Bal.Acc: 69.6%  F1: 0.6960  |  LR: 1.00e-04  (14.4s)
 Best checkpoint saved (val_loss: 0.8128)
Epoch [  7/20]  Train Loss: 0.5974  Bal.Acc: 77.5%  F1: 0.7641  |  Val Loss: 0.7377  Bal.Acc: 71.3%  F1: 0.7097  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 0.7377)
Epoch [  8/20]  Train Loss: 0.4947  Bal.Acc: 81.7%  F1: 0.8064  |  Val Loss: 0.7256  Bal.Acc: 72.7%  F1: 0.7315  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 0.7256)
Epoch [  9/20]  Train Loss: 0.4890  Bal.Acc: 80.6%  F1: 0.8005  |  Val Loss: 0.7072  Bal.Acc: 74.2%  F1: 0.7457  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 0.7072)
Epoch [ 10/20]  Train Loss: 0.4195  Bal.Acc: 84.7%  F1: 0.8397  |  Val Loss: 0.8258  Bal.Acc: 73.9%  F1: 0.7315  |  LR: 1.00e-04  (15.0s)
Epoch [ 11/20]  Train Loss: 0.4385  Bal.Acc: 83.5%  F1: 0.8283  |  Val Loss: 0.7320  Bal.Acc: 75.1%  F1: 0.7388  |  LR: 1.00e-04  (15.8s)
Epoch [ 12/20]  Train Loss: 0.3433  Bal.Acc: 87.3%  F1: 0.8705  |  Val Loss: 0.6143  Bal.Acc: 77.7%  F1: 0.7747  |  LR: 1.00e-04  (13.0s)
 Best checkpoint saved (val_loss: 0.6143)
Epoch [ 13/20]  Train Loss: 0.3238  Bal.Acc: 86.9%  F1: 0.8610  |  Val Loss: 0.6306  Bal.Acc: 77.6%  F1: 0.7757  |  LR: 1.00e-04  (14.5s)
Epoch [ 14/20]  Train Loss: 0.2815  Bal.Acc: 89.0%  F1: 0.8803  |  Val Loss: 0.6278  Bal.Acc: 78.9%  F1: 0.7822  |  LR: 1.00e-04  (13.1s)
Epoch [ 15/20]  Train Loss: 0.3133  Bal.Acc: 88.8%  F1: 0.8857  |  Val Loss: 0.6244  Bal.Acc: 80.2%  F1: 0.7961  |  LR: 1.00e-04  (12.6s)
Epoch [ 16/20]  Train Loss: 0.2663  Bal.Acc: 90.1%  F1: 0.8944  |  Val Loss: 0.6485  Bal.Acc: 76.9%  F1: 0.7651  |  LR: 5.00e-05  (12.7s)
Epoch [ 17/20]  Train Loss: 0.2308  Bal.Acc: 91.4%  F1: 0.9100  |  Val Loss: 0.6675  Bal.Acc: 79.5%  F1: 0.7945  |  LR: 5.00e-05  (12.3s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.6143

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold1_best.pt
Log CSV: results/mobilenetv3_large_fold1_training_log.csv
Best weights loaded from  12

Model evaluation: mobilenetv3_large_fold1
----------------------------------------
  Balanced Accuracy:       77.67%
  F1 (macro):              0.7747
  Quadratic Cohen's Kappa: 0.8852
  ECE:                     0.0868
  Brier Score (mean):      0.0655

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.87      0.87        87
    Doubtful       0.79      0.74      0.76        81
        Mild       0.56      0.68      0.61        40
    Moderate       0.85      0.74      0.79        38
      Severe       0.81      0.86      0.83        35

    accuracy                           0.79       281
   macro avg       0.78      0.78      0.77       281
weighted avg       0.79      0.79      0.79       281

  Metrics saved to: results/mobilenetv3_large_fold1_metrics.json
  Probability saved to: results/mobilenetv3_large_fold1_test_probs.npz

--- mobilenetv3_large | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: mobilenetv3_large

============================================================
TRAINING: mobilenetv3_large_fold2
============================================================
Epoch [  1/20]  Train Loss: 2.2789  Bal.Acc: 32.1%  F1: 0.3054  |  Val Loss: 2.5582  Bal.Acc: 30.3%  F1: 0.3152  |  LR: 1.00e-04  (12.5s)
 Best checkpoint saved (val_loss: 2.5582)
Epoch [  2/20]  Train Loss: 1.3834  Bal.Acc: 55.0%  F1: 0.5351  |  Val Loss: 1.9563  Bal.Acc: 39.5%  F1: 0.4003  |  LR: 1.00e-04  (12.5s)
 Best checkpoint saved (val_loss: 1.9563)
Epoch [  3/20]  Train Loss: 1.0485  Bal.Acc: 62.7%  F1: 0.6157  |  Val Loss: 1.5859  Bal.Acc: 51.1%  F1: 0.5159  |  LR: 1.00e-04  (14.3s)
 Best checkpoint saved (val_loss: 1.5859)
Epoch [  4/20]  Train Loss: 0.8921  Bal.Acc: 67.8%  F1: 0.6638  |  Val Loss: 1.1046  Bal.Acc: 63.1%  F1: 0.6182  |  LR: 1.00e-04  (14.3s)
 Best checkpoint saved (val_loss: 1.1046)
Epoch [  5/20]  Train Loss: 0.7548  Bal.Acc: 71.4%  F1: 0.7000  |  Val Loss: 0.9079  Bal.Acc: 67.0%  F1: 0.6689  |  LR: 1.00e-04  (14.2s)
 Best checkpoint saved (val_loss: 0.9079)
Epoch [  6/20]  Train Loss: 0.6695  Bal.Acc: 74.3%  F1: 0.7372  |  Val Loss: 0.8030  Bal.Acc: 69.1%  F1: 0.6940  |  LR: 1.00e-04  (14.2s)
 Best checkpoint saved (val_loss: 0.8030)
Epoch [  7/20]  Train Loss: 0.6094  Bal.Acc: 77.4%  F1: 0.7604  |  Val Loss: 0.7634  Bal.Acc: 70.1%  F1: 0.7012  |  LR: 1.00e-04  (14.4s)
 Best checkpoint saved (val_loss: 0.7634)
Epoch [  8/20]  Train Loss: 0.5312  Bal.Acc: 80.7%  F1: 0.7957  |  Val Loss: 0.7075  Bal.Acc: 72.6%  F1: 0.7291  |  LR: 1.00e-04  (14.4s)
 Best checkpoint saved (val_loss: 0.7075)
Epoch [  9/20]  Train Loss: 0.4754  Bal.Acc: 81.6%  F1: 0.8098  |  Val Loss: 0.6640  Bal.Acc: 73.7%  F1: 0.7380  |  LR: 1.00e-04  (14.6s)
 Best checkpoint saved (val_loss: 0.6640)
Epoch [ 10/20]  Train Loss: 0.4529  Bal.Acc: 81.9%  F1: 0.8081  |  Val Loss: 0.6186  Bal.Acc: 76.9%  F1: 0.7690  |  LR: 1.00e-04  (14.5s)
 Best checkpoint saved (val_loss: 0.6186)
Epoch [ 11/20]  Train Loss: 0.4184  Bal.Acc: 83.6%  F1: 0.8295  |  Val Loss: 0.5991  Bal.Acc: 76.4%  F1: 0.7574  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 0.5991)
Epoch [ 12/20]  Train Loss: 0.3521  Bal.Acc: 86.1%  F1: 0.8549  |  Val Loss: 0.6216  Bal.Acc: 77.5%  F1: 0.7653  |  LR: 1.00e-04  (16.1s)
Epoch [ 13/20]  Train Loss: 0.3341  Bal.Acc: 87.9%  F1: 0.8748  |  Val Loss: 0.6275  Bal.Acc: 77.7%  F1: 0.7594  |  LR: 1.00e-04  (13.3s)
Epoch [ 14/20]  Train Loss: 0.3199  Bal.Acc: 88.3%  F1: 0.8762  |  Val Loss: 0.5461  Bal.Acc: 79.4%  F1: 0.7992  |  LR: 1.00e-04  (12.6s)
 Best checkpoint saved (val_loss: 0.5461)
Epoch [ 15/20]  Train Loss: 0.2530  Bal.Acc: 90.9%  F1: 0.9039  |  Val Loss: 0.5290  Bal.Acc: 81.9%  F1: 0.8194  |  LR: 1.00e-04  (14.5s)
 Best checkpoint saved (val_loss: 0.5290)
Epoch [ 16/20]  Train Loss: 0.2812  Bal.Acc: 89.7%  F1: 0.8916  |  Val Loss: 0.5559  Bal.Acc: 79.5%  F1: 0.7889  |  LR: 1.00e-04  (16.3s)
Epoch [ 17/20]  Train Loss: 0.2519  Bal.Acc: 90.3%  F1: 0.8987  |  Val Loss: 0.5082  Bal.Acc: 82.2%  F1: 0.8248  |  LR: 1.00e-04  (12.8s)
 Best checkpoint saved (val_loss: 0.5082)
Epoch [ 18/20]  Train Loss: 0.2477  Bal.Acc: 91.0%  F1: 0.9029  |  Val Loss: 0.5308  Bal.Acc: 80.2%  F1: 0.8112  |  LR: 1.00e-04  (14.4s)
Epoch [ 19/20]  Train Loss: 0.2132  Bal.Acc: 91.8%  F1: 0.9119  |  Val Loss: 0.5568  Bal.Acc: 78.8%  F1: 0.7974  |  LR: 1.00e-04  (12.9s)
Epoch [ 20/20]  Train Loss: 0.2116  Bal.Acc: 91.9%  F1: 0.9155  |  Val Loss: 0.5705  Bal.Acc: 79.3%  F1: 0.7940  |  LR: 1.00e-04  (11.8s)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold2_best.pt
Log CSV: results/mobilenetv3_large_fold2_training_log.csv
Best weights loaded from  17

Model evaluation: mobilenetv3_large_fold2
----------------------------------------
  Balanced Accuracy:       82.22%
  F1 (macro):              0.8248
  Quadratic Cohen's Kappa: 0.9146
  ECE:                     0.0654
  Brier Score (mean):      0.0522

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.84      0.93      0.88        87
    Doubtful       0.81      0.75      0.78        81
        Mild       0.77      0.68      0.72        40
    Moderate       0.85      0.89      0.87        38
      Severe       0.88      0.86      0.87        35

    accuracy                           0.83       281
   macro avg       0.83      0.82      0.82       281
weighted avg       0.83      0.83      0.83       281

  Metrics saved to: results/mobilenetv3_large_fold2_metrics.json
  Probability saved to: results/mobilenetv3_large_fold2_test_probs.npz

--- mobilenetv3_large | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: mobilenetv3_large

============================================================
TRAINING: mobilenetv3_large_fold3
============================================================
Epoch [  1/20]  Train Loss: 2.1763  Bal.Acc: 36.1%  F1: 0.3483  |  Val Loss: 2.6456  Bal.Acc: 29.8%  F1: 0.2919  |  LR: 1.00e-04  (12.3s)
 Best checkpoint saved (val_loss: 2.6456)
Epoch [  2/20]  Train Loss: 1.3398  Bal.Acc: 54.2%  F1: 0.5193  |  Val Loss: 1.9188  Bal.Acc: 39.5%  F1: 0.4151  |  LR: 1.00e-04  (12.4s)
 Best checkpoint saved (val_loss: 1.9188)
Epoch [  3/20]  Train Loss: 1.0136  Bal.Acc: 65.6%  F1: 0.6434  |  Val Loss: 1.4194  Bal.Acc: 51.6%  F1: 0.5253  |  LR: 1.00e-04  (12.0s)
 Best checkpoint saved (val_loss: 1.4194)
Epoch [  4/20]  Train Loss: 0.8504  Bal.Acc: 68.6%  F1: 0.6776  |  Val Loss: 1.1427  Bal.Acc: 62.3%  F1: 0.6245  |  LR: 1.00e-04  (12.9s)
 Best checkpoint saved (val_loss: 1.1427)
Epoch [  5/20]  Train Loss: 0.7035  Bal.Acc: 72.4%  F1: 0.7113  |  Val Loss: 1.0887  Bal.Acc: 67.0%  F1: 0.6760  |  LR: 1.00e-04  (13.6s)
 Best checkpoint saved (val_loss: 1.0887)
Epoch [  6/20]  Train Loss: 0.6017  Bal.Acc: 77.8%  F1: 0.7720  |  Val Loss: 1.0404  Bal.Acc: 68.0%  F1: 0.6606  |  LR: 1.00e-04  (13.4s)
 Best checkpoint saved (val_loss: 1.0404)
Epoch [  7/20]  Train Loss: 0.5298  Bal.Acc: 79.6%  F1: 0.7819  |  Val Loss: 1.0413  Bal.Acc: 70.0%  F1: 0.6811  |  LR: 1.00e-04  (13.2s)
Epoch [  8/20]  Train Loss: 0.4676  Bal.Acc: 83.7%  F1: 0.8280  |  Val Loss: 1.0435  Bal.Acc: 69.9%  F1: 0.7037  |  LR: 1.00e-04  (13.3s)
Epoch [  9/20]  Train Loss: 0.4408  Bal.Acc: 83.7%  F1: 0.8307  |  Val Loss: 0.9352  Bal.Acc: 72.9%  F1: 0.7300  |  LR: 1.00e-04  (12.8s)
 Best checkpoint saved (val_loss: 0.9352)
Epoch [ 10/20]  Train Loss: 0.4333  Bal.Acc: 83.0%  F1: 0.8209  |  Val Loss: 1.0004  Bal.Acc: 73.1%  F1: 0.7286  |  LR: 1.00e-04  (14.8s)
Epoch [ 11/20]  Train Loss: 0.4237  Bal.Acc: 85.0%  F1: 0.8428  |  Val Loss: 0.9293  Bal.Acc: 73.5%  F1: 0.7342  |  LR: 1.00e-04  (12.4s)
 Best checkpoint saved (val_loss: 0.9293)
Epoch [ 12/20]  Train Loss: 0.2995  Bal.Acc: 88.6%  F1: 0.8829  |  Val Loss: 0.9536  Bal.Acc: 72.6%  F1: 0.7286  |  LR: 1.00e-04  (14.7s)
Epoch [ 13/20]  Train Loss: 0.3043  Bal.Acc: 88.8%  F1: 0.8825  |  Val Loss: 0.9260  Bal.Acc: 75.5%  F1: 0.7591  |  LR: 1.00e-04  (12.6s)
 Best checkpoint saved (val_loss: 0.9260)
Epoch [ 14/20]  Train Loss: 0.2720  Bal.Acc: 90.9%  F1: 0.9025  |  Val Loss: 0.8893  Bal.Acc: 76.1%  F1: 0.7613  |  LR: 1.00e-04  (14.5s)
 Best checkpoint saved (val_loss: 0.8893)
Epoch [ 15/20]  Train Loss: 0.2488  Bal.Acc: 90.5%  F1: 0.8986  |  Val Loss: 0.9522  Bal.Acc: 76.9%  F1: 0.7715  |  LR: 1.00e-04  (14.2s)
Epoch [ 16/20]  Train Loss: 0.2684  Bal.Acc: 91.0%  F1: 0.9043  |  Val Loss: 0.9432  Bal.Acc: 76.2%  F1: 0.7657  |  LR: 1.00e-04  (11.8s)
Epoch [ 17/20]  Train Loss: 0.2243  Bal.Acc: 91.8%  F1: 0.9127  |  Val Loss: 0.8987  Bal.Acc: 77.8%  F1: 0.7824  |  LR: 1.00e-04  (12.3s)
Epoch [ 18/20]  Train Loss: 0.2358  Bal.Acc: 91.5%  F1: 0.9139  |  Val Loss: 0.8631  Bal.Acc: 76.3%  F1: 0.7617  |  LR: 1.00e-04  (12.9s)
 Best checkpoint saved (val_loss: 0.8631)
Epoch [ 19/20]  Train Loss: 0.2133  Bal.Acc: 91.8%  F1: 0.9139  |  Val Loss: 0.9501  Bal.Acc: 74.6%  F1: 0.7552  |  LR: 1.00e-04  (14.6s)
Epoch [ 20/20]  Train Loss: 0.2126  Bal.Acc: 92.3%  F1: 0.9176  |  Val Loss: 0.9012  Bal.Acc: 75.9%  F1: 0.7577  |  LR: 1.00e-04  (12.9s)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold3_best.pt
Log CSV: results/mobilenetv3_large_fold3_training_log.csv
Best weights loaded from  18

Model evaluation: mobilenetv3_large_fold3
----------------------------------------
  Balanced Accuracy:       76.30%
  F1 (macro):              0.7617
  Quadratic Cohen's Kappa: 0.8744
  ECE:                     0.1236
  Brier Score (mean):      0.0733

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.79      0.83      0.81        88
    Doubtful       0.69      0.63      0.66        81
        Mild       0.63      0.74      0.68        39
    Moderate       0.88      0.78      0.83        37
      Severe       0.83      0.83      0.83        35

    accuracy                           0.75       280
   macro avg       0.76      0.76      0.76       280
weighted avg       0.76      0.75      0.75       280

  Metrics saved to: results/mobilenetv3_large_fold3_metrics.json
  Probability saved to: results/mobilenetv3_large_fold3_test_probs.npz

--- mobilenetv3_large | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: mobilenetv3_large

============================================================
TRAINING: mobilenetv3_large_fold4
============================================================
Epoch [  1/20]  Train Loss: 2.2981  Bal.Acc: 30.0%  F1: 0.2883  |  Val Loss: 2.9626  Bal.Acc: 25.7%  F1: 0.2465  |  LR: 1.00e-04  (13.1s)
 Best checkpoint saved (val_loss: 2.9626)
Epoch [  2/20]  Train Loss: 1.3703  Bal.Acc: 52.9%  F1: 0.5084  |  Val Loss: 1.9782  Bal.Acc: 35.2%  F1: 0.3630  |  LR: 1.00e-04  (13.3s)
 Best checkpoint saved (val_loss: 1.9782)
Epoch [  3/20]  Train Loss: 1.0300  Bal.Acc: 61.2%  F1: 0.5937  |  Val Loss: 1.5421  Bal.Acc: 50.2%  F1: 0.5050  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 1.5421)
Epoch [  4/20]  Train Loss: 0.8687  Bal.Acc: 69.8%  F1: 0.6887  |  Val Loss: 1.1125  Bal.Acc: 59.7%  F1: 0.5734  |  LR: 1.00e-04  (14.8s)
 Best checkpoint saved (val_loss: 1.1125)
Epoch [  5/20]  Train Loss: 0.7623  Bal.Acc: 71.4%  F1: 0.7048  |  Val Loss: 1.1046  Bal.Acc: 63.4%  F1: 0.6224  |  LR: 1.00e-04  (14.8s)
 Best checkpoint saved (val_loss: 1.1046)
Epoch [  6/20]  Train Loss: 0.6346  Bal.Acc: 75.7%  F1: 0.7398  |  Val Loss: 1.2156  Bal.Acc: 57.3%  F1: 0.5793  |  LR: 1.00e-04  (15.0s)
Epoch [  7/20]  Train Loss: 0.5985  Bal.Acc: 77.5%  F1: 0.7679  |  Val Loss: 1.0391  Bal.Acc: 64.8%  F1: 0.6430  |  LR: 1.00e-04  (12.9s)
 Best checkpoint saved (val_loss: 1.0391)
Epoch [  8/20]  Train Loss: 0.5387  Bal.Acc: 79.5%  F1: 0.7859  |  Val Loss: 1.0059  Bal.Acc: 66.3%  F1: 0.6520  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 1.0059)
Epoch [  9/20]  Train Loss: 0.5182  Bal.Acc: 81.0%  F1: 0.8026  |  Val Loss: 0.9750  Bal.Acc: 68.0%  F1: 0.6767  |  LR: 1.00e-04  (14.7s)
 Best checkpoint saved (val_loss: 0.9750)
Epoch [ 10/20]  Train Loss: 0.4653  Bal.Acc: 81.1%  F1: 0.8062  |  Val Loss: 0.7977  Bal.Acc: 71.6%  F1: 0.7112  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 0.7977)
Epoch [ 11/20]  Train Loss: 0.4271  Bal.Acc: 83.5%  F1: 0.8216  |  Val Loss: 0.8486  Bal.Acc: 68.7%  F1: 0.6951  |  LR: 1.00e-04  (14.7s)
Epoch [ 12/20]  Train Loss: 0.3225  Bal.Acc: 86.6%  F1: 0.8628  |  Val Loss: 0.8075  Bal.Acc: 74.0%  F1: 0.7398  |  LR: 1.00e-04  (12.9s)
Epoch [ 13/20]  Train Loss: 0.3590  Bal.Acc: 86.5%  F1: 0.8607  |  Val Loss: 0.8218  Bal.Acc: 72.5%  F1: 0.7271  |  LR: 1.00e-04  (13.2s)
Epoch [ 14/20]  Train Loss: 0.3454  Bal.Acc: 87.2%  F1: 0.8618  |  Val Loss: 0.8097  Bal.Acc: 73.0%  F1: 0.7368  |  LR: 5.00e-05  (12.7s)
Epoch [ 15/20]  Train Loss: 0.2732  Bal.Acc: 89.6%  F1: 0.8955  |  Val Loss: 0.7769  Bal.Acc: 72.2%  F1: 0.7300  |  LR: 5.00e-05  (11.9s)
 Best checkpoint saved (val_loss: 0.7769)
Epoch [ 16/20]  Train Loss: 0.2400  Bal.Acc: 91.2%  F1: 0.9056  |  Val Loss: 0.8161  Bal.Acc: 73.1%  F1: 0.7425  |  LR: 5.00e-05  (15.2s)
Epoch [ 17/20]  Train Loss: 0.2430  Bal.Acc: 91.0%  F1: 0.9045  |  Val Loss: 0.8002  Bal.Acc: 71.2%  F1: 0.7230  |  LR: 5.00e-05  (12.3s)
Epoch [ 18/20]  Train Loss: 0.2394  Bal.Acc: 91.2%  F1: 0.9055  |  Val Loss: 0.8006  Bal.Acc: 70.4%  F1: 0.7136  |  LR: 5.00e-05  (12.9s)
Epoch [ 19/20]  Train Loss: 0.2286  Bal.Acc: 91.9%  F1: 0.9117  |  Val Loss: 0.7183  Bal.Acc: 74.8%  F1: 0.7501  |  LR: 5.00e-05  (12.9s)
 Best checkpoint saved (val_loss: 0.7183)
Epoch [ 20/20]  Train Loss: 0.2173  Bal.Acc: 92.3%  F1: 0.9195  |  Val Loss: 0.7621  Bal.Acc: 73.9%  F1: 0.7450  |  LR: 5.00e-05  (14.7s)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold4_best.pt
Log CSV: results/mobilenetv3_large_fold4_training_log.csv
Best weights loaded from  19

Model evaluation: mobilenetv3_large_fold4
----------------------------------------
  Balanced Accuracy:       74.78%
  F1 (macro):              0.7501
  Quadratic Cohen's Kappa: 0.8708
  ECE:                     0.1114
  Brier Score (mean):      0.0726

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.77      0.82      0.80        88
    Doubtful       0.73      0.64      0.68        81
        Mild       0.50      0.64      0.56        39
    Moderate       0.91      0.84      0.87        37
      Severe       0.88      0.80      0.84        35

    accuracy                           0.74       280
   macro avg       0.76      0.75      0.75       280
weighted avg       0.75      0.74      0.75       280

  Metrics saved to: results/mobilenetv3_large_fold4_metrics.json
  Probability saved to: results/mobilenetv3_large_fold4_test_probs.npz

--- mobilenetv3_large | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 5):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: mobilenetv3_large

============================================================
TRAINING: mobilenetv3_large_fold5
============================================================
Epoch [  1/20]  Train Loss: 2.2039  Bal.Acc: 34.9%  F1: 0.3342  |  Val Loss: 2.7683  Bal.Acc: 23.3%  F1: 0.2193  |  LR: 1.00e-04  (13.0s)
 Best checkpoint saved (val_loss: 2.7683)
Epoch [  2/20]  Train Loss: 1.3281  Bal.Acc: 54.9%  F1: 0.5343  |  Val Loss: 1.6614  Bal.Acc: 44.0%  F1: 0.4419  |  LR: 1.00e-04  (13.2s)
 Best checkpoint saved (val_loss: 1.6614)
Epoch [  3/20]  Train Loss: 1.0606  Bal.Acc: 62.5%  F1: 0.6026  |  Val Loss: 1.4776  Bal.Acc: 46.8%  F1: 0.4724  |  LR: 1.00e-04  (14.5s)
 Best checkpoint saved (val_loss: 1.4776)
Epoch [  4/20]  Train Loss: 0.9242  Bal.Acc: 67.4%  F1: 0.6617  |  Val Loss: 1.1946  Bal.Acc: 59.2%  F1: 0.5869  |  LR: 1.00e-04  (14.5s)
 Best checkpoint saved (val_loss: 1.1946)
Epoch [  5/20]  Train Loss: 0.7812  Bal.Acc: 71.2%  F1: 0.6999  |  Val Loss: 1.1752  Bal.Acc: 62.4%  F1: 0.6141  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 1.1752)
Epoch [  6/20]  Train Loss: 0.6676  Bal.Acc: 75.1%  F1: 0.7404  |  Val Loss: 0.9847  Bal.Acc: 68.4%  F1: 0.6646  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 0.9847)
Epoch [  7/20]  Train Loss: 0.6286  Bal.Acc: 75.8%  F1: 0.7466  |  Val Loss: 0.9447  Bal.Acc: 69.8%  F1: 0.6943  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 0.9447)
Epoch [  8/20]  Train Loss: 0.4917  Bal.Acc: 81.6%  F1: 0.8067  |  Val Loss: 0.9218  Bal.Acc: 71.9%  F1: 0.7166  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 0.9218)
Epoch [  9/20]  Train Loss: 0.4977  Bal.Acc: 80.6%  F1: 0.7991  |  Val Loss: 0.8488  Bal.Acc: 73.1%  F1: 0.7236  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 0.8488)
Epoch [ 10/20]  Train Loss: 0.4225  Bal.Acc: 84.4%  F1: 0.8370  |  Val Loss: 0.8766  Bal.Acc: 71.6%  F1: 0.7127  |  LR: 1.00e-04  (12.3s)
Epoch [ 11/20]  Train Loss: 0.3643  Bal.Acc: 86.9%  F1: 0.8562  |  Val Loss: 0.9385  Bal.Acc: 69.5%  F1: 0.6904  |  LR: 1.00e-04  (17.0s)
Epoch [ 12/20]  Train Loss: 0.3635  Bal.Acc: 87.0%  F1: 0.8650  |  Val Loss: 0.8950  Bal.Acc: 71.4%  F1: 0.7099  |  LR: 1.00e-04  (12.2s)
Epoch [ 13/20]  Train Loss: 0.3564  Bal.Acc: 87.6%  F1: 0.8760  |  Val Loss: 0.8478  Bal.Acc: 72.3%  F1: 0.7104  |  LR: 1.00e-04  (12.3s)
 Best checkpoint saved (val_loss: 0.8478)
Epoch [ 14/20]  Train Loss: 0.2704  Bal.Acc: 89.5%  F1: 0.8842  |  Val Loss: 0.8735  Bal.Acc: 73.3%  F1: 0.7420  |  LR: 1.00e-04  (14.5s)
Epoch [ 15/20]  Train Loss: 0.3209  Bal.Acc: 87.5%  F1: 0.8687  |  Val Loss: 0.8313  Bal.Acc: 76.3%  F1: 0.7631  |  LR: 1.00e-04  (13.3s)
 Best checkpoint saved (val_loss: 0.8313)
Epoch [ 16/20]  Train Loss: 0.2527  Bal.Acc: 91.3%  F1: 0.9061  |  Val Loss: 0.8195  Bal.Acc: 74.7%  F1: 0.7388  |  LR: 1.00e-04  (14.8s)
 Best checkpoint saved (val_loss: 0.8195)
Epoch [ 17/20]  Train Loss: 0.2332  Bal.Acc: 91.4%  F1: 0.9066  |  Val Loss: 0.8729  Bal.Acc: 74.7%  F1: 0.7486  |  LR: 1.00e-04  (16.7s)
Epoch [ 18/20]  Train Loss: 0.2240  Bal.Acc: 91.2%  F1: 0.9102  |  Val Loss: 0.8627  Bal.Acc: 72.9%  F1: 0.7166  |  LR: 1.00e-04  (12.9s)
Epoch [ 19/20]  Train Loss: 0.1830  Bal.Acc: 93.1%  F1: 0.9314  |  Val Loss: 0.8207  Bal.Acc: 76.0%  F1: 0.7570  |  LR: 1.00e-04  (12.8s)
Epoch [ 20/20]  Train Loss: 0.1976  Bal.Acc: 91.6%  F1: 0.9118  |  Val Loss: 0.7450  Bal.Acc: 76.7%  F1: 0.7615  |  LR: 1.00e-04  (12.9s)
 Best checkpoint saved (val_loss: 0.7450)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold5_best.pt
Log CSV: results/mobilenetv3_large_fold5_training_log.csv
Best weights loaded from  20

Model evaluation: mobilenetv3_large_fold5
----------------------------------------
  Balanced Accuracy:       76.71%
  F1 (macro):              0.7615
  Quadratic Cohen's Kappa: 0.8808
  ECE:                     0.1214
  Brier Score (mean):      0.0704

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.79      0.84      0.82        87
    Doubtful       0.72      0.63      0.67        81
        Mild       0.60      0.64      0.62        39
    Moderate       0.82      0.87      0.85        38
      Severe       0.86      0.86      0.86        35

    accuracy                           0.76       280
   macro avg       0.76      0.77      0.76       280
weighted avg       0.76      0.76      0.76       280

  Metrics saved to: results/mobilenetv3_large_fold5_metrics.json
  Probability saved to: results/mobilenetv3_large_fold5_test_probs.npz

 FINISHED: mobilenetv3_large. Average kappa out of 5 folds: 0.8852 ±0.0155

================================================================================
MODEL TRAINING START: convnext_tiny
================================================================================

--- convnext_tiny | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: convnext_tiny
model.safetensors: 100% 114M/114M [00:02<00:00, 47.4MB/s]

============================================================
TRAINING: convnext_tiny_fold1
============================================================
Epoch [  1/20]  Train Loss: 1.9012  Bal.Acc: 20.3%  F1: 0.1770  |  Val Loss: 1.6491  Bal.Acc: 28.3%  F1: 0.1180  |  LR: 1.00e-04  (35.2s)
 Best checkpoint saved (val_loss: 1.6491)
Epoch [  2/20]  Train Loss: 1.6071  Bal.Acc: 23.2%  F1: 0.2259  |  Val Loss: 1.5929  Bal.Acc: 24.3%  F1: 0.1578  |  LR: 1.00e-04  (22.5s)
 Best checkpoint saved (val_loss: 1.5929)
Epoch [  3/20]  Train Loss: 1.4952  Bal.Acc: 33.7%  F1: 0.2761  |  Val Loss: 1.4199  Bal.Acc: 34.1%  F1: 0.2778  |  LR: 1.00e-04  (28.8s)
 Best checkpoint saved (val_loss: 1.4199)
Epoch [  4/20]  Train Loss: 1.4769  Bal.Acc: 33.1%  F1: 0.2942  |  Val Loss: 1.2884  Bal.Acc: 50.0%  F1: 0.4471  |  LR: 1.00e-04  (35.2s)
 Best checkpoint saved (val_loss: 1.2884)
Epoch [  5/20]  Train Loss: 1.2518  Bal.Acc: 47.2%  F1: 0.4477  |  Val Loss: 0.9910  Bal.Acc: 60.0%  F1: 0.6160  |  LR: 1.00e-04  (35.3s)
 Best checkpoint saved (val_loss: 0.9910)
Epoch [  6/20]  Train Loss: 0.9964  Bal.Acc: 59.1%  F1: 0.5809  |  Val Loss: 1.0102  Bal.Acc: 59.0%  F1: 0.5407  |  LR: 1.00e-04  (34.0s)
Epoch [  7/20]  Train Loss: 0.7993  Bal.Acc: 67.0%  F1: 0.6582  |  Val Loss: 0.8152  Bal.Acc: 66.8%  F1: 0.6152  |  LR: 1.00e-04  (23.4s)
 Best checkpoint saved (val_loss: 0.8152)
Epoch [  8/20]  Train Loss: 0.6566  Bal.Acc: 72.8%  F1: 0.7180  |  Val Loss: 0.8089  Bal.Acc: 70.2%  F1: 0.6729  |  LR: 1.00e-04  (34.5s)
 Best checkpoint saved (val_loss: 0.8089)
Epoch [  9/20]  Train Loss: 0.6141  Bal.Acc: 75.3%  F1: 0.7421  |  Val Loss: 0.6271  Bal.Acc: 72.0%  F1: 0.6902  |  LR: 1.00e-04  (34.2s)
 Best checkpoint saved (val_loss: 0.6271)
Epoch [ 10/20]  Train Loss: 0.5617  Bal.Acc: 77.9%  F1: 0.7733  |  Val Loss: 0.6000  Bal.Acc: 73.1%  F1: 0.7187  |  LR: 1.00e-04  (33.9s)
 Best checkpoint saved (val_loss: 0.6000)
Epoch [ 11/20]  Train Loss: 0.5016  Bal.Acc: 79.0%  F1: 0.7809  |  Val Loss: 0.6486  Bal.Acc: 76.3%  F1: 0.7499  |  LR: 1.00e-04  (33.2s)
Epoch [ 12/20]  Train Loss: 0.4227  Bal.Acc: 84.2%  F1: 0.8326  |  Val Loss: 0.6792  Bal.Acc: 70.6%  F1: 0.6946  |  LR: 1.00e-04  (22.8s)
Epoch [ 13/20]  Train Loss: 0.4775  Bal.Acc: 80.8%  F1: 0.8022  |  Val Loss: 0.6333  Bal.Acc: 76.6%  F1: 0.7423  |  LR: 1.00e-04  (23.5s)
Epoch [ 14/20]  Train Loss: 0.3602  Bal.Acc: 86.5%  F1: 0.8544  |  Val Loss: 0.8180  Bal.Acc: 71.1%  F1: 0.7183  |  LR: 5.00e-05  (22.4s)
Epoch [ 15/20]  Train Loss: 0.3277  Bal.Acc: 87.4%  F1: 0.8729  |  Val Loss: 0.5305  Bal.Acc: 82.7%  F1: 0.8222  |  LR: 5.00e-05  (22.5s)
 Best checkpoint saved (val_loss: 0.5305)
Epoch [ 16/20]  Train Loss: 0.2403  Bal.Acc: 91.4%  F1: 0.9109  |  Val Loss: 0.4846  Bal.Acc: 83.8%  F1: 0.8323  |  LR: 5.00e-05  (27.7s)
 Best checkpoint saved (val_loss: 0.4846)
Epoch [ 17/20]  Train Loss: 0.1923  Bal.Acc: 93.0%  F1: 0.9287  |  Val Loss: 0.6807  Bal.Acc: 79.7%  F1: 0.7757  |  LR: 5.00e-05  (35.2s)
Epoch [ 18/20]  Train Loss: 0.2367  Bal.Acc: 90.5%  F1: 0.9000  |  Val Loss: 0.4749  Bal.Acc: 80.9%  F1: 0.8141  |  LR: 5.00e-05  (22.8s)
 Best checkpoint saved (val_loss: 0.4749)
Epoch [ 19/20]  Train Loss: 0.1548  Bal.Acc: 94.0%  F1: 0.9363  |  Val Loss: 0.5367  Bal.Acc: 82.8%  F1: 0.8145  |  LR: 5.00e-05  (35.5s)
Epoch [ 20/20]  Train Loss: 0.1418  Bal.Acc: 94.9%  F1: 0.9453  |  Val Loss: 0.4927  Bal.Acc: 85.6%  F1: 0.8509  |  LR: 5.00e-05  (23.4s)

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold1_best.pt
Log CSV: results/convnext_tiny_fold1_training_log.csv
Best weights loaded from  18

Model evaluation: convnext_tiny_fold1
----------------------------------------
  Balanced Accuracy:       80.91%
  F1 (macro):              0.8141
  Quadratic Cohen's Kappa: 0.9334
  ECE:                     0.0657
  Brier Score (mean):      0.0517

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.86      0.93      0.90        87
    Doubtful       0.79      0.74      0.76        81
        Mild       0.64      0.72      0.68        40
    Moderate       0.88      0.76      0.82        38
      Severe       0.94      0.89      0.91        35

    accuracy                           0.82       281
   macro avg       0.82      0.81      0.81       281
weighted avg       0.82      0.82      0.82       281

  Metrics saved to: results/convnext_tiny_fold1_metrics.json
  Probability saved to: results/convnext_tiny_fold1_test_probs.npz

--- convnext_tiny | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.495  (count = 150)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Buliding model: convnext_tiny

============================================================
TRAINING: convnext_tiny_fold2
============================================================
Epoch [  1/20]  Train Loss: 1.6381  Bal.Acc: 28.6%  F1: 0.2561  |  Val Loss: 1.5042  Bal.Acc: 38.8%  F1: 0.3286  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 1.5042)
Epoch [  2/20]  Train Loss: 1.3225  Bal.Acc: 41.5%  F1: 0.3885  |  Val Loss: 1.3306  Bal.Acc: 42.4%  F1: 0.3839  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 1.3306)
Epoch [  3/20]  Train Loss: 0.9863  Bal.Acc: 58.7%  F1: 0.5699  |  Val Loss: 1.1427  Bal.Acc: 52.5%  F1: 0.5323  |  LR: 1.00e-04  (29.2s)
 Best checkpoint saved (val_loss: 1.1427)
Epoch [  4/20]  Train Loss: 0.7280  Bal.Acc: 69.0%  F1: 0.6801  |  Val Loss: 0.9202  Bal.Acc: 59.0%  F1: 0.5798  |  LR: 1.00e-04  (35.8s)
 Best checkpoint saved (val_loss: 0.9202)
Epoch [  5/20]  Train Loss: 0.6882  Bal.Acc: 72.5%  F1: 0.7142  |  Val Loss: 0.5923  Bal.Acc: 77.4%  F1: 0.7662  |  LR: 1.00e-04  (34.5s)
 Best checkpoint saved (val_loss: 0.5923)
Epoch [  6/20]  Train Loss: 0.4677  Bal.Acc: 81.5%  F1: 0.8077  |  Val Loss: 0.7376  Bal.Acc: 67.3%  F1: 0.6775  |  LR: 1.00e-04  (34.0s)
Epoch [  7/20]  Train Loss: 0.4834  Bal.Acc: 80.2%  F1: 0.7951  |  Val Loss: 0.5972  Bal.Acc: 75.9%  F1: 0.7472  |  LR: 1.00e-04  (22.6s)
Epoch [  8/20]  Train Loss: 0.3872  Bal.Acc: 85.1%  F1: 0.8418  |  Val Loss: 0.6111  Bal.Acc: 76.9%  F1: 0.7451  |  LR: 1.00e-04  (22.9s)
Epoch [  9/20]  Train Loss: 0.3532  Bal.Acc: 85.5%  F1: 0.8490  |  Val Loss: 0.5296  Bal.Acc: 79.6%  F1: 0.7788  |  LR: 1.00e-04  (22.4s)
 Best checkpoint saved (val_loss: 0.5296)
Epoch [ 10/20]  Train Loss: 0.2980  Bal.Acc: 89.7%  F1: 0.8915  |  Val Loss: 0.4824  Bal.Acc: 82.0%  F1: 0.7978  |  LR: 1.00e-04  (29.5s)
 Best checkpoint saved (val_loss: 0.4824)
Epoch [ 11/20]  Train Loss: 0.3815  Bal.Acc: 84.3%  F1: 0.8365  |  Val Loss: 0.6707  Bal.Acc: 76.4%  F1: 0.7412  |  LR: 1.00e-04  (35.2s)
Epoch [ 12/20]  Train Loss: 0.3662  Bal.Acc: 86.4%  F1: 0.8579  |  Val Loss: 0.5171  Bal.Acc: 80.4%  F1: 0.7925  |  LR: 1.00e-04  (22.8s)
Epoch [ 13/20]  Train Loss: 0.2812  Bal.Acc: 89.9%  F1: 0.8956  |  Val Loss: 0.4051  Bal.Acc: 84.1%  F1: 0.8313  |  LR: 1.00e-04  (23.2s)
 Best checkpoint saved (val_loss: 0.4051)
Epoch [ 14/20]  Train Loss: 0.2018  Bal.Acc: 93.0%  F1: 0.9252  |  Val Loss: 0.4536  Bal.Acc: 84.5%  F1: 0.8460  |  LR: 1.00e-04  (28.3s)
Epoch [ 15/20]  Train Loss: 0.1592  Bal.Acc: 94.5%  F1: 0.9394  |  Val Loss: 0.4437  Bal.Acc: 85.4%  F1: 0.8399  |  LR: 1.00e-04  (23.6s)
Epoch [ 16/20]  Train Loss: 0.1483  Bal.Acc: 94.0%  F1: 0.9379  |  Val Loss: 0.6704  Bal.Acc: 83.3%  F1: 0.8358  |  LR: 1.00e-04  (23.2s)
Epoch [ 17/20]  Train Loss: 0.1418  Bal.Acc: 94.6%  F1: 0.9443  |  Val Loss: 0.4806  Bal.Acc: 82.6%  F1: 0.8138  |  LR: 5.00e-05  (22.3s)
Epoch [ 18/20]  Train Loss: 0.1281  Bal.Acc: 95.3%  F1: 0.9512  |  Val Loss: 0.4741  Bal.Acc: 85.7%  F1: 0.8532  |  LR: 5.00e-05  (22.4s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.4051

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold2_best.pt
Log CSV: results/convnext_tiny_fold2_training_log.csv
Best weights loaded from  13

Model evaluation: convnext_tiny_fold2
----------------------------------------
  Balanced Accuracy:       84.07%
  F1 (macro):              0.8313
  Quadratic Cohen's Kappa: 0.9488
  ECE:                     0.0608
  Brier Score (mean):      0.0442

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.91      0.95      0.93        87
    Doubtful       0.87      0.74      0.80        81
        Mild       0.67      0.75      0.71        40
    Moderate       0.86      0.82      0.84        38
      Severe       0.82      0.94      0.88        35

    accuracy                           0.84       281
   macro avg       0.83      0.84      0.83       281
weighted avg       0.85      0.84      0.84       281

  Metrics saved to: results/convnext_tiny_fold2_metrics.json
  Probability saved to: results/convnext_tiny_fold2_test_probs.npz

--- convnext_tiny | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: convnext_tiny

============================================================
TRAINING: convnext_tiny_fold3
============================================================
Epoch [  1/20]  Train Loss: 1.6912  Bal.Acc: 25.2%  F1: 0.2264  |  Val Loss: 1.4737  Bal.Acc: 37.4%  F1: 0.3159  |  LR: 1.00e-04  (28.8s)
 Best checkpoint saved (val_loss: 1.4737)
Epoch [  2/20]  Train Loss: 1.4104  Bal.Acc: 37.4%  F1: 0.3479  |  Val Loss: 1.8359  Bal.Acc: 27.9%  F1: 0.2359  |  LR: 1.00e-04  (22.7s)
Epoch [  3/20]  Train Loss: 1.1182  Bal.Acc: 52.3%  F1: 0.4949  |  Val Loss: 1.0505  Bal.Acc: 54.5%  F1: 0.5149  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 1.0505)
Epoch [  4/20]  Train Loss: 0.7118  Bal.Acc: 70.7%  F1: 0.6932  |  Val Loss: 0.7768  Bal.Acc: 72.1%  F1: 0.7008  |  LR: 1.00e-04  (29.0s)
 Best checkpoint saved (val_loss: 0.7768)
Epoch [  5/20]  Train Loss: 0.7225  Bal.Acc: 71.2%  F1: 0.7050  |  Val Loss: 0.9654  Bal.Acc: 64.4%  F1: 0.6027  |  LR: 1.00e-04  (34.8s)
Epoch [  6/20]  Train Loss: 0.6118  Bal.Acc: 76.2%  F1: 0.7539  |  Val Loss: 0.6763  Bal.Acc: 70.2%  F1: 0.6933  |  LR: 1.00e-04  (23.7s)
 Best checkpoint saved (val_loss: 0.6763)
Epoch [  7/20]  Train Loss: 0.4916  Bal.Acc: 80.1%  F1: 0.7937  |  Val Loss: 0.6636  Bal.Acc: 75.8%  F1: 0.7418  |  LR: 1.00e-04  (28.4s)
 Best checkpoint saved (val_loss: 0.6636)
Epoch [  8/20]  Train Loss: 0.4006  Bal.Acc: 83.5%  F1: 0.8285  |  Val Loss: 0.8954  Bal.Acc: 69.5%  F1: 0.6668  |  LR: 1.00e-04  (35.1s)
Epoch [  9/20]  Train Loss: 0.3964  Bal.Acc: 85.3%  F1: 0.8454  |  Val Loss: 0.7835  Bal.Acc: 72.4%  F1: 0.7279  |  LR: 1.00e-04  (22.9s)
Epoch [ 10/20]  Train Loss: 0.2930  Bal.Acc: 89.4%  F1: 0.8874  |  Val Loss: 0.6865  Bal.Acc: 75.2%  F1: 0.7531  |  LR: 1.00e-04  (23.2s)
Epoch [ 11/20]  Train Loss: 0.2641  Bal.Acc: 89.9%  F1: 0.8953  |  Val Loss: 0.7466  Bal.Acc: 75.4%  F1: 0.7473  |  LR: 5.00e-05  (22.6s)
Epoch [ 12/20]  Train Loss: 0.2068  Bal.Acc: 93.1%  F1: 0.9277  |  Val Loss: 0.7501  Bal.Acc: 75.4%  F1: 0.7560  |  LR: 5.00e-05  (23.0s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.6636

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold3_best.pt
Log CSV: results/convnext_tiny_fold3_training_log.csv
Best weights loaded from  7

Model evaluation: convnext_tiny_fold3
----------------------------------------
  Balanced Accuracy:       75.77%
  F1 (macro):              0.7418
  Quadratic Cohen's Kappa: 0.8931
  ECE:                     0.0646
  Brier Score (mean):      0.0777

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.86      0.78      0.82        88
    Doubtful       0.71      0.59      0.64        81
        Mild       0.50      0.79      0.61        39
    Moderate       0.84      0.70      0.76        37
      Severe       0.82      0.91      0.86        35

    accuracy                           0.74       280
   macro avg       0.75      0.76      0.74       280
weighted avg       0.76      0.74      0.74       280

  Metrics saved to: results/convnext_tiny_fold3_metrics.json
  Probability saved to: results/convnext_tiny_fold3_test_probs.npz

--- convnext_tiny | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.643  (count = 349)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.486  (count = 151)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: convnext_tiny

============================================================
TRAINING: convnext_tiny_fold4
============================================================
Epoch [  1/20]  Train Loss: 1.7439  Bal.Acc: 26.2%  F1: 0.2486  |  Val Loss: 1.5039  Bal.Acc: 33.3%  F1: 0.1615  |  LR: 1.00e-04  (23.2s)
 Best checkpoint saved (val_loss: 1.5039)
Epoch [  2/20]  Train Loss: 1.3850  Bal.Acc: 40.4%  F1: 0.3494  |  Val Loss: 1.3256  Bal.Acc: 41.4%  F1: 0.4086  |  LR: 1.00e-04  (23.6s)
 Best checkpoint saved (val_loss: 1.3256)
Epoch [  3/20]  Train Loss: 1.0706  Bal.Acc: 55.3%  F1: 0.5296  |  Val Loss: 1.0447  Bal.Acc: 55.1%  F1: 0.5080  |  LR: 1.00e-04  (28.0s)
 Best checkpoint saved (val_loss: 1.0447)
Epoch [  4/20]  Train Loss: 0.8263  Bal.Acc: 67.1%  F1: 0.6546  |  Val Loss: 1.1180  Bal.Acc: 53.2%  F1: 0.5333  |  LR: 1.00e-04  (36.0s)
Epoch [  5/20]  Train Loss: 0.7856  Bal.Acc: 67.0%  F1: 0.6581  |  Val Loss: 0.7560  Bal.Acc: 65.6%  F1: 0.6304  |  LR: 1.00e-04  (22.8s)
 Best checkpoint saved (val_loss: 0.7560)
Epoch [  6/20]  Train Loss: 0.6251  Bal.Acc: 76.1%  F1: 0.7494  |  Val Loss: 0.9248  Bal.Acc: 69.4%  F1: 0.7118  |  LR: 1.00e-04  (29.4s)
Epoch [  7/20]  Train Loss: 0.5556  Bal.Acc: 76.7%  F1: 0.7588  |  Val Loss: 0.6323  Bal.Acc: 77.2%  F1: 0.7828  |  LR: 1.00e-04  (23.1s)
 Best checkpoint saved (val_loss: 0.6323)
Epoch [  8/20]  Train Loss: 0.4674  Bal.Acc: 80.5%  F1: 0.7965  |  Val Loss: 0.6248  Bal.Acc: 73.2%  F1: 0.7210  |  LR: 1.00e-04  (29.0s)
 Best checkpoint saved (val_loss: 0.6248)
Epoch [  9/20]  Train Loss: 0.4074  Bal.Acc: 84.2%  F1: 0.8346  |  Val Loss: 0.6164  Bal.Acc: 78.6%  F1: 0.7975  |  LR: 1.00e-04  (34.4s)
 Best checkpoint saved (val_loss: 0.6164)
Epoch [ 10/20]  Train Loss: 0.3777  Bal.Acc: 84.2%  F1: 0.8355  |  Val Loss: 0.5844  Bal.Acc: 80.3%  F1: 0.7935  |  LR: 1.00e-04  (34.6s)
 Best checkpoint saved (val_loss: 0.5844)
Epoch [ 11/20]  Train Loss: 0.3158  Bal.Acc: 89.2%  F1: 0.8850  |  Val Loss: 0.7880  Bal.Acc: 78.5%  F1: 0.7879  |  LR: 1.00e-04  (33.7s)
Epoch [ 12/20]  Train Loss: 0.3826  Bal.Acc: 85.4%  F1: 0.8488  |  Val Loss: 0.6626  Bal.Acc: 74.2%  F1: 0.7165  |  LR: 1.00e-04  (23.5s)
Epoch [ 13/20]  Train Loss: 0.3375  Bal.Acc: 86.3%  F1: 0.8618  |  Val Loss: 0.8951  Bal.Acc: 73.7%  F1: 0.7144  |  LR: 1.00e-04  (23.5s)
Epoch [ 14/20]  Train Loss: 0.2667  Bal.Acc: 90.1%  F1: 0.8943  |  Val Loss: 0.7332  Bal.Acc: 78.0%  F1: 0.7966  |  LR: 5.00e-05  (23.0s)
Epoch [ 15/20]  Train Loss: 0.2060  Bal.Acc: 91.7%  F1: 0.9158  |  Val Loss: 0.7635  Bal.Acc: 74.3%  F1: 0.7554  |  LR: 5.00e-05  (23.1s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.5844

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold4_best.pt
Log CSV: results/convnext_tiny_fold4_training_log.csv
Best weights loaded from  10

Model evaluation: convnext_tiny_fold4
----------------------------------------
  Balanced Accuracy:       80.34%
  F1 (macro):              0.7935
  Quadratic Cohen's Kappa: 0.9197
  ECE:                     0.0730
  Brier Score (mean):      0.0607

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.92      0.92      0.92        88
    Doubtful       0.85      0.69      0.76        81
        Mild       0.55      0.74      0.63        39
    Moderate       0.76      0.92      0.83        37
      Severe       0.93      0.74      0.83        35

    accuracy                           0.81       280
   macro avg       0.80      0.80      0.79       280
weighted avg       0.83      0.81      0.81       280

  Metrics saved to: results/convnext_tiny_fold4_metrics.json
  Probability saved to: results/convnext_tiny_fold4_test_probs.npz

--- convnext_tiny | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 5):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Buliding model: convnext_tiny

============================================================
TRAINING: convnext_tiny_fold5
============================================================
Epoch [  1/20]  Train Loss: 1.6603  Bal.Acc: 23.3%  F1: 0.2095  |  Val Loss: 1.5868  Bal.Acc: 30.8%  F1: 0.2106  |  LR: 1.00e-04  (23.3s)
 Best checkpoint saved (val_loss: 1.5868)
Epoch [  2/20]  Train Loss: 1.4366  Bal.Acc: 37.9%  F1: 0.3403  |  Val Loss: 1.4845  Bal.Acc: 31.4%  F1: 0.2560  |  LR: 1.00e-04  (22.6s)
 Best checkpoint saved (val_loss: 1.4845)
Epoch [  3/20]  Train Loss: 1.1465  Bal.Acc: 51.7%  F1: 0.4878  |  Val Loss: 1.0506  Bal.Acc: 58.0%  F1: 0.5231  |  LR: 1.00e-04  (23.0s)
 Best checkpoint saved (val_loss: 1.0506)
Epoch [  4/20]  Train Loss: 0.8513  Bal.Acc: 65.8%  F1: 0.6403  |  Val Loss: 1.3115  Bal.Acc: 50.1%  F1: 0.4787  |  LR: 1.00e-04  (24.1s)
Epoch [  5/20]  Train Loss: 0.7897  Bal.Acc: 67.6%  F1: 0.6590  |  Val Loss: 0.8244  Bal.Acc: 66.6%  F1: 0.6221  |  LR: 1.00e-04  (24.1s)
 Best checkpoint saved (val_loss: 0.8244)
Epoch [  6/20]  Train Loss: 0.5999  Bal.Acc: 76.1%  F1: 0.7568  |  Val Loss: 0.7504  Bal.Acc: 73.0%  F1: 0.7131  |  LR: 1.00e-04  (29.3s)
 Best checkpoint saved (val_loss: 0.7504)
Epoch [  7/20]  Train Loss: 0.4946  Bal.Acc: 79.2%  F1: 0.7792  |  Val Loss: 0.7349  Bal.Acc: 78.0%  F1: 0.7809  |  LR: 1.00e-04  (35.5s)
 Best checkpoint saved (val_loss: 0.7349)
Epoch [  8/20]  Train Loss: 0.4451  Bal.Acc: 81.7%  F1: 0.8084  |  Val Loss: 0.7686  Bal.Acc: 70.5%  F1: 0.6876  |  LR: 1.00e-04  (34.0s)
Epoch [  9/20]  Train Loss: 0.4846  Bal.Acc: 80.0%  F1: 0.7873  |  Val Loss: 0.7078  Bal.Acc: 73.9%  F1: 0.7258  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 0.7078)
Epoch [ 10/20]  Train Loss: 0.3518  Bal.Acc: 87.2%  F1: 0.8632  |  Val Loss: 0.7621  Bal.Acc: 76.7%  F1: 0.7495  |  LR: 1.00e-04  (28.5s)
Epoch [ 11/20]  Train Loss: 0.3188  Bal.Acc: 87.8%  F1: 0.8730  |  Val Loss: 0.8524  Bal.Acc: 75.8%  F1: 0.7474  |  LR: 1.00e-04  (23.5s)
Epoch [ 12/20]  Train Loss: 0.3085  Bal.Acc: 88.2%  F1: 0.8756  |  Val Loss: 0.9232  Bal.Acc: 70.5%  F1: 0.7027  |  LR: 1.00e-04  (22.7s)
Epoch [ 13/20]  Train Loss: 0.2860  Bal.Acc: 88.8%  F1: 0.8832  |  Val Loss: 0.6195  Bal.Acc: 80.3%  F1: 0.7871  |  LR: 1.00e-04  (23.0s)
 Best checkpoint saved (val_loss: 0.6195)
Epoch [ 14/20]  Train Loss: 0.2161  Bal.Acc: 90.7%  F1: 0.9020  |  Val Loss: 0.8975  Bal.Acc: 76.8%  F1: 0.7592  |  LR: 1.00e-04  (27.4s)
Epoch [ 15/20]  Train Loss: 0.2082  Bal.Acc: 92.5%  F1: 0.9211  |  Val Loss: 0.8215  Bal.Acc: 82.3%  F1: 0.8281  |  LR: 1.00e-04  (23.6s)
Epoch [ 16/20]  Train Loss: 0.2098  Bal.Acc: 91.8%  F1: 0.9140  |  Val Loss: 0.8046  Bal.Acc: 77.9%  F1: 0.7678  |  LR: 1.00e-04  (23.1s)
Epoch [ 17/20]  Train Loss: 0.1474  Bal.Acc: 94.2%  F1: 0.9390  |  Val Loss: 0.9136  Bal.Acc: 79.9%  F1: 0.7821  |  LR: 5.00e-05  (23.4s)
Epoch [ 18/20]  Train Loss: 0.1315  Bal.Acc: 95.5%  F1: 0.9512  |  Val Loss: 0.8996  Bal.Acc: 80.6%  F1: 0.8074  |  LR: 5.00e-05  (22.4s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.6195

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold5_best.pt
Log CSV: results/convnext_tiny_fold5_training_log.csv
Best weights loaded from  13

Model evaluation: convnext_tiny_fold5
----------------------------------------
  Balanced Accuracy:       80.28%
  F1 (macro):              0.7871
  Quadratic Cohen's Kappa: 0.9247
  ECE:                     0.0765
  Brier Score (mean):      0.0582

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.90      0.91      0.90        87
    Doubtful       0.81      0.70      0.75        81
        Mild       0.57      0.54      0.55        39
    Moderate       0.76      0.92      0.83        38
      Severe       0.85      0.94      0.89        35

    accuracy                           0.80       280
   macro avg       0.78      0.80      0.79       280
weighted avg       0.80      0.80      0.80       280

  Metrics saved to: results/convnext_tiny_fold5_metrics.json
  Probability saved to: results/convnext_tiny_fold5_test_probs.npz

 FINISHED: convnext_tiny. Average kappa out of 5 folds: 0.9239 ±0.0183

========================================================================================================================
Single Fold Summary:
========================================================================================================================
Model                        Kappa   F1-Mac     ECE   Brier |     KL0     KL1     KL2     KL3     KL4
------------------------------------------------------------------------------------------------------------------------
convnext_tiny_fold2         0.9488   0.8313  0.0608  0.0442 |  0.9326  0.8000  0.7059  0.8378  0.8800
densenet121_fold5           0.9434   0.8433  0.0512  0.0467 |  0.8814  0.7901  0.7042  0.9114  0.9296
densenet121_fold4           0.9386   0.8484  0.0472  0.0470 |  0.9080  0.8242  0.7013  0.8831  0.9254
convnext_tiny_fold1         0.9334   0.8141  0.0657  0.0517 |  0.8950  0.7643  0.6824  0.8169  0.9118
densenet121_fold2           0.9333   0.8431  0.0206  0.0461 |  0.8966  0.7975  0.7619  0.8451  0.9143
densenet121_fold1           0.9316   0.8329  0.0631  0.0502 |  0.8810  0.8121  0.7089  0.8642  0.8986
convnext_tiny_fold5         0.9247   0.7871  0.0765  0.0582 |  0.9029  0.7550  0.5526  0.8333  0.8919
convnext_tiny_fold4         0.9197   0.7935  0.0730  0.0607 |  0.9205  0.7619  0.6304  0.8293  0.8254
mobilenetv3_large_fold2     0.9146   0.8248  0.0654  0.0522 |  0.8804  0.7821  0.7200  0.8718  0.8696
efficientnet_b3_fold2       0.9139   0.8313  0.0797  0.0528 |  0.8889  0.7750  0.7595  0.8718  0.8615
efficientnet_b3_fold4       0.9022   0.7623  0.1281  0.0742 |  0.8315  0.6879  0.5250  0.8718  0.8955
efficientnet_b3_fold1       0.9019   0.7688  0.1242  0.0687 |  0.7816  0.6753  0.6437  0.8267  0.9167
resnet50_fold1              0.9008   0.7220  0.0596  0.0728 |  0.8394  0.6803  0.4737  0.7397  0.8767
efficientnet_b3_fold3       0.8991   0.7732  0.1388  0.0718 |  0.8261  0.6968  0.6053  0.8649  0.8732
resnet50_fold2              0.8987   0.7320  0.0502  0.0712 |  0.8163  0.5630  0.6591  0.7606  0.8611
convnext_tiny_fold3         0.8931   0.7418  0.0646  0.0777 |  0.8214  0.6443  0.6139  0.7647  0.8649
mobilenetv3_large_fold1     0.8852   0.7747  0.0868  0.0655 |  0.8736  0.7643  0.6136  0.7887  0.8333
mobilenetv3_large_fold5     0.8808   0.7615  0.1214  0.0704 |  0.8156  0.6711  0.6173  0.8462  0.8571
resnet50_fold5              0.8784   0.7400  0.0393  0.0739 |  0.8743  0.6901  0.5301  0.8052  0.8000
densenet121_fold3           0.8760   0.7518  0.0595  0.0655 |  0.8421  0.6622  0.6087  0.8354  0.8108
mobilenetv3_large_fold3     0.8744   0.7617  0.1236  0.0733 |  0.8111  0.6581  0.6824  0.8286  0.8286
mobilenetv3_large_fold4     0.8708   0.7501  0.1114  0.0726 |  0.7956  0.6842  0.5618  0.8732  0.8358
resnet50_fold4              0.8674   0.7397  0.0489  0.0698 |  0.8521  0.7073  0.4878  0.8800  0.7714
resnet50_fold3              0.8537   0.6472  0.0756  0.0917 |  0.7830  0.5147  0.4595  0.7143  0.7647
efficientnet_b3_fold5       0.7338   0.5204  0.1862  0.1223 |  0.7135  0.4722  0.3030  0.5946  0.5185
========================================================================================================================
Sorted by Cohen's Kappa

===================================================================================================================
 CROSS-VALIDATION SUMMARY — AVERAGE OUT OF 5 FOLDs
===================================================================================================================
Model                         Kappa         F1-Mac |      KL0      KL1      KL2      KL3      KL4
-------------------------------------------------------------------------------------------------------------------
resnet50              0.8798 ±0.0181  0.7162 ±0.0351 |   0.8330   0.6311   0.5220   0.7800   0.8148
efficientnet_b3       0.8702 ±0.0684  0.7312 ±0.1083 |   0.8083   0.6614   0.5673   0.8060   0.8131
densenet121           0.9246 ±0.0246  0.8239 ±0.0364 |   0.8818   0.7772   0.6970   0.8678   0.8957
mobilenetv3_large     0.8852 ±0.0155  0.7746 ±0.0263 |   0.8353   0.7120   0.6390   0.8417   0.8449
convnext_tiny         0.9239 ±0.0183  0.7936 ±0.0302 |   0.8945   0.7451   0.6370   0.8164   0.8748
===================================================================================================================
Results saved to: results
