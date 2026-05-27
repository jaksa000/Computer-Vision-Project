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
  Together: 1650 images, 5-fold CV

============================================================
 DATA SPLIT
============================================================
  K-Fold CV data (85%): 1402 images
  Hold-out data  (15%): 248 images
  [OK] Split integrity verified — no overlap between CV (1402) and holdout (248) sets.
  Split manifest saved: results/split_manifest.json
  [OK] Holdout matches split manifest exactly (248 images).

================================================================================
MODEL TRAINING START: resnet50
================================================================================

--- resnet50 | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

============================================================
TRAINING: resnet50_fold1
============================================================
Epoch [  1/30]  Train Loss: 1.6010  Bal.Acc: 26.1%  F1: 0.1798  |  Val Loss: 1.5949  Bal.Acc: 28.5%  F1: 0.2312  |  LR: 1.00e-04  (60.8s)
 Best checkpoint saved (val_loss: 1.5949)
Epoch [  2/30]  Train Loss: 1.5826  Bal.Acc: 35.5%  F1: 0.3263  |  Val Loss: 1.5813  Bal.Acc: 40.7%  F1: 0.3648  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 1.5813)
Epoch [  3/30]  Train Loss: 1.5633  Bal.Acc: 39.8%  F1: 0.3494  |  Val Loss: 1.5514  Bal.Acc: 37.7%  F1: 0.3365  |  LR: 1.00e-04  (21.7s)
 Best checkpoint saved (val_loss: 1.5514)
Epoch [  4/30]  Train Loss: 1.5341  Bal.Acc: 46.3%  F1: 0.3988  |  Val Loss: 1.5127  Bal.Acc: 39.3%  F1: 0.3445  |  LR: 1.00e-04  (22.6s)
 Best checkpoint saved (val_loss: 1.5127)
Epoch [  5/30]  Train Loss: 1.4950  Bal.Acc: 45.9%  F1: 0.3876  |  Val Loss: 1.4770  Bal.Acc: 44.5%  F1: 0.4078  |  LR: 1.00e-04  (22.4s)
 Best checkpoint saved (val_loss: 1.4770)
Epoch [  6/30]  Train Loss: 1.4468  Bal.Acc: 48.1%  F1: 0.4046  |  Val Loss: 1.3962  Bal.Acc: 48.8%  F1: 0.4221  |  LR: 1.00e-04  (29.1s)
 Best checkpoint saved (val_loss: 1.3962)
Epoch [  7/30]  Train Loss: 1.3691  Bal.Acc: 52.7%  F1: 0.4420  |  Val Loss: 1.3884  Bal.Acc: 50.3%  F1: 0.4482  |  LR: 1.00e-04  (21.4s)
 Best checkpoint saved (val_loss: 1.3884)
Epoch [  8/30]  Train Loss: 1.2540  Bal.Acc: 55.3%  F1: 0.4646  |  Val Loss: 1.1926  Bal.Acc: 55.0%  F1: 0.5005  |  LR: 1.00e-04  (29.2s)
 Best checkpoint saved (val_loss: 1.1926)
Epoch [  9/30]  Train Loss: 1.1405  Bal.Acc: 57.3%  F1: 0.4967  |  Val Loss: 1.2505  Bal.Acc: 44.5%  F1: 0.3990  |  LR: 1.00e-04  (28.6s)
Epoch [ 10/30]  Train Loss: 1.0234  Bal.Acc: 61.0%  F1: 0.5514  |  Val Loss: 1.1208  Bal.Acc: 51.5%  F1: 0.5030  |  LR: 1.00e-04  (15.4s)
 Best checkpoint saved (val_loss: 1.1208)
Epoch [ 11/30]  Train Loss: 0.9396  Bal.Acc: 63.4%  F1: 0.5994  |  Val Loss: 0.9592  Bal.Acc: 67.3%  F1: 0.6536  |  LR: 1.00e-04  (22.4s)
 Best checkpoint saved (val_loss: 0.9592)
Epoch [ 12/30]  Train Loss: 0.8678  Bal.Acc: 66.1%  F1: 0.6278  |  Val Loss: 0.8857  Bal.Acc: 69.7%  F1: 0.6931  |  LR: 1.00e-04  (28.1s)
 Best checkpoint saved (val_loss: 0.8857)
Epoch [ 13/30]  Train Loss: 0.7936  Bal.Acc: 69.9%  F1: 0.6782  |  Val Loss: 0.9859  Bal.Acc: 58.9%  F1: 0.6103  |  LR: 1.00e-04  (22.6s)
Epoch [ 14/30]  Train Loss: 0.7361  Bal.Acc: 71.8%  F1: 0.7039  |  Val Loss: 0.8563  Bal.Acc: 64.5%  F1: 0.6584  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 0.8563)
Epoch [ 15/30]  Train Loss: 0.6961  Bal.Acc: 73.0%  F1: 0.7238  |  Val Loss: 0.6636  Bal.Acc: 73.1%  F1: 0.7315  |  LR: 1.00e-04  (23.5s)
 Best checkpoint saved (val_loss: 0.6636)
Epoch [ 16/30]  Train Loss: 0.6538  Bal.Acc: 74.7%  F1: 0.7425  |  Val Loss: 0.6779  Bal.Acc: 77.4%  F1: 0.7730  |  LR: 1.00e-04  (22.0s)
Epoch [ 17/30]  Train Loss: 0.6063  Bal.Acc: 76.7%  F1: 0.7592  |  Val Loss: 0.6606  Bal.Acc: 78.4%  F1: 0.7923  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 0.6606)
Epoch [ 18/30]  Train Loss: 0.5642  Bal.Acc: 79.0%  F1: 0.7828  |  Val Loss: 0.6432  Bal.Acc: 76.9%  F1: 0.7538  |  LR: 1.00e-04  (23.7s)
 Best checkpoint saved (val_loss: 0.6432)
Epoch [ 19/30]  Train Loss: 0.5374  Bal.Acc: 79.5%  F1: 0.7851  |  Val Loss: 0.5777  Bal.Acc: 81.5%  F1: 0.8038  |  LR: 1.00e-04  (28.8s)
 Best checkpoint saved (val_loss: 0.5777)
Epoch [ 20/30]  Train Loss: 0.5085  Bal.Acc: 80.1%  F1: 0.7863  |  Val Loss: 0.6842  Bal.Acc: 72.6%  F1: 0.7316  |  LR: 1.00e-04  (23.0s)
Epoch [ 21/30]  Train Loss: 0.4929  Bal.Acc: 80.8%  F1: 0.8015  |  Val Loss: 0.7772  Bal.Acc: 70.6%  F1: 0.7130  |  LR: 1.00e-04  (16.5s)
Epoch [ 22/30]  Train Loss: 0.4836  Bal.Acc: 80.0%  F1: 0.7923  |  Val Loss: 0.6748  Bal.Acc: 74.2%  F1: 0.7446  |  LR: 1.00e-04  (16.4s)
Epoch [ 23/30]  Train Loss: 0.4310  Bal.Acc: 85.6%  F1: 0.8479  |  Val Loss: 0.5338  Bal.Acc: 80.0%  F1: 0.7911  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 0.5338)
Epoch [ 24/30]  Train Loss: 0.4045  Bal.Acc: 84.6%  F1: 0.8380  |  Val Loss: 0.6238  Bal.Acc: 73.5%  F1: 0.7216  |  LR: 1.00e-04  (22.8s)
Epoch [ 25/30]  Train Loss: 0.4017  Bal.Acc: 85.4%  F1: 0.8471  |  Val Loss: 0.4967  Bal.Acc: 82.0%  F1: 0.8212  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 0.4967)
Epoch [ 26/30]  Train Loss: 0.3752  Bal.Acc: 85.4%  F1: 0.8460  |  Val Loss: 0.5473  Bal.Acc: 77.6%  F1: 0.7790  |  LR: 1.00e-04  (22.9s)
Epoch [ 27/30]  Train Loss: 0.3516  Bal.Acc: 88.5%  F1: 0.8790  |  Val Loss: 0.4435  Bal.Acc: 84.8%  F1: 0.8430  |  LR: 1.00e-04  (15.4s)
 Best checkpoint saved (val_loss: 0.4435)
Epoch [ 28/30]  Train Loss: 0.3259  Bal.Acc: 88.7%  F1: 0.8782  |  Val Loss: 0.5005  Bal.Acc: 81.4%  F1: 0.8072  |  LR: 1.00e-04  (22.0s)
Epoch [ 29/30]  Train Loss: 0.3450  Bal.Acc: 87.2%  F1: 0.8657  |  Val Loss: 0.6740  Bal.Acc: 76.2%  F1: 0.7561  |  LR: 1.00e-04  (16.0s)
Epoch [ 30/30]  Train Loss: 0.3128  Bal.Acc: 89.7%  F1: 0.8897  |  Val Loss: 0.4792  Bal.Acc: 83.3%  F1: 0.8369  |  LR: 1.00e-04  (16.1s)

 Training finished. Checkpoint: checkpoints/resnet50_fold1_best.pt
Log CSV: results/logs/resnet50_fold1_training_log.csv
Best weights loaded from epoch 27

Model evaluation: resnet50_fold1
----------------------------------------
  Balanced Accuracy:       84.85%
  F1 (macro):              0.8430
  Quadratic Cohen's Kappa: 0.9333
  MAE (ordinal):           0.1851
  Off-by-one accuracy:     97.51%
  ECE:                     0.0439
  Brier Score (mean):      0.0480

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.84      0.93      0.88        88
    Doubtful       0.89      0.72      0.79        81
        Mild       0.72      0.85      0.78        40
    Moderate       0.84      0.97      0.90        37
      Severe       0.96      0.77      0.86        35

    accuracy                           0.84       281
   macro avg       0.85      0.85      0.84       281
weighted avg       0.85      0.84      0.84       281

  Metrics saved to: results/individual_models/resnet50_fold1_metrics.json
  Probabilities saved to: results/individual_models/resnet50_fold1_test_probs.npz

--- resnet50 | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

============================================================
TRAINING: resnet50_fold2
============================================================
Epoch [  1/30]  Train Loss: 1.6008  Bal.Acc: 24.5%  F1: 0.1897  |  Val Loss: 1.5910  Bal.Acc: 30.3%  F1: 0.2989  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 1.5910)
Epoch [  2/30]  Train Loss: 1.5829  Bal.Acc: 31.7%  F1: 0.2976  |  Val Loss: 1.5657  Bal.Acc: 33.7%  F1: 0.3131  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 1.5657)
Epoch [  3/30]  Train Loss: 1.5690  Bal.Acc: 34.6%  F1: 0.3438  |  Val Loss: 1.5631  Bal.Acc: 33.2%  F1: 0.3306  |  LR: 1.00e-04  (21.2s)
 Best checkpoint saved (val_loss: 1.5631)
Epoch [  4/30]  Train Loss: 1.5464  Bal.Acc: 42.0%  F1: 0.3966  |  Val Loss: 1.5150  Bal.Acc: 36.0%  F1: 0.3498  |  LR: 1.00e-04  (28.1s)
 Best checkpoint saved (val_loss: 1.5150)
Epoch [  5/30]  Train Loss: 1.5162  Bal.Acc: 45.9%  F1: 0.4278  |  Val Loss: 1.4851  Bal.Acc: 36.9%  F1: 0.3433  |  LR: 1.00e-04  (21.7s)
 Best checkpoint saved (val_loss: 1.4851)
Epoch [  6/30]  Train Loss: 1.4726  Bal.Acc: 49.3%  F1: 0.4306  |  Val Loss: 1.4424  Bal.Acc: 52.4%  F1: 0.5189  |  LR: 1.00e-04  (27.8s)
 Best checkpoint saved (val_loss: 1.4424)
Epoch [  7/30]  Train Loss: 1.4156  Bal.Acc: 51.7%  F1: 0.4638  |  Val Loss: 1.3469  Bal.Acc: 54.6%  F1: 0.5273  |  LR: 1.00e-04  (28.8s)
 Best checkpoint saved (val_loss: 1.3469)
Epoch [  8/30]  Train Loss: 1.3359  Bal.Acc: 54.3%  F1: 0.4854  |  Val Loss: 1.3094  Bal.Acc: 44.6%  F1: 0.4204  |  LR: 1.00e-04  (22.2s)
 Best checkpoint saved (val_loss: 1.3094)
Epoch [  9/30]  Train Loss: 1.2180  Bal.Acc: 57.5%  F1: 0.5085  |  Val Loss: 1.1822  Bal.Acc: 55.3%  F1: 0.5320  |  LR: 1.00e-04  (29.6s)
 Best checkpoint saved (val_loss: 1.1822)
Epoch [ 10/30]  Train Loss: 1.0858  Bal.Acc: 60.8%  F1: 0.5537  |  Val Loss: 1.1020  Bal.Acc: 61.4%  F1: 0.6072  |  LR: 1.00e-04  (28.9s)
 Best checkpoint saved (val_loss: 1.1020)
Epoch [ 11/30]  Train Loss: 0.9946  Bal.Acc: 64.3%  F1: 0.6109  |  Val Loss: 0.9604  Bal.Acc: 61.8%  F1: 0.6153  |  LR: 1.00e-04  (29.1s)
 Best checkpoint saved (val_loss: 0.9604)
Epoch [ 12/30]  Train Loss: 0.9123  Bal.Acc: 67.4%  F1: 0.6472  |  Val Loss: 0.9838  Bal.Acc: 59.9%  F1: 0.6098  |  LR: 1.00e-04  (28.4s)
Epoch [ 13/30]  Train Loss: 0.8458  Bal.Acc: 68.8%  F1: 0.6701  |  Val Loss: 0.8067  Bal.Acc: 68.0%  F1: 0.6675  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 0.8067)
Epoch [ 14/30]  Train Loss: 0.7693  Bal.Acc: 72.8%  F1: 0.7146  |  Val Loss: 0.8690  Bal.Acc: 63.2%  F1: 0.6517  |  LR: 1.00e-04  (22.8s)
Epoch [ 15/30]  Train Loss: 0.7152  Bal.Acc: 74.5%  F1: 0.7334  |  Val Loss: 0.9484  Bal.Acc: 60.0%  F1: 0.6234  |  LR: 1.00e-04  (15.7s)
Epoch [ 16/30]  Train Loss: 0.6623  Bal.Acc: 75.1%  F1: 0.7451  |  Val Loss: 0.7213  Bal.Acc: 72.8%  F1: 0.7482  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 0.7213)
Epoch [ 17/30]  Train Loss: 0.6062  Bal.Acc: 77.4%  F1: 0.7667  |  Val Loss: 0.7600  Bal.Acc: 68.2%  F1: 0.6847  |  LR: 1.00e-04  (22.4s)
Epoch [ 18/30]  Train Loss: 0.5892  Bal.Acc: 77.1%  F1: 0.7607  |  Val Loss: 0.8563  Bal.Acc: 64.0%  F1: 0.6599  |  LR: 1.00e-04  (17.6s)
Epoch [ 19/30]  Train Loss: 0.5492  Bal.Acc: 78.4%  F1: 0.7788  |  Val Loss: 0.6013  Bal.Acc: 76.0%  F1: 0.7731  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 0.6013)
Epoch [ 20/30]  Train Loss: 0.5236  Bal.Acc: 80.0%  F1: 0.7901  |  Val Loss: 0.6129  Bal.Acc: 76.3%  F1: 0.7626  |  LR: 1.00e-04  (21.8s)
Epoch [ 21/30]  Train Loss: 0.4993  Bal.Acc: 81.7%  F1: 0.8101  |  Val Loss: 0.6311  Bal.Acc: 71.2%  F1: 0.7181  |  LR: 1.00e-04  (16.4s)
Epoch [ 22/30]  Train Loss: 0.4540  Bal.Acc: 82.6%  F1: 0.8196  |  Val Loss: 0.5566  Bal.Acc: 77.6%  F1: 0.7802  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 0.5566)
Epoch [ 23/30]  Train Loss: 0.4464  Bal.Acc: 84.3%  F1: 0.8345  |  Val Loss: 0.7464  Bal.Acc: 68.8%  F1: 0.7008  |  LR: 1.00e-04  (22.8s)
Epoch [ 24/30]  Train Loss: 0.4214  Bal.Acc: 84.8%  F1: 0.8426  |  Val Loss: 0.5689  Bal.Acc: 78.6%  F1: 0.7921  |  LR: 1.00e-04  (14.9s)
Epoch [ 25/30]  Train Loss: 0.3903  Bal.Acc: 85.4%  F1: 0.8465  |  Val Loss: 0.6983  Bal.Acc: 72.4%  F1: 0.7337  |  LR: 1.00e-04  (15.9s)
Epoch [ 26/30]  Train Loss: 0.3838  Bal.Acc: 85.3%  F1: 0.8463  |  Val Loss: 0.7330  Bal.Acc: 69.9%  F1: 0.7081  |  LR: 5.00e-05  (15.6s)
Epoch [ 27/30]  Train Loss: 0.3603  Bal.Acc: 86.5%  F1: 0.8591  |  Val Loss: 0.5393  Bal.Acc: 79.9%  F1: 0.8020  |  LR: 5.00e-05  (16.0s)
 Best checkpoint saved (val_loss: 0.5393)
Epoch [ 28/30]  Train Loss: 0.3184  Bal.Acc: 89.5%  F1: 0.8865  |  Val Loss: 0.5948  Bal.Acc: 77.6%  F1: 0.7870  |  LR: 5.00e-05  (21.4s)
Epoch [ 29/30]  Train Loss: 0.3249  Bal.Acc: 88.6%  F1: 0.8782  |  Val Loss: 0.5323  Bal.Acc: 80.3%  F1: 0.8058  |  LR: 5.00e-05  (17.0s)
 Best checkpoint saved (val_loss: 0.5323)
Epoch [ 30/30]  Train Loss: 0.3091  Bal.Acc: 89.4%  F1: 0.8877  |  Val Loss: 0.6075  Bal.Acc: 75.1%  F1: 0.7701  |  LR: 5.00e-05  (23.6s)

 Training finished. Checkpoint: checkpoints/resnet50_fold2_best.pt
Log CSV: results/logs/resnet50_fold2_training_log.csv
Best weights loaded from epoch 29

Model evaluation: resnet50_fold2
----------------------------------------
  Balanced Accuracy:       80.28%
  F1 (macro):              0.8058
  Quadratic Cohen's Kappa: 0.9308
  MAE (ordinal):           0.2242
  Off-by-one accuracy:     98.58%
  ECE:                     0.0539
  Brier Score (mean):      0.0603

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.77      0.93      0.84        88
    Doubtful       0.75      0.60      0.67        81
        Mild       0.71      0.72      0.72        40
    Moderate       0.91      0.84      0.87        37
      Severe       0.94      0.91      0.93        35

    accuracy                           0.79       281
   macro avg       0.82      0.80      0.81       281
weighted avg       0.80      0.79      0.79       281

  Metrics saved to: results/individual_models/resnet50_fold2_metrics.json
  Probabilities saved to: results/individual_models/resnet50_fold2_test_probs.npz

--- resnet50 | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

============================================================
TRAINING: resnet50_fold3
============================================================
Epoch [  1/30]  Train Loss: 1.6048  Bal.Acc: 22.1%  F1: 0.1192  |  Val Loss: 1.6037  Bal.Acc: 24.3%  F1: 0.2243  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 1.6037)
Epoch [  2/30]  Train Loss: 1.5891  Bal.Acc: 32.3%  F1: 0.2907  |  Val Loss: 1.5914  Bal.Acc: 31.3%  F1: 0.2853  |  LR: 1.00e-04  (18.1s)
 Best checkpoint saved (val_loss: 1.5914)
Epoch [  3/30]  Train Loss: 1.5666  Bal.Acc: 42.1%  F1: 0.3965  |  Val Loss: 1.5635  Bal.Acc: 35.1%  F1: 0.3343  |  LR: 1.00e-04  (22.4s)
 Best checkpoint saved (val_loss: 1.5635)
Epoch [  4/30]  Train Loss: 1.5437  Bal.Acc: 44.0%  F1: 0.4009  |  Val Loss: 1.5347  Bal.Acc: 43.3%  F1: 0.3876  |  LR: 1.00e-04  (23.5s)
 Best checkpoint saved (val_loss: 1.5347)
Epoch [  5/30]  Train Loss: 1.5127  Bal.Acc: 45.9%  F1: 0.4161  |  Val Loss: 1.4893  Bal.Acc: 42.1%  F1: 0.3933  |  LR: 1.00e-04  (22.5s)
 Best checkpoint saved (val_loss: 1.4893)
Epoch [  6/30]  Train Loss: 1.4625  Bal.Acc: 49.8%  F1: 0.4509  |  Val Loss: 1.4303  Bal.Acc: 48.9%  F1: 0.4435  |  LR: 1.00e-04  (28.9s)
 Best checkpoint saved (val_loss: 1.4303)
Epoch [  7/30]  Train Loss: 1.3829  Bal.Acc: 56.0%  F1: 0.4966  |  Val Loss: 1.3494  Bal.Acc: 51.8%  F1: 0.4468  |  LR: 1.00e-04  (28.8s)
 Best checkpoint saved (val_loss: 1.3494)
Epoch [  8/30]  Train Loss: 1.2798  Bal.Acc: 57.2%  F1: 0.5112  |  Val Loss: 1.2581  Bal.Acc: 50.9%  F1: 0.4194  |  LR: 1.00e-04  (29.1s)
 Best checkpoint saved (val_loss: 1.2581)
Epoch [  9/30]  Train Loss: 1.1404  Bal.Acc: 58.6%  F1: 0.5193  |  Val Loss: 1.2051  Bal.Acc: 53.7%  F1: 0.4891  |  LR: 1.00e-04  (28.1s)
 Best checkpoint saved (val_loss: 1.2051)
Epoch [ 10/30]  Train Loss: 1.0424  Bal.Acc: 62.1%  F1: 0.5769  |  Val Loss: 1.0838  Bal.Acc: 53.2%  F1: 0.4950  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 1.0838)
Epoch [ 11/30]  Train Loss: 0.9271  Bal.Acc: 66.6%  F1: 0.6348  |  Val Loss: 1.0044  Bal.Acc: 58.4%  F1: 0.5665  |  LR: 1.00e-04  (28.1s)
 Best checkpoint saved (val_loss: 1.0044)
Epoch [ 12/30]  Train Loss: 0.8563  Bal.Acc: 67.8%  F1: 0.6514  |  Val Loss: 1.0016  Bal.Acc: 58.4%  F1: 0.5870  |  LR: 1.00e-04  (28.3s)
 Best checkpoint saved (val_loss: 1.0016)
Epoch [ 13/30]  Train Loss: 0.8055  Bal.Acc: 68.9%  F1: 0.6732  |  Val Loss: 0.8678  Bal.Acc: 67.6%  F1: 0.6668  |  LR: 1.00e-04  (27.0s)
 Best checkpoint saved (val_loss: 0.8678)
Epoch [ 14/30]  Train Loss: 0.7361  Bal.Acc: 72.8%  F1: 0.7159  |  Val Loss: 0.8524  Bal.Acc: 69.5%  F1: 0.6985  |  LR: 1.00e-04  (28.7s)
 Best checkpoint saved (val_loss: 0.8524)
Epoch [ 15/30]  Train Loss: 0.6637  Bal.Acc: 74.4%  F1: 0.7365  |  Val Loss: 0.7775  Bal.Acc: 71.5%  F1: 0.7132  |  LR: 1.00e-04  (27.3s)
 Best checkpoint saved (val_loss: 0.7775)
Epoch [ 16/30]  Train Loss: 0.6174  Bal.Acc: 76.6%  F1: 0.7581  |  Val Loss: 0.7554  Bal.Acc: 68.8%  F1: 0.6854  |  LR: 1.00e-04  (27.5s)
 Best checkpoint saved (val_loss: 0.7554)
Epoch [ 17/30]  Train Loss: 0.5802  Bal.Acc: 78.7%  F1: 0.7763  |  Val Loss: 0.8521  Bal.Acc: 64.2%  F1: 0.6496  |  LR: 1.00e-04  (29.1s)
Epoch [ 18/30]  Train Loss: 0.5701  Bal.Acc: 78.4%  F1: 0.7751  |  Val Loss: 0.8603  Bal.Acc: 63.1%  F1: 0.6311  |  LR: 1.00e-04  (16.5s)
Epoch [ 19/30]  Train Loss: 0.5019  Bal.Acc: 81.0%  F1: 0.8040  |  Val Loss: 0.7604  Bal.Acc: 67.7%  F1: 0.6844  |  LR: 1.00e-04  (15.7s)
Epoch [ 20/30]  Train Loss: 0.5126  Bal.Acc: 81.3%  F1: 0.8029  |  Val Loss: 0.7937  Bal.Acc: 67.6%  F1: 0.6798  |  LR: 5.00e-05  (16.0s)
Epoch [ 21/30]  Train Loss: 0.4686  Bal.Acc: 83.2%  F1: 0.8221  |  Val Loss: 0.6991  Bal.Acc: 72.6%  F1: 0.7245  |  LR: 5.00e-05  (15.9s)
 Best checkpoint saved (val_loss: 0.6991)
Epoch [ 22/30]  Train Loss: 0.4420  Bal.Acc: 83.4%  F1: 0.8252  |  Val Loss: 0.6615  Bal.Acc: 75.3%  F1: 0.7556  |  LR: 5.00e-05  (23.1s)
 Best checkpoint saved (val_loss: 0.6615)
Epoch [ 23/30]  Train Loss: 0.4463  Bal.Acc: 83.4%  F1: 0.8287  |  Val Loss: 0.6210  Bal.Acc: 78.0%  F1: 0.7832  |  LR: 5.00e-05  (21.6s)
 Best checkpoint saved (val_loss: 0.6210)
Epoch [ 24/30]  Train Loss: 0.4227  Bal.Acc: 85.8%  F1: 0.8458  |  Val Loss: 0.6233  Bal.Acc: 76.4%  F1: 0.7676  |  LR: 5.00e-05  (22.7s)
Epoch [ 25/30]  Train Loss: 0.4247  Bal.Acc: 83.9%  F1: 0.8300  |  Val Loss: 0.6044  Bal.Acc: 74.8%  F1: 0.7462  |  LR: 5.00e-05  (14.8s)
 Best checkpoint saved (val_loss: 0.6044)
Epoch [ 26/30]  Train Loss: 0.4182  Bal.Acc: 83.8%  F1: 0.8343  |  Val Loss: 0.6042  Bal.Acc: 76.5%  F1: 0.7670  |  LR: 5.00e-05  (22.2s)
 Best checkpoint saved (val_loss: 0.6042)
Epoch [ 27/30]  Train Loss: 0.3863  Bal.Acc: 86.2%  F1: 0.8569  |  Val Loss: 0.6047  Bal.Acc: 77.8%  F1: 0.7741  |  LR: 5.00e-05  (23.2s)
Epoch [ 28/30]  Train Loss: 0.3565  Bal.Acc: 87.1%  F1: 0.8633  |  Val Loss: 0.6239  Bal.Acc: 78.9%  F1: 0.7887  |  LR: 5.00e-05  (15.4s)
Epoch [ 29/30]  Train Loss: 0.3803  Bal.Acc: 86.2%  F1: 0.8535  |  Val Loss: 0.6487  Bal.Acc: 78.2%  F1: 0.7817  |  LR: 5.00e-05  (16.4s)
Epoch [ 30/30]  Train Loss: 0.3710  Bal.Acc: 86.4%  F1: 0.8569  |  Val Loss: 0.5524  Bal.Acc: 77.8%  F1: 0.7719  |  LR: 5.00e-05  (15.7s)
 Best checkpoint saved (val_loss: 0.5524)

 Training finished. Checkpoint: checkpoints/resnet50_fold3_best.pt
Log CSV: results/logs/resnet50_fold3_training_log.csv
Best weights loaded from epoch 30

Model evaluation: resnet50_fold3
----------------------------------------
  Balanced Accuracy:       77.78%
  F1 (macro):              0.7719
  Quadratic Cohen's Kappa: 0.9148
  MAE (ordinal):           0.2500
  Off-by-one accuracy:     97.50%
  ECE:                     0.0313
  Brier Score (mean):      0.0606

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.83      0.85        87
    Doubtful       0.71      0.77      0.74        81
        Mild       0.70      0.49      0.58        39
    Moderate       0.74      0.89      0.81        38
      Severe       0.86      0.91      0.89        35

    accuracy                           0.78       280
   macro avg       0.78      0.78      0.77       280
weighted avg       0.78      0.78      0.78       280

  Metrics saved to: results/individual_models/resnet50_fold3_metrics.json
  Probabilities saved to: results/individual_models/resnet50_fold3_test_probs.npz

--- resnet50 | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

============================================================
TRAINING: resnet50_fold4
============================================================
Epoch [  1/30]  Train Loss: 1.6062  Bal.Acc: 24.7%  F1: 0.2068  |  Val Loss: 1.6021  Bal.Acc: 25.8%  F1: 0.2501  |  LR: 1.00e-04  (23.0s)
 Best checkpoint saved (val_loss: 1.6021)
Epoch [  2/30]  Train Loss: 1.5896  Bal.Acc: 30.5%  F1: 0.2955  |  Val Loss: 1.5880  Bal.Acc: 29.9%  F1: 0.2995  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 1.5880)
Epoch [  3/30]  Train Loss: 1.5682  Bal.Acc: 38.9%  F1: 0.3754  |  Val Loss: 1.5702  Bal.Acc: 34.2%  F1: 0.3265  |  LR: 1.00e-04  (21.8s)
 Best checkpoint saved (val_loss: 1.5702)
Epoch [  4/30]  Train Loss: 1.5429  Bal.Acc: 42.7%  F1: 0.3977  |  Val Loss: 1.5400  Bal.Acc: 42.8%  F1: 0.3863  |  LR: 1.00e-04  (29.0s)
 Best checkpoint saved (val_loss: 1.5400)
Epoch [  5/30]  Train Loss: 1.5060  Bal.Acc: 46.4%  F1: 0.4171  |  Val Loss: 1.5043  Bal.Acc: 37.0%  F1: 0.3487  |  LR: 1.00e-04  (23.2s)
 Best checkpoint saved (val_loss: 1.5043)
Epoch [  6/30]  Train Loss: 1.4484  Bal.Acc: 49.1%  F1: 0.4343  |  Val Loss: 1.4352  Bal.Acc: 43.8%  F1: 0.4033  |  LR: 1.00e-04  (21.8s)
 Best checkpoint saved (val_loss: 1.4352)
Epoch [  7/30]  Train Loss: 1.3865  Bal.Acc: 53.0%  F1: 0.4747  |  Val Loss: 1.3808  Bal.Acc: 43.6%  F1: 0.4087  |  LR: 1.00e-04  (28.6s)
 Best checkpoint saved (val_loss: 1.3808)
Epoch [  8/30]  Train Loss: 1.2826  Bal.Acc: 55.5%  F1: 0.4987  |  Val Loss: 1.2361  Bal.Acc: 55.8%  F1: 0.4956  |  LR: 1.00e-04  (28.2s)
 Best checkpoint saved (val_loss: 1.2361)
Epoch [  9/30]  Train Loss: 1.1735  Bal.Acc: 60.2%  F1: 0.5485  |  Val Loss: 1.1398  Bal.Acc: 58.0%  F1: 0.5507  |  LR: 1.00e-04  (27.3s)
 Best checkpoint saved (val_loss: 1.1398)
Epoch [ 10/30]  Train Loss: 1.0551  Bal.Acc: 61.9%  F1: 0.5694  |  Val Loss: 1.0703  Bal.Acc: 59.4%  F1: 0.5658  |  LR: 1.00e-04  (28.4s)
 Best checkpoint saved (val_loss: 1.0703)
Epoch [ 11/30]  Train Loss: 0.9574  Bal.Acc: 63.9%  F1: 0.5970  |  Val Loss: 0.9728  Bal.Acc: 62.9%  F1: 0.6111  |  LR: 1.00e-04  (27.9s)
 Best checkpoint saved (val_loss: 0.9728)
Epoch [ 12/30]  Train Loss: 0.8757  Bal.Acc: 68.1%  F1: 0.6543  |  Val Loss: 0.9191  Bal.Acc: 64.5%  F1: 0.6353  |  LR: 1.00e-04  (25.7s)
 Best checkpoint saved (val_loss: 0.9191)
Epoch [ 13/30]  Train Loss: 0.7888  Bal.Acc: 71.4%  F1: 0.6976  |  Val Loss: 0.9427  Bal.Acc: 61.1%  F1: 0.6301  |  LR: 1.00e-04  (27.2s)
Epoch [ 14/30]  Train Loss: 0.7438  Bal.Acc: 72.2%  F1: 0.7121  |  Val Loss: 0.7909  Bal.Acc: 70.9%  F1: 0.7112  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 0.7909)
Epoch [ 15/30]  Train Loss: 0.6891  Bal.Acc: 73.5%  F1: 0.7262  |  Val Loss: 0.7864  Bal.Acc: 69.4%  F1: 0.6931  |  LR: 1.00e-04  (21.9s)
 Best checkpoint saved (val_loss: 0.7864)
Epoch [ 16/30]  Train Loss: 0.6249  Bal.Acc: 76.8%  F1: 0.7613  |  Val Loss: 0.6731  Bal.Acc: 75.4%  F1: 0.7444  |  LR: 1.00e-04  (28.6s)
 Best checkpoint saved (val_loss: 0.6731)
Epoch [ 17/30]  Train Loss: 0.5715  Bal.Acc: 80.2%  F1: 0.7941  |  Val Loss: 0.6922  Bal.Acc: 77.1%  F1: 0.7676  |  LR: 1.00e-04  (28.2s)
Epoch [ 18/30]  Train Loss: 0.5347  Bal.Acc: 78.7%  F1: 0.7773  |  Val Loss: 0.6815  Bal.Acc: 76.2%  F1: 0.7565  |  LR: 1.00e-04  (16.1s)
Epoch [ 19/30]  Train Loss: 0.5215  Bal.Acc: 79.3%  F1: 0.7847  |  Val Loss: 0.6493  Bal.Acc: 76.4%  F1: 0.7572  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 0.6493)
Epoch [ 20/30]  Train Loss: 0.4624  Bal.Acc: 83.8%  F1: 0.8330  |  Val Loss: 0.6371  Bal.Acc: 74.2%  F1: 0.7386  |  LR: 1.00e-04  (21.4s)
 Best checkpoint saved (val_loss: 0.6371)
Epoch [ 21/30]  Train Loss: 0.4375  Bal.Acc: 84.0%  F1: 0.8326  |  Val Loss: 0.6660  Bal.Acc: 77.0%  F1: 0.7732  |  LR: 1.00e-04  (28.0s)
Epoch [ 22/30]  Train Loss: 0.4360  Bal.Acc: 83.5%  F1: 0.8305  |  Val Loss: 0.6330  Bal.Acc: 74.3%  F1: 0.7262  |  LR: 1.00e-04  (15.9s)
 Best checkpoint saved (val_loss: 0.6330)
Epoch [ 23/30]  Train Loss: 0.3819  Bal.Acc: 87.6%  F1: 0.8703  |  Val Loss: 0.6507  Bal.Acc: 74.4%  F1: 0.7445  |  LR: 1.00e-04  (22.0s)
Epoch [ 24/30]  Train Loss: 0.3758  Bal.Acc: 86.1%  F1: 0.8570  |  Val Loss: 0.6347  Bal.Acc: 78.6%  F1: 0.7615  |  LR: 1.00e-04  (16.1s)
Epoch [ 25/30]  Train Loss: 0.3523  Bal.Acc: 85.9%  F1: 0.8484  |  Val Loss: 0.6952  Bal.Acc: 75.5%  F1: 0.7492  |  LR: 1.00e-04  (15.9s)
Epoch [ 26/30]  Train Loss: 0.3444  Bal.Acc: 87.0%  F1: 0.8668  |  Val Loss: 0.6175  Bal.Acc: 77.0%  F1: 0.7693  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 0.6175)
Epoch [ 27/30]  Train Loss: 0.3520  Bal.Acc: 86.5%  F1: 0.8607  |  Val Loss: 0.6814  Bal.Acc: 72.0%  F1: 0.7061  |  LR: 1.00e-04  (23.0s)
Epoch [ 28/30]  Train Loss: 0.3095  Bal.Acc: 88.8%  F1: 0.8789  |  Val Loss: 0.6203  Bal.Acc: 77.9%  F1: 0.7732  |  LR: 1.00e-04  (15.0s)
Epoch [ 29/30]  Train Loss: 0.2844  Bal.Acc: 90.2%  F1: 0.8966  |  Val Loss: 0.5937  Bal.Acc: 77.4%  F1: 0.7599  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 0.5937)
Epoch [ 30/30]  Train Loss: 0.2722  Bal.Acc: 90.3%  F1: 0.8981  |  Val Loss: 0.5967  Bal.Acc: 77.8%  F1: 0.7696  |  LR: 1.00e-04  (22.1s)

 Training finished. Checkpoint: checkpoints/resnet50_fold4_best.pt
Log CSV: results/logs/resnet50_fold4_training_log.csv
Best weights loaded from epoch 29

Model evaluation: resnet50_fold4
----------------------------------------
  Balanced Accuracy:       77.38%
  F1 (macro):              0.7599
  Quadratic Cohen's Kappa: 0.9055
  MAE (ordinal):           0.2714
  Off-by-one accuracy:     97.14%
  ECE:                     0.0867
  Brier Score (mean):      0.0653

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.84      0.94      0.89        87
    Doubtful       0.79      0.56      0.65        81
        Mild       0.51      0.64      0.57        39
    Moderate       0.82      0.82      0.82        38
      Severe       0.84      0.91      0.88        35

    accuracy                           0.77       280
   macro avg       0.76      0.77      0.76       280
weighted avg       0.78      0.77      0.76       280

  Metrics saved to: results/individual_models/resnet50_fold4_metrics.json
  Probabilities saved to: results/individual_models/resnet50_fold4_test_probs.npz

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

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable

============================================================
TRAINING: resnet50_fold5
============================================================
Epoch [  1/30]  Train Loss: 1.6019  Bal.Acc: 23.8%  F1: 0.1983  |  Val Loss: 1.5966  Bal.Acc: 24.4%  F1: 0.2286  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 1.5966)
Epoch [  2/30]  Train Loss: 1.5800  Bal.Acc: 29.6%  F1: 0.2907  |  Val Loss: 1.5700  Bal.Acc: 39.3%  F1: 0.3905  |  LR: 1.00e-04  (15.9s)
 Best checkpoint saved (val_loss: 1.5700)
Epoch [  3/30]  Train Loss: 1.5583  Bal.Acc: 36.1%  F1: 0.3581  |  Val Loss: 1.5539  Bal.Acc: 38.9%  F1: 0.3987  |  LR: 1.00e-04  (21.8s)
 Best checkpoint saved (val_loss: 1.5539)
Epoch [  4/30]  Train Loss: 1.5316  Bal.Acc: 41.2%  F1: 0.4030  |  Val Loss: 1.5221  Bal.Acc: 52.0%  F1: 0.5202  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 1.5221)
Epoch [  5/30]  Train Loss: 1.4992  Bal.Acc: 46.6%  F1: 0.4487  |  Val Loss: 1.4819  Bal.Acc: 52.5%  F1: 0.5151  |  LR: 1.00e-04  (21.7s)
 Best checkpoint saved (val_loss: 1.4819)
Epoch [  6/30]  Train Loss: 1.4478  Bal.Acc: 50.0%  F1: 0.4806  |  Val Loss: 1.4413  Bal.Acc: 37.8%  F1: 0.3644  |  LR: 1.00e-04  (22.6s)
 Best checkpoint saved (val_loss: 1.4413)
Epoch [  7/30]  Train Loss: 1.3987  Bal.Acc: 50.1%  F1: 0.4711  |  Val Loss: 1.3368  Bal.Acc: 46.5%  F1: 0.4421  |  LR: 1.00e-04  (22.1s)
 Best checkpoint saved (val_loss: 1.3368)
Epoch [  8/30]  Train Loss: 1.3097  Bal.Acc: 54.4%  F1: 0.5006  |  Val Loss: 1.2578  Bal.Acc: 57.4%  F1: 0.5320  |  LR: 1.00e-04  (28.7s)
 Best checkpoint saved (val_loss: 1.2578)
Epoch [  9/30]  Train Loss: 1.2067  Bal.Acc: 57.7%  F1: 0.5242  |  Val Loss: 1.2462  Bal.Acc: 55.0%  F1: 0.5156  |  LR: 1.00e-04  (28.9s)
 Best checkpoint saved (val_loss: 1.2462)
Epoch [ 10/30]  Train Loss: 1.0895  Bal.Acc: 60.6%  F1: 0.5593  |  Val Loss: 1.1099  Bal.Acc: 55.1%  F1: 0.5167  |  LR: 1.00e-04  (28.0s)
 Best checkpoint saved (val_loss: 1.1099)
Epoch [ 11/30]  Train Loss: 0.9857  Bal.Acc: 63.5%  F1: 0.5930  |  Val Loss: 0.9658  Bal.Acc: 61.9%  F1: 0.5748  |  LR: 1.00e-04  (27.6s)
 Best checkpoint saved (val_loss: 0.9658)
Epoch [ 12/30]  Train Loss: 0.9175  Bal.Acc: 65.7%  F1: 0.6346  |  Val Loss: 0.9418  Bal.Acc: 59.7%  F1: 0.5488  |  LR: 1.00e-04  (27.7s)
 Best checkpoint saved (val_loss: 0.9418)
Epoch [ 13/30]  Train Loss: 0.8487  Bal.Acc: 68.3%  F1: 0.6624  |  Val Loss: 0.8995  Bal.Acc: 62.2%  F1: 0.6016  |  LR: 1.00e-04  (28.2s)
 Best checkpoint saved (val_loss: 0.8995)
Epoch [ 14/30]  Train Loss: 0.7704  Bal.Acc: 71.0%  F1: 0.6942  |  Val Loss: 0.8979  Bal.Acc: 62.7%  F1: 0.6013  |  LR: 1.00e-04  (27.0s)
 Best checkpoint saved (val_loss: 0.8979)
Epoch [ 15/30]  Train Loss: 0.7015  Bal.Acc: 75.4%  F1: 0.7411  |  Val Loss: 0.8387  Bal.Acc: 64.4%  F1: 0.6390  |  LR: 1.00e-04  (27.4s)
 Best checkpoint saved (val_loss: 0.8387)
Epoch [ 16/30]  Train Loss: 0.6854  Bal.Acc: 74.1%  F1: 0.7291  |  Val Loss: 0.9137  Bal.Acc: 63.7%  F1: 0.6300  |  LR: 1.00e-04  (28.0s)
Epoch [ 17/30]  Train Loss: 0.6289  Bal.Acc: 76.1%  F1: 0.7544  |  Val Loss: 0.7598  Bal.Acc: 70.1%  F1: 0.6884  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 0.7598)
Epoch [ 18/30]  Train Loss: 0.5953  Bal.Acc: 77.3%  F1: 0.7635  |  Val Loss: 0.7584  Bal.Acc: 72.4%  F1: 0.7133  |  LR: 1.00e-04  (21.0s)
 Best checkpoint saved (val_loss: 0.7584)
Epoch [ 19/30]  Train Loss: 0.5691  Bal.Acc: 78.8%  F1: 0.7757  |  Val Loss: 0.7472  Bal.Acc: 72.6%  F1: 0.7164  |  LR: 1.00e-04  (27.5s)
 Best checkpoint saved (val_loss: 0.7472)
Epoch [ 20/30]  Train Loss: 0.5212  Bal.Acc: 80.4%  F1: 0.7986  |  Val Loss: 0.6876  Bal.Acc: 75.4%  F1: 0.7496  |  LR: 1.00e-04  (28.0s)
 Best checkpoint saved (val_loss: 0.6876)
Epoch [ 21/30]  Train Loss: 0.4969  Bal.Acc: 81.6%  F1: 0.8072  |  Val Loss: 0.7428  Bal.Acc: 73.8%  F1: 0.7324  |  LR: 1.00e-04  (28.2s)
Epoch [ 22/30]  Train Loss: 0.4811  Bal.Acc: 82.6%  F1: 0.8160  |  Val Loss: 0.9139  Bal.Acc: 65.0%  F1: 0.6595  |  LR: 1.00e-04  (17.0s)
Epoch [ 23/30]  Train Loss: 0.4752  Bal.Acc: 82.7%  F1: 0.8262  |  Val Loss: 0.6270  Bal.Acc: 78.4%  F1: 0.7710  |  LR: 1.00e-04  (15.4s)
 Best checkpoint saved (val_loss: 0.6270)
Epoch [ 24/30]  Train Loss: 0.4522  Bal.Acc: 81.9%  F1: 0.8072  |  Val Loss: 0.6729  Bal.Acc: 78.7%  F1: 0.7801  |  LR: 1.00e-04  (21.7s)
Epoch [ 25/30]  Train Loss: 0.4086  Bal.Acc: 84.2%  F1: 0.8329  |  Val Loss: 0.6513  Bal.Acc: 79.6%  F1: 0.7822  |  LR: 1.00e-04  (17.8s)
Epoch [ 26/30]  Train Loss: 0.3962  Bal.Acc: 84.5%  F1: 0.8379  |  Val Loss: 0.6362  Bal.Acc: 79.6%  F1: 0.7817  |  LR: 1.00e-04  (16.9s)
Epoch [ 27/30]  Train Loss: 0.4000  Bal.Acc: 85.4%  F1: 0.8443  |  Val Loss: 0.6456  Bal.Acc: 77.4%  F1: 0.7628  |  LR: 5.00e-05  (16.4s)
Epoch [ 28/30]  Train Loss: 0.3535  Bal.Acc: 86.9%  F1: 0.8680  |  Val Loss: 0.6041  Bal.Acc: 82.1%  F1: 0.8092  |  LR: 5.00e-05  (15.3s)
 Best checkpoint saved (val_loss: 0.6041)
Epoch [ 29/30]  Train Loss: 0.3388  Bal.Acc: 87.4%  F1: 0.8676  |  Val Loss: 0.6012  Bal.Acc: 81.5%  F1: 0.8060  |  LR: 5.00e-05  (21.9s)
 Best checkpoint saved (val_loss: 0.6012)
Epoch [ 30/30]  Train Loss: 0.3265  Bal.Acc: 88.3%  F1: 0.8742  |  Val Loss: 0.6048  Bal.Acc: 81.1%  F1: 0.8010  |  LR: 5.00e-05  (23.3s)

 Training finished. Checkpoint: checkpoints/resnet50_fold5_best.pt
Log CSV: results/logs/resnet50_fold5_training_log.csv
Best weights loaded from epoch 29

Model evaluation: resnet50_fold5
----------------------------------------
  Balanced Accuracy:       81.52%
  F1 (macro):              0.8060
  Quadratic Cohen's Kappa: 0.9091
  MAE (ordinal):           0.2357
  Off-by-one accuracy:     96.07%
  ECE:                     0.0507
  Brier Score (mean):      0.0598

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.94      0.89        87
    Doubtful       0.84      0.67      0.74        81
        Mild       0.69      0.74      0.72        39
    Moderate       0.87      0.89      0.88        38
      Severe       0.76      0.83      0.79        35

    accuracy                           0.81       280
   macro avg       0.80      0.82      0.81       280
weighted avg       0.82      0.81      0.81       280

  Metrics saved to: results/individual_models/resnet50_fold5_metrics.json
  Probabilities saved to: results/individual_models/resnet50_fold5_test_probs.npz

 FINISHED: resnet50. Average kappa out of 5 folds: 0.9187 ±0.0113

================================================================================
MODEL TRAINING START: efficientnet_b3
================================================================================

--- efficientnet_b3 | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: efficientnet_b3
model.safetensors: 100% 49.3M/49.3M [00:01<00:00, 36.8MB/s]
  Parameters: 10,703,917 total, 10,703,917 trainable

============================================================
TRAINING: efficientnet_b3_fold1
============================================================
Epoch [  1/30]  Train Loss: 2.6010  Bal.Acc: 29.1%  F1: 0.2745  |  Val Loss: 2.1388  Bal.Acc: 34.9%  F1: 0.3205  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 2.1388)
Epoch [  2/30]  Train Loss: 1.6426  Bal.Acc: 47.0%  F1: 0.4505  |  Val Loss: 1.7589  Bal.Acc: 41.8%  F1: 0.3872  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 1.7589)
Epoch [  3/30]  Train Loss: 1.2373  Bal.Acc: 54.6%  F1: 0.5308  |  Val Loss: 1.5170  Bal.Acc: 49.9%  F1: 0.4863  |  LR: 1.00e-04  (19.6s)
 Best checkpoint saved (val_loss: 1.5170)
Epoch [  4/30]  Train Loss: 1.0983  Bal.Acc: 62.5%  F1: 0.6077  |  Val Loss: 1.5004  Bal.Acc: 54.9%  F1: 0.5676  |  LR: 1.00e-04  (19.4s)
 Best checkpoint saved (val_loss: 1.5004)
Epoch [  5/30]  Train Loss: 0.8491  Bal.Acc: 69.3%  F1: 0.6829  |  Val Loss: 1127.8426  Bal.Acc: 62.7%  F1: 0.6241  |  LR: 1.00e-04  (19.3s)
Epoch [  6/30]  Train Loss: 0.7117  Bal.Acc: 71.8%  F1: 0.7040  |  Val Loss: 47.2578  Bal.Acc: 67.1%  F1: 0.6697  |  LR: 1.00e-04  (16.0s)
Epoch [  7/30]  Train Loss: 0.6394  Bal.Acc: 77.5%  F1: 0.7623  |  Val Loss: 4090.8634  Bal.Acc: 68.3%  F1: 0.6870  |  LR: 1.00e-04  (16.4s)
Epoch [  8/30]  Train Loss: 0.5786  Bal.Acc: 78.6%  F1: 0.7766  |  Val Loss: 9676.4589  Bal.Acc: 72.0%  F1: 0.7199  |  LR: 5.00e-05  (15.9s)
Epoch [  9/30]  Train Loss: 0.5232  Bal.Acc: 79.5%  F1: 0.7833  |  Val Loss: 6898.2474  Bal.Acc: 71.8%  F1: 0.7210  |  LR: 5.00e-05  (15.5s)
Epoch [ 10/30]  Train Loss: 0.4685  Bal.Acc: 80.8%  F1: 0.7979  |  Val Loss: 39207.9820  Bal.Acc: 72.4%  F1: 0.7256  |  LR: 5.00e-05  (15.2s)
Epoch [ 11/30]  Train Loss: 0.4337  Bal.Acc: 82.9%  F1: 0.8170  |  Val Loss: 21063.6788  Bal.Acc: 75.3%  F1: 0.7484  |  LR: 5.00e-05  (15.2s)
Epoch [ 12/30]  Train Loss: 0.4561  Bal.Acc: 83.0%  F1: 0.8216  |  Val Loss: 29982.0442  Bal.Acc: 78.0%  F1: 0.7782  |  LR: 2.50e-05  (14.8s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 1.5004

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold1_best.pt
Log CSV: results/logs/efficientnet_b3_fold1_training_log.csv
Best weights loaded from epoch 4

Model evaluation: efficientnet_b3_fold1
----------------------------------------
  Balanced Accuracy:       54.87%
  F1 (macro):              0.5676
  Quadratic Cohen's Kappa: 0.6274
  MAE (ordinal):           0.6726
  Off-by-one accuracy:     83.27%
  ECE:                     0.1932
  Brier Score (mean):      0.1282

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.55      0.88      0.68        88
    Doubtful       0.57      0.41      0.47        81
        Mild       0.39      0.33      0.36        40
    Moderate       0.79      0.62      0.70        37
      Severe       0.82      0.51      0.63        35

    accuracy                           0.58       281
   macro avg       0.63      0.55      0.57       281
weighted avg       0.60      0.58      0.57       281

  Metrics saved to: results/individual_models/efficientnet_b3_fold1_metrics.json
  Probabilities saved to: results/individual_models/efficientnet_b3_fold1_test_probs.npz

--- efficientnet_b3 | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

============================================================
TRAINING: efficientnet_b3_fold2
============================================================
Epoch [  1/30]  Train Loss: 2.5174  Bal.Acc: 27.8%  F1: 0.2641  |  Val Loss: 2.3096  Bal.Acc: 30.3%  F1: 0.2876  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 2.3096)
Epoch [  2/30]  Train Loss: 1.5782  Bal.Acc: 48.9%  F1: 0.4623  |  Val Loss: 1.8916  Bal.Acc: 40.8%  F1: 0.4229  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 1.8916)
Epoch [  3/30]  Train Loss: 1.2259  Bal.Acc: 58.0%  F1: 0.5686  |  Val Loss: 1.5767  Bal.Acc: 47.1%  F1: 0.4880  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 1.5767)
Epoch [  4/30]  Train Loss: 0.9031  Bal.Acc: 67.6%  F1: 0.6517  |  Val Loss: 1.3464  Bal.Acc: 57.9%  F1: 0.5679  |  LR: 1.00e-04  (18.8s)
 Best checkpoint saved (val_loss: 1.3464)
Epoch [  5/30]  Train Loss: 0.8389  Bal.Acc: 69.5%  F1: 0.6854  |  Val Loss: 1.1972  Bal.Acc: 62.6%  F1: 0.6271  |  LR: 1.00e-04  (19.5s)
 Best checkpoint saved (val_loss: 1.1972)
Epoch [  6/30]  Train Loss: 0.6971  Bal.Acc: 75.0%  F1: 0.7356  |  Val Loss: 1.1089  Bal.Acc: 65.2%  F1: 0.6529  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 1.1089)
Epoch [  7/30]  Train Loss: 0.6021  Bal.Acc: 76.8%  F1: 0.7565  |  Val Loss: 1.0520  Bal.Acc: 61.5%  F1: 0.6256  |  LR: 1.00e-04  (19.0s)
 Best checkpoint saved (val_loss: 1.0520)
Epoch [  8/30]  Train Loss: 0.5173  Bal.Acc: 80.2%  F1: 0.7897  |  Val Loss: 1.0337  Bal.Acc: 67.5%  F1: 0.6907  |  LR: 1.00e-04  (19.0s)
 Best checkpoint saved (val_loss: 1.0337)
Epoch [  9/30]  Train Loss: 0.5040  Bal.Acc: 80.7%  F1: 0.8032  |  Val Loss: 1.0034  Bal.Acc: 67.8%  F1: 0.6965  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 1.0034)
Epoch [ 10/30]  Train Loss: 0.4613  Bal.Acc: 82.7%  F1: 0.8149  |  Val Loss: 0.9799  Bal.Acc: 68.3%  F1: 0.6969  |  LR: 1.00e-04  (18.9s)
 Best checkpoint saved (val_loss: 0.9799)
Epoch [ 11/30]  Train Loss: 0.3950  Bal.Acc: 85.4%  F1: 0.8434  |  Val Loss: 0.9333  Bal.Acc: 68.9%  F1: 0.6992  |  LR: 1.00e-04  (19.0s)
 Best checkpoint saved (val_loss: 0.9333)
Epoch [ 12/30]  Train Loss: 0.3389  Bal.Acc: 88.1%  F1: 0.8708  |  Val Loss: 0.8666  Bal.Acc: 71.5%  F1: 0.7293  |  LR: 1.00e-04  (20.3s)
 Best checkpoint saved (val_loss: 0.8666)
Epoch [ 13/30]  Train Loss: 0.3272  Bal.Acc: 87.6%  F1: 0.8685  |  Val Loss: 0.8391  Bal.Acc: 73.2%  F1: 0.7406  |  LR: 1.00e-04  (19.4s)
 Best checkpoint saved (val_loss: 0.8391)
Epoch [ 14/30]  Train Loss: 0.2952  Bal.Acc: 87.8%  F1: 0.8688  |  Val Loss: 0.9063  Bal.Acc: 73.5%  F1: 0.7501  |  LR: 1.00e-04  (19.2s)
Epoch [ 15/30]  Train Loss: 0.2912  Bal.Acc: 88.7%  F1: 0.8841  |  Val Loss: 0.8073  Bal.Acc: 75.4%  F1: 0.7630  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 0.8073)
Epoch [ 16/30]  Train Loss: 0.2539  Bal.Acc: 91.5%  F1: 0.9061  |  Val Loss: 0.8763  Bal.Acc: 74.0%  F1: 0.7513  |  LR: 1.00e-04  (19.6s)
Epoch [ 17/30]  Train Loss: 0.2133  Bal.Acc: 92.1%  F1: 0.9167  |  Val Loss: 0.8771  Bal.Acc: 72.5%  F1: 0.7347  |  LR: 1.00e-04  (14.9s)
Epoch [ 18/30]  Train Loss: 0.2248  Bal.Acc: 91.9%  F1: 0.9165  |  Val Loss: 0.8262  Bal.Acc: 74.4%  F1: 0.7517  |  LR: 1.00e-04  (14.8s)
Epoch [ 19/30]  Train Loss: 0.1993  Bal.Acc: 92.5%  F1: 0.9196  |  Val Loss: 0.8610  Bal.Acc: 75.5%  F1: 0.7585  |  LR: 5.00e-05  (14.8s)
Epoch [ 20/30]  Train Loss: 0.1723  Bal.Acc: 94.0%  F1: 0.9370  |  Val Loss: 0.8762  Bal.Acc: 74.3%  F1: 0.7547  |  LR: 5.00e-05  (14.9s)
Epoch [ 21/30]  Train Loss: 0.1590  Bal.Acc: 93.5%  F1: 0.9322  |  Val Loss: 0.8483  Bal.Acc: 76.1%  F1: 0.7677  |  LR: 5.00e-05  (14.7s)
Epoch [ 22/30]  Train Loss: 0.1719  Bal.Acc: 93.5%  F1: 0.9309  |  Val Loss: 0.8436  Bal.Acc: 75.8%  F1: 0.7648  |  LR: 5.00e-05  (15.3s)
Epoch [ 23/30]  Train Loss: 0.1536  Bal.Acc: 95.3%  F1: 0.9484  |  Val Loss: 0.8512  Bal.Acc: 77.0%  F1: 0.7826  |  LR: 2.50e-05  (15.3s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.8073

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold2_best.pt
Log CSV: results/logs/efficientnet_b3_fold2_training_log.csv
Best weights loaded from epoch 15

Model evaluation: efficientnet_b3_fold2
----------------------------------------
  Balanced Accuracy:       75.41%
  F1 (macro):              0.7630
  Quadratic Cohen's Kappa: 0.8958
  MAE (ordinal):           0.2847
  Off-by-one accuracy:     96.80%
  ECE:                     0.1252
  Brier Score (mean):      0.0732

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.79      0.90      0.84        88
    Doubtful       0.68      0.63      0.65        81
        Mild       0.56      0.57      0.57        40
    Moderate       0.91      0.81      0.86        37
      Severe       0.94      0.86      0.90        35

    accuracy                           0.76       281
   macro avg       0.78      0.75      0.76       281
weighted avg       0.76      0.76      0.76       281

  Metrics saved to: results/individual_models/efficientnet_b3_fold2_metrics.json
  Probabilities saved to: results/individual_models/efficientnet_b3_fold2_test_probs.npz

--- efficientnet_b3 | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

============================================================
TRAINING: efficientnet_b3_fold3
============================================================
Epoch [  1/30]  Train Loss: 2.6984  Bal.Acc: 30.8%  F1: 0.2911  |  Val Loss: 2.4751  Bal.Acc: 31.7%  F1: 0.3297  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 2.4751)
Epoch [  2/30]  Train Loss: 1.5976  Bal.Acc: 48.2%  F1: 0.4648  |  Val Loss: 1.9025  Bal.Acc: 39.5%  F1: 0.4053  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 1.9025)
Epoch [  3/30]  Train Loss: 1.1706  Bal.Acc: 59.3%  F1: 0.5720  |  Val Loss: 1.5258  Bal.Acc: 52.1%  F1: 0.5137  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 1.5258)
Epoch [  4/30]  Train Loss: 1.0075  Bal.Acc: 63.2%  F1: 0.6166  |  Val Loss: 1.3610  Bal.Acc: 60.5%  F1: 0.6012  |  LR: 1.00e-04  (19.2s)
 Best checkpoint saved (val_loss: 1.3610)
Epoch [  5/30]  Train Loss: 0.8628  Bal.Acc: 68.4%  F1: 0.6678  |  Val Loss: 1.2254  Bal.Acc: 61.4%  F1: 0.6033  |  LR: 1.00e-04  (19.2s)
 Best checkpoint saved (val_loss: 1.2254)
Epoch [  6/30]  Train Loss: 0.7421  Bal.Acc: 71.9%  F1: 0.7045  |  Val Loss: 1.1543  Bal.Acc: 64.1%  F1: 0.6360  |  LR: 1.00e-04  (19.6s)
 Best checkpoint saved (val_loss: 1.1543)
Epoch [  7/30]  Train Loss: 0.6511  Bal.Acc: 76.8%  F1: 0.7586  |  Val Loss: 1.0371  Bal.Acc: 66.2%  F1: 0.6611  |  LR: 1.00e-04  (19.5s)
 Best checkpoint saved (val_loss: 1.0371)
Epoch [  8/30]  Train Loss: 0.5636  Bal.Acc: 78.0%  F1: 0.7720  |  Val Loss: 0.9660  Bal.Acc: 67.4%  F1: 0.6664  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 0.9660)
Epoch [  9/30]  Train Loss: 0.4869  Bal.Acc: 82.4%  F1: 0.8113  |  Val Loss: 0.9455  Bal.Acc: 69.1%  F1: 0.6850  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 0.9455)
Epoch [ 10/30]  Train Loss: 0.4686  Bal.Acc: 83.2%  F1: 0.8239  |  Val Loss: 0.9413  Bal.Acc: 70.0%  F1: 0.6950  |  LR: 1.00e-04  (19.5s)
 Best checkpoint saved (val_loss: 0.9413)
Epoch [ 11/30]  Train Loss: 0.3923  Bal.Acc: 84.8%  F1: 0.8406  |  Val Loss: 0.9091  Bal.Acc: 73.7%  F1: 0.7273  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 0.9091)
Epoch [ 12/30]  Train Loss: 0.3996  Bal.Acc: 85.5%  F1: 0.8456  |  Val Loss: 0.8984  Bal.Acc: 74.5%  F1: 0.7369  |  LR: 1.00e-04  (19.2s)
 Best checkpoint saved (val_loss: 0.8984)
Epoch [ 13/30]  Train Loss: 0.3724  Bal.Acc: 85.7%  F1: 0.8531  |  Val Loss: 0.9362  Bal.Acc: 70.0%  F1: 0.6950  |  LR: 1.00e-04  (19.4s)
Epoch [ 14/30]  Train Loss: 0.2825  Bal.Acc: 89.2%  F1: 0.8848  |  Val Loss: 0.9060  Bal.Acc: 76.0%  F1: 0.7472  |  LR: 1.00e-04  (15.9s)
Epoch [ 15/30]  Train Loss: 0.2930  Bal.Acc: 89.3%  F1: 0.8876  |  Val Loss: 0.9365  Bal.Acc: 75.4%  F1: 0.7450  |  LR: 1.00e-04  (16.4s)
Epoch [ 16/30]  Train Loss: 0.2742  Bal.Acc: 90.1%  F1: 0.8943  |  Val Loss: 0.9200  Bal.Acc: 75.5%  F1: 0.7442  |  LR: 5.00e-05  (15.8s)
Epoch [ 17/30]  Train Loss: 0.2302  Bal.Acc: 90.7%  F1: 0.9018  |  Val Loss: 0.8965  Bal.Acc: 75.9%  F1: 0.7549  |  LR: 5.00e-05  (16.5s)
 Best checkpoint saved (val_loss: 0.8965)
Epoch [ 18/30]  Train Loss: 0.2330  Bal.Acc: 92.0%  F1: 0.9156  |  Val Loss: 0.8512  Bal.Acc: 76.2%  F1: 0.7561  |  LR: 5.00e-05  (18.9s)
 Best checkpoint saved (val_loss: 0.8512)
Epoch [ 19/30]  Train Loss: 0.2396  Bal.Acc: 91.3%  F1: 0.9066  |  Val Loss: 0.8501  Bal.Acc: 77.6%  F1: 0.7673  |  LR: 5.00e-05  (19.3s)
 Best checkpoint saved (val_loss: 0.8501)
Epoch [ 20/30]  Train Loss: 0.2077  Bal.Acc: 92.7%  F1: 0.9194  |  Val Loss: 0.8396  Bal.Acc: 72.8%  F1: 0.7273  |  LR: 5.00e-05  (20.1s)
 Best checkpoint saved (val_loss: 0.8396)
Epoch [ 21/30]  Train Loss: 0.2112  Bal.Acc: 92.8%  F1: 0.9237  |  Val Loss: 0.8560  Bal.Acc: 73.8%  F1: 0.7408  |  LR: 5.00e-05  (19.1s)
Epoch [ 22/30]  Train Loss: 0.1960  Bal.Acc: 92.6%  F1: 0.9223  |  Val Loss: 0.8451  Bal.Acc: 76.4%  F1: 0.7633  |  LR: 5.00e-05  (15.7s)
Epoch [ 23/30]  Train Loss: 0.1730  Bal.Acc: 93.6%  F1: 0.9325  |  Val Loss: 0.8080  Bal.Acc: 77.9%  F1: 0.7768  |  LR: 5.00e-05  (15.9s)
 Best checkpoint saved (val_loss: 0.8080)
Epoch [ 24/30]  Train Loss: 0.1505  Bal.Acc: 94.4%  F1: 0.9417  |  Val Loss: 0.8255  Bal.Acc: 76.7%  F1: 0.7649  |  LR: 5.00e-05  (19.1s)
Epoch [ 25/30]  Train Loss: 0.1868  Bal.Acc: 92.7%  F1: 0.9198  |  Val Loss: 0.8199  Bal.Acc: 75.0%  F1: 0.7507  |  LR: 5.00e-05  (15.9s)
Epoch [ 26/30]  Train Loss: 0.1510  Bal.Acc: 94.5%  F1: 0.9425  |  Val Loss: 0.8158  Bal.Acc: 76.8%  F1: 0.7696  |  LR: 5.00e-05  (15.7s)
Epoch [ 27/30]  Train Loss: 0.1581  Bal.Acc: 94.2%  F1: 0.9412  |  Val Loss: 0.8570  Bal.Acc: 76.8%  F1: 0.7662  |  LR: 2.50e-05  (16.2s)
Epoch [ 28/30]  Train Loss: 0.1579  Bal.Acc: 94.8%  F1: 0.9443  |  Val Loss: 0.8553  Bal.Acc: 77.6%  F1: 0.7723  |  LR: 2.50e-05  (16.1s)
Epoch [ 29/30]  Train Loss: 0.1243  Bal.Acc: 95.2%  F1: 0.9496  |  Val Loss: 0.8644  Bal.Acc: 79.5%  F1: 0.7908  |  LR: 2.50e-05  (15.7s)
Epoch [ 30/30]  Train Loss: 0.1232  Bal.Acc: 95.9%  F1: 0.9551  |  Val Loss: 0.8975  Bal.Acc: 77.6%  F1: 0.7750  |  LR: 2.50e-05  (15.4s)

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold3_best.pt
Log CSV: results/logs/efficientnet_b3_fold3_training_log.csv
Best weights loaded from epoch 23

Model evaluation: efficientnet_b3_fold3
----------------------------------------
  Balanced Accuracy:       77.95%
  F1 (macro):              0.7768
  Quadratic Cohen's Kappa: 0.8851
  MAE (ordinal):           0.2857
  Off-by-one accuracy:     95.00%
  ECE:                     0.1131
  Brier Score (mean):      0.0698

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.80      0.85      0.82        87
    Doubtful       0.74      0.70      0.72        81
        Mild       0.64      0.54      0.58        39
    Moderate       0.86      0.95      0.90        38
      Severe       0.86      0.86      0.86        35

    accuracy                           0.78       280
   macro avg       0.78      0.78      0.78       280
weighted avg       0.77      0.78      0.77       280

  Metrics saved to: results/individual_models/efficientnet_b3_fold3_metrics.json
  Probabilities saved to: results/individual_models/efficientnet_b3_fold3_test_probs.npz

--- efficientnet_b3 | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

============================================================
TRAINING: efficientnet_b3_fold4
============================================================
Epoch [  1/30]  Train Loss: 2.4392  Bal.Acc: 29.7%  F1: 0.2799  |  Val Loss: 2.0799  Bal.Acc: 37.4%  F1: 0.3544  |  LR: 1.00e-04  (15.3s)
 Best checkpoint saved (val_loss: 2.0799)
Epoch [  2/30]  Train Loss: 1.5242  Bal.Acc: 50.1%  F1: 0.4785  |  Val Loss: 1.6451  Bal.Acc: 43.8%  F1: 0.4341  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 1.6451)
Epoch [  3/30]  Train Loss: 1.2300  Bal.Acc: 58.0%  F1: 0.5602  |  Val Loss: 1.3534  Bal.Acc: 49.6%  F1: 0.4938  |  LR: 1.00e-04  (19.2s)
 Best checkpoint saved (val_loss: 1.3534)
Epoch [  4/30]  Train Loss: 1.0068  Bal.Acc: 64.0%  F1: 0.6265  |  Val Loss: 1.2364  Bal.Acc: 55.5%  F1: 0.5645  |  LR: 1.00e-04  (20.1s)
 Best checkpoint saved (val_loss: 1.2364)
Epoch [  5/30]  Train Loss: 0.8140  Bal.Acc: 71.3%  F1: 0.6987  |  Val Loss: 1.0420  Bal.Acc: 59.9%  F1: 0.5945  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 1.0420)
Epoch [  6/30]  Train Loss: 0.7285  Bal.Acc: 73.5%  F1: 0.7190  |  Val Loss: 1.0382  Bal.Acc: 65.6%  F1: 0.6554  |  LR: 1.00e-04  (19.5s)
 Best checkpoint saved (val_loss: 1.0382)
Epoch [  7/30]  Train Loss: 0.6621  Bal.Acc: 76.2%  F1: 0.7519  |  Val Loss: 0.9007  Bal.Acc: 72.5%  F1: 0.7152  |  LR: 1.00e-04  (20.1s)
 Best checkpoint saved (val_loss: 0.9007)
Epoch [  8/30]  Train Loss: 0.5592  Bal.Acc: 78.6%  F1: 0.7749  |  Val Loss: 0.9068  Bal.Acc: 70.6%  F1: 0.6996  |  LR: 1.00e-04  (19.3s)
Epoch [  9/30]  Train Loss: 0.4950  Bal.Acc: 81.4%  F1: 0.8036  |  Val Loss: 0.8686  Bal.Acc: 70.2%  F1: 0.7006  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 0.8686)
Epoch [ 10/30]  Train Loss: 0.4314  Bal.Acc: 83.7%  F1: 0.8303  |  Val Loss: 0.7687  Bal.Acc: 75.3%  F1: 0.7493  |  LR: 1.00e-04  (20.0s)
 Best checkpoint saved (val_loss: 0.7687)
Epoch [ 11/30]  Train Loss: 0.4148  Bal.Acc: 85.2%  F1: 0.8413  |  Val Loss: 0.8204  Bal.Acc: 73.2%  F1: 0.7222  |  LR: 1.00e-04  (20.0s)
Epoch [ 12/30]  Train Loss: 0.3657  Bal.Acc: 86.9%  F1: 0.8602  |  Val Loss: 0.8080  Bal.Acc: 73.2%  F1: 0.7215  |  LR: 1.00e-04  (15.6s)
Epoch [ 13/30]  Train Loss: 0.3378  Bal.Acc: 86.6%  F1: 0.8596  |  Val Loss: 0.7111  Bal.Acc: 75.4%  F1: 0.7515  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 0.7111)
Epoch [ 14/30]  Train Loss: 0.3110  Bal.Acc: 88.8%  F1: 0.8806  |  Val Loss: 0.7149  Bal.Acc: 74.9%  F1: 0.7488  |  LR: 1.00e-04  (19.3s)
Epoch [ 15/30]  Train Loss: 0.3083  Bal.Acc: 89.2%  F1: 0.8876  |  Val Loss: 0.7414  Bal.Acc: 74.5%  F1: 0.7482  |  LR: 1.00e-04  (15.9s)
Epoch [ 16/30]  Train Loss: 0.2952  Bal.Acc: 88.6%  F1: 0.8803  |  Val Loss: 0.6715  Bal.Acc: 77.9%  F1: 0.7755  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 0.6715)
Epoch [ 17/30]  Train Loss: 0.2316  Bal.Acc: 91.8%  F1: 0.9121  |  Val Loss: 0.7371  Bal.Acc: 76.7%  F1: 0.7691  |  LR: 1.00e-04  (19.7s)
Epoch [ 18/30]  Train Loss: 0.2205  Bal.Acc: 92.0%  F1: 0.9154  |  Val Loss: 0.7240  Bal.Acc: 76.3%  F1: 0.7676  |  LR: 1.00e-04  (15.1s)
Epoch [ 19/30]  Train Loss: 0.1916  Bal.Acc: 92.5%  F1: 0.9215  |  Val Loss: 0.6264  Bal.Acc: 77.8%  F1: 0.7784  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 0.6264)
Epoch [ 20/30]  Train Loss: 0.2031  Bal.Acc: 92.9%  F1: 0.9234  |  Val Loss: 0.6986  Bal.Acc: 78.5%  F1: 0.7918  |  LR: 1.00e-04  (19.5s)
Epoch [ 21/30]  Train Loss: 0.1673  Bal.Acc: 93.6%  F1: 0.9313  |  Val Loss: 0.6882  Bal.Acc: 78.2%  F1: 0.7885  |  LR: 1.00e-04  (16.2s)
Epoch [ 22/30]  Train Loss: 0.1502  Bal.Acc: 94.8%  F1: 0.9443  |  Val Loss: 0.6801  Bal.Acc: 81.2%  F1: 0.8053  |  LR: 1.00e-04  (16.1s)
Epoch [ 23/30]  Train Loss: 0.1587  Bal.Acc: 94.7%  F1: 0.9434  |  Val Loss: 0.7030  Bal.Acc: 78.9%  F1: 0.7913  |  LR: 5.00e-05  (16.3s)
Epoch [ 24/30]  Train Loss: 0.1356  Bal.Acc: 95.4%  F1: 0.9512  |  Val Loss: 0.7041  Bal.Acc: 78.6%  F1: 0.7876  |  LR: 5.00e-05  (16.7s)
Epoch [ 25/30]  Train Loss: 0.1176  Bal.Acc: 96.0%  F1: 0.9590  |  Val Loss: 0.7138  Bal.Acc: 79.5%  F1: 0.7987  |  LR: 5.00e-05  (16.5s)
Epoch [ 26/30]  Train Loss: 0.1124  Bal.Acc: 95.7%  F1: 0.9568  |  Val Loss: 0.7249  Bal.Acc: 77.7%  F1: 0.7830  |  LR: 5.00e-05  (16.1s)
Epoch [ 27/30]  Train Loss: 0.1190  Bal.Acc: 95.5%  F1: 0.9538  |  Val Loss: 0.7263  Bal.Acc: 79.9%  F1: 0.7985  |  LR: 2.50e-05  (15.3s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.6264

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold4_best.pt
Log CSV: results/logs/efficientnet_b3_fold4_training_log.csv
Best weights loaded from epoch 19

Model evaluation: efficientnet_b3_fold4
----------------------------------------
  Balanced Accuracy:       77.83%
  F1 (macro):              0.7784
  Quadratic Cohen's Kappa: 0.9085
  MAE (ordinal):           0.2500
  Off-by-one accuracy:     97.50%
  ECE:                     0.1097
  Brier Score (mean):      0.0620

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.89      0.89        87
    Doubtful       0.72      0.72      0.72        81
        Mild       0.55      0.54      0.55        39
    Moderate       0.81      0.89      0.85        38
      Severe       0.94      0.86      0.90        35

    accuracy                           0.79       280
   macro avg       0.78      0.78      0.78       280
weighted avg       0.79      0.79      0.79       280

  Metrics saved to: results/individual_models/efficientnet_b3_fold4_metrics.json
  Probabilities saved to: results/individual_models/efficientnet_b3_fold4_test_probs.npz

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

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable

============================================================
TRAINING: efficientnet_b3_fold5
============================================================
Epoch [  1/30]  Train Loss: 2.7402  Bal.Acc: 25.8%  F1: 0.2470  |  Val Loss: 2.4251  Bal.Acc: 34.9%  F1: 0.2778  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 2.4251)
Epoch [  2/30]  Train Loss: 1.6454  Bal.Acc: 45.3%  F1: 0.4270  |  Val Loss: 1.7863  Bal.Acc: 41.1%  F1: 0.3616  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 1.7863)
Epoch [  3/30]  Train Loss: 1.2325  Bal.Acc: 59.1%  F1: 0.5738  |  Val Loss: 1.5736  Bal.Acc: 46.8%  F1: 0.4338  |  LR: 1.00e-04  (20.6s)
 Best checkpoint saved (val_loss: 1.5736)
Epoch [  4/30]  Train Loss: 1.0162  Bal.Acc: 64.6%  F1: 0.6269  |  Val Loss: 1.3029  Bal.Acc: 57.3%  F1: 0.5581  |  LR: 1.00e-04  (19.8s)
 Best checkpoint saved (val_loss: 1.3029)
Epoch [  5/30]  Train Loss: 0.8023  Bal.Acc: 70.4%  F1: 0.6872  |  Val Loss: 1.0675  Bal.Acc: 64.1%  F1: 0.6285  |  LR: 1.00e-04  (19.4s)
 Best checkpoint saved (val_loss: 1.0675)
Epoch [  6/30]  Train Loss: 0.7527  Bal.Acc: 72.6%  F1: 0.7170  |  Val Loss: 0.9599  Bal.Acc: 67.0%  F1: 0.6582  |  LR: 1.00e-04  (20.2s)
 Best checkpoint saved (val_loss: 0.9599)
Epoch [  7/30]  Train Loss: 0.6342  Bal.Acc: 76.0%  F1: 0.7437  |  Val Loss: 0.9174  Bal.Acc: 70.4%  F1: 0.6969  |  LR: 1.00e-04  (19.2s)
 Best checkpoint saved (val_loss: 0.9174)
Epoch [  8/30]  Train Loss: 0.5554  Bal.Acc: 77.8%  F1: 0.7683  |  Val Loss: 0.8085  Bal.Acc: 71.7%  F1: 0.7069  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 0.8085)
Epoch [  9/30]  Train Loss: 0.4869  Bal.Acc: 81.2%  F1: 0.8049  |  Val Loss: 0.7792  Bal.Acc: 73.3%  F1: 0.7261  |  LR: 1.00e-04  (20.4s)
 Best checkpoint saved (val_loss: 0.7792)
Epoch [ 10/30]  Train Loss: 0.4728  Bal.Acc: 82.8%  F1: 0.8169  |  Val Loss: 0.7979  Bal.Acc: 72.9%  F1: 0.7192  |  LR: 1.00e-04  (19.2s)
Epoch [ 11/30]  Train Loss: 0.4156  Bal.Acc: 83.3%  F1: 0.8214  |  Val Loss: 0.7849  Bal.Acc: 74.2%  F1: 0.7322  |  LR: 1.00e-04  (16.2s)
Epoch [ 12/30]  Train Loss: 0.3359  Bal.Acc: 87.8%  F1: 0.8711  |  Val Loss: 0.7907  Bal.Acc: 76.4%  F1: 0.7566  |  LR: 1.00e-04  (16.4s)
Epoch [ 13/30]  Train Loss: 0.3365  Bal.Acc: 87.8%  F1: 0.8709  |  Val Loss: 0.7637  Bal.Acc: 78.6%  F1: 0.7736  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 0.7637)
Epoch [ 14/30]  Train Loss: 0.2924  Bal.Acc: 89.4%  F1: 0.8858  |  Val Loss: 0.7658  Bal.Acc: 77.9%  F1: 0.7696  |  LR: 1.00e-04  (20.6s)
Epoch [ 15/30]  Train Loss: 0.2660  Bal.Acc: 89.6%  F1: 0.8882  |  Val Loss: 0.7752  Bal.Acc: 76.3%  F1: 0.7612  |  LR: 1.00e-04  (16.8s)
Epoch [ 16/30]  Train Loss: 0.2635  Bal.Acc: 91.0%  F1: 0.9043  |  Val Loss: 0.7291  Bal.Acc: 78.8%  F1: 0.7869  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 0.7291)
Epoch [ 17/30]  Train Loss: 0.2239  Bal.Acc: 91.9%  F1: 0.9170  |  Val Loss: 0.7358  Bal.Acc: 80.4%  F1: 0.8017  |  LR: 1.00e-04  (19.7s)
Epoch [ 18/30]  Train Loss: 0.2436  Bal.Acc: 91.3%  F1: 0.9086  |  Val Loss: 0.7706  Bal.Acc: 78.5%  F1: 0.7894  |  LR: 1.00e-04  (16.5s)
Epoch [ 19/30]  Train Loss: 0.2094  Bal.Acc: 93.3%  F1: 0.9242  |  Val Loss: 0.7605  Bal.Acc: 79.7%  F1: 0.7962  |  LR: 1.00e-04  (16.3s)
Epoch [ 20/30]  Train Loss: 0.1990  Bal.Acc: 92.6%  F1: 0.9206  |  Val Loss: 0.7974  Bal.Acc: 80.0%  F1: 0.7987  |  LR: 5.00e-05  (16.6s)
Epoch [ 21/30]  Train Loss: 0.2051  Bal.Acc: 92.1%  F1: 0.9176  |  Val Loss: 0.7839  Bal.Acc: 78.3%  F1: 0.7805  |  LR: 5.00e-05  (16.1s)
Epoch [ 22/30]  Train Loss: 0.1689  Bal.Acc: 94.1%  F1: 0.9384  |  Val Loss: 0.7563  Bal.Acc: 78.8%  F1: 0.7856  |  LR: 5.00e-05  (16.7s)
Epoch [ 23/30]  Train Loss: 0.1654  Bal.Acc: 94.3%  F1: 0.9375  |  Val Loss: 0.7580  Bal.Acc: 80.6%  F1: 0.8021  |  LR: 5.00e-05  (16.4s)
Epoch [ 24/30]  Train Loss: 0.1494  Bal.Acc: 94.5%  F1: 0.9404  |  Val Loss: 0.7435  Bal.Acc: 80.2%  F1: 0.8016  |  LR: 2.50e-05  (16.6s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.7291

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold5_best.pt
Log CSV: results/logs/efficientnet_b3_fold5_training_log.csv
Best weights loaded from epoch 16

Model evaluation: efficientnet_b3_fold5
----------------------------------------
  Balanced Accuracy:       78.80%
  F1 (macro):              0.7869
  Quadratic Cohen's Kappa: 0.8957
  MAE (ordinal):           0.2643
  Off-by-one accuracy:     96.79%
  ECE:                     0.1009
  Brier Score (mean):      0.0663

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.80      0.90      0.84        87
    Doubtful       0.75      0.65      0.70        81
        Mild       0.75      0.69      0.72        39
    Moderate       0.87      0.87      0.87        38
      Severe       0.78      0.83      0.81        35

    accuracy                           0.79       280
   macro avg       0.79      0.79      0.79       280
weighted avg       0.78      0.79      0.78       280

  Metrics saved to: results/individual_models/efficientnet_b3_fold5_metrics.json
  Probabilities saved to: results/individual_models/efficientnet_b3_fold5_test_probs.npz

 FINISHED: efficientnet_b3. Average kappa out of 5 folds: 0.8425 ±0.1078

================================================================================
MODEL TRAINING START: densenet121
================================================================================

--- densenet121 | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: densenet121
model.safetensors: 100% 32.3M/32.3M [00:01<00:00, 22.2MB/s]
  Parameters: 6,958,981 total, 6,958,981 trainable

============================================================
TRAINING: densenet121_fold1
============================================================
Epoch [  1/30]  Train Loss: 1.5239  Bal.Acc: 32.7%  F1: 0.2849  |  Val Loss: 1.4806  Bal.Acc: 33.3%  F1: 0.2820  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 1.4806)
Epoch [  2/30]  Train Loss: 1.2342  Bal.Acc: 52.3%  F1: 0.4708  |  Val Loss: 1.2405  Bal.Acc: 48.5%  F1: 0.4261  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 1.2405)
Epoch [  3/30]  Train Loss: 1.0248  Bal.Acc: 63.8%  F1: 0.6129  |  Val Loss: 1.0303  Bal.Acc: 62.5%  F1: 0.6056  |  LR: 1.00e-04  (18.6s)
 Best checkpoint saved (val_loss: 1.0303)
Epoch [  4/30]  Train Loss: 0.8337  Bal.Acc: 71.0%  F1: 0.6931  |  Val Loss: 0.8260  Bal.Acc: 70.6%  F1: 0.7022  |  LR: 1.00e-04  (19.3s)
 Best checkpoint saved (val_loss: 0.8260)
Epoch [  5/30]  Train Loss: 0.6926  Bal.Acc: 75.4%  F1: 0.7391  |  Val Loss: 0.6622  Bal.Acc: 77.8%  F1: 0.7617  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.6622)
Epoch [  6/30]  Train Loss: 0.5822  Bal.Acc: 78.7%  F1: 0.7773  |  Val Loss: 0.6682  Bal.Acc: 75.7%  F1: 0.7395  |  LR: 1.00e-04  (19.1s)
Epoch [  7/30]  Train Loss: 0.5000  Bal.Acc: 83.8%  F1: 0.8266  |  Val Loss: 0.6065  Bal.Acc: 81.6%  F1: 0.8158  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 0.6065)
Epoch [  8/30]  Train Loss: 0.4240  Bal.Acc: 84.7%  F1: 0.8381  |  Val Loss: 0.5084  Bal.Acc: 83.2%  F1: 0.8204  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 0.5084)
Epoch [  9/30]  Train Loss: 0.3875  Bal.Acc: 86.2%  F1: 0.8539  |  Val Loss: 0.4979  Bal.Acc: 81.5%  F1: 0.8040  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 0.4979)
Epoch [ 10/30]  Train Loss: 0.3425  Bal.Acc: 88.1%  F1: 0.8751  |  Val Loss: 0.4610  Bal.Acc: 83.6%  F1: 0.8248  |  LR: 1.00e-04  (18.7s)
 Best checkpoint saved (val_loss: 0.4610)
Epoch [ 11/30]  Train Loss: 0.3234  Bal.Acc: 89.5%  F1: 0.8915  |  Val Loss: 0.4364  Bal.Acc: 85.8%  F1: 0.8425  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.4364)
Epoch [ 12/30]  Train Loss: 0.3142  Bal.Acc: 89.0%  F1: 0.8807  |  Val Loss: 0.5120  Bal.Acc: 79.4%  F1: 0.8029  |  LR: 1.00e-04  (18.4s)
Epoch [ 13/30]  Train Loss: 0.2628  Bal.Acc: 91.7%  F1: 0.9126  |  Val Loss: 0.4051  Bal.Acc: 88.7%  F1: 0.8746  |  LR: 1.00e-04  (15.2s)
 Best checkpoint saved (val_loss: 0.4051)
Epoch [ 14/30]  Train Loss: 0.2596  Bal.Acc: 90.6%  F1: 0.9044  |  Val Loss: 0.4293  Bal.Acc: 83.7%  F1: 0.8297  |  LR: 1.00e-04  (18.0s)
Epoch [ 15/30]  Train Loss: 0.2063  Bal.Acc: 93.6%  F1: 0.9295  |  Val Loss: 0.4140  Bal.Acc: 86.8%  F1: 0.8629  |  LR: 1.00e-04  (16.3s)
Epoch [ 16/30]  Train Loss: 0.1867  Bal.Acc: 94.1%  F1: 0.9389  |  Val Loss: 0.4163  Bal.Acc: 84.8%  F1: 0.8299  |  LR: 1.00e-04  (16.3s)
Epoch [ 17/30]  Train Loss: 0.1737  Bal.Acc: 94.8%  F1: 0.9424  |  Val Loss: 0.3250  Bal.Acc: 89.8%  F1: 0.8966  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 0.3250)
Epoch [ 18/30]  Train Loss: 0.1567  Bal.Acc: 94.7%  F1: 0.9472  |  Val Loss: 0.4014  Bal.Acc: 87.3%  F1: 0.8654  |  LR: 1.00e-04  (18.5s)
Epoch [ 19/30]  Train Loss: 0.1581  Bal.Acc: 95.4%  F1: 0.9489  |  Val Loss: 0.4293  Bal.Acc: 85.7%  F1: 0.8528  |  LR: 1.00e-04  (16.8s)
Epoch [ 20/30]  Train Loss: 0.1506  Bal.Acc: 95.1%  F1: 0.9483  |  Val Loss: 0.3911  Bal.Acc: 87.5%  F1: 0.8650  |  LR: 1.00e-04  (16.6s)
Epoch [ 21/30]  Train Loss: 0.1324  Bal.Acc: 95.9%  F1: 0.9556  |  Val Loss: 0.3954  Bal.Acc: 86.2%  F1: 0.8555  |  LR: 5.00e-05  (16.0s)
Epoch [ 22/30]  Train Loss: 0.1146  Bal.Acc: 96.5%  F1: 0.9618  |  Val Loss: 0.3376  Bal.Acc: 87.8%  F1: 0.8757  |  LR: 5.00e-05  (15.6s)
Epoch [ 23/30]  Train Loss: 0.0953  Bal.Acc: 97.4%  F1: 0.9725  |  Val Loss: 0.3697  Bal.Acc: 86.8%  F1: 0.8601  |  LR: 5.00e-05  (15.5s)
Epoch [ 24/30]  Train Loss: 0.1001  Bal.Acc: 97.0%  F1: 0.9655  |  Val Loss: 0.3803  Bal.Acc: 86.2%  F1: 0.8597  |  LR: 5.00e-05  (15.4s)
Epoch [ 25/30]  Train Loss: 0.0862  Bal.Acc: 97.5%  F1: 0.9739  |  Val Loss: 0.3718  Bal.Acc: 85.6%  F1: 0.8570  |  LR: 2.50e-05  (15.5s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.3250

 Training finished. Checkpoint: checkpoints/densenet121_fold1_best.pt
Log CSV: results/logs/densenet121_fold1_training_log.csv
Best weights loaded from epoch 17

Model evaluation: densenet121_fold1
----------------------------------------
  Balanced Accuracy:       89.81%
  F1 (macro):              0.8966
  Quadratic Cohen's Kappa: 0.9517
  MAE (ordinal):           0.1281
  Off-by-one accuracy:     97.15%
  ECE:                     0.0396
  Brier Score (mean):      0.0322

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.97      0.93        88
    Doubtful       0.93      0.84      0.88        81
        Mild       0.85      0.82      0.84        40
    Moderate       0.90      0.95      0.92        37
      Severe       0.91      0.91      0.91        35

    accuracy                           0.90       281
   macro avg       0.90      0.90      0.90       281
weighted avg       0.90      0.90      0.90       281

  Metrics saved to: results/individual_models/densenet121_fold1_metrics.json
  Probabilities saved to: results/individual_models/densenet121_fold1_test_probs.npz

--- densenet121 | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

============================================================
TRAINING: densenet121_fold2
============================================================
Epoch [  1/30]  Train Loss: 1.4801  Bal.Acc: 38.4%  F1: 0.3411  |  Val Loss: 1.4484  Bal.Acc: 30.2%  F1: 0.2761  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 1.4484)
Epoch [  2/30]  Train Loss: 1.2145  Bal.Acc: 52.6%  F1: 0.4878  |  Val Loss: 1.1017  Bal.Acc: 56.9%  F1: 0.5796  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 1.1017)
Epoch [  3/30]  Train Loss: 0.9748  Bal.Acc: 62.8%  F1: 0.6035  |  Val Loss: 0.9886  Bal.Acc: 59.0%  F1: 0.6048  |  LR: 1.00e-04  (19.2s)
 Best checkpoint saved (val_loss: 0.9886)
Epoch [  4/30]  Train Loss: 0.8055  Bal.Acc: 70.6%  F1: 0.6935  |  Val Loss: 0.8484  Bal.Acc: 63.8%  F1: 0.6493  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 0.8484)
Epoch [  5/30]  Train Loss: 0.6538  Bal.Acc: 78.0%  F1: 0.7735  |  Val Loss: 0.7542  Bal.Acc: 67.2%  F1: 0.6828  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 0.7542)
Epoch [  6/30]  Train Loss: 0.5710  Bal.Acc: 80.2%  F1: 0.7950  |  Val Loss: 0.6715  Bal.Acc: 72.4%  F1: 0.7371  |  LR: 1.00e-04  (19.0s)
 Best checkpoint saved (val_loss: 0.6715)
Epoch [  7/30]  Train Loss: 0.4723  Bal.Acc: 83.3%  F1: 0.8262  |  Val Loss: 0.5827  Bal.Acc: 75.1%  F1: 0.7522  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.5827)
Epoch [  8/30]  Train Loss: 0.4408  Bal.Acc: 85.0%  F1: 0.8442  |  Val Loss: 0.6772  Bal.Acc: 74.4%  F1: 0.7318  |  LR: 1.00e-04  (18.1s)
Epoch [  9/30]  Train Loss: 0.3784  Bal.Acc: 87.0%  F1: 0.8580  |  Val Loss: 0.7062  Bal.Acc: 75.2%  F1: 0.7693  |  LR: 1.00e-04  (15.7s)
Epoch [ 10/30]  Train Loss: 0.3250  Bal.Acc: 89.6%  F1: 0.8917  |  Val Loss: 0.5694  Bal.Acc: 79.0%  F1: 0.7843  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 0.5694)
Epoch [ 11/30]  Train Loss: 0.2905  Bal.Acc: 90.7%  F1: 0.9050  |  Val Loss: 0.6270  Bal.Acc: 74.9%  F1: 0.7650  |  LR: 1.00e-04  (18.1s)
Epoch [ 12/30]  Train Loss: 0.2512  Bal.Acc: 91.6%  F1: 0.9125  |  Val Loss: 0.5249  Bal.Acc: 81.0%  F1: 0.8175  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 0.5249)
Epoch [ 13/30]  Train Loss: 0.2250  Bal.Acc: 93.6%  F1: 0.9327  |  Val Loss: 0.5085  Bal.Acc: 80.1%  F1: 0.8074  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.5085)
Epoch [ 14/30]  Train Loss: 0.1996  Bal.Acc: 93.2%  F1: 0.9290  |  Val Loss: 0.5075  Bal.Acc: 81.5%  F1: 0.8210  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.5075)
Epoch [ 15/30]  Train Loss: 0.2174  Bal.Acc: 92.1%  F1: 0.9145  |  Val Loss: 0.9316  Bal.Acc: 65.8%  F1: 0.6836  |  LR: 1.00e-04  (18.7s)
Epoch [ 16/30]  Train Loss: 0.1988  Bal.Acc: 93.9%  F1: 0.9388  |  Val Loss: 0.5389  Bal.Acc: 81.0%  F1: 0.8091  |  LR: 1.00e-04  (16.5s)
Epoch [ 17/30]  Train Loss: 0.1706  Bal.Acc: 94.4%  F1: 0.9393  |  Val Loss: 0.5747  Bal.Acc: 80.0%  F1: 0.8127  |  LR: 1.00e-04  (16.5s)
Epoch [ 18/30]  Train Loss: 0.1446  Bal.Acc: 95.3%  F1: 0.9484  |  Val Loss: 0.5810  Bal.Acc: 80.4%  F1: 0.8216  |  LR: 5.00e-05  (16.3s)
Epoch [ 19/30]  Train Loss: 0.1307  Bal.Acc: 96.1%  F1: 0.9599  |  Val Loss: 0.5334  Bal.Acc: 81.5%  F1: 0.8198  |  LR: 5.00e-05  (16.5s)
Epoch [ 20/30]  Train Loss: 0.1156  Bal.Acc: 96.6%  F1: 0.9635  |  Val Loss: 0.4933  Bal.Acc: 84.1%  F1: 0.8379  |  LR: 5.00e-05  (16.2s)
 Best checkpoint saved (val_loss: 0.4933)
Epoch [ 21/30]  Train Loss: 0.1104  Bal.Acc: 96.8%  F1: 0.9669  |  Val Loss: 0.5095  Bal.Acc: 85.3%  F1: 0.8564  |  LR: 5.00e-05  (18.2s)
Epoch [ 22/30]  Train Loss: 0.1077  Bal.Acc: 96.7%  F1: 0.9636  |  Val Loss: 0.5409  Bal.Acc: 83.5%  F1: 0.8488  |  LR: 5.00e-05  (16.0s)
Epoch [ 23/30]  Train Loss: 0.1176  Bal.Acc: 96.0%  F1: 0.9573  |  Val Loss: 0.5178  Bal.Acc: 84.1%  F1: 0.8488  |  LR: 5.00e-05  (16.1s)
Epoch [ 24/30]  Train Loss: 0.0972  Bal.Acc: 97.5%  F1: 0.9723  |  Val Loss: 0.5545  Bal.Acc: 84.3%  F1: 0.8509  |  LR: 2.50e-05  (16.4s)
Epoch [ 25/30]  Train Loss: 0.1003  Bal.Acc: 97.0%  F1: 0.9689  |  Val Loss: 0.5467  Bal.Acc: 84.7%  F1: 0.8527  |  LR: 2.50e-05  (16.4s)
Epoch [ 26/30]  Train Loss: 0.0788  Bal.Acc: 97.8%  F1: 0.9754  |  Val Loss: 0.5283  Bal.Acc: 86.3%  F1: 0.8683  |  LR: 2.50e-05  (15.8s)
Epoch [ 27/30]  Train Loss: 0.0871  Bal.Acc: 97.4%  F1: 0.9721  |  Val Loss: 0.5349  Bal.Acc: 85.6%  F1: 0.8609  |  LR: 2.50e-05  (16.2s)
Epoch [ 28/30]  Train Loss: 0.0729  Bal.Acc: 98.0%  F1: 0.9783  |  Val Loss: 0.5529  Bal.Acc: 85.1%  F1: 0.8609  |  LR: 1.25e-05  (16.3s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.4933

 Training finished. Checkpoint: checkpoints/densenet121_fold2_best.pt
Log CSV: results/logs/densenet121_fold2_training_log.csv
Best weights loaded from epoch 20

Model evaluation: densenet121_fold2
----------------------------------------
  Balanced Accuracy:       84.11%
  F1 (macro):              0.8379
  Quadratic Cohen's Kappa: 0.9320
  MAE (ordinal):           0.1957
  Off-by-one accuracy:     97.86%
  ECE:                     0.0573
  Brier Score (mean):      0.0528

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.92      0.82      0.87        88
    Doubtful       0.75      0.80      0.77        81
        Mild       0.68      0.75      0.71        40
    Moderate       0.92      0.89      0.90        37
      Severe       0.92      0.94      0.93        35

    accuracy                           0.83       281
   macro avg       0.84      0.84      0.84       281
weighted avg       0.84      0.83      0.83       281

  Metrics saved to: results/individual_models/densenet121_fold2_metrics.json
  Probabilities saved to: results/individual_models/densenet121_fold2_test_probs.npz

--- densenet121 | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: densenet121
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  Parameters: 6,958,981 total, 6,958,981 trainable

============================================================
TRAINING: densenet121_fold3
============================================================
Epoch [  1/30]  Train Loss: 1.5142  Bal.Acc: 34.0%  F1: 0.3092  |  Val Loss: 1.4796  Bal.Acc: 36.4%  F1: 0.3120  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 1.4796)
Epoch [  2/30]  Train Loss: 1.2450  Bal.Acc: 53.0%  F1: 0.4756  |  Val Loss: 1.3138  Bal.Acc: 47.2%  F1: 0.4316  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 1.3138)
Epoch [  3/30]  Train Loss: 0.9947  Bal.Acc: 64.6%  F1: 0.6167  |  Val Loss: 1.1712  Bal.Acc: 53.0%  F1: 0.5253  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 1.1712)
Epoch [  4/30]  Train Loss: 0.7885  Bal.Acc: 73.7%  F1: 0.7209  |  Val Loss: 0.9239  Bal.Acc: 64.0%  F1: 0.6388  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.9239)
Epoch [  5/30]  Train Loss: 0.6497  Bal.Acc: 78.7%  F1: 0.7748  |  Val Loss: 0.8738  Bal.Acc: 64.0%  F1: 0.6278  |  LR: 1.00e-04  (19.0s)
 Best checkpoint saved (val_loss: 0.8738)
Epoch [  6/30]  Train Loss: 0.5610  Bal.Acc: 80.2%  F1: 0.7924  |  Val Loss: 0.7643  Bal.Acc: 71.6%  F1: 0.7227  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 0.7643)
Epoch [  7/30]  Train Loss: 0.4979  Bal.Acc: 82.1%  F1: 0.8173  |  Val Loss: 0.7011  Bal.Acc: 73.6%  F1: 0.7034  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.7011)
Epoch [  8/30]  Train Loss: 0.4594  Bal.Acc: 83.1%  F1: 0.8179  |  Val Loss: 0.6840  Bal.Acc: 72.3%  F1: 0.7268  |  LR: 1.00e-04  (18.1s)
 Best checkpoint saved (val_loss: 0.6840)
Epoch [  9/30]  Train Loss: 0.3874  Bal.Acc: 85.8%  F1: 0.8496  |  Val Loss: 0.5963  Bal.Acc: 76.6%  F1: 0.7601  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.5963)
Epoch [ 10/30]  Train Loss: 0.3230  Bal.Acc: 89.0%  F1: 0.8880  |  Val Loss: 0.6286  Bal.Acc: 77.4%  F1: 0.7678  |  LR: 1.00e-04  (18.3s)
Epoch [ 11/30]  Train Loss: 0.3131  Bal.Acc: 89.3%  F1: 0.8871  |  Val Loss: 0.5434  Bal.Acc: 78.4%  F1: 0.7797  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 0.5434)
Epoch [ 12/30]  Train Loss: 0.2684  Bal.Acc: 90.7%  F1: 0.8992  |  Val Loss: 0.7974  Bal.Acc: 72.7%  F1: 0.7392  |  LR: 1.00e-04  (18.4s)
Epoch [ 13/30]  Train Loss: 0.2516  Bal.Acc: 91.7%  F1: 0.9100  |  Val Loss: 0.5790  Bal.Acc: 78.6%  F1: 0.7893  |  LR: 1.00e-04  (16.9s)
Epoch [ 14/30]  Train Loss: 0.2075  Bal.Acc: 94.0%  F1: 0.9360  |  Val Loss: 0.6284  Bal.Acc: 78.5%  F1: 0.7861  |  LR: 1.00e-04  (16.3s)
Epoch [ 15/30]  Train Loss: 0.1919  Bal.Acc: 94.1%  F1: 0.9380  |  Val Loss: 0.6166  Bal.Acc: 76.9%  F1: 0.7685  |  LR: 5.00e-05  (15.8s)
Epoch [ 16/30]  Train Loss: 0.1883  Bal.Acc: 93.2%  F1: 0.9283  |  Val Loss: 0.5720  Bal.Acc: 80.8%  F1: 0.8155  |  LR: 5.00e-05  (15.2s)
Epoch [ 17/30]  Train Loss: 0.1572  Bal.Acc: 94.7%  F1: 0.9453  |  Val Loss: 0.5638  Bal.Acc: 80.7%  F1: 0.8105  |  LR: 5.00e-05  (15.0s)
Epoch [ 18/30]  Train Loss: 0.1492  Bal.Acc: 94.8%  F1: 0.9460  |  Val Loss: 0.5315  Bal.Acc: 80.3%  F1: 0.7968  |  LR: 5.00e-05  (15.2s)
 Best checkpoint saved (val_loss: 0.5315)
Epoch [ 19/30]  Train Loss: 0.1529  Bal.Acc: 95.3%  F1: 0.9487  |  Val Loss: 0.5588  Bal.Acc: 82.1%  F1: 0.8200  |  LR: 5.00e-05  (18.1s)
Epoch [ 20/30]  Train Loss: 0.1417  Bal.Acc: 95.2%  F1: 0.9479  |  Val Loss: 0.6020  Bal.Acc: 82.1%  F1: 0.8284  |  LR: 5.00e-05  (15.9s)
Epoch [ 21/30]  Train Loss: 0.1398  Bal.Acc: 95.8%  F1: 0.9558  |  Val Loss: 0.5765  Bal.Acc: 80.3%  F1: 0.8076  |  LR: 5.00e-05  (16.0s)
Epoch [ 22/30]  Train Loss: 0.1218  Bal.Acc: 96.5%  F1: 0.9631  |  Val Loss: 0.5620  Bal.Acc: 81.9%  F1: 0.8197  |  LR: 2.50e-05  (16.1s)
Epoch [ 23/30]  Train Loss: 0.1184  Bal.Acc: 96.0%  F1: 0.9565  |  Val Loss: 0.5720  Bal.Acc: 81.6%  F1: 0.8172  |  LR: 2.50e-05  (16.1s)
Epoch [ 24/30]  Train Loss: 0.1068  Bal.Acc: 97.2%  F1: 0.9696  |  Val Loss: 0.5792  Bal.Acc: 82.1%  F1: 0.8236  |  LR: 2.50e-05  (17.0s)
Epoch [ 25/30]  Train Loss: 0.1205  Bal.Acc: 95.8%  F1: 0.9557  |  Val Loss: 0.5908  Bal.Acc: 80.8%  F1: 0.8117  |  LR: 2.50e-05  (17.3s)
Epoch [ 26/30]  Train Loss: 0.1131  Bal.Acc: 96.4%  F1: 0.9598  |  Val Loss: 0.5491  Bal.Acc: 82.2%  F1: 0.8236  |  LR: 1.25e-05  (16.1s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.5315

 Training finished. Checkpoint: checkpoints/densenet121_fold3_best.pt
Log CSV: results/logs/densenet121_fold3_training_log.csv
Best weights loaded from epoch 18

Model evaluation: densenet121_fold3
----------------------------------------
  Balanced Accuracy:       80.35%
  F1 (macro):              0.7968
  Quadratic Cohen's Kappa: 0.9086
  MAE (ordinal):           0.2429
  Off-by-one accuracy:     97.14%
  ECE:                     0.0963
  Brier Score (mean):      0.0584

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.89      0.87        87
    Doubtful       0.73      0.70      0.72        81
        Mild       0.72      0.54      0.62        39
    Moderate       0.80      0.95      0.87        38
      Severe       0.89      0.94      0.92        35

    accuracy                           0.80       280
   macro avg       0.80      0.80      0.80       280
weighted avg       0.80      0.80      0.79       280

  Metrics saved to: results/individual_models/densenet121_fold3_metrics.json
  Probabilities saved to: results/individual_models/densenet121_fold3_test_probs.npz

--- densenet121 | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

============================================================
TRAINING: densenet121_fold4
============================================================
Epoch [  1/30]  Train Loss: 1.4992  Bal.Acc: 35.2%  F1: 0.3239  |  Val Loss: 1.5608  Bal.Acc: 32.0%  F1: 0.2694  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 1.5608)
Epoch [  2/30]  Train Loss: 1.2021  Bal.Acc: 57.8%  F1: 0.5341  |  Val Loss: 1.1723  Bal.Acc: 55.8%  F1: 0.5298  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 1.1723)
Epoch [  3/30]  Train Loss: 0.9758  Bal.Acc: 66.8%  F1: 0.6440  |  Val Loss: 0.9592  Bal.Acc: 67.2%  F1: 0.6655  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.9592)
Epoch [  4/30]  Train Loss: 0.8001  Bal.Acc: 73.2%  F1: 0.7234  |  Val Loss: 0.8381  Bal.Acc: 67.0%  F1: 0.6795  |  LR: 1.00e-04  (18.9s)
 Best checkpoint saved (val_loss: 0.8381)
Epoch [  5/30]  Train Loss: 0.6478  Bal.Acc: 79.6%  F1: 0.7863  |  Val Loss: 0.7429  Bal.Acc: 75.6%  F1: 0.7600  |  LR: 1.00e-04  (17.5s)
 Best checkpoint saved (val_loss: 0.7429)
Epoch [  6/30]  Train Loss: 0.5258  Bal.Acc: 83.6%  F1: 0.8324  |  Val Loss: 0.6577  Bal.Acc: 77.2%  F1: 0.7567  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 0.6577)
Epoch [  7/30]  Train Loss: 0.4556  Bal.Acc: 84.7%  F1: 0.8360  |  Val Loss: 0.6775  Bal.Acc: 75.2%  F1: 0.7588  |  LR: 1.00e-04  (18.1s)
Epoch [  8/30]  Train Loss: 0.4150  Bal.Acc: 85.7%  F1: 0.8479  |  Val Loss: 0.6258  Bal.Acc: 75.1%  F1: 0.7382  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 0.6258)
Epoch [  9/30]  Train Loss: 0.3630  Bal.Acc: 87.9%  F1: 0.8685  |  Val Loss: 0.5998  Bal.Acc: 78.5%  F1: 0.7896  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 0.5998)
Epoch [ 10/30]  Train Loss: 0.3168  Bal.Acc: 89.2%  F1: 0.8871  |  Val Loss: 0.5151  Bal.Acc: 81.6%  F1: 0.7986  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.5151)
Epoch [ 11/30]  Train Loss: 0.2701  Bal.Acc: 91.5%  F1: 0.9112  |  Val Loss: 0.5207  Bal.Acc: 82.8%  F1: 0.8195  |  LR: 1.00e-04  (18.3s)
Epoch [ 12/30]  Train Loss: 0.2154  Bal.Acc: 93.5%  F1: 0.9292  |  Val Loss: 0.5562  Bal.Acc: 78.7%  F1: 0.7895  |  LR: 1.00e-04  (16.0s)
Epoch [ 13/30]  Train Loss: 0.2031  Bal.Acc: 93.7%  F1: 0.9354  |  Val Loss: 0.5081  Bal.Acc: 83.1%  F1: 0.8222  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 0.5081)
Epoch [ 14/30]  Train Loss: 0.2029  Bal.Acc: 94.1%  F1: 0.9373  |  Val Loss: 0.5475  Bal.Acc: 81.6%  F1: 0.8167  |  LR: 1.00e-04  (18.2s)
Epoch [ 15/30]  Train Loss: 0.2076  Bal.Acc: 93.0%  F1: 0.9300  |  Val Loss: 0.5024  Bal.Acc: 82.8%  F1: 0.8173  |  LR: 1.00e-04  (15.3s)
 Best checkpoint saved (val_loss: 0.5024)
Epoch [ 16/30]  Train Loss: 0.1771  Bal.Acc: 93.8%  F1: 0.9339  |  Val Loss: 0.5053  Bal.Acc: 82.3%  F1: 0.8158  |  LR: 1.00e-04  (17.8s)
Epoch [ 17/30]  Train Loss: 0.1775  Bal.Acc: 94.2%  F1: 0.9396  |  Val Loss: 0.5197  Bal.Acc: 82.8%  F1: 0.8125  |  LR: 1.00e-04  (17.3s)
Epoch [ 18/30]  Train Loss: 0.1717  Bal.Acc: 94.7%  F1: 0.9426  |  Val Loss: 0.5242  Bal.Acc: 85.2%  F1: 0.8420  |  LR: 1.00e-04  (16.3s)
Epoch [ 19/30]  Train Loss: 0.1768  Bal.Acc: 94.9%  F1: 0.9464  |  Val Loss: 0.4899  Bal.Acc: 84.5%  F1: 0.8371  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 0.4899)
Epoch [ 20/30]  Train Loss: 0.1365  Bal.Acc: 96.2%  F1: 0.9598  |  Val Loss: 0.5032  Bal.Acc: 82.6%  F1: 0.8273  |  LR: 1.00e-04  (18.5s)
Epoch [ 21/30]  Train Loss: 0.1226  Bal.Acc: 96.7%  F1: 0.9648  |  Val Loss: 0.4906  Bal.Acc: 86.4%  F1: 0.8557  |  LR: 1.00e-04  (15.4s)
Epoch [ 22/30]  Train Loss: 0.1146  Bal.Acc: 96.9%  F1: 0.9679  |  Val Loss: 0.4864  Bal.Acc: 88.4%  F1: 0.8641  |  LR: 1.00e-04  (15.3s)
 Best checkpoint saved (val_loss: 0.4864)
Epoch [ 23/30]  Train Loss: 0.0951  Bal.Acc: 97.1%  F1: 0.9679  |  Val Loss: 0.5115  Bal.Acc: 84.9%  F1: 0.8413  |  LR: 1.00e-04  (18.1s)
Epoch [ 24/30]  Train Loss: 0.0862  Bal.Acc: 97.4%  F1: 0.9736  |  Val Loss: 0.4867  Bal.Acc: 85.0%  F1: 0.8426  |  LR: 1.00e-04  (16.0s)
Epoch [ 25/30]  Train Loss: 0.1051  Bal.Acc: 96.5%  F1: 0.9621  |  Val Loss: 0.4962  Bal.Acc: 85.9%  F1: 0.8479  |  LR: 1.00e-04  (16.1s)
Epoch [ 26/30]  Train Loss: 0.1153  Bal.Acc: 96.8%  F1: 0.9641  |  Val Loss: 0.4630  Bal.Acc: 84.1%  F1: 0.8352  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 0.4630)
Epoch [ 27/30]  Train Loss: 0.0979  Bal.Acc: 97.4%  F1: 0.9720  |  Val Loss: 0.5182  Bal.Acc: 85.2%  F1: 0.8488  |  LR: 1.00e-04  (19.2s)
Epoch [ 28/30]  Train Loss: 0.0729  Bal.Acc: 98.3%  F1: 0.9829  |  Val Loss: 0.5377  Bal.Acc: 84.3%  F1: 0.8345  |  LR: 1.00e-04  (15.9s)
Epoch [ 29/30]  Train Loss: 0.0669  Bal.Acc: 98.0%  F1: 0.9781  |  Val Loss: 0.5844  Bal.Acc: 85.1%  F1: 0.8485  |  LR: 1.00e-04  (15.2s)
Epoch [ 30/30]  Train Loss: 0.0738  Bal.Acc: 98.0%  F1: 0.9781  |  Val Loss: 0.6530  Bal.Acc: 82.1%  F1: 0.8161  |  LR: 5.00e-05  (15.2s)

 Training finished. Checkpoint: checkpoints/densenet121_fold4_best.pt
Log CSV: results/logs/densenet121_fold4_training_log.csv
Best weights loaded from epoch 26

Model evaluation: densenet121_fold4
----------------------------------------
  Balanced Accuracy:       84.13%
  F1 (macro):              0.8352
  Quadratic Cohen's Kappa: 0.9332
  MAE (ordinal):           0.1857
  Off-by-one accuracy:     97.86%
  ECE:                     0.0750
  Brier Score (mean):      0.0479

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.97      0.93        87
    Doubtful       0.84      0.72      0.77        81
        Mild       0.67      0.72      0.69        39
    Moderate       0.88      0.92      0.90        38
      Severe       0.89      0.89      0.89        35

    accuracy                           0.84       280
   macro avg       0.83      0.84      0.84       280
weighted avg       0.84      0.84      0.84       280

  Metrics saved to: results/individual_models/densenet121_fold4_metrics.json
  Probabilities saved to: results/individual_models/densenet121_fold4_test_probs.npz

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

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable

============================================================
TRAINING: densenet121_fold5
============================================================
Epoch [  1/30]  Train Loss: 1.4858  Bal.Acc: 34.2%  F1: 0.3299  |  Val Loss: 1.4159  Bal.Acc: 40.5%  F1: 0.3711  |  LR: 1.00e-04  (16.0s)
 Best checkpoint saved (val_loss: 1.4159)
Epoch [  2/30]  Train Loss: 1.2096  Bal.Acc: 54.5%  F1: 0.4975  |  Val Loss: 1.1271  Bal.Acc: 61.1%  F1: 0.6060  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 1.1271)
Epoch [  3/30]  Train Loss: 1.0026  Bal.Acc: 63.8%  F1: 0.6087  |  Val Loss: 0.9420  Bal.Acc: 62.3%  F1: 0.6172  |  LR: 1.00e-04  (18.8s)
 Best checkpoint saved (val_loss: 0.9420)
Epoch [  4/30]  Train Loss: 0.8015  Bal.Acc: 74.0%  F1: 0.7253  |  Val Loss: 0.8206  Bal.Acc: 68.5%  F1: 0.6707  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 0.8206)
Epoch [  5/30]  Train Loss: 0.6871  Bal.Acc: 77.0%  F1: 0.7640  |  Val Loss: 0.7227  Bal.Acc: 74.8%  F1: 0.7448  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 0.7227)
Epoch [  6/30]  Train Loss: 0.5801  Bal.Acc: 79.4%  F1: 0.7831  |  Val Loss: 0.6769  Bal.Acc: 72.6%  F1: 0.7095  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.6769)
Epoch [  7/30]  Train Loss: 0.5189  Bal.Acc: 82.8%  F1: 0.8242  |  Val Loss: 0.6265  Bal.Acc: 76.5%  F1: 0.7454  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 0.6265)
Epoch [  8/30]  Train Loss: 0.4487  Bal.Acc: 84.5%  F1: 0.8368  |  Val Loss: 0.5966  Bal.Acc: 77.4%  F1: 0.7640  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.5966)
Epoch [  9/30]  Train Loss: 0.3614  Bal.Acc: 88.4%  F1: 0.8823  |  Val Loss: 0.5239  Bal.Acc: 81.1%  F1: 0.7993  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 0.5239)
Epoch [ 10/30]  Train Loss: 0.3206  Bal.Acc: 90.1%  F1: 0.8963  |  Val Loss: 0.6006  Bal.Acc: 76.8%  F1: 0.7570  |  LR: 1.00e-04  (18.1s)
Epoch [ 11/30]  Train Loss: 0.2909  Bal.Acc: 90.7%  F1: 0.9023  |  Val Loss: 0.5064  Bal.Acc: 82.2%  F1: 0.8026  |  LR: 1.00e-04  (15.9s)
 Best checkpoint saved (val_loss: 0.5064)
Epoch [ 12/30]  Train Loss: 0.2924  Bal.Acc: 89.8%  F1: 0.8919  |  Val Loss: 0.6909  Bal.Acc: 75.1%  F1: 0.7597  |  LR: 1.00e-04  (18.5s)
Epoch [ 13/30]  Train Loss: 0.2414  Bal.Acc: 92.4%  F1: 0.9209  |  Val Loss: 0.4681  Bal.Acc: 84.2%  F1: 0.8320  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 0.4681)
Epoch [ 14/30]  Train Loss: 0.2317  Bal.Acc: 92.2%  F1: 0.9169  |  Val Loss: 0.5045  Bal.Acc: 82.9%  F1: 0.8224  |  LR: 1.00e-04  (17.4s)
Epoch [ 15/30]  Train Loss: 0.1882  Bal.Acc: 94.0%  F1: 0.9368  |  Val Loss: 0.5093  Bal.Acc: 82.2%  F1: 0.8184  |  LR: 1.00e-04  (15.2s)
Epoch [ 16/30]  Train Loss: 0.1931  Bal.Acc: 93.9%  F1: 0.9364  |  Val Loss: 0.4942  Bal.Acc: 83.4%  F1: 0.8267  |  LR: 1.00e-04  (14.8s)
Epoch [ 17/30]  Train Loss: 0.1529  Bal.Acc: 95.2%  F1: 0.9486  |  Val Loss: 0.4928  Bal.Acc: 84.2%  F1: 0.8359  |  LR: 5.00e-05  (15.1s)
Epoch [ 18/30]  Train Loss: 0.1261  Bal.Acc: 96.5%  F1: 0.9621  |  Val Loss: 0.4901  Bal.Acc: 85.7%  F1: 0.8545  |  LR: 5.00e-05  (15.2s)
Epoch [ 19/30]  Train Loss: 0.1236  Bal.Acc: 96.0%  F1: 0.9564  |  Val Loss: 0.5082  Bal.Acc: 85.5%  F1: 0.8468  |  LR: 5.00e-05  (15.9s)
Epoch [ 20/30]  Train Loss: 0.1149  Bal.Acc: 96.8%  F1: 0.9645  |  Val Loss: 0.5331  Bal.Acc: 84.6%  F1: 0.8427  |  LR: 5.00e-05  (15.4s)
Epoch [ 21/30]  Train Loss: 0.1016  Bal.Acc: 97.3%  F1: 0.9728  |  Val Loss: 0.5467  Bal.Acc: 83.6%  F1: 0.8265  |  LR: 2.50e-05  (15.5s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.4681

 Training finished. Checkpoint: checkpoints/densenet121_fold5_best.pt
Log CSV: results/logs/densenet121_fold5_training_log.csv
Best weights loaded from epoch 13

Model evaluation: densenet121_fold5
----------------------------------------
  Balanced Accuracy:       84.25%
  F1 (macro):              0.8320
  Quadratic Cohen's Kappa: 0.9281
  MAE (ordinal):           0.2000
  Off-by-one accuracy:     96.79%
  ECE:                     0.0419
  Brier Score (mean):      0.0517

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.91      0.90        87
    Doubtful       0.83      0.73      0.78        81
        Mild       0.73      0.77      0.75        39
    Moderate       0.83      0.92      0.88        38
      Severe       0.84      0.89      0.86        35

    accuracy                           0.84       280
   macro avg       0.82      0.84      0.83       280
weighted avg       0.84      0.84      0.83       280

  Metrics saved to: results/individual_models/densenet121_fold5_metrics.json
  Probabilities saved to: results/individual_models/densenet121_fold5_test_probs.npz

 FINISHED: densenet121. Average kappa out of 5 folds: 0.9307 ±0.0137

================================================================================
MODEL TRAINING START: mobilenetv3_large
================================================================================

--- mobilenetv3_large | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: mobilenetv3_large
model.safetensors: 100% 22.1M/22.1M [00:00<00:00, 32.6MB/s]
  Parameters: 4,208,437 total, 4,208,437 trainable

============================================================
TRAINING: mobilenetv3_large_fold1
============================================================
Epoch [  1/30]  Train Loss: 2.3153  Bal.Acc: 32.7%  F1: 0.3107  |  Val Loss: 2.1328  Bal.Acc: 37.3%  F1: 0.3372  |  LR: 1.00e-04  (12.9s)
 Best checkpoint saved (val_loss: 2.1328)
Epoch [  2/30]  Train Loss: 1.3460  Bal.Acc: 53.5%  F1: 0.5269  |  Val Loss: 1.7643  Bal.Acc: 45.4%  F1: 0.3900  |  LR: 1.00e-04  (13.1s)
 Best checkpoint saved (val_loss: 1.7643)
Epoch [  3/30]  Train Loss: 1.1012  Bal.Acc: 61.3%  F1: 0.5881  |  Val Loss: 1.2652  Bal.Acc: 58.2%  F1: 0.5683  |  LR: 1.00e-04  (14.5s)
 Best checkpoint saved (val_loss: 1.2652)
Epoch [  4/30]  Train Loss: 0.8895  Bal.Acc: 67.0%  F1: 0.6586  |  Val Loss: 1.0473  Bal.Acc: 62.9%  F1: 0.6101  |  LR: 1.00e-04  (14.8s)
 Best checkpoint saved (val_loss: 1.0473)
Epoch [  5/30]  Train Loss: 0.7219  Bal.Acc: 72.5%  F1: 0.7164  |  Val Loss: 0.7332  Bal.Acc: 70.3%  F1: 0.6929  |  LR: 1.00e-04  (14.4s)
 Best checkpoint saved (val_loss: 0.7332)
Epoch [  6/30]  Train Loss: 0.6418  Bal.Acc: 76.3%  F1: 0.7450  |  Val Loss: 0.7340  Bal.Acc: 73.3%  F1: 0.7326  |  LR: 1.00e-04  (22.8s)
Epoch [  7/30]  Train Loss: 0.6172  Bal.Acc: 78.5%  F1: 0.7774  |  Val Loss: 0.6363  Bal.Acc: 76.2%  F1: 0.7528  |  LR: 1.00e-04  (12.0s)
 Best checkpoint saved (val_loss: 0.6363)
Epoch [  8/30]  Train Loss: 0.5264  Bal.Acc: 80.2%  F1: 0.7937  |  Val Loss: 0.6588  Bal.Acc: 73.6%  F1: 0.7390  |  LR: 1.00e-04  (15.0s)
Epoch [  9/30]  Train Loss: 0.4285  Bal.Acc: 83.1%  F1: 0.8209  |  Val Loss: 0.6329  Bal.Acc: 75.8%  F1: 0.7689  |  LR: 1.00e-04  (12.3s)
 Best checkpoint saved (val_loss: 0.6329)
Epoch [ 10/30]  Train Loss: 0.4515  Bal.Acc: 82.8%  F1: 0.8213  |  Val Loss: 0.5163  Bal.Acc: 78.9%  F1: 0.7844  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 0.5163)
Epoch [ 11/30]  Train Loss: 0.3376  Bal.Acc: 87.4%  F1: 0.8656  |  Val Loss: 0.5334  Bal.Acc: 78.3%  F1: 0.7891  |  LR: 1.00e-04  (15.4s)
Epoch [ 12/30]  Train Loss: 0.3360  Bal.Acc: 87.4%  F1: 0.8671  |  Val Loss: 0.5481  Bal.Acc: 77.9%  F1: 0.7738  |  LR: 1.00e-04  (12.8s)
Epoch [ 13/30]  Train Loss: 0.3301  Bal.Acc: 87.0%  F1: 0.8609  |  Val Loss: 0.4731  Bal.Acc: 81.3%  F1: 0.8074  |  LR: 1.00e-04  (12.8s)
 Best checkpoint saved (val_loss: 0.4731)
Epoch [ 14/30]  Train Loss: 0.3181  Bal.Acc: 88.3%  F1: 0.8726  |  Val Loss: 0.5927  Bal.Acc: 79.9%  F1: 0.8003  |  LR: 1.00e-04  (15.2s)
Epoch [ 15/30]  Train Loss: 0.2786  Bal.Acc: 89.3%  F1: 0.8926  |  Val Loss: 0.4461  Bal.Acc: 82.2%  F1: 0.8192  |  LR: 1.00e-04  (13.1s)
 Best checkpoint saved (val_loss: 0.4461)
Epoch [ 16/30]  Train Loss: 0.2567  Bal.Acc: 90.6%  F1: 0.8951  |  Val Loss: 0.4765  Bal.Acc: 81.2%  F1: 0.8180  |  LR: 1.00e-04  (14.9s)
Epoch [ 17/30]  Train Loss: 0.2465  Bal.Acc: 90.8%  F1: 0.9074  |  Val Loss: 0.4890  Bal.Acc: 80.0%  F1: 0.7947  |  LR: 1.00e-04  (13.0s)
Epoch [ 18/30]  Train Loss: 0.2375  Bal.Acc: 91.9%  F1: 0.9113  |  Val Loss: 0.5058  Bal.Acc: 82.1%  F1: 0.8270  |  LR: 1.00e-04  (12.7s)
Epoch [ 19/30]  Train Loss: 0.2000  Bal.Acc: 92.2%  F1: 0.9185  |  Val Loss: 0.6504  Bal.Acc: 80.8%  F1: 0.8012  |  LR: 5.00e-05  (12.3s)
Epoch [ 20/30]  Train Loss: 0.1991  Bal.Acc: 93.6%  F1: 0.9297  |  Val Loss: 0.5016  Bal.Acc: 83.3%  F1: 0.8391  |  LR: 5.00e-05  (11.6s)
Epoch [ 21/30]  Train Loss: 0.1527  Bal.Acc: 94.5%  F1: 0.9400  |  Val Loss: 0.4508  Bal.Acc: 84.7%  F1: 0.8467  |  LR: 5.00e-05  (12.6s)
Epoch [ 22/30]  Train Loss: 0.1692  Bal.Acc: 94.0%  F1: 0.9386  |  Val Loss: 0.5262  Bal.Acc: 83.2%  F1: 0.8315  |  LR: 5.00e-05  (13.1s)
Epoch [ 23/30]  Train Loss: 0.1546  Bal.Acc: 95.1%  F1: 0.9479  |  Val Loss: 0.5938  Bal.Acc: 83.8%  F1: 0.8373  |  LR: 2.50e-05  (12.8s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.4461

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold1_best.pt
Log CSV: results/logs/mobilenetv3_large_fold1_training_log.csv
Best weights loaded from epoch 15

Model evaluation: mobilenetv3_large_fold1
----------------------------------------
  Balanced Accuracy:       82.20%
  F1 (macro):              0.8192
  Quadratic Cohen's Kappa: 0.9246
  MAE (ordinal):           0.2135
  Off-by-one accuracy:     97.15%
  ECE:                     0.0559
  Brier Score (mean):      0.0515

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.90      0.87        88
    Doubtful       0.79      0.73      0.76        81
        Mild       0.68      0.62      0.65        40
    Moderate       0.80      0.97      0.88        37
      Severe       1.00      0.89      0.94        35

    accuracy                           0.82       281
   macro avg       0.82      0.82      0.82       281
weighted avg       0.82      0.82      0.82       281

  Metrics saved to: results/individual_models/mobilenetv3_large_fold1_metrics.json
  Probabilities saved to: results/individual_models/mobilenetv3_large_fold1_test_probs.npz

--- mobilenetv3_large | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

============================================================
TRAINING: mobilenetv3_large_fold2
============================================================
Epoch [  1/30]  Train Loss: 2.2521  Bal.Acc: 34.0%  F1: 0.3253  |  Val Loss: 2.4367  Bal.Acc: 33.7%  F1: 0.3379  |  LR: 1.00e-04  (13.0s)
 Best checkpoint saved (val_loss: 2.4367)
Epoch [  2/30]  Train Loss: 1.3433  Bal.Acc: 53.7%  F1: 0.5271  |  Val Loss: 1.7816  Bal.Acc: 41.6%  F1: 0.3880  |  LR: 1.00e-04  (13.4s)
 Best checkpoint saved (val_loss: 1.7816)
Epoch [  3/30]  Train Loss: 1.0539  Bal.Acc: 60.7%  F1: 0.5886  |  Val Loss: 1.5277  Bal.Acc: 45.4%  F1: 0.4599  |  LR: 1.00e-04  (14.7s)
 Best checkpoint saved (val_loss: 1.5277)
Epoch [  4/30]  Train Loss: 0.8636  Bal.Acc: 67.4%  F1: 0.6672  |  Val Loss: 1.1624  Bal.Acc: 60.8%  F1: 0.6078  |  LR: 1.00e-04  (14.7s)
 Best checkpoint saved (val_loss: 1.1624)
Epoch [  5/30]  Train Loss: 0.7023  Bal.Acc: 74.1%  F1: 0.7261  |  Val Loss: 1.1716  Bal.Acc: 60.6%  F1: 0.6127  |  LR: 1.00e-04  (15.1s)
Epoch [  6/30]  Train Loss: 0.6406  Bal.Acc: 75.1%  F1: 0.7381  |  Val Loss: 1.1543  Bal.Acc: 61.8%  F1: 0.6144  |  LR: 1.00e-04  (12.7s)
 Best checkpoint saved (val_loss: 1.1543)
Epoch [  7/30]  Train Loss: 0.5297  Bal.Acc: 81.0%  F1: 0.7979  |  Val Loss: 1.0216  Bal.Acc: 69.0%  F1: 0.6946  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 1.0216)
Epoch [  8/30]  Train Loss: 0.4829  Bal.Acc: 80.7%  F1: 0.7993  |  Val Loss: 1.0202  Bal.Acc: 68.3%  F1: 0.6872  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 1.0202)
Epoch [  9/30]  Train Loss: 0.4419  Bal.Acc: 83.2%  F1: 0.8246  |  Val Loss: 0.9742  Bal.Acc: 68.4%  F1: 0.6953  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 0.9742)
Epoch [ 10/30]  Train Loss: 0.3809  Bal.Acc: 85.1%  F1: 0.8473  |  Val Loss: 0.9240  Bal.Acc: 71.4%  F1: 0.7069  |  LR: 1.00e-04  (14.6s)
 Best checkpoint saved (val_loss: 0.9240)
Epoch [ 11/30]  Train Loss: 0.3518  Bal.Acc: 84.9%  F1: 0.8386  |  Val Loss: 0.8971  Bal.Acc: 73.9%  F1: 0.7366  |  LR: 1.00e-04  (14.7s)
 Best checkpoint saved (val_loss: 0.8971)
Epoch [ 12/30]  Train Loss: 0.3173  Bal.Acc: 88.2%  F1: 0.8756  |  Val Loss: 0.7915  Bal.Acc: 75.0%  F1: 0.7513  |  LR: 1.00e-04  (15.2s)
 Best checkpoint saved (val_loss: 0.7915)
Epoch [ 13/30]  Train Loss: 0.3592  Bal.Acc: 86.5%  F1: 0.8617  |  Val Loss: 0.7773  Bal.Acc: 75.5%  F1: 0.7516  |  LR: 1.00e-04  (14.6s)
 Best checkpoint saved (val_loss: 0.7773)
Epoch [ 14/30]  Train Loss: 0.2855  Bal.Acc: 89.5%  F1: 0.8885  |  Val Loss: 0.7740  Bal.Acc: 76.4%  F1: 0.7712  |  LR: 1.00e-04  (14.7s)
 Best checkpoint saved (val_loss: 0.7740)
Epoch [ 15/30]  Train Loss: 0.2432  Bal.Acc: 91.5%  F1: 0.9051  |  Val Loss: 0.8565  Bal.Acc: 76.5%  F1: 0.7732  |  LR: 1.00e-04  (14.2s)
Epoch [ 16/30]  Train Loss: 0.2343  Bal.Acc: 90.6%  F1: 0.9027  |  Val Loss: 0.7740  Bal.Acc: 76.4%  F1: 0.7725  |  LR: 1.00e-04  (12.7s)
 Best checkpoint saved (val_loss: 0.7740)
Epoch [ 17/30]  Train Loss: 0.2729  Bal.Acc: 90.2%  F1: 0.8947  |  Val Loss: 0.8445  Bal.Acc: 75.0%  F1: 0.7665  |  LR: 1.00e-04  (14.5s)
Epoch [ 18/30]  Train Loss: 0.2280  Bal.Acc: 91.0%  F1: 0.9049  |  Val Loss: 0.8276  Bal.Acc: 73.6%  F1: 0.7372  |  LR: 5.00e-05  (12.0s)
Epoch [ 19/30]  Train Loss: 0.1781  Bal.Acc: 94.3%  F1: 0.9381  |  Val Loss: 0.8309  Bal.Acc: 75.2%  F1: 0.7619  |  LR: 5.00e-05  (12.4s)
Epoch [ 20/30]  Train Loss: 0.1842  Bal.Acc: 92.6%  F1: 0.9216  |  Val Loss: 0.7388  Bal.Acc: 77.8%  F1: 0.7831  |  LR: 5.00e-05  (13.1s)
 Best checkpoint saved (val_loss: 0.7388)
Epoch [ 21/30]  Train Loss: 0.1436  Bal.Acc: 94.5%  F1: 0.9428  |  Val Loss: 0.7397  Bal.Acc: 77.5%  F1: 0.7799  |  LR: 5.00e-05  (14.6s)
Epoch [ 22/30]  Train Loss: 0.1878  Bal.Acc: 93.1%  F1: 0.9278  |  Val Loss: 0.7995  Bal.Acc: 77.3%  F1: 0.7765  |  LR: 5.00e-05  (12.6s)
Epoch [ 23/30]  Train Loss: 0.1423  Bal.Acc: 95.8%  F1: 0.9531  |  Val Loss: 0.7685  Bal.Acc: 78.6%  F1: 0.7905  |  LR: 5.00e-05  (13.0s)
Epoch [ 24/30]  Train Loss: 0.1668  Bal.Acc: 94.1%  F1: 0.9381  |  Val Loss: 0.7411  Bal.Acc: 78.4%  F1: 0.7880  |  LR: 2.50e-05  (12.5s)
Epoch [ 25/30]  Train Loss: 0.1489  Bal.Acc: 94.9%  F1: 0.9460  |  Val Loss: 0.7586  Bal.Acc: 78.6%  F1: 0.7935  |  LR: 2.50e-05  (12.1s)
Epoch [ 26/30]  Train Loss: 0.1592  Bal.Acc: 94.0%  F1: 0.9369  |  Val Loss: 0.7365  Bal.Acc: 76.8%  F1: 0.7711  |  LR: 2.50e-05  (11.8s)
 Best checkpoint saved (val_loss: 0.7365)
Epoch [ 27/30]  Train Loss: 0.1466  Bal.Acc: 94.3%  F1: 0.9422  |  Val Loss: 0.7562  Bal.Acc: 77.6%  F1: 0.7805  |  LR: 2.50e-05  (15.0s)
Epoch [ 28/30]  Train Loss: 0.1407  Bal.Acc: 95.4%  F1: 0.9508  |  Val Loss: 0.7753  Bal.Acc: 76.0%  F1: 0.7656  |  LR: 2.50e-05  (12.5s)
Epoch [ 29/30]  Train Loss: 0.1410  Bal.Acc: 94.9%  F1: 0.9465  |  Val Loss: 0.7508  Bal.Acc: 79.1%  F1: 0.7984  |  LR: 2.50e-05  (13.2s)
Epoch [ 30/30]  Train Loss: 0.1390  Bal.Acc: 95.7%  F1: 0.9568  |  Val Loss: 0.7628  Bal.Acc: 79.6%  F1: 0.7994  |  LR: 1.25e-05  (13.1s)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold2_best.pt
Log CSV: results/logs/mobilenetv3_large_fold2_training_log.csv
Best weights loaded from epoch 26

Model evaluation: mobilenetv3_large_fold2
----------------------------------------
  Balanced Accuracy:       76.82%
  F1 (macro):              0.7711
  Quadratic Cohen's Kappa: 0.9018
  MAE (ordinal):           0.2776
  Off-by-one accuracy:     96.80%
  ECE:                     0.1296
  Brier Score (mean):      0.0698

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.79      0.85      0.82        88
    Doubtful       0.71      0.64      0.68        81
        Mild       0.57      0.65      0.60        40
    Moderate       0.91      0.81      0.86        37
      Severe       0.91      0.89      0.90        35

    accuracy                           0.76       281
   macro avg       0.78      0.77      0.77       281
weighted avg       0.77      0.76      0.76       281

  Metrics saved to: results/individual_models/mobilenetv3_large_fold2_metrics.json
  Probabilities saved to: results/individual_models/mobilenetv3_large_fold2_test_probs.npz

--- mobilenetv3_large | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

============================================================
TRAINING: mobilenetv3_large_fold3
============================================================
Epoch [  1/30]  Train Loss: 2.3125  Bal.Acc: 32.1%  F1: 0.3082  |  Val Loss: 2.4602  Bal.Acc: 37.0%  F1: 0.2996  |  LR: 1.00e-04  (12.7s)
 Best checkpoint saved (val_loss: 2.4602)
Epoch [  2/30]  Train Loss: 1.3337  Bal.Acc: 56.6%  F1: 0.5491  |  Val Loss: 1.8975  Bal.Acc: 41.0%  F1: 0.3749  |  LR: 1.00e-04  (13.2s)
 Best checkpoint saved (val_loss: 1.8975)
Epoch [  3/30]  Train Loss: 1.1044  Bal.Acc: 61.2%  F1: 0.5929  |  Val Loss: 1.4957  Bal.Acc: 52.6%  F1: 0.5054  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 1.4957)
Epoch [  4/30]  Train Loss: 0.9024  Bal.Acc: 67.3%  F1: 0.6554  |  Val Loss: 1.1247  Bal.Acc: 57.9%  F1: 0.5660  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 1.1247)
Epoch [  5/30]  Train Loss: 0.7652  Bal.Acc: 71.6%  F1: 0.7091  |  Val Loss: 0.9522  Bal.Acc: 67.8%  F1: 0.6701  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 0.9522)
Epoch [  6/30]  Train Loss: 0.6427  Bal.Acc: 76.5%  F1: 0.7564  |  Val Loss: 0.9276  Bal.Acc: 68.2%  F1: 0.6873  |  LR: 1.00e-04  (15.2s)
 Best checkpoint saved (val_loss: 0.9276)
Epoch [  7/30]  Train Loss: 0.5750  Bal.Acc: 77.6%  F1: 0.7646  |  Val Loss: 0.9763  Bal.Acc: 67.6%  F1: 0.6815  |  LR: 1.00e-04  (15.0s)
Epoch [  8/30]  Train Loss: 0.5258  Bal.Acc: 79.4%  F1: 0.7841  |  Val Loss: 0.8998  Bal.Acc: 67.8%  F1: 0.6777  |  LR: 1.00e-04  (12.9s)
 Best checkpoint saved (val_loss: 0.8998)
Epoch [  9/30]  Train Loss: 0.4566  Bal.Acc: 83.4%  F1: 0.8268  |  Val Loss: 0.8526  Bal.Acc: 67.1%  F1: 0.6746  |  LR: 1.00e-04  (14.5s)
 Best checkpoint saved (val_loss: 0.8526)
Epoch [ 10/30]  Train Loss: 0.4058  Bal.Acc: 84.6%  F1: 0.8406  |  Val Loss: 0.8016  Bal.Acc: 72.0%  F1: 0.7190  |  LR: 1.00e-04  (14.8s)
 Best checkpoint saved (val_loss: 0.8016)
Epoch [ 11/30]  Train Loss: 0.3624  Bal.Acc: 87.3%  F1: 0.8671  |  Val Loss: 0.7893  Bal.Acc: 72.4%  F1: 0.7182  |  LR: 1.00e-04  (14.6s)
 Best checkpoint saved (val_loss: 0.7893)
Epoch [ 12/30]  Train Loss: 0.3597  Bal.Acc: 86.0%  F1: 0.8548  |  Val Loss: 0.7315  Bal.Acc: 73.1%  F1: 0.7264  |  LR: 1.00e-04  (14.6s)
 Best checkpoint saved (val_loss: 0.7315)
Epoch [ 13/30]  Train Loss: 0.3101  Bal.Acc: 88.7%  F1: 0.8832  |  Val Loss: 0.7113  Bal.Acc: 76.7%  F1: 0.7594  |  LR: 1.00e-04  (14.3s)
 Best checkpoint saved (val_loss: 0.7113)
Epoch [ 14/30]  Train Loss: 0.3146  Bal.Acc: 88.6%  F1: 0.8788  |  Val Loss: 0.7319  Bal.Acc: 75.0%  F1: 0.7482  |  LR: 1.00e-04  (14.7s)
Epoch [ 15/30]  Train Loss: 0.2372  Bal.Acc: 91.8%  F1: 0.9143  |  Val Loss: 0.6939  Bal.Acc: 76.9%  F1: 0.7671  |  LR: 1.00e-04  (12.6s)
 Best checkpoint saved (val_loss: 0.6939)
Epoch [ 16/30]  Train Loss: 0.2549  Bal.Acc: 90.5%  F1: 0.9014  |  Val Loss: 0.6900  Bal.Acc: 79.0%  F1: 0.7849  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 0.6900)
Epoch [ 17/30]  Train Loss: 0.2362  Bal.Acc: 90.9%  F1: 0.9028  |  Val Loss: 0.7166  Bal.Acc: 77.8%  F1: 0.7789  |  LR: 1.00e-04  (14.7s)
Epoch [ 18/30]  Train Loss: 0.2327  Bal.Acc: 90.7%  F1: 0.9057  |  Val Loss: 0.6802  Bal.Acc: 78.9%  F1: 0.7948  |  LR: 1.00e-04  (12.5s)
 Best checkpoint saved (val_loss: 0.6802)
Epoch [ 19/30]  Train Loss: 0.1999  Bal.Acc: 92.7%  F1: 0.9222  |  Val Loss: 0.6359  Bal.Acc: 79.8%  F1: 0.8007  |  LR: 1.00e-04  (14.3s)
 Best checkpoint saved (val_loss: 0.6359)
Epoch [ 20/30]  Train Loss: 0.2012  Bal.Acc: 91.8%  F1: 0.9165  |  Val Loss: 0.6658  Bal.Acc: 80.4%  F1: 0.8085  |  LR: 1.00e-04  (14.2s)
Epoch [ 21/30]  Train Loss: 0.1817  Bal.Acc: 93.6%  F1: 0.9326  |  Val Loss: 0.7553  Bal.Acc: 79.4%  F1: 0.7934  |  LR: 1.00e-04  (12.1s)
Epoch [ 22/30]  Train Loss: 0.1543  Bal.Acc: 94.2%  F1: 0.9411  |  Val Loss: 0.6750  Bal.Acc: 80.2%  F1: 0.8015  |  LR: 1.00e-04  (12.2s)
Epoch [ 23/30]  Train Loss: 0.1577  Bal.Acc: 95.0%  F1: 0.9468  |  Val Loss: 0.7222  Bal.Acc: 81.1%  F1: 0.8134  |  LR: 5.00e-05  (12.9s)
Epoch [ 24/30]  Train Loss: 0.1581  Bal.Acc: 94.2%  F1: 0.9424  |  Val Loss: 0.6853  Bal.Acc: 79.2%  F1: 0.7897  |  LR: 5.00e-05  (12.9s)
Epoch [ 25/30]  Train Loss: 0.1284  Bal.Acc: 95.8%  F1: 0.9554  |  Val Loss: 0.6534  Bal.Acc: 79.9%  F1: 0.8004  |  LR: 5.00e-05  (12.8s)
Epoch [ 26/30]  Train Loss: 0.1232  Bal.Acc: 95.6%  F1: 0.9530  |  Val Loss: 0.6793  Bal.Acc: 78.8%  F1: 0.7902  |  LR: 5.00e-05  (12.4s)
Epoch [ 27/30]  Train Loss: 0.1302  Bal.Acc: 95.1%  F1: 0.9520  |  Val Loss: 0.6622  Bal.Acc: 80.6%  F1: 0.8092  |  LR: 2.50e-05  (11.9s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.6359

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold3_best.pt
Log CSV: results/logs/mobilenetv3_large_fold3_training_log.csv
Best weights loaded from epoch 19

Model evaluation: mobilenetv3_large_fold3
----------------------------------------
  Balanced Accuracy:       79.81%
  F1 (macro):              0.8007
  Quadratic Cohen's Kappa: 0.8986
  MAE (ordinal):           0.2464
  Off-by-one accuracy:     95.71%
  ECE:                     0.0838
  Brier Score (mean):      0.0608

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.88      0.84      0.86        87
    Doubtful       0.75      0.81      0.78        81
        Mild       0.67      0.62      0.64        39
    Moderate       0.83      0.92      0.88        38
      Severe       0.90      0.80      0.85        35

    accuracy                           0.81       280
   macro avg       0.81      0.80      0.80       280
weighted avg       0.81      0.81      0.81       280

  Metrics saved to: results/individual_models/mobilenetv3_large_fold3_metrics.json
  Probabilities saved to: results/individual_models/mobilenetv3_large_fold3_test_probs.npz

--- mobilenetv3_large | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

============================================================
TRAINING: mobilenetv3_large_fold4
============================================================
Epoch [  1/30]  Train Loss: 2.2899  Bal.Acc: 35.7%  F1: 0.3476  |  Val Loss: 2.1734  Bal.Acc: 40.7%  F1: 0.3650  |  LR: 1.00e-04  (12.7s)
 Best checkpoint saved (val_loss: 2.1734)
Epoch [  2/30]  Train Loss: 1.4581  Bal.Acc: 53.6%  F1: 0.5210  |  Val Loss: 1.8798  Bal.Acc: 41.1%  F1: 0.3631  |  LR: 1.00e-04  (12.3s)
 Best checkpoint saved (val_loss: 1.8798)
Epoch [  3/30]  Train Loss: 1.0987  Bal.Acc: 60.9%  F1: 0.5938  |  Val Loss: 1.5978  Bal.Acc: 47.5%  F1: 0.4464  |  LR: 1.00e-04  (14.4s)
 Best checkpoint saved (val_loss: 1.5978)
Epoch [  4/30]  Train Loss: 0.9134  Bal.Acc: 67.6%  F1: 0.6657  |  Val Loss: 1.1871  Bal.Acc: 62.8%  F1: 0.6331  |  LR: 1.00e-04  (14.4s)
 Best checkpoint saved (val_loss: 1.1871)
Epoch [  5/30]  Train Loss: 0.7502  Bal.Acc: 70.9%  F1: 0.6960  |  Val Loss: 1.0235  Bal.Acc: 62.3%  F1: 0.6217  |  LR: 1.00e-04  (14.5s)
 Best checkpoint saved (val_loss: 1.0235)
Epoch [  6/30]  Train Loss: 0.6315  Bal.Acc: 75.5%  F1: 0.7492  |  Val Loss: 0.9858  Bal.Acc: 64.6%  F1: 0.6429  |  LR: 1.00e-04  (14.2s)
 Best checkpoint saved (val_loss: 0.9858)
Epoch [  7/30]  Train Loss: 0.5996  Bal.Acc: 77.3%  F1: 0.7612  |  Val Loss: 0.8975  Bal.Acc: 68.3%  F1: 0.6897  |  LR: 1.00e-04  (14.4s)
 Best checkpoint saved (val_loss: 0.8975)
Epoch [  8/30]  Train Loss: 0.5059  Bal.Acc: 80.2%  F1: 0.7929  |  Val Loss: 0.8857  Bal.Acc: 70.9%  F1: 0.7108  |  LR: 1.00e-04  (14.6s)
 Best checkpoint saved (val_loss: 0.8857)
Epoch [  9/30]  Train Loss: 0.4616  Bal.Acc: 81.6%  F1: 0.8108  |  Val Loss: 0.7619  Bal.Acc: 74.6%  F1: 0.7419  |  LR: 1.00e-04  (14.3s)
 Best checkpoint saved (val_loss: 0.7619)
Epoch [ 10/30]  Train Loss: 0.4380  Bal.Acc: 83.5%  F1: 0.8228  |  Val Loss: 0.7866  Bal.Acc: 74.1%  F1: 0.7420  |  LR: 1.00e-04  (14.1s)
Epoch [ 11/30]  Train Loss: 0.3692  Bal.Acc: 86.3%  F1: 0.8609  |  Val Loss: 0.7639  Bal.Acc: 74.3%  F1: 0.7338  |  LR: 1.00e-04  (11.9s)
Epoch [ 12/30]  Train Loss: 0.3217  Bal.Acc: 87.6%  F1: 0.8715  |  Val Loss: 0.8627  Bal.Acc: 71.4%  F1: 0.7096  |  LR: 1.00e-04  (12.3s)
Epoch [ 13/30]  Train Loss: 0.3179  Bal.Acc: 88.0%  F1: 0.8736  |  Val Loss: 0.7830  Bal.Acc: 77.3%  F1: 0.7687  |  LR: 5.00e-05  (12.6s)
Epoch [ 14/30]  Train Loss: 0.2960  Bal.Acc: 89.4%  F1: 0.8927  |  Val Loss: 0.7460  Bal.Acc: 75.1%  F1: 0.7511  |  LR: 5.00e-05  (12.9s)
 Best checkpoint saved (val_loss: 0.7460)
Epoch [ 15/30]  Train Loss: 0.2978  Bal.Acc: 89.2%  F1: 0.8865  |  Val Loss: 0.7085  Bal.Acc: 77.6%  F1: 0.7750  |  LR: 5.00e-05  (14.7s)
 Best checkpoint saved (val_loss: 0.7085)
Epoch [ 16/30]  Train Loss: 0.2441  Bal.Acc: 90.9%  F1: 0.9034  |  Val Loss: 0.6992  Bal.Acc: 77.5%  F1: 0.7722  |  LR: 5.00e-05  (14.4s)
 Best checkpoint saved (val_loss: 0.6992)
Epoch [ 17/30]  Train Loss: 0.2292  Bal.Acc: 91.5%  F1: 0.9129  |  Val Loss: 0.6982  Bal.Acc: 77.3%  F1: 0.7710  |  LR: 5.00e-05  (14.7s)
 Best checkpoint saved (val_loss: 0.6982)
Epoch [ 18/30]  Train Loss: 0.2427  Bal.Acc: 91.0%  F1: 0.9076  |  Val Loss: 0.6808  Bal.Acc: 78.1%  F1: 0.7787  |  LR: 5.00e-05  (14.9s)
 Best checkpoint saved (val_loss: 0.6808)
Epoch [ 19/30]  Train Loss: 0.2201  Bal.Acc: 92.0%  F1: 0.9176  |  Val Loss: 0.7044  Bal.Acc: 77.8%  F1: 0.7734  |  LR: 5.00e-05  (15.4s)
Epoch [ 20/30]  Train Loss: 0.2177  Bal.Acc: 91.7%  F1: 0.9101  |  Val Loss: 0.6794  Bal.Acc: 78.9%  F1: 0.7930  |  LR: 5.00e-05  (13.5s)
 Best checkpoint saved (val_loss: 0.6794)
Epoch [ 21/30]  Train Loss: 0.2127  Bal.Acc: 91.6%  F1: 0.9147  |  Val Loss: 0.6655  Bal.Acc: 79.1%  F1: 0.7926  |  LR: 5.00e-05  (14.7s)
 Best checkpoint saved (val_loss: 0.6655)
Epoch [ 22/30]  Train Loss: 0.2044  Bal.Acc: 93.1%  F1: 0.9269  |  Val Loss: 0.6602  Bal.Acc: 79.0%  F1: 0.7894  |  LR: 5.00e-05  (15.1s)
 Best checkpoint saved (val_loss: 0.6602)
Epoch [ 23/30]  Train Loss: 0.1817  Bal.Acc: 93.5%  F1: 0.9321  |  Val Loss: 0.6746  Bal.Acc: 78.7%  F1: 0.7851  |  LR: 5.00e-05  (15.2s)
Epoch [ 24/30]  Train Loss: 0.1841  Bal.Acc: 93.7%  F1: 0.9347  |  Val Loss: 0.6724  Bal.Acc: 79.4%  F1: 0.7946  |  LR: 5.00e-05  (13.2s)
Epoch [ 25/30]  Train Loss: 0.1895  Bal.Acc: 93.0%  F1: 0.9275  |  Val Loss: 0.6593  Bal.Acc: 78.8%  F1: 0.7895  |  LR: 5.00e-05  (13.2s)
 Best checkpoint saved (val_loss: 0.6593)
Epoch [ 26/30]  Train Loss: 0.1729  Bal.Acc: 93.5%  F1: 0.9323  |  Val Loss: 0.6518  Bal.Acc: 80.1%  F1: 0.8009  |  LR: 5.00e-05  (15.3s)
 Best checkpoint saved (val_loss: 0.6518)
Epoch [ 27/30]  Train Loss: 0.1664  Bal.Acc: 94.8%  F1: 0.9450  |  Val Loss: 0.6456  Bal.Acc: 79.7%  F1: 0.7969  |  LR: 5.00e-05  (15.0s)
 Best checkpoint saved (val_loss: 0.6456)
Epoch [ 28/30]  Train Loss: 0.1724  Bal.Acc: 93.5%  F1: 0.9341  |  Val Loss: 0.6424  Bal.Acc: 79.6%  F1: 0.7941  |  LR: 5.00e-05  (15.1s)
 Best checkpoint saved (val_loss: 0.6424)
Epoch [ 29/30]  Train Loss: 0.1721  Bal.Acc: 93.8%  F1: 0.9340  |  Val Loss: 0.6604  Bal.Acc: 78.1%  F1: 0.7764  |  LR: 5.00e-05  (15.1s)
Epoch [ 30/30]  Train Loss: 0.1535  Bal.Acc: 95.2%  F1: 0.9469  |  Val Loss: 0.7200  Bal.Acc: 78.7%  F1: 0.7879  |  LR: 5.00e-05  (12.8s)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold4_best.pt
Log CSV: results/logs/mobilenetv3_large_fold4_training_log.csv
Best weights loaded from epoch 28

Model evaluation: mobilenetv3_large_fold4
----------------------------------------
  Balanced Accuracy:       79.64%
  F1 (macro):              0.7941
  Quadratic Cohen's Kappa: 0.8829
  MAE (ordinal):           0.2643
  Off-by-one accuracy:     95.00%
  ECE:                     0.1039
  Brier Score (mean):      0.0614

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.88      0.87      0.88        87
    Doubtful       0.77      0.75      0.76        81
        Mild       0.67      0.72      0.69        39
    Moderate       0.81      0.89      0.85        38
      Severe       0.84      0.74      0.79        35

    accuracy                           0.80       280
   macro avg       0.79      0.80      0.79       280
weighted avg       0.81      0.80      0.80       280

  Metrics saved to: results/individual_models/mobilenetv3_large_fold4_metrics.json
  Probabilities saved to: results/individual_models/mobilenetv3_large_fold4_test_probs.npz

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

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable

============================================================
TRAINING: mobilenetv3_large_fold5
============================================================
Epoch [  1/30]  Train Loss: 2.3656  Bal.Acc: 38.3%  F1: 0.3648  |  Val Loss: 2.1479  Bal.Acc: 33.2%  F1: 0.2867  |  LR: 1.00e-04  (13.3s)
 Best checkpoint saved (val_loss: 2.1479)
Epoch [  2/30]  Train Loss: 1.4349  Bal.Acc: 52.1%  F1: 0.5089  |  Val Loss: 2.0616  Bal.Acc: 37.7%  F1: 0.3244  |  LR: 1.00e-04  (13.4s)
 Best checkpoint saved (val_loss: 2.0616)
Epoch [  3/30]  Train Loss: 1.1553  Bal.Acc: 61.2%  F1: 0.5925  |  Val Loss: 1.6156  Bal.Acc: 47.6%  F1: 0.4691  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 1.6156)
Epoch [  4/30]  Train Loss: 0.9194  Bal.Acc: 65.9%  F1: 0.6491  |  Val Loss: 1.0663  Bal.Acc: 61.9%  F1: 0.6106  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 1.0663)
Epoch [  5/30]  Train Loss: 0.7988  Bal.Acc: 70.7%  F1: 0.7004  |  Val Loss: 0.9734  Bal.Acc: 67.6%  F1: 0.6776  |  LR: 1.00e-04  (15.5s)
 Best checkpoint saved (val_loss: 0.9734)
Epoch [  6/30]  Train Loss: 0.7146  Bal.Acc: 72.5%  F1: 0.7134  |  Val Loss: 0.8963  Bal.Acc: 67.4%  F1: 0.6795  |  LR: 1.00e-04  (14.8s)
 Best checkpoint saved (val_loss: 0.8963)
Epoch [  7/30]  Train Loss: 0.5561  Bal.Acc: 79.9%  F1: 0.7853  |  Val Loss: 0.9268  Bal.Acc: 68.3%  F1: 0.6945  |  LR: 1.00e-04  (14.8s)
Epoch [  8/30]  Train Loss: 0.5155  Bal.Acc: 79.9%  F1: 0.7923  |  Val Loss: 0.8603  Bal.Acc: 72.2%  F1: 0.7182  |  LR: 1.00e-04  (13.2s)
 Best checkpoint saved (val_loss: 0.8603)
Epoch [  9/30]  Train Loss: 0.4425  Bal.Acc: 84.0%  F1: 0.8335  |  Val Loss: 0.8206  Bal.Acc: 70.4%  F1: 0.7137  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 0.8206)
Epoch [ 10/30]  Train Loss: 0.4388  Bal.Acc: 82.9%  F1: 0.8213  |  Val Loss: 0.8071  Bal.Acc: 73.3%  F1: 0.7301  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 0.8071)
Epoch [ 11/30]  Train Loss: 0.3841  Bal.Acc: 86.1%  F1: 0.8548  |  Val Loss: 0.7906  Bal.Acc: 73.9%  F1: 0.7340  |  LR: 1.00e-04  (15.1s)
 Best checkpoint saved (val_loss: 0.7906)
Epoch [ 12/30]  Train Loss: 0.3736  Bal.Acc: 87.0%  F1: 0.8625  |  Val Loss: 0.8016  Bal.Acc: 73.4%  F1: 0.7298  |  LR: 1.00e-04  (15.2s)
Epoch [ 13/30]  Train Loss: 0.3363  Bal.Acc: 87.0%  F1: 0.8651  |  Val Loss: 0.7271  Bal.Acc: 74.1%  F1: 0.7360  |  LR: 1.00e-04  (13.0s)
 Best checkpoint saved (val_loss: 0.7271)
Epoch [ 14/30]  Train Loss: 0.2999  Bal.Acc: 88.9%  F1: 0.8785  |  Val Loss: 0.7893  Bal.Acc: 72.7%  F1: 0.7293  |  LR: 1.00e-04  (14.9s)
Epoch [ 15/30]  Train Loss: 0.3227  Bal.Acc: 88.0%  F1: 0.8749  |  Val Loss: 0.7336  Bal.Acc: 77.8%  F1: 0.7694  |  LR: 1.00e-04  (12.4s)
Epoch [ 16/30]  Train Loss: 0.2545  Bal.Acc: 90.6%  F1: 0.9045  |  Val Loss: 0.7331  Bal.Acc: 77.6%  F1: 0.7750  |  LR: 1.00e-04  (11.9s)
Epoch [ 17/30]  Train Loss: 0.2389  Bal.Acc: 91.7%  F1: 0.9088  |  Val Loss: 0.7295  Bal.Acc: 77.9%  F1: 0.7787  |  LR: 5.00e-05  (12.5s)
Epoch [ 18/30]  Train Loss: 0.2404  Bal.Acc: 91.2%  F1: 0.9071  |  Val Loss: 0.7251  Bal.Acc: 78.4%  F1: 0.7832  |  LR: 5.00e-05  (12.8s)
 Best checkpoint saved (val_loss: 0.7251)
Epoch [ 19/30]  Train Loss: 0.2006  Bal.Acc: 93.3%  F1: 0.9296  |  Val Loss: 0.7196  Bal.Acc: 77.9%  F1: 0.7752  |  LR: 5.00e-05  (15.0s)
 Best checkpoint saved (val_loss: 0.7196)
Epoch [ 20/30]  Train Loss: 0.1953  Bal.Acc: 93.1%  F1: 0.9245  |  Val Loss: 0.7077  Bal.Acc: 78.5%  F1: 0.7801  |  LR: 5.00e-05  (14.7s)
 Best checkpoint saved (val_loss: 0.7077)
Epoch [ 21/30]  Train Loss: 0.1735  Bal.Acc: 93.7%  F1: 0.9345  |  Val Loss: 0.7173  Bal.Acc: 79.3%  F1: 0.7911  |  LR: 5.00e-05  (14.7s)
Epoch [ 22/30]  Train Loss: 0.1956  Bal.Acc: 93.4%  F1: 0.9328  |  Val Loss: 0.7306  Bal.Acc: 77.2%  F1: 0.7707  |  LR: 5.00e-05  (12.9s)
Epoch [ 23/30]  Train Loss: 0.1859  Bal.Acc: 93.6%  F1: 0.9283  |  Val Loss: 0.6957  Bal.Acc: 78.8%  F1: 0.7818  |  LR: 5.00e-05  (12.9s)
 Best checkpoint saved (val_loss: 0.6957)
Epoch [ 24/30]  Train Loss: 0.1781  Bal.Acc: 94.3%  F1: 0.9425  |  Val Loss: 0.7139  Bal.Acc: 78.4%  F1: 0.7796  |  LR: 5.00e-05  (14.8s)
Epoch [ 25/30]  Train Loss: 0.1608  Bal.Acc: 94.0%  F1: 0.9361  |  Val Loss: 0.7041  Bal.Acc: 79.6%  F1: 0.7969  |  LR: 5.00e-05  (12.8s)
Epoch [ 26/30]  Train Loss: 0.1632  Bal.Acc: 95.1%  F1: 0.9461  |  Val Loss: 0.7172  Bal.Acc: 79.3%  F1: 0.7964  |  LR: 5.00e-05  (12.1s)
Epoch [ 27/30]  Train Loss: 0.1510  Bal.Acc: 94.1%  F1: 0.9392  |  Val Loss: 0.6963  Bal.Acc: 79.4%  F1: 0.7939  |  LR: 2.50e-05  (12.3s)
Epoch [ 28/30]  Train Loss: 0.1476  Bal.Acc: 94.7%  F1: 0.9402  |  Val Loss: 0.6934  Bal.Acc: 79.8%  F1: 0.7999  |  LR: 2.50e-05  (12.9s)
 Best checkpoint saved (val_loss: 0.6934)
Epoch [ 29/30]  Train Loss: 0.1230  Bal.Acc: 95.6%  F1: 0.9569  |  Val Loss: 0.7096  Bal.Acc: 80.3%  F1: 0.7998  |  LR: 2.50e-05  (14.8s)
Epoch [ 30/30]  Train Loss: 0.1401  Bal.Acc: 95.2%  F1: 0.9510  |  Val Loss: 0.6771  Bal.Acc: 80.8%  F1: 0.8036  |  LR: 2.50e-05  (13.1s)
 Best checkpoint saved (val_loss: 0.6771)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold5_best.pt
Log CSV: results/logs/mobilenetv3_large_fold5_training_log.csv
Best weights loaded from epoch 30

Model evaluation: mobilenetv3_large_fold5
----------------------------------------
  Balanced Accuracy:       80.77%
  F1 (macro):              0.8036
  Quadratic Cohen's Kappa: 0.8923
  MAE (ordinal):           0.2464
  Off-by-one accuracy:     95.36%
  ECE:                     0.1035
  Brier Score (mean):      0.0577

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.91      0.90        87
    Doubtful       0.79      0.77      0.78        81
        Mild       0.69      0.62      0.65        39
    Moderate       0.88      0.92      0.90        38
      Severe       0.76      0.83      0.79        35

    accuracy                           0.82       280
   macro avg       0.80      0.81      0.80       280
weighted avg       0.82      0.82      0.82       280

  Metrics saved to: results/individual_models/mobilenetv3_large_fold5_metrics.json
  Probabilities saved to: results/individual_models/mobilenetv3_large_fold5_test_probs.npz

 FINISHED: mobilenetv3_large. Average kappa out of 5 folds: 0.9000 ±0.0139

================================================================================
MODEL TRAINING START: convnext_tiny
================================================================================

--- convnext_tiny | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 1):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: convnext_tiny
model.safetensors: 100% 114M/114M [00:01<00:00, 75.5MB/s]
  Parameters: 27,823,973 total, 27,823,973 trainable

============================================================
TRAINING: convnext_tiny_fold1
============================================================
Epoch [  1/30]  Train Loss: 1.6539  Bal.Acc: 25.8%  F1: 0.2199  |  Val Loss: 1.4977  Bal.Acc: 37.9%  F1: 0.2959  |  LR: 1.00e-04  (36.1s)
 Best checkpoint saved (val_loss: 1.4977)
Epoch [  2/30]  Train Loss: 1.4118  Bal.Acc: 37.7%  F1: 0.3514  |  Val Loss: 1.1279  Bal.Acc: 59.1%  F1: 0.5443  |  LR: 1.00e-04  (22.4s)
 Best checkpoint saved (val_loss: 1.1279)
Epoch [  3/30]  Train Loss: 1.0202  Bal.Acc: 54.4%  F1: 0.5202  |  Val Loss: 1.1899  Bal.Acc: 52.8%  F1: 0.4995  |  LR: 1.00e-04  (28.5s)
Epoch [  4/30]  Train Loss: 0.8108  Bal.Acc: 67.5%  F1: 0.6639  |  Val Loss: 0.6823  Bal.Acc: 74.1%  F1: 0.7164  |  LR: 1.00e-04  (22.8s)
 Best checkpoint saved (val_loss: 0.6823)
Epoch [  5/30]  Train Loss: 0.7634  Bal.Acc: 66.7%  F1: 0.6579  |  Val Loss: 0.6751  Bal.Acc: 73.8%  F1: 0.7060  |  LR: 1.00e-04  (27.6s)
 Best checkpoint saved (val_loss: 0.6751)
Epoch [  6/30]  Train Loss: 0.5900  Bal.Acc: 77.3%  F1: 0.7549  |  Val Loss: 0.6813  Bal.Acc: 74.2%  F1: 0.7450  |  LR: 1.00e-04  (28.2s)
Epoch [  7/30]  Train Loss: 0.5055  Bal.Acc: 80.5%  F1: 0.8013  |  Val Loss: 0.4590  Bal.Acc: 80.6%  F1: 0.7970  |  LR: 1.00e-04  (22.6s)
 Best checkpoint saved (val_loss: 0.4590)
Epoch [  8/30]  Train Loss: 0.4410  Bal.Acc: 81.7%  F1: 0.8083  |  Val Loss: 0.5766  Bal.Acc: 78.8%  F1: 0.7869  |  LR: 1.00e-04  (27.5s)
Epoch [  9/30]  Train Loss: 0.3792  Bal.Acc: 84.3%  F1: 0.8391  |  Val Loss: 1.1886  Bal.Acc: 61.9%  F1: 0.5883  |  LR: 1.00e-04  (22.9s)
Epoch [ 10/30]  Train Loss: 0.4434  Bal.Acc: 81.7%  F1: 0.8070  |  Val Loss: 0.5730  Bal.Acc: 76.1%  F1: 0.7621  |  LR: 1.00e-04  (22.7s)
Epoch [ 11/30]  Train Loss: 0.3490  Bal.Acc: 85.8%  F1: 0.8521  |  Val Loss: 0.4768  Bal.Acc: 81.6%  F1: 0.8086  |  LR: 5.00e-05  (22.5s)
Epoch [ 12/30]  Train Loss: 0.2076  Bal.Acc: 93.1%  F1: 0.9262  |  Val Loss: 0.4371  Bal.Acc: 84.9%  F1: 0.8367  |  LR: 5.00e-05  (22.7s)
 Best checkpoint saved (val_loss: 0.4371)
Epoch [ 13/30]  Train Loss: 0.1875  Bal.Acc: 92.9%  F1: 0.9232  |  Val Loss: 0.4038  Bal.Acc: 85.4%  F1: 0.8472  |  LR: 5.00e-05  (28.7s)
 Best checkpoint saved (val_loss: 0.4038)
Epoch [ 14/30]  Train Loss: 0.1579  Bal.Acc: 93.5%  F1: 0.9334  |  Val Loss: 0.5086  Bal.Acc: 82.0%  F1: 0.8131  |  LR: 5.00e-05  (29.6s)
Epoch [ 15/30]  Train Loss: 0.1513  Bal.Acc: 94.8%  F1: 0.9429  |  Val Loss: 0.4822  Bal.Acc: 83.6%  F1: 0.8475  |  LR: 5.00e-05  (23.1s)
Epoch [ 16/30]  Train Loss: 0.1436  Bal.Acc: 95.1%  F1: 0.9480  |  Val Loss: 0.4760  Bal.Acc: 84.7%  F1: 0.8399  |  LR: 5.00e-05  (22.7s)
Epoch [ 17/30]  Train Loss: 0.1664  Bal.Acc: 93.9%  F1: 0.9339  |  Val Loss: 0.3783  Bal.Acc: 86.8%  F1: 0.8620  |  LR: 5.00e-05  (22.7s)
 Best checkpoint saved (val_loss: 0.3783)
Epoch [ 18/30]  Train Loss: 0.1386  Bal.Acc: 94.1%  F1: 0.9395  |  Val Loss: 0.5125  Bal.Acc: 81.4%  F1: 0.8020  |  LR: 5.00e-05  (28.4s)
Epoch [ 19/30]  Train Loss: 0.1116  Bal.Acc: 95.8%  F1: 0.9526  |  Val Loss: 0.4589  Bal.Acc: 85.0%  F1: 0.8454  |  LR: 5.00e-05  (23.1s)
Epoch [ 20/30]  Train Loss: 0.0711  Bal.Acc: 97.8%  F1: 0.9780  |  Val Loss: 0.5091  Bal.Acc: 84.9%  F1: 0.8469  |  LR: 5.00e-05  (23.0s)
Epoch [ 21/30]  Train Loss: 0.0953  Bal.Acc: 96.7%  F1: 0.9660  |  Val Loss: 0.6294  Bal.Acc: 83.4%  F1: 0.8375  |  LR: 2.50e-05  (22.9s)
Epoch [ 22/30]  Train Loss: 0.0792  Bal.Acc: 97.0%  F1: 0.9686  |  Val Loss: 0.4030  Bal.Acc: 87.8%  F1: 0.8720  |  LR: 2.50e-05  (22.3s)
Epoch [ 23/30]  Train Loss: 0.0573  Bal.Acc: 98.1%  F1: 0.9791  |  Val Loss: 0.4367  Bal.Acc: 86.5%  F1: 0.8653  |  LR: 2.50e-05  (23.0s)
Epoch [ 24/30]  Train Loss: 0.0417  Bal.Acc: 98.6%  F1: 0.9845  |  Val Loss: 0.4705  Bal.Acc: 86.7%  F1: 0.8684  |  LR: 2.50e-05  (22.0s)
Epoch [ 25/30]  Train Loss: 0.0431  Bal.Acc: 98.6%  F1: 0.9862  |  Val Loss: 0.5006  Bal.Acc: 83.9%  F1: 0.8342  |  LR: 1.25e-05  (22.8s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.3783

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold1_best.pt
Log CSV: results/logs/convnext_tiny_fold1_training_log.csv
Best weights loaded from epoch 17

Model evaluation: convnext_tiny_fold1
----------------------------------------
  Balanced Accuracy:       86.77%
  F1 (macro):              0.8620
  Quadratic Cohen's Kappa: 0.9514
  MAE (ordinal):           0.1495
  Off-by-one accuracy:     98.22%
  ECE:                     0.0307
  Brier Score (mean):      0.0383

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.90      0.95      0.93        88
    Doubtful       0.91      0.78      0.84        81
        Mild       0.68      0.80      0.74        40
    Moderate       0.89      0.89      0.89        37
      Severe       0.91      0.91      0.91        35

    accuracy                           0.87       281
   macro avg       0.86      0.87      0.86       281
weighted avg       0.87      0.87      0.87       281

  Metrics saved to: results/individual_models/convnext_tiny_fold1_metrics.json
  Probabilities saved to: results/individual_models/convnext_tiny_fold1_test_probs.npz

--- convnext_tiny | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 images
    Val:   281 images

    Class weights (fold 2):
      Class 0 (Normal): weight = 0.642  (count = 349)
      Class 1 (Doubtful): weight = 0.692  (count = 324)
      Class 2 (Mild): weight = 1.428  (count = 157)
      Class 3 (Moderate): weight = 1.485  (count = 151)
      Class 4 (Severe): weight = 1.601  (count = 140)

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

============================================================
TRAINING: convnext_tiny_fold2
============================================================
Epoch [  1/30]  Train Loss: 1.6952  Bal.Acc: 26.2%  F1: 0.2341  |  Val Loss: 1.5859  Bal.Acc: 23.8%  F1: 0.1650  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 1.5859)
Epoch [  2/30]  Train Loss: 1.4480  Bal.Acc: 34.4%  F1: 0.3124  |  Val Loss: 1.2391  Bal.Acc: 48.5%  F1: 0.4527  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 1.2391)
Epoch [  3/30]  Train Loss: 1.1075  Bal.Acc: 53.7%  F1: 0.5094  |  Val Loss: 1.0096  Bal.Acc: 58.5%  F1: 0.6010  |  LR: 1.00e-04  (28.8s)
 Best checkpoint saved (val_loss: 1.0096)
Epoch [  4/30]  Train Loss: 0.8203  Bal.Acc: 66.5%  F1: 0.6542  |  Val Loss: 0.6706  Bal.Acc: 75.3%  F1: 0.7228  |  LR: 1.00e-04  (28.7s)
 Best checkpoint saved (val_loss: 0.6706)
Epoch [  5/30]  Train Loss: 0.6797  Bal.Acc: 73.0%  F1: 0.7199  |  Val Loss: 0.7394  Bal.Acc: 68.5%  F1: 0.6950  |  LR: 1.00e-04  (35.5s)
Epoch [  6/30]  Train Loss: 0.5996  Bal.Acc: 75.9%  F1: 0.7493  |  Val Loss: 0.7331  Bal.Acc: 68.3%  F1: 0.6672  |  LR: 1.00e-04  (22.9s)
Epoch [  7/30]  Train Loss: 0.5822  Bal.Acc: 76.5%  F1: 0.7549  |  Val Loss: 0.6019  Bal.Acc: 77.9%  F1: 0.7398  |  LR: 1.00e-04  (22.3s)
 Best checkpoint saved (val_loss: 0.6019)
Epoch [  8/30]  Train Loss: 0.5793  Bal.Acc: 77.6%  F1: 0.7646  |  Val Loss: 0.9414  Bal.Acc: 63.6%  F1: 0.6586  |  LR: 1.00e-04  (29.7s)
Epoch [  9/30]  Train Loss: 0.4332  Bal.Acc: 83.3%  F1: 0.8297  |  Val Loss: 0.5798  Bal.Acc: 79.3%  F1: 0.7591  |  LR: 1.00e-04  (23.0s)
 Best checkpoint saved (val_loss: 0.5798)
Epoch [ 10/30]  Train Loss: 0.3825  Bal.Acc: 83.4%  F1: 0.8270  |  Val Loss: 0.6529  Bal.Acc: 77.3%  F1: 0.7545  |  LR: 1.00e-04  (29.2s)
Epoch [ 11/30]  Train Loss: 0.3544  Bal.Acc: 86.5%  F1: 0.8607  |  Val Loss: 0.6071  Bal.Acc: 77.7%  F1: 0.7600  |  LR: 1.00e-04  (22.8s)
Epoch [ 12/30]  Train Loss: 0.3071  Bal.Acc: 87.5%  F1: 0.8710  |  Val Loss: 0.4831  Bal.Acc: 81.7%  F1: 0.8084  |  LR: 1.00e-04  (23.2s)
 Best checkpoint saved (val_loss: 0.4831)
Epoch [ 13/30]  Train Loss: 0.2715  Bal.Acc: 89.7%  F1: 0.8866  |  Val Loss: 0.7020  Bal.Acc: 79.6%  F1: 0.8075  |  LR: 1.00e-04  (29.9s)
Epoch [ 14/30]  Train Loss: 0.2291  Bal.Acc: 91.6%  F1: 0.9121  |  Val Loss: 0.7804  Bal.Acc: 74.6%  F1: 0.7533  |  LR: 1.00e-04  (23.0s)
Epoch [ 15/30]  Train Loss: 0.2135  Bal.Acc: 91.1%  F1: 0.9095  |  Val Loss: 0.5846  Bal.Acc: 82.5%  F1: 0.8270  |  LR: 1.00e-04  (23.3s)
Epoch [ 16/30]  Train Loss: 0.2209  Bal.Acc: 92.3%  F1: 0.9194  |  Val Loss: 0.5767  Bal.Acc: 79.1%  F1: 0.7835  |  LR: 5.00e-05  (21.9s)
Epoch [ 17/30]  Train Loss: 0.1364  Bal.Acc: 95.2%  F1: 0.9469  |  Val Loss: 0.6204  Bal.Acc: 81.0%  F1: 0.8168  |  LR: 5.00e-05  (22.9s)
Epoch [ 18/30]  Train Loss: 0.1078  Bal.Acc: 96.0%  F1: 0.9581  |  Val Loss: 0.6683  Bal.Acc: 80.5%  F1: 0.8186  |  LR: 5.00e-05  (22.4s)
Epoch [ 19/30]  Train Loss: 0.0768  Bal.Acc: 97.6%  F1: 0.9742  |  Val Loss: 0.7827  Bal.Acc: 81.1%  F1: 0.8218  |  LR: 5.00e-05  (23.1s)
Epoch [ 20/30]  Train Loss: 0.0718  Bal.Acc: 97.4%  F1: 0.9717  |  Val Loss: 0.7668  Bal.Acc: 80.7%  F1: 0.8169  |  LR: 2.50e-05  (21.9s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.4831

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold2_best.pt
Log CSV: results/logs/convnext_tiny_fold2_training_log.csv
Best weights loaded from epoch 12

Model evaluation: convnext_tiny_fold2
----------------------------------------
  Balanced Accuracy:       81.73%
  F1 (macro):              0.8084
  Quadratic Cohen's Kappa: 0.9273
  MAE (ordinal):           0.2242
  Off-by-one accuracy:     98.22%
  ECE:                     0.0708
  Brier Score (mean):      0.0572

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.84      0.85        88
    Doubtful       0.75      0.67      0.71        81
        Mild       0.64      0.80      0.71        40
    Moderate       0.86      0.86      0.86        37
      Severe       0.91      0.91      0.91        35

    accuracy                           0.80       281
   macro avg       0.80      0.82      0.81       281
weighted avg       0.80      0.80      0.80       281

  Metrics saved to: results/individual_models/convnext_tiny_fold2_metrics.json
  Probabilities saved to: results/individual_models/convnext_tiny_fold2_test_probs.npz

--- convnext_tiny | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 3):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

============================================================
TRAINING: convnext_tiny_fold3
============================================================
Epoch [  1/30]  Train Loss: 1.7617  Bal.Acc: 20.4%  F1: 0.2017  |  Val Loss: 1.6210  Bal.Acc: 20.0%  F1: 0.0953  |  LR: 1.00e-04  (29.1s)
 Best checkpoint saved (val_loss: 1.6210)
Epoch [  2/30]  Train Loss: 1.5993  Bal.Acc: 22.4%  F1: 0.2014  |  Val Loss: 1.5401  Bal.Acc: 29.2%  F1: 0.1851  |  LR: 1.00e-04  (22.9s)
 Best checkpoint saved (val_loss: 1.5401)
Epoch [  3/30]  Train Loss: 1.5050  Bal.Acc: 31.7%  F1: 0.2830  |  Val Loss: 1.4348  Bal.Acc: 32.1%  F1: 0.3187  |  LR: 1.00e-04  (29.0s)
 Best checkpoint saved (val_loss: 1.4348)
Epoch [  4/30]  Train Loss: 1.4422  Bal.Acc: 36.4%  F1: 0.3227  |  Val Loss: 1.4001  Bal.Acc: 34.7%  F1: 0.3130  |  LR: 1.00e-04  (29.2s)
 Best checkpoint saved (val_loss: 1.4001)
Epoch [  5/30]  Train Loss: 1.3170  Bal.Acc: 42.8%  F1: 0.3932  |  Val Loss: 1.1580  Bal.Acc: 49.9%  F1: 0.4584  |  LR: 1.00e-04  (29.2s)
 Best checkpoint saved (val_loss: 1.1580)
Epoch [  6/30]  Train Loss: 0.9644  Bal.Acc: 59.7%  F1: 0.5744  |  Val Loss: 1.0792  Bal.Acc: 59.5%  F1: 0.5976  |  LR: 1.00e-04  (30.3s)
 Best checkpoint saved (val_loss: 1.0792)
Epoch [  7/30]  Train Loss: 0.8921  Bal.Acc: 64.3%  F1: 0.6316  |  Val Loss: 0.8565  Bal.Acc: 63.9%  F1: 0.6303  |  LR: 1.00e-04  (28.9s)
 Best checkpoint saved (val_loss: 0.8565)
Epoch [  8/30]  Train Loss: 0.7552  Bal.Acc: 67.6%  F1: 0.6588  |  Val Loss: 0.6384  Bal.Acc: 74.4%  F1: 0.7238  |  LR: 1.00e-04  (36.9s)
 Best checkpoint saved (val_loss: 0.6384)
Epoch [  9/30]  Train Loss: 0.5812  Bal.Acc: 75.8%  F1: 0.7456  |  Val Loss: 1.0365  Bal.Acc: 55.8%  F1: 0.5788  |  LR: 1.00e-04  (29.8s)
Epoch [ 10/30]  Train Loss: 0.5438  Bal.Acc: 77.1%  F1: 0.7639  |  Val Loss: 0.5350  Bal.Acc: 78.2%  F1: 0.7664  |  LR: 1.00e-04  (22.9s)
 Best checkpoint saved (val_loss: 0.5350)
Epoch [ 11/30]  Train Loss: 0.5189  Bal.Acc: 80.3%  F1: 0.7931  |  Val Loss: 0.5838  Bal.Acc: 75.5%  F1: 0.7097  |  LR: 1.00e-04  (28.8s)
Epoch [ 12/30]  Train Loss: 0.5197  Bal.Acc: 78.0%  F1: 0.7704  |  Val Loss: 0.7969  Bal.Acc: 68.7%  F1: 0.7019  |  LR: 1.00e-04  (23.4s)
Epoch [ 13/30]  Train Loss: 0.4408  Bal.Acc: 83.4%  F1: 0.8293  |  Val Loss: 0.5265  Bal.Acc: 76.7%  F1: 0.7591  |  LR: 1.00e-04  (22.5s)
 Best checkpoint saved (val_loss: 0.5265)
Epoch [ 14/30]  Train Loss: 0.3577  Bal.Acc: 86.0%  F1: 0.8521  |  Val Loss: 0.7363  Bal.Acc: 75.2%  F1: 0.7275  |  LR: 1.00e-04  (28.5s)
Epoch [ 15/30]  Train Loss: 0.3564  Bal.Acc: 87.3%  F1: 0.8640  |  Val Loss: 0.5687  Bal.Acc: 78.5%  F1: 0.7802  |  LR: 1.00e-04  (23.7s)
Epoch [ 16/30]  Train Loss: 0.2681  Bal.Acc: 89.8%  F1: 0.8914  |  Val Loss: 0.8083  Bal.Acc: 75.3%  F1: 0.7588  |  LR: 1.00e-04  (22.3s)
Epoch [ 17/30]  Train Loss: 0.2595  Bal.Acc: 90.1%  F1: 0.8963  |  Val Loss: 0.4731  Bal.Acc: 82.9%  F1: 0.8157  |  LR: 1.00e-04  (22.9s)
 Best checkpoint saved (val_loss: 0.4731)
Epoch [ 18/30]  Train Loss: 0.2214  Bal.Acc: 92.0%  F1: 0.9155  |  Val Loss: 0.5911  Bal.Acc: 79.7%  F1: 0.7999  |  LR: 1.00e-04  (28.4s)
Epoch [ 19/30]  Train Loss: 0.1938  Bal.Acc: 92.6%  F1: 0.9226  |  Val Loss: 0.7500  Bal.Acc: 77.1%  F1: 0.7636  |  LR: 1.00e-04  (23.3s)
Epoch [ 20/30]  Train Loss: 0.2394  Bal.Acc: 91.6%  F1: 0.9088  |  Val Loss: 0.6172  Bal.Acc: 77.4%  F1: 0.7664  |  LR: 1.00e-04  (22.8s)
Epoch [ 21/30]  Train Loss: 0.1800  Bal.Acc: 92.8%  F1: 0.9260  |  Val Loss: 0.5627  Bal.Acc: 84.5%  F1: 0.8493  |  LR: 5.00e-05  (22.9s)
Epoch [ 22/30]  Train Loss: 0.1529  Bal.Acc: 94.5%  F1: 0.9397  |  Val Loss: 0.6177  Bal.Acc: 78.4%  F1: 0.7842  |  LR: 5.00e-05  (23.0s)
Epoch [ 23/30]  Train Loss: 0.1013  Bal.Acc: 96.3%  F1: 0.9606  |  Val Loss: 0.7100  Bal.Acc: 81.8%  F1: 0.8266  |  LR: 5.00e-05  (22.6s)
Epoch [ 24/30]  Train Loss: 0.0857  Bal.Acc: 96.4%  F1: 0.9630  |  Val Loss: 0.6087  Bal.Acc: 83.7%  F1: 0.8336  |  LR: 5.00e-05  (22.9s)
Epoch [ 25/30]  Train Loss: 0.0858  Bal.Acc: 96.5%  F1: 0.9635  |  Val Loss: 0.6336  Bal.Acc: 82.9%  F1: 0.8177  |  LR: 2.50e-05  (22.1s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.4731

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold3_best.pt
Log CSV: results/logs/convnext_tiny_fold3_training_log.csv
Best weights loaded from epoch 17

Model evaluation: convnext_tiny_fold3
----------------------------------------
  Balanced Accuracy:       82.93%
  F1 (macro):              0.8157
  Quadratic Cohen's Kappa: 0.9347
  MAE (ordinal):           0.2036
  Off-by-one accuracy:     98.57%
  ECE:                     0.0671
  Brier Score (mean):      0.0553

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.90      0.88        87
    Doubtful       0.83      0.65      0.73        81
        Mild       0.63      0.85      0.73        39
    Moderate       0.83      0.92      0.88        38
      Severe       0.91      0.83      0.87        35

    accuracy                           0.81       280
   macro avg       0.81      0.83      0.82       280
weighted avg       0.82      0.81      0.81       280

  Metrics saved to: results/individual_models/convnext_tiny_fold3_metrics.json
  Probabilities saved to: results/individual_models/convnext_tiny_fold3_test_probs.npz

--- convnext_tiny | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 images
    Val:   280 images

    Class weights (fold 4):
      Class 0 (Normal): weight = 0.641  (count = 350)
      Class 1 (Doubtful): weight = 0.693  (count = 324)
      Class 2 (Mild): weight = 1.420  (count = 158)
      Class 3 (Moderate): weight = 1.496  (count = 150)
      Class 4 (Severe): weight = 1.603  (count = 140)

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

============================================================
TRAINING: convnext_tiny_fold4
============================================================
Epoch [  1/30]  Train Loss: 1.7555  Bal.Acc: 26.2%  F1: 0.2404  |  Val Loss: 1.5392  Bal.Acc: 30.4%  F1: 0.1868  |  LR: 1.00e-04  (23.1s)
 Best checkpoint saved (val_loss: 1.5392)
Epoch [  2/30]  Train Loss: 1.4402  Bal.Acc: 35.3%  F1: 0.3213  |  Val Loss: 1.4314  Bal.Acc: 35.7%  F1: 0.2900  |  LR: 1.00e-04  (23.5s)
 Best checkpoint saved (val_loss: 1.4314)
Epoch [  3/30]  Train Loss: 1.0636  Bal.Acc: 51.9%  F1: 0.5046  |  Val Loss: 0.9961  Bal.Acc: 61.8%  F1: 0.5633  |  LR: 1.00e-04  (28.7s)
 Best checkpoint saved (val_loss: 0.9961)
Epoch [  4/30]  Train Loss: 0.9246  Bal.Acc: 62.2%  F1: 0.6048  |  Val Loss: 0.7728  Bal.Acc: 66.8%  F1: 0.6438  |  LR: 1.00e-04  (30.5s)
 Best checkpoint saved (val_loss: 0.7728)
Epoch [  5/30]  Train Loss: 0.6111  Bal.Acc: 75.8%  F1: 0.7431  |  Val Loss: 0.9213  Bal.Acc: 61.5%  F1: 0.6078  |  LR: 1.00e-04  (28.0s)
Epoch [  6/30]  Train Loss: 0.5625  Bal.Acc: 76.9%  F1: 0.7592  |  Val Loss: 0.6942  Bal.Acc: 72.0%  F1: 0.7058  |  LR: 1.00e-04  (22.6s)
 Best checkpoint saved (val_loss: 0.6942)
Epoch [  7/30]  Train Loss: 0.4988  Bal.Acc: 80.2%  F1: 0.7939  |  Val Loss: 1.1579  Bal.Acc: 59.3%  F1: 0.5944  |  LR: 1.00e-04  (28.0s)
Epoch [  8/30]  Train Loss: 0.4815  Bal.Acc: 82.0%  F1: 0.8126  |  Val Loss: 0.6803  Bal.Acc: 75.6%  F1: 0.7550  |  LR: 1.00e-04  (23.2s)
 Best checkpoint saved (val_loss: 0.6803)
Epoch [  9/30]  Train Loss: 0.3451  Bal.Acc: 87.0%  F1: 0.8641  |  Val Loss: 0.7728  Bal.Acc: 71.4%  F1: 0.6877  |  LR: 1.00e-04  (28.4s)
Epoch [ 10/30]  Train Loss: 0.3447  Bal.Acc: 86.5%  F1: 0.8571  |  Val Loss: 0.5875  Bal.Acc: 77.4%  F1: 0.7623  |  LR: 1.00e-04  (22.4s)
 Best checkpoint saved (val_loss: 0.5875)
Epoch [ 11/30]  Train Loss: 0.2911  Bal.Acc: 89.7%  F1: 0.8926  |  Val Loss: 0.5534  Bal.Acc: 81.7%  F1: 0.7967  |  LR: 1.00e-04  (28.7s)
 Best checkpoint saved (val_loss: 0.5534)
Epoch [ 12/30]  Train Loss: 0.2077  Bal.Acc: 92.3%  F1: 0.9181  |  Val Loss: 0.8133  Bal.Acc: 72.2%  F1: 0.7265  |  LR: 1.00e-04  (27.8s)
Epoch [ 13/30]  Train Loss: 0.2258  Bal.Acc: 91.6%  F1: 0.9132  |  Val Loss: 0.5890  Bal.Acc: 81.0%  F1: 0.7993  |  LR: 1.00e-04  (22.8s)
Epoch [ 14/30]  Train Loss: 0.2729  Bal.Acc: 90.3%  F1: 0.8985  |  Val Loss: 0.5823  Bal.Acc: 78.3%  F1: 0.7493  |  LR: 1.00e-04  (22.8s)
Epoch [ 15/30]  Train Loss: 0.2358  Bal.Acc: 91.5%  F1: 0.9075  |  Val Loss: 0.8284  Bal.Acc: 75.7%  F1: 0.7572  |  LR: 5.00e-05  (22.7s)
Epoch [ 16/30]  Train Loss: 0.1955  Bal.Acc: 92.8%  F1: 0.9274  |  Val Loss: 0.5704  Bal.Acc: 81.3%  F1: 0.8022  |  LR: 5.00e-05  (22.9s)
Epoch [ 17/30]  Train Loss: 0.1330  Bal.Acc: 94.9%  F1: 0.9443  |  Val Loss: 0.5271  Bal.Acc: 81.8%  F1: 0.8089  |  LR: 5.00e-05  (22.9s)
 Best checkpoint saved (val_loss: 0.5271)
Epoch [ 18/30]  Train Loss: 0.0854  Bal.Acc: 96.8%  F1: 0.9667  |  Val Loss: 0.6352  Bal.Acc: 82.0%  F1: 0.8164  |  LR: 5.00e-05  (29.3s)
Epoch [ 19/30]  Train Loss: 0.0655  Bal.Acc: 97.7%  F1: 0.9748  |  Val Loss: 0.6441  Bal.Acc: 81.9%  F1: 0.8168  |  LR: 5.00e-05  (22.9s)
Epoch [ 20/30]  Train Loss: 0.0782  Bal.Acc: 97.1%  F1: 0.9687  |  Val Loss: 0.7444  Bal.Acc: 82.0%  F1: 0.8150  |  LR: 5.00e-05  (22.9s)
Epoch [ 21/30]  Train Loss: 0.0993  Bal.Acc: 96.6%  F1: 0.9646  |  Val Loss: 0.6563  Bal.Acc: 84.0%  F1: 0.8302  |  LR: 2.50e-05  (21.9s)
Epoch [ 22/30]  Train Loss: 0.0556  Bal.Acc: 98.1%  F1: 0.9796  |  Val Loss: 0.6342  Bal.Acc: 81.1%  F1: 0.8103  |  LR: 2.50e-05  (23.0s)
Epoch [ 23/30]  Train Loss: 0.0444  Bal.Acc: 98.5%  F1: 0.9843  |  Val Loss: 0.6727  Bal.Acc: 83.1%  F1: 0.8248  |  LR: 2.50e-05  (22.3s)
Epoch [ 24/30]  Train Loss: 0.0522  Bal.Acc: 97.9%  F1: 0.9778  |  Val Loss: 0.6664  Bal.Acc: 82.6%  F1: 0.8245  |  LR: 2.50e-05  (22.9s)
Epoch [ 25/30]  Train Loss: 0.0402  Bal.Acc: 98.5%  F1: 0.9847  |  Val Loss: 0.6845  Bal.Acc: 83.5%  F1: 0.8286  |  LR: 1.25e-05  (22.1s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.5271

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold4_best.pt
Log CSV: results/logs/convnext_tiny_fold4_training_log.csv
Best weights loaded from epoch 17

Model evaluation: convnext_tiny_fold4
----------------------------------------
  Balanced Accuracy:       81.76%
  F1 (macro):              0.8089
  Quadratic Cohen's Kappa: 0.9404
  MAE (ordinal):           0.2000
  Off-by-one accuracy:     99.29%
  ECE:                     0.1067
  Brier Score (mean):      0.0586

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.92      0.89        87
    Doubtful       0.79      0.67      0.72        81
        Mild       0.60      0.67      0.63        39
    Moderate       0.81      0.92      0.86        38
      Severe       0.94      0.91      0.93        35

    accuracy                           0.81       280
   macro avg       0.80      0.82      0.81       280
weighted avg       0.81      0.81      0.81       280

  Metrics saved to: results/individual_models/convnext_tiny_fold4_metrics.json
  Probabilities saved to: results/individual_models/convnext_tiny_fold4_test_probs.npz

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

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable

============================================================
TRAINING: convnext_tiny_fold5
============================================================
Epoch [  1/30]  Train Loss: 1.6634  Bal.Acc: 26.5%  F1: 0.2304  |  Val Loss: 1.4882  Bal.Acc: 38.1%  F1: 0.3021  |  LR: 1.00e-04  (22.9s)
 Best checkpoint saved (val_loss: 1.4882)
Epoch [  2/30]  Train Loss: 1.3670  Bal.Acc: 43.0%  F1: 0.4046  |  Val Loss: 1.1502  Bal.Acc: 55.2%  F1: 0.4983  |  LR: 1.00e-04  (35.2s)
 Best checkpoint saved (val_loss: 1.1502)
Epoch [  3/30]  Train Loss: 0.9937  Bal.Acc: 58.1%  F1: 0.5557  |  Val Loss: 0.9105  Bal.Acc: 60.9%  F1: 0.6035  |  LR: 1.00e-04  (28.2s)
 Best checkpoint saved (val_loss: 0.9105)
Epoch [  4/30]  Train Loss: 0.7241  Bal.Acc: 71.0%  F1: 0.6972  |  Val Loss: 0.6943  Bal.Acc: 75.1%  F1: 0.7418  |  LR: 1.00e-04  (29.3s)
 Best checkpoint saved (val_loss: 0.6943)
Epoch [  5/30]  Train Loss: 0.7410  Bal.Acc: 71.2%  F1: 0.7018  |  Val Loss: 0.6682  Bal.Acc: 72.6%  F1: 0.7243  |  LR: 1.00e-04  (36.2s)
 Best checkpoint saved (val_loss: 0.6682)
Epoch [  6/30]  Train Loss: 0.5566  Bal.Acc: 77.6%  F1: 0.7679  |  Val Loss: 0.6456  Bal.Acc: 76.8%  F1: 0.7647  |  LR: 1.00e-04  (34.9s)
 Best checkpoint saved (val_loss: 0.6456)
Epoch [  7/30]  Train Loss: 0.5008  Bal.Acc: 80.3%  F1: 0.7954  |  Val Loss: 0.7156  Bal.Acc: 71.9%  F1: 0.7014  |  LR: 1.00e-04  (28.2s)
Epoch [  8/30]  Train Loss: 0.4573  Bal.Acc: 81.5%  F1: 0.8091  |  Val Loss: 0.7613  Bal.Acc: 74.0%  F1: 0.7246  |  LR: 1.00e-04  (23.0s)
Epoch [  9/30]  Train Loss: 0.3522  Bal.Acc: 86.2%  F1: 0.8557  |  Val Loss: 0.6420  Bal.Acc: 79.9%  F1: 0.7841  |  LR: 1.00e-04  (23.0s)
 Best checkpoint saved (val_loss: 0.6420)
Epoch [ 10/30]  Train Loss: 0.3562  Bal.Acc: 85.3%  F1: 0.8454  |  Val Loss: 0.7000  Bal.Acc: 78.5%  F1: 0.7907  |  LR: 1.00e-04  (29.3s)
Epoch [ 11/30]  Train Loss: 0.4027  Bal.Acc: 84.2%  F1: 0.8398  |  Val Loss: 0.5849  Bal.Acc: 80.3%  F1: 0.7915  |  LR: 1.00e-04  (23.3s)
 Best checkpoint saved (val_loss: 0.5849)
Epoch [ 12/30]  Train Loss: 0.3972  Bal.Acc: 84.9%  F1: 0.8420  |  Val Loss: 0.6236  Bal.Acc: 82.3%  F1: 0.8286  |  LR: 1.00e-04  (27.7s)
Epoch [ 13/30]  Train Loss: 0.2794  Bal.Acc: 89.7%  F1: 0.8951  |  Val Loss: 0.5889  Bal.Acc: 81.6%  F1: 0.8072  |  LR: 1.00e-04  (23.0s)
Epoch [ 14/30]  Train Loss: 0.3416  Bal.Acc: 87.5%  F1: 0.8732  |  Val Loss: 0.5230  Bal.Acc: 84.0%  F1: 0.8156  |  LR: 1.00e-04  (23.2s)
 Best checkpoint saved (val_loss: 0.5230)
Epoch [ 15/30]  Train Loss: 0.2947  Bal.Acc: 88.1%  F1: 0.8730  |  Val Loss: 0.5809  Bal.Acc: 81.6%  F1: 0.8065  |  LR: 1.00e-04  (29.8s)
Epoch [ 16/30]  Train Loss: 0.1849  Bal.Acc: 93.9%  F1: 0.9362  |  Val Loss: 0.6784  Bal.Acc: 81.4%  F1: 0.8133  |  LR: 1.00e-04  (23.3s)
Epoch [ 17/30]  Train Loss: 0.1645  Bal.Acc: 94.0%  F1: 0.9368  |  Val Loss: 0.7345  Bal.Acc: 80.8%  F1: 0.8036  |  LR: 1.00e-04  (23.0s)
Epoch [ 18/30]  Train Loss: 0.1741  Bal.Acc: 93.4%  F1: 0.9320  |  Val Loss: 0.7444  Bal.Acc: 82.3%  F1: 0.8124  |  LR: 5.00e-05  (22.2s)
Epoch [ 19/30]  Train Loss: 0.1421  Bal.Acc: 94.6%  F1: 0.9441  |  Val Loss: 0.6342  Bal.Acc: 84.3%  F1: 0.8344  |  LR: 5.00e-05  (23.0s)
Epoch [ 20/30]  Train Loss: 0.0780  Bal.Acc: 97.2%  F1: 0.9711  |  Val Loss: 0.6636  Bal.Acc: 85.3%  F1: 0.8486  |  LR: 5.00e-05  (22.4s)
Epoch [ 21/30]  Train Loss: 0.0754  Bal.Acc: 97.2%  F1: 0.9691  |  Val Loss: 0.8010  Bal.Acc: 82.2%  F1: 0.8171  |  LR: 5.00e-05  (23.0s)
Epoch [ 22/30]  Train Loss: 0.0679  Bal.Acc: 97.3%  F1: 0.9736  |  Val Loss: 0.7617  Bal.Acc: 85.8%  F1: 0.8452  |  LR: 2.50e-05  (21.9s)

  Early stopping due to lack of improvement 8 epoch.
  Best val_loss: 0.5230

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold5_best.pt
Log CSV: results/logs/convnext_tiny_fold5_training_log.csv
Best weights loaded from epoch 14

Model evaluation: convnext_tiny_fold5
----------------------------------------
  Balanced Accuracy:       84.02%
  F1 (macro):              0.8156
  Quadratic Cohen's Kappa: 0.9205
  MAE (ordinal):           0.2179
  Off-by-one accuracy:     97.14%
  ECE:                     0.0400
  Brier Score (mean):      0.0570

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.92      0.91      0.91        87
    Doubtful       0.85      0.62      0.71        81
        Mild       0.61      0.90      0.73        39
    Moderate       0.81      0.92      0.86        38
      Severe       0.86      0.86      0.86        35

    accuracy                           0.82       280
   macro avg       0.81      0.84      0.82       280
weighted avg       0.83      0.82      0.82       280

  Metrics saved to: results/individual_models/convnext_tiny_fold5_metrics.json
  Probabilities saved to: results/individual_models/convnext_tiny_fold5_test_probs.npz

 FINISHED: convnext_tiny. Average kappa out of 5 folds: 0.9349 ±0.0107

=======================================================================================================================================
Single Fold Summary:
=======================================================================================================================================
Model                        Kappa   F1-Mac     MAE   Off-1     ECE   Brier |     KL0     KL1     KL2     KL3     KL4
---------------------------------------------------------------------------------------------------------------------------------------
densenet121_fold1           0.9517   0.8966  0.1281  0.9715  0.0396  0.0322 |  0.9290  0.8831  0.8354  0.9211  0.9143
convnext_tiny_fold1         0.9514   0.8620  0.1495  0.9822  0.0307  0.0383 |  0.9282  0.8400  0.7356  0.8919  0.9143
convnext_tiny_fold4         0.9404   0.8089  0.2000  0.9929  0.1067  0.0586 |  0.8939  0.7248  0.6341  0.8642  0.9275
convnext_tiny_fold3         0.9347   0.8157  0.2036  0.9857  0.0671  0.0553 |  0.8814  0.7310  0.7253  0.8750  0.8657
resnet50_fold1              0.9333   0.8430  0.1851  0.9751  0.0439  0.0480 |  0.8817  0.7945  0.7816  0.9000  0.8571
densenet121_fold4           0.9332   0.8352  0.1857  0.9786  0.0750  0.0479 |  0.9282  0.7733  0.6914  0.8974  0.8857
densenet121_fold2           0.9320   0.8379  0.1957  0.9786  0.0573  0.0528 |  0.8675  0.7738  0.7143  0.9041  0.9296
resnet50_fold2              0.9308   0.8058  0.2242  0.9858  0.0539  0.0603 |  0.8410  0.6712  0.7160  0.8732  0.9275
densenet121_fold5           0.9281   0.8320  0.2000  0.9679  0.0419  0.0517 |  0.8977  0.7763  0.7500  0.8750  0.8611
convnext_tiny_fold2         0.9273   0.8084  0.2242  0.9822  0.0708  0.0572 |  0.8457  0.7059  0.7111  0.8649  0.9143
mobilenetv3_large_fold1     0.9246   0.8192  0.2135  0.9715  0.0559  0.0515 |  0.8729  0.7564  0.6494  0.8780  0.9394
convnext_tiny_fold5         0.9205   0.8156  0.2179  0.9714  0.0400  0.0570 |  0.9133  0.7143  0.7292  0.8642  0.8571
resnet50_fold3              0.9148   0.7719  0.2500  0.9750  0.0313  0.0606 |  0.8471  0.7381  0.5758  0.8095  0.8889
resnet50_fold5              0.9091   0.8060  0.2357  0.9607  0.0507  0.0598 |  0.8913  0.7448  0.7160  0.8831  0.7945
densenet121_fold3           0.9086   0.7968  0.2429  0.9714  0.0963  0.0584 |  0.8652  0.7170  0.6176  0.8675  0.9167
efficientnet_b3_fold4       0.9085   0.7784  0.2500  0.9750  0.1097  0.0620 |  0.8851  0.7160  0.5455  0.8500  0.8955
resnet50_fold4              0.9055   0.7599  0.2714  0.9714  0.0867  0.0653 |  0.8865  0.6522  0.5682  0.8158  0.8767
mobilenetv3_large_fold2     0.9018   0.7711  0.2776  0.9680  0.1296  0.0698 |  0.8197  0.6753  0.6047  0.8571  0.8986
mobilenetv3_large_fold3     0.8986   0.8007  0.2464  0.9571  0.0838  0.0608 |  0.8588  0.7811  0.6400  0.8750  0.8485
efficientnet_b3_fold2       0.8958   0.7630  0.2847  0.9680  0.1252  0.0732 |  0.8404  0.6538  0.5679  0.8571  0.8955
efficientnet_b3_fold5       0.8957   0.7869  0.2643  0.9679  0.1009  0.0663 |  0.8432  0.6974  0.7200  0.8684  0.8056
mobilenetv3_large_fold5     0.8923   0.8036  0.2464  0.9536  0.1035  0.0577 |  0.8977  0.7799  0.6486  0.8974  0.7945
efficientnet_b3_fold3       0.8851   0.7768  0.2857  0.9500  0.1131  0.0698 |  0.8222  0.7215  0.5833  0.9000  0.8571
mobilenetv3_large_fold4     0.8829   0.7941  0.2643  0.9500  0.1039  0.0614 |  0.8786  0.7625  0.6914  0.8500  0.7879
efficientnet_b3_fold1       0.6274   0.5676  0.6726  0.8327  0.1932  0.1282 |  0.6784  0.4748  0.3562  0.6970  0.6316
=======================================================================================================================================
Sorted by Cohen's Kappa

=============================================================================================================================
 CROSS-VALIDATION SUMMARY — AVERAGE OUT OF 5 FOLDs
=============================================================================================================================
Model                         Kappa         F1-Mac        MAE      Off-1 |      KL0      KL1      KL2      KL3      KL4
-----------------------------------------------------------------------------------------------------------------------------
resnet50              0.9187 ±0.0113  0.7973 ±0.0293  0.2333 ±0.0288  0.9736 ±0.0081 |   0.8695   0.7202   0.6715   0.8563   0.8689
efficientnet_b3       0.8425 ±0.1078  0.7345 ±0.0838  0.3515 ±0.1611  0.9387 ±0.0537 |   0.8139   0.6527   0.5546   0.8345   0.8171
densenet121           0.9307 ±0.0137  0.8397 ±0.0321  0.1905 ±0.0368  0.9736 ±0.0043 |   0.8975   0.7847   0.7217   0.8930   0.9015
mobilenetv3_large     0.9000 ±0.0139  0.7977 ±0.0157  0.2496 ±0.0216  0.9600 ±0.0083 |   0.8655   0.7510   0.6468   0.8715   0.8538
convnext_tiny         0.9349 ±0.0107  0.8221 ±0.0202  0.1990 ±0.0263  0.9829 ±0.0069 |   0.8925   0.7432   0.7071   0.8720   0.8958
=============================================================================================================================

================================================================================
 BEST-FOLD HOLDOUT EVALUATION — Individual Models
================================================================================

  resnet50  (best CV fold: 1)

 Building model: resnet50
  Parameters: 23,518,277 total, 23,518,277 trainable
  Holdout Kappa: 0.8979  F1: 0.7791  MAE: 0.2742  Off-1: 0.9677  ECE: 0.0475
  Saved: resnet50_best_fold_holdout_metrics.json

  efficientnet_b3  (best CV fold: 4)

 Building model: efficientnet_b3
  Parameters: 10,703,917 total, 10,703,917 trainable
  Holdout Kappa: 0.8911  F1: 0.8140  MAE: 0.2500  Off-1: 0.9597  ECE: 0.1000
  Saved: efficientnet_b3_best_fold_holdout_metrics.json

  densenet121  (best CV fold: 1)

 Building model: densenet121
  Parameters: 6,958,981 total, 6,958,981 trainable
  Holdout Kappa: 0.9238  F1: 0.8610  MAE: 0.1895  Off-1: 0.9677  ECE: 0.0662
  Saved: densenet121_best_fold_holdout_metrics.json

  mobilenetv3_large  (best CV fold: 1)

 Building model: mobilenetv3_large
  Parameters: 4,208,437 total, 4,208,437 trainable
  Holdout Kappa: 0.8724  F1: 0.7670  MAE: 0.3105  Off-1: 0.9355  ECE: 0.1097
  Saved: mobilenetv3_large_best_fold_holdout_metrics.json

  convnext_tiny  (best CV fold: 1)

 Building model: convnext_tiny
  Parameters: 27,823,973 total, 27,823,973 trainable
  Holdout Kappa: 0.9288  F1: 0.8420  MAE: 0.1895  Off-1: 0.9758  ECE: 0.0693
  Saved: convnext_tiny_best_fold_holdout_metrics.json

Results saved to: results
