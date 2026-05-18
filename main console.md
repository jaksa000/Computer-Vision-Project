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
Epoch [  1/20]  Train Loss: 1.6109  Bal.Acc: 18.0%  F1: 0.1515  |  Val Loss: 1.6060  Bal.Acc: 18.6%  F1: 0.1200  |  LR: 1.00e-04  (123.4s)
 Best checkpoint saved (val_loss: 1.6060)
Epoch [  2/20]  Train Loss: 1.5888  Bal.Acc: 28.8%  F1: 0.2843  |  Val Loss: 1.5955  Bal.Acc: 23.2%  F1: 0.1659  |  LR: 1.00e-04  (19.4s)
 Best checkpoint saved (val_loss: 1.5955)
Epoch [  3/20]  Train Loss: 1.5741  Bal.Acc: 35.2%  F1: 0.3586  |  Val Loss: 1.5650  Bal.Acc: 32.0%  F1: 0.3159  |  LR: 1.00e-04  (25.1s)
 Best checkpoint saved (val_loss: 1.5650)
Epoch [  4/20]  Train Loss: 1.5492  Bal.Acc: 44.0%  F1: 0.4290  |  Val Loss: 1.5330  Bal.Acc: 41.0%  F1: 0.4092  |  LR: 1.00e-04  (33.4s)
 Best checkpoint saved (val_loss: 1.5330)
Epoch [  5/20]  Train Loss: 1.5180  Bal.Acc: 49.7%  F1: 0.4724  |  Val Loss: 1.4981  Bal.Acc: 47.5%  F1: 0.4630  |  LR: 1.00e-04  (26.2s)
 Best checkpoint saved (val_loss: 1.4981)
Epoch [  6/20]  Train Loss: 1.4736  Bal.Acc: 49.9%  F1: 0.4706  |  Val Loss: 1.4461  Bal.Acc: 49.9%  F1: 0.4791  |  LR: 1.00e-04  (24.6s)
 Best checkpoint saved (val_loss: 1.4461)
Epoch [  7/20]  Train Loss: 1.3981  Bal.Acc: 56.6%  F1: 0.5179  |  Val Loss: 1.3488  Bal.Acc: 53.8%  F1: 0.5079  |  LR: 1.00e-04  (26.7s)
 Best checkpoint saved (val_loss: 1.3488)
Epoch [  8/20]  Train Loss: 1.3046  Bal.Acc: 57.6%  F1: 0.5229  |  Val Loss: 1.2558  Bal.Acc: 56.9%  F1: 0.5408  |  LR: 1.00e-04  (26.0s)
 Best checkpoint saved (val_loss: 1.2558)
Epoch [  9/20]  Train Loss: 1.1924  Bal.Acc: 60.8%  F1: 0.5551  |  Val Loss: 1.1731  Bal.Acc: 55.4%  F1: 0.5269  |  LR: 1.00e-04  (24.8s)
 Best checkpoint saved (val_loss: 1.1731)
Epoch [ 10/20]  Train Loss: 1.0609  Bal.Acc: 63.6%  F1: 0.6051  |  Val Loss: 1.0467  Bal.Acc: 59.9%  F1: 0.5856  |  LR: 1.00e-04  (33.1s)
 Best checkpoint saved (val_loss: 1.0467)
Epoch [ 11/20]  Train Loss: 0.9586  Bal.Acc: 65.9%  F1: 0.6289  |  Val Loss: 1.0902  Bal.Acc: 56.4%  F1: 0.5598  |  LR: 1.00e-04  (26.7s)
Epoch [ 12/20]  Train Loss: 0.8682  Bal.Acc: 68.9%  F1: 0.6768  |  Val Loss: 0.9090  Bal.Acc: 64.5%  F1: 0.6332  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.9090)
Epoch [ 13/20]  Train Loss: 0.7808  Bal.Acc: 74.4%  F1: 0.7326  |  Val Loss: 0.9226  Bal.Acc: 60.4%  F1: 0.6002  |  LR: 1.00e-04  (25.0s)
Epoch [ 14/20]  Train Loss: 0.7354  Bal.Acc: 73.4%  F1: 0.7285  |  Val Loss: 0.8701  Bal.Acc: 61.1%  F1: 0.5975  |  LR: 1.00e-04  (18.1s)
 Best checkpoint saved (val_loss: 0.8701)
Epoch [ 15/20]  Train Loss: 0.6742  Bal.Acc: 73.6%  F1: 0.7241  |  Val Loss: 0.8695  Bal.Acc: 63.5%  F1: 0.6326  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 0.8695)
Epoch [ 16/20]  Train Loss: 0.6028  Bal.Acc: 77.6%  F1: 0.7693  |  Val Loss: 0.8749  Bal.Acc: 62.9%  F1: 0.6305  |  LR: 1.00e-04  (32.9s)
Epoch [ 17/20]  Train Loss: 0.5772  Bal.Acc: 79.4%  F1: 0.7864  |  Val Loss: 0.8065  Bal.Acc: 70.3%  F1: 0.7020  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 0.8065)
Epoch [ 18/20]  Train Loss: 0.5321  Bal.Acc: 80.4%  F1: 0.7945  |  Val Loss: 0.7590  Bal.Acc: 73.9%  F1: 0.7384  |  LR: 1.00e-04  (26.0s)
 Best checkpoint saved (val_loss: 0.7590)
Epoch [ 19/20]  Train Loss: 0.5089  Bal.Acc: 80.8%  F1: 0.8001  |  Val Loss: 0.7444  Bal.Acc: 72.1%  F1: 0.7175  |  LR: 1.00e-04  (24.2s)
 Best checkpoint saved (val_loss: 0.7444)
Epoch [ 20/20]  Train Loss: 0.4923  Bal.Acc: 82.3%  F1: 0.8089  |  Val Loss: 0.9454  Bal.Acc: 62.3%  F1: 0.6341  |  LR: 1.00e-04  (27.1s)

 Training finished. Checkpoint: checkpoints/resnet50_fold1_best.pt
Log CSV: results/logs/resnet50_fold1_training_log.csv
Best weights loaded from epoch 19

Model evaluation: resnet50_fold1
----------------------------------------
  Balanced Accuracy:       72.14%
  F1 (macro):              0.7175
  Quadratic Cohen's Kappa: 0.9082
  ECE:                     0.0626
  Brier Score (mean):      0.0748

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.77      0.93      0.85        88
    Doubtful       0.78      0.62      0.69        81
        Mild       0.57      0.53      0.55        40
    Moderate       0.69      0.68      0.68        37
      Severe       0.79      0.86      0.82        35

    accuracy                           0.74       281
   macro avg       0.72      0.72      0.72       281
weighted avg       0.74      0.74      0.73       281

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
Epoch [  1/20]  Train Loss: 1.6050  Bal.Acc: 23.3%  F1: 0.1760  |  Val Loss: 1.6037  Bal.Acc: 27.7%  F1: 0.2374  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 1.6037)
Epoch [  2/20]  Train Loss: 1.5907  Bal.Acc: 30.3%  F1: 0.2786  |  Val Loss: 1.5855  Bal.Acc: 25.8%  F1: 0.2142  |  LR: 1.00e-04  (19.4s)
 Best checkpoint saved (val_loss: 1.5855)
Epoch [  3/20]  Train Loss: 1.5679  Bal.Acc: 36.7%  F1: 0.3677  |  Val Loss: 1.5839  Bal.Acc: 26.9%  F1: 0.2287  |  LR: 1.00e-04  (23.8s)
 Best checkpoint saved (val_loss: 1.5839)
Epoch [  4/20]  Train Loss: 1.5497  Bal.Acc: 41.2%  F1: 0.3881  |  Val Loss: 1.5394  Bal.Acc: 41.5%  F1: 0.3765  |  LR: 1.00e-04  (32.4s)
 Best checkpoint saved (val_loss: 1.5394)
Epoch [  5/20]  Train Loss: 1.5136  Bal.Acc: 48.5%  F1: 0.4440  |  Val Loss: 1.4881  Bal.Acc: 42.3%  F1: 0.4009  |  LR: 1.00e-04  (32.9s)
 Best checkpoint saved (val_loss: 1.4881)
Epoch [  6/20]  Train Loss: 1.4716  Bal.Acc: 49.0%  F1: 0.4406  |  Val Loss: 1.4584  Bal.Acc: 43.2%  F1: 0.3909  |  LR: 1.00e-04  (33.8s)
 Best checkpoint saved (val_loss: 1.4584)
Epoch [  7/20]  Train Loss: 1.4006  Bal.Acc: 53.0%  F1: 0.4572  |  Val Loss: 1.4057  Bal.Acc: 40.3%  F1: 0.3657  |  LR: 1.00e-04  (32.9s)
 Best checkpoint saved (val_loss: 1.4057)
Epoch [  8/20]  Train Loss: 1.3015  Bal.Acc: 53.8%  F1: 0.4652  |  Val Loss: 1.2929  Bal.Acc: 45.4%  F1: 0.4167  |  LR: 1.00e-04  (26.6s)
 Best checkpoint saved (val_loss: 1.2929)
Epoch [  9/20]  Train Loss: 1.1837  Bal.Acc: 57.3%  F1: 0.5029  |  Val Loss: 1.1791  Bal.Acc: 52.7%  F1: 0.4953  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 1.1791)
Epoch [ 10/20]  Train Loss: 1.0782  Bal.Acc: 59.9%  F1: 0.5414  |  Val Loss: 1.0967  Bal.Acc: 55.1%  F1: 0.5221  |  LR: 1.00e-04  (26.2s)
 Best checkpoint saved (val_loss: 1.0967)
Epoch [ 11/20]  Train Loss: 0.9826  Bal.Acc: 63.1%  F1: 0.5912  |  Val Loss: 1.1458  Bal.Acc: 50.3%  F1: 0.4921  |  LR: 1.00e-04  (24.8s)
Epoch [ 12/20]  Train Loss: 0.8953  Bal.Acc: 65.9%  F1: 0.6316  |  Val Loss: 1.1224  Bal.Acc: 55.7%  F1: 0.5503  |  LR: 1.00e-04  (18.7s)
Epoch [ 13/20]  Train Loss: 0.8295  Bal.Acc: 69.4%  F1: 0.6708  |  Val Loss: 1.0028  Bal.Acc: 62.7%  F1: 0.6343  |  LR: 1.00e-04  (17.4s)
 Best checkpoint saved (val_loss: 1.0028)
Epoch [ 14/20]  Train Loss: 0.7607  Bal.Acc: 71.8%  F1: 0.7045  |  Val Loss: 0.9259  Bal.Acc: 66.1%  F1: 0.6606  |  LR: 1.00e-04  (24.5s)
 Best checkpoint saved (val_loss: 0.9259)
Epoch [ 15/20]  Train Loss: 0.7191  Bal.Acc: 72.8%  F1: 0.7184  |  Val Loss: 0.8744  Bal.Acc: 63.4%  F1: 0.6442  |  LR: 1.00e-04  (33.1s)
 Best checkpoint saved (val_loss: 0.8744)
Epoch [ 16/20]  Train Loss: 0.6423  Bal.Acc: 75.0%  F1: 0.7438  |  Val Loss: 0.8138  Bal.Acc: 65.8%  F1: 0.6646  |  LR: 1.00e-04  (32.7s)
 Best checkpoint saved (val_loss: 0.8138)
Epoch [ 17/20]  Train Loss: 0.6368  Bal.Acc: 75.5%  F1: 0.7453  |  Val Loss: 0.6993  Bal.Acc: 69.6%  F1: 0.6927  |  LR: 1.00e-04  (32.7s)
 Best checkpoint saved (val_loss: 0.6993)
Epoch [ 18/20]  Train Loss: 0.6026  Bal.Acc: 75.9%  F1: 0.7576  |  Val Loss: 0.6861  Bal.Acc: 75.1%  F1: 0.7404  |  LR: 1.00e-04  (32.3s)
 Best checkpoint saved (val_loss: 0.6861)
Epoch [ 19/20]  Train Loss: 0.5485  Bal.Acc: 80.3%  F1: 0.7934  |  Val Loss: 0.8214  Bal.Acc: 66.2%  F1: 0.6750  |  LR: 1.00e-04  (31.4s)
Epoch [ 20/20]  Train Loss: 0.5147  Bal.Acc: 79.8%  F1: 0.7939  |  Val Loss: 0.6975  Bal.Acc: 71.5%  F1: 0.7174  |  LR: 1.00e-04  (18.2s)

 Training finished. Checkpoint: checkpoints/resnet50_fold2_best.pt
Log CSV: results/logs/resnet50_fold2_training_log.csv
Best weights loaded from epoch 18

Model evaluation: resnet50_fold2
----------------------------------------
  Balanced Accuracy:       75.07%
  F1 (macro):              0.7404
  Quadratic Cohen's Kappa: 0.8890
  ECE:                     0.0898
  Brier Score (mean):      0.0773

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.88      0.76      0.82        88
    Doubtful       0.67      0.67      0.67        81
        Mild       0.50      0.57      0.53        40
    Moderate       0.76      0.86      0.81        37
      Severe       0.86      0.89      0.87        35

    accuracy                           0.74       281
   macro avg       0.73      0.75      0.74       281
weighted avg       0.75      0.74      0.74       281

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
Epoch [  1/20]  Train Loss: 1.6052  Bal.Acc: 24.8%  F1: 0.1954  |  Val Loss: 1.6025  Bal.Acc: 29.0%  F1: 0.2541  |  LR: 1.00e-04  (18.1s)
 Best checkpoint saved (val_loss: 1.6025)
Epoch [  2/20]  Train Loss: 1.5854  Bal.Acc: 34.1%  F1: 0.2997  |  Val Loss: 1.5914  Bal.Acc: 31.0%  F1: 0.2879  |  LR: 1.00e-04  (19.4s)
 Best checkpoint saved (val_loss: 1.5914)
Epoch [  3/20]  Train Loss: 1.5704  Bal.Acc: 39.0%  F1: 0.3622  |  Val Loss: 1.5554  Bal.Acc: 35.1%  F1: 0.3335  |  LR: 1.00e-04  (25.0s)
 Best checkpoint saved (val_loss: 1.5554)
Epoch [  4/20]  Train Loss: 1.5462  Bal.Acc: 43.0%  F1: 0.3905  |  Val Loss: 1.5305  Bal.Acc: 41.8%  F1: 0.4211  |  LR: 1.00e-04  (32.2s)
 Best checkpoint saved (val_loss: 1.5305)
Epoch [  5/20]  Train Loss: 1.5188  Bal.Acc: 45.0%  F1: 0.4104  |  Val Loss: 1.5092  Bal.Acc: 50.3%  F1: 0.4642  |  LR: 1.00e-04  (31.2s)
 Best checkpoint saved (val_loss: 1.5092)
Epoch [  6/20]  Train Loss: 1.4822  Bal.Acc: 49.3%  F1: 0.4451  |  Val Loss: 1.4575  Bal.Acc: 53.0%  F1: 0.4782  |  LR: 1.00e-04  (30.7s)
 Best checkpoint saved (val_loss: 1.4575)
Epoch [  7/20]  Train Loss: 1.4205  Bal.Acc: 53.3%  F1: 0.4681  |  Val Loss: 1.4080  Bal.Acc: 54.2%  F1: 0.4703  |  LR: 1.00e-04  (31.8s)
 Best checkpoint saved (val_loss: 1.4080)
Epoch [  8/20]  Train Loss: 1.3499  Bal.Acc: 55.7%  F1: 0.4917  |  Val Loss: 1.3601  Bal.Acc: 46.6%  F1: 0.4402  |  LR: 1.00e-04  (31.8s)
 Best checkpoint saved (val_loss: 1.3601)
Epoch [  9/20]  Train Loss: 1.2359  Bal.Acc: 57.8%  F1: 0.5129  |  Val Loss: 1.2428  Bal.Acc: 51.1%  F1: 0.4446  |  LR: 1.00e-04  (31.7s)
 Best checkpoint saved (val_loss: 1.2428)
Epoch [ 10/20]  Train Loss: 1.1211  Bal.Acc: 60.8%  F1: 0.5488  |  Val Loss: 1.1109  Bal.Acc: 55.0%  F1: 0.5143  |  LR: 1.00e-04  (30.8s)
 Best checkpoint saved (val_loss: 1.1109)
Epoch [ 11/20]  Train Loss: 1.0157  Bal.Acc: 64.3%  F1: 0.6015  |  Val Loss: 1.0318  Bal.Acc: 59.7%  F1: 0.5584  |  LR: 1.00e-04  (31.4s)
 Best checkpoint saved (val_loss: 1.0318)
Epoch [ 12/20]  Train Loss: 0.9162  Bal.Acc: 67.0%  F1: 0.6387  |  Val Loss: 1.0429  Bal.Acc: 58.0%  F1: 0.5828  |  LR: 1.00e-04  (30.3s)
Epoch [ 13/20]  Train Loss: 0.8325  Bal.Acc: 70.0%  F1: 0.6822  |  Val Loss: 0.9330  Bal.Acc: 61.7%  F1: 0.6229  |  LR: 1.00e-04  (18.8s)
 Best checkpoint saved (val_loss: 0.9330)
Epoch [ 14/20]  Train Loss: 0.7445  Bal.Acc: 73.1%  F1: 0.7191  |  Val Loss: 0.9392  Bal.Acc: 61.5%  F1: 0.6103  |  LR: 1.00e-04  (24.7s)
Epoch [ 15/20]  Train Loss: 0.6932  Bal.Acc: 73.5%  F1: 0.7211  |  Val Loss: 0.8870  Bal.Acc: 65.4%  F1: 0.6535  |  LR: 1.00e-04  (19.0s)
 Best checkpoint saved (val_loss: 0.8870)
Epoch [ 16/20]  Train Loss: 0.6490  Bal.Acc: 75.6%  F1: 0.7478  |  Val Loss: 0.7582  Bal.Acc: 65.7%  F1: 0.6447  |  LR: 1.00e-04  (24.7s)
 Best checkpoint saved (val_loss: 0.7582)
Epoch [ 17/20]  Train Loss: 0.5873  Bal.Acc: 79.3%  F1: 0.7855  |  Val Loss: 0.9057  Bal.Acc: 65.7%  F1: 0.6173  |  LR: 1.00e-04  (32.9s)
Epoch [ 18/20]  Train Loss: 0.5618  Bal.Acc: 79.3%  F1: 0.7826  |  Val Loss: 0.7905  Bal.Acc: 66.6%  F1: 0.6627  |  LR: 1.00e-04  (18.4s)
Epoch [ 19/20]  Train Loss: 0.5405  Bal.Acc: 79.2%  F1: 0.7850  |  Val Loss: 0.6947  Bal.Acc: 71.2%  F1: 0.7005  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 0.6947)
Epoch [ 20/20]  Train Loss: 0.5208  Bal.Acc: 80.8%  F1: 0.7964  |  Val Loss: 0.7842  Bal.Acc: 69.3%  F1: 0.7029  |  LR: 1.00e-04  (25.7s)

 Training finished. Checkpoint: checkpoints/resnet50_fold3_best.pt
Log CSV: results/logs/resnet50_fold3_training_log.csv
Best weights loaded from epoch 19

Model evaluation: resnet50_fold3
----------------------------------------
  Balanced Accuracy:       71.21%
  F1 (macro):              0.7005
  Quadratic Cohen's Kappa: 0.8752
  ECE:                     0.0814
  Brier Score (mean):      0.0808

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.75      0.79      0.77        87
    Doubtful       0.58      0.46      0.51        81
        Mild       0.47      0.59      0.52        39
    Moderate       0.83      0.92      0.88        38
      Severe       0.85      0.80      0.82        35

    accuracy                           0.69       280
   macro avg       0.70      0.71      0.70       280
weighted avg       0.68      0.69      0.68       280

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
Epoch [  1/20]  Train Loss: 1.6049  Bal.Acc: 24.7%  F1: 0.2295  |  Val Loss: 1.6049  Bal.Acc: 24.5%  F1: 0.2083  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 1.6049)
Epoch [  2/20]  Train Loss: 1.5901  Bal.Acc: 30.2%  F1: 0.3019  |  Val Loss: 1.5947  Bal.Acc: 28.1%  F1: 0.2670  |  LR: 1.00e-04  (19.2s)
 Best checkpoint saved (val_loss: 1.5947)
Epoch [  3/20]  Train Loss: 1.5744  Bal.Acc: 35.6%  F1: 0.3462  |  Val Loss: 1.5752  Bal.Acc: 31.1%  F1: 0.2952  |  LR: 1.00e-04  (24.0s)
 Best checkpoint saved (val_loss: 1.5752)
Epoch [  4/20]  Train Loss: 1.5552  Bal.Acc: 39.2%  F1: 0.3750  |  Val Loss: 1.5435  Bal.Acc: 35.4%  F1: 0.3468  |  LR: 1.00e-04  (33.3s)
 Best checkpoint saved (val_loss: 1.5435)
Epoch [  5/20]  Train Loss: 1.5275  Bal.Acc: 44.4%  F1: 0.4191  |  Val Loss: 1.5121  Bal.Acc: 40.6%  F1: 0.4057  |  LR: 1.00e-04  (32.3s)
 Best checkpoint saved (val_loss: 1.5121)
Epoch [  6/20]  Train Loss: 1.4952  Bal.Acc: 48.0%  F1: 0.4550  |  Val Loss: 1.4792  Bal.Acc: 41.3%  F1: 0.4079  |  LR: 1.00e-04  (32.2s)
 Best checkpoint saved (val_loss: 1.4792)
Epoch [  7/20]  Train Loss: 1.4549  Bal.Acc: 51.1%  F1: 0.4666  |  Val Loss: 1.4226  Bal.Acc: 45.7%  F1: 0.4621  |  LR: 1.00e-04  (31.5s)
 Best checkpoint saved (val_loss: 1.4226)
Epoch [  8/20]  Train Loss: 1.3851  Bal.Acc: 53.6%  F1: 0.4937  |  Val Loss: 1.3605  Bal.Acc: 52.4%  F1: 0.5291  |  LR: 1.00e-04  (30.8s)
 Best checkpoint saved (val_loss: 1.3605)
Epoch [  9/20]  Train Loss: 1.2871  Bal.Acc: 57.4%  F1: 0.5254  |  Val Loss: 1.2692  Bal.Acc: 56.0%  F1: 0.5580  |  LR: 1.00e-04  (31.6s)
 Best checkpoint saved (val_loss: 1.2692)
Epoch [ 10/20]  Train Loss: 1.1666  Bal.Acc: 60.2%  F1: 0.5512  |  Val Loss: 1.2851  Bal.Acc: 51.6%  F1: 0.5145  |  LR: 1.00e-04  (31.1s)
Epoch [ 11/20]  Train Loss: 1.0582  Bal.Acc: 60.6%  F1: 0.5661  |  Val Loss: 1.0896  Bal.Acc: 54.4%  F1: 0.5310  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 1.0896)
Epoch [ 12/20]  Train Loss: 0.9554  Bal.Acc: 64.5%  F1: 0.6142  |  Val Loss: 1.0567  Bal.Acc: 57.2%  F1: 0.5814  |  LR: 1.00e-04  (23.6s)
 Best checkpoint saved (val_loss: 1.0567)
Epoch [ 13/20]  Train Loss: 0.8902  Bal.Acc: 67.2%  F1: 0.6511  |  Val Loss: 0.8950  Bal.Acc: 69.3%  F1: 0.6665  |  LR: 1.00e-04  (32.0s)
 Best checkpoint saved (val_loss: 0.8950)
Epoch [ 14/20]  Train Loss: 0.8071  Bal.Acc: 69.5%  F1: 0.6780  |  Val Loss: 0.9709  Bal.Acc: 60.4%  F1: 0.6152  |  LR: 1.00e-04  (33.0s)
Epoch [ 15/20]  Train Loss: 0.7560  Bal.Acc: 72.0%  F1: 0.7091  |  Val Loss: 0.8810  Bal.Acc: 63.6%  F1: 0.6429  |  LR: 1.00e-04  (19.4s)
 Best checkpoint saved (val_loss: 0.8810)
Epoch [ 16/20]  Train Loss: 0.6893  Bal.Acc: 74.4%  F1: 0.7382  |  Val Loss: 0.6849  Bal.Acc: 76.0%  F1: 0.7543  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 0.6849)
Epoch [ 17/20]  Train Loss: 0.6595  Bal.Acc: 76.6%  F1: 0.7507  |  Val Loss: 0.7432  Bal.Acc: 73.2%  F1: 0.7421  |  LR: 1.00e-04  (32.3s)
Epoch [ 18/20]  Train Loss: 0.6136  Bal.Acc: 76.3%  F1: 0.7539  |  Val Loss: 0.6811  Bal.Acc: 74.9%  F1: 0.7488  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 0.6811)
Epoch [ 19/20]  Train Loss: 0.5751  Bal.Acc: 79.0%  F1: 0.7797  |  Val Loss: 0.8064  Bal.Acc: 67.9%  F1: 0.6871  |  LR: 1.00e-04  (25.7s)
Epoch [ 20/20]  Train Loss: 0.5429  Bal.Acc: 79.8%  F1: 0.7894  |  Val Loss: 0.7979  Bal.Acc: 69.3%  F1: 0.7034  |  LR: 1.00e-04  (17.5s)

 Training finished. Checkpoint: checkpoints/resnet50_fold4_best.pt
Log CSV: results/logs/resnet50_fold4_training_log.csv
Best weights loaded from epoch 18

Model evaluation: resnet50_fold4
----------------------------------------
  Balanced Accuracy:       74.93%
  F1 (macro):              0.7488
  Quadratic Cohen's Kappa: 0.8820
  ECE:                     0.0745
  Brier Score (mean):      0.0769

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.86      0.77      0.81        87
    Doubtful       0.70      0.64      0.67        81
        Mild       0.44      0.69      0.53        39
    Moderate       0.89      0.84      0.86        38
      Severe       0.93      0.80      0.86        35

    accuracy                           0.74       280
   macro avg       0.76      0.75      0.75       280
weighted avg       0.77      0.74      0.75       280

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
Epoch [  1/20]  Train Loss: 1.6067  Bal.Acc: 22.4%  F1: 0.1807  |  Val Loss: 1.5989  Bal.Acc: 26.5%  F1: 0.2275  |  LR: 1.00e-04  (17.6s)
 Best checkpoint saved (val_loss: 1.5989)
Epoch [  2/20]  Train Loss: 1.5868  Bal.Acc: 34.9%  F1: 0.3190  |  Val Loss: 1.5856  Bal.Acc: 32.7%  F1: 0.3091  |  LR: 1.00e-04  (19.3s)
 Best checkpoint saved (val_loss: 1.5856)
Epoch [  3/20]  Train Loss: 1.5730  Bal.Acc: 34.7%  F1: 0.3412  |  Val Loss: 1.5609  Bal.Acc: 37.1%  F1: 0.3706  |  LR: 1.00e-04  (24.7s)
 Best checkpoint saved (val_loss: 1.5609)
Epoch [  4/20]  Train Loss: 1.5529  Bal.Acc: 38.2%  F1: 0.3766  |  Val Loss: 1.5335  Bal.Acc: 42.6%  F1: 0.4266  |  LR: 1.00e-04  (32.8s)
 Best checkpoint saved (val_loss: 1.5335)
Epoch [  5/20]  Train Loss: 1.5184  Bal.Acc: 46.2%  F1: 0.4425  |  Val Loss: 1.4971  Bal.Acc: 50.7%  F1: 0.4921  |  LR: 1.00e-04  (32.4s)
 Best checkpoint saved (val_loss: 1.4971)
Epoch [  6/20]  Train Loss: 1.4791  Bal.Acc: 47.8%  F1: 0.4388  |  Val Loss: 1.4435  Bal.Acc: 49.1%  F1: 0.4699  |  LR: 1.00e-04  (32.6s)
 Best checkpoint saved (val_loss: 1.4435)
Epoch [  7/20]  Train Loss: 1.4056  Bal.Acc: 55.4%  F1: 0.5160  |  Val Loss: 1.3449  Bal.Acc: 50.0%  F1: 0.4609  |  LR: 1.00e-04  (31.9s)
 Best checkpoint saved (val_loss: 1.3449)
Epoch [  8/20]  Train Loss: 1.3001  Bal.Acc: 56.9%  F1: 0.5188  |  Val Loss: 1.2243  Bal.Acc: 59.1%  F1: 0.5433  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 1.2243)
Epoch [  9/20]  Train Loss: 1.1890  Bal.Acc: 57.8%  F1: 0.5213  |  Val Loss: 1.1320  Bal.Acc: 56.0%  F1: 0.5152  |  LR: 1.00e-04  (31.2s)
 Best checkpoint saved (val_loss: 1.1320)
Epoch [ 10/20]  Train Loss: 1.0892  Bal.Acc: 60.6%  F1: 0.5724  |  Val Loss: 0.9897  Bal.Acc: 62.1%  F1: 0.5882  |  LR: 1.00e-04  (30.7s)
 Best checkpoint saved (val_loss: 0.9897)
Epoch [ 11/20]  Train Loss: 0.9671  Bal.Acc: 63.8%  F1: 0.5961  |  Val Loss: 1.0827  Bal.Acc: 55.1%  F1: 0.5404  |  LR: 1.00e-04  (30.7s)
Epoch [ 12/20]  Train Loss: 0.8808  Bal.Acc: 67.4%  F1: 0.6602  |  Val Loss: 0.8680  Bal.Acc: 66.3%  F1: 0.6274  |  LR: 1.00e-04  (19.3s)
 Best checkpoint saved (val_loss: 0.8680)
Epoch [ 13/20]  Train Loss: 0.7959  Bal.Acc: 70.9%  F1: 0.6982  |  Val Loss: 0.7474  Bal.Acc: 68.9%  F1: 0.6670  |  LR: 1.00e-04  (24.2s)
 Best checkpoint saved (val_loss: 0.7474)
Epoch [ 14/20]  Train Loss: 0.7410  Bal.Acc: 72.0%  F1: 0.7060  |  Val Loss: 0.7999  Bal.Acc: 63.8%  F1: 0.6288  |  LR: 1.00e-04  (31.2s)
Epoch [ 15/20]  Train Loss: 0.6989  Bal.Acc: 73.0%  F1: 0.7257  |  Val Loss: 0.6626  Bal.Acc: 72.1%  F1: 0.7079  |  LR: 1.00e-04  (18.6s)
 Best checkpoint saved (val_loss: 0.6626)
Epoch [ 16/20]  Train Loss: 0.6765  Bal.Acc: 74.8%  F1: 0.7389  |  Val Loss: 0.6338  Bal.Acc: 73.6%  F1: 0.7245  |  LR: 1.00e-04  (23.6s)
 Best checkpoint saved (val_loss: 0.6338)
Epoch [ 17/20]  Train Loss: 0.6163  Bal.Acc: 76.4%  F1: 0.7534  |  Val Loss: 0.6472  Bal.Acc: 74.6%  F1: 0.7395  |  LR: 1.00e-04  (32.1s)
Epoch [ 18/20]  Train Loss: 0.5605  Bal.Acc: 79.9%  F1: 0.7940  |  Val Loss: 0.6090  Bal.Acc: 75.2%  F1: 0.7410  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.6090)
Epoch [ 19/20]  Train Loss: 0.5152  Bal.Acc: 80.6%  F1: 0.7983  |  Val Loss: 0.6514  Bal.Acc: 74.4%  F1: 0.7095  |  LR: 1.00e-04  (25.4s)
Epoch [ 20/20]  Train Loss: 0.5013  Bal.Acc: 80.9%  F1: 0.8012  |  Val Loss: 0.5853  Bal.Acc: 79.2%  F1: 0.7749  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 0.5853)

 Training finished. Checkpoint: checkpoints/resnet50_fold5_best.pt
Log CSV: results/logs/resnet50_fold5_training_log.csv
Best weights loaded from epoch 20

Model evaluation: resnet50_fold5
----------------------------------------
  Balanced Accuracy:       79.20%
  F1 (macro):              0.7749
  Quadratic Cohen's Kappa: 0.8877
  ECE:                     0.0687
  Brier Score (mean):      0.0701

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.85      0.87        87
    Doubtful       0.74      0.65      0.69        81
        Mild       0.57      0.62      0.59        39
    Moderate       0.87      0.87      0.87        38
      Severe       0.76      0.97      0.85        35

    accuracy                           0.78       280
   macro avg       0.76      0.79      0.77       280
weighted avg       0.78      0.78      0.78       280

  Metrics saved to: results/individual_models/resnet50_fold5_metrics.json
  Probabilities saved to: results/individual_models/resnet50_fold5_test_probs.npz

 FINISHED: resnet50. Average kappa out of 5 folds: 0.8884 ±0.0110

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
model.safetensors: 100% 49.3M/49.3M [00:01<00:00, 40.2MB/s]
  Parameters: 10,703,917 total, 10,703,917 trainable

============================================================
TRAINING: efficientnet_b3_fold1
============================================================
Epoch [  1/20]  Train Loss: 2.4980  Bal.Acc: 30.8%  F1: 0.2916  |  Val Loss: 2.1879  Bal.Acc: 35.9%  F1: 0.3167  |  LR: 1.00e-04  (26.0s)
 Best checkpoint saved (val_loss: 2.1879)
Epoch [  2/20]  Train Loss: 1.4933  Bal.Acc: 49.0%  F1: 0.4688  |  Val Loss: 1.8853  Bal.Acc: 43.4%  F1: 0.3845  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 1.8853)
Epoch [  3/20]  Train Loss: 1.1287  Bal.Acc: 60.8%  F1: 0.5930  |  Val Loss: 1.7003  Bal.Acc: 50.4%  F1: 0.4648  |  LR: 1.00e-04  (21.2s)
 Best checkpoint saved (val_loss: 1.7003)
Epoch [  4/20]  Train Loss: 0.9160  Bal.Acc: 67.1%  F1: 0.6540  |  Val Loss: 1.4158  Bal.Acc: 57.7%  F1: 0.5694  |  LR: 1.00e-04  (23.8s)
 Best checkpoint saved (val_loss: 1.4158)
Epoch [  5/20]  Train Loss: 0.8107  Bal.Acc: 70.9%  F1: 0.6941  |  Val Loss: 1.1972  Bal.Acc: 66.9%  F1: 0.6500  |  LR: 1.00e-04  (21.6s)
 Best checkpoint saved (val_loss: 1.1972)
Epoch [  6/20]  Train Loss: 0.7118  Bal.Acc: 73.6%  F1: 0.7179  |  Val Loss: 1.1234  Bal.Acc: 65.6%  F1: 0.6482  |  LR: 1.00e-04  (22.5s)
 Best checkpoint saved (val_loss: 1.1234)
Epoch [  7/20]  Train Loss: 0.6277  Bal.Acc: 76.9%  F1: 0.7662  |  Val Loss: 1.1241  Bal.Acc: 67.8%  F1: 0.6774  |  LR: 1.00e-04  (21.3s)
Epoch [  8/20]  Train Loss: 0.5684  Bal.Acc: 78.4%  F1: 0.7663  |  Val Loss: 1.0535  Bal.Acc: 70.5%  F1: 0.7023  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 1.0535)
Epoch [  9/20]  Train Loss: 0.4586  Bal.Acc: 82.0%  F1: 0.8145  |  Val Loss: 1.0681  Bal.Acc: 68.9%  F1: 0.6927  |  LR: 1.00e-04  (22.0s)
Epoch [ 10/20]  Train Loss: 0.4225  Bal.Acc: 83.5%  F1: 0.8244  |  Val Loss: 1.0372  Bal.Acc: 72.0%  F1: 0.7128  |  LR: 1.00e-04  (17.2s)
 Best checkpoint saved (val_loss: 1.0372)
Epoch [ 11/20]  Train Loss: 0.3887  Bal.Acc: 86.3%  F1: 0.8519  |  Val Loss: 1.0699  Bal.Acc: 74.2%  F1: 0.7377  |  LR: 1.00e-04  (21.8s)
Epoch [ 12/20]  Train Loss: 0.3430  Bal.Acc: 86.9%  F1: 0.8602  |  Val Loss: 1.0463  Bal.Acc: 73.8%  F1: 0.7318  |  LR: 1.00e-04  (17.9s)
Epoch [ 13/20]  Train Loss: 0.2972  Bal.Acc: 88.6%  F1: 0.8784  |  Val Loss: 1.0816  Bal.Acc: 74.2%  F1: 0.7428  |  LR: 1.00e-04  (17.7s)
Epoch [ 14/20]  Train Loss: 0.2779  Bal.Acc: 90.1%  F1: 0.8927  |  Val Loss: 1.0638  Bal.Acc: 73.2%  F1: 0.7287  |  LR: 5.00e-05  (17.7s)
Epoch [ 15/20]  Train Loss: 0.2308  Bal.Acc: 91.2%  F1: 0.9112  |  Val Loss: 1.0599  Bal.Acc: 75.3%  F1: 0.7490  |  LR: 5.00e-05  (16.9s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 1.0372

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold1_best.pt
Log CSV: results/logs/efficientnet_b3_fold1_training_log.csv
Best weights loaded from epoch 10

Model evaluation: efficientnet_b3_fold1
----------------------------------------
  Balanced Accuracy:       72.01%
  F1 (macro):              0.7128
  Quadratic Cohen's Kappa: 0.8431
  ECE:                     0.1197
  Brier Score (mean):      0.0856

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.74      0.85      0.79        88
    Doubtful       0.76      0.54      0.63        81
        Mild       0.51      0.65      0.57        40
    Moderate       0.81      0.78      0.79        37
      Severe       0.77      0.77      0.77        35

    accuracy                           0.72       281
   macro avg       0.72      0.72      0.71       281
weighted avg       0.73      0.72      0.71       281

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
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  Parameters: 10,703,917 total, 10,703,917 trainable

============================================================
TRAINING: efficientnet_b3_fold2
============================================================
Epoch [  1/20]  Train Loss: 2.3129  Bal.Acc: 32.2%  F1: 0.3064  |  Val Loss: 2.5400  Bal.Acc: 29.5%  F1: 0.2691  |  LR: 1.00e-04  (17.5s)
 Best checkpoint saved (val_loss: 2.5400)
Epoch [  2/20]  Train Loss: 1.5387  Bal.Acc: 48.2%  F1: 0.4583  |  Val Loss: 1.6600  Bal.Acc: 42.8%  F1: 0.4227  |  LR: 1.00e-04  (18.8s)
 Best checkpoint saved (val_loss: 1.6600)
Epoch [  3/20]  Train Loss: 1.2089  Bal.Acc: 57.6%  F1: 0.5627  |  Val Loss: 1.6925  Bal.Acc: 44.8%  F1: 0.4504  |  LR: 1.00e-04  (22.4s)
Epoch [  4/20]  Train Loss: 0.9116  Bal.Acc: 68.6%  F1: 0.6663  |  Val Loss: 1.2455  Bal.Acc: 56.7%  F1: 0.5561  |  LR: 1.00e-04  (17.1s)
 Best checkpoint saved (val_loss: 1.2455)
Epoch [  5/20]  Train Loss: 0.6999  Bal.Acc: 74.2%  F1: 0.7310  |  Val Loss: 1.0680  Bal.Acc: 66.4%  F1: 0.6632  |  LR: 1.00e-04  (21.3s)
 Best checkpoint saved (val_loss: 1.0680)
Epoch [  6/20]  Train Loss: 0.6845  Bal.Acc: 74.2%  F1: 0.7321  |  Val Loss: 0.9990  Bal.Acc: 69.5%  F1: 0.6947  |  LR: 1.00e-04  (23.1s)
 Best checkpoint saved (val_loss: 0.9990)
Epoch [  7/20]  Train Loss: 0.5150  Bal.Acc: 81.4%  F1: 0.8005  |  Val Loss: 0.9272  Bal.Acc: 67.3%  F1: 0.6704  |  LR: 1.00e-04  (21.6s)
 Best checkpoint saved (val_loss: 0.9272)
Epoch [  8/20]  Train Loss: 0.4897  Bal.Acc: 81.2%  F1: 0.8004  |  Val Loss: 0.8556  Bal.Acc: 73.7%  F1: 0.7315  |  LR: 1.00e-04  (22.0s)
 Best checkpoint saved (val_loss: 0.8556)
Epoch [  9/20]  Train Loss: 0.4413  Bal.Acc: 85.2%  F1: 0.8435  |  Val Loss: 0.8707  Bal.Acc: 72.8%  F1: 0.7298  |  LR: 1.00e-04  (22.2s)
Epoch [ 10/20]  Train Loss: 0.3768  Bal.Acc: 86.4%  F1: 0.8556  |  Val Loss: 0.8752  Bal.Acc: 70.9%  F1: 0.7095  |  LR: 1.00e-04  (17.6s)
Epoch [ 11/20]  Train Loss: 0.3548  Bal.Acc: 87.8%  F1: 0.8730  |  Val Loss: 0.8776  Bal.Acc: 75.4%  F1: 0.7462  |  LR: 1.00e-04  (17.4s)
Epoch [ 12/20]  Train Loss: 0.3211  Bal.Acc: 88.1%  F1: 0.8736  |  Val Loss: 0.8228  Bal.Acc: 77.3%  F1: 0.7717  |  LR: 1.00e-04  (17.6s)
 Best checkpoint saved (val_loss: 0.8228)
Epoch [ 13/20]  Train Loss: 0.2877  Bal.Acc: 88.9%  F1: 0.8818  |  Val Loss: 0.8284  Bal.Acc: 77.6%  F1: 0.7761  |  LR: 1.00e-04  (22.6s)
Epoch [ 14/20]  Train Loss: 0.2417  Bal.Acc: 91.2%  F1: 0.9036  |  Val Loss: 0.7908  Bal.Acc: 78.1%  F1: 0.7764  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 0.7908)
Epoch [ 15/20]  Train Loss: 0.2379  Bal.Acc: 91.8%  F1: 0.9088  |  Val Loss: 0.7723  Bal.Acc: 79.1%  F1: 0.7891  |  LR: 1.00e-04  (21.7s)
 Best checkpoint saved (val_loss: 0.7723)
Epoch [ 16/20]  Train Loss: 0.2430  Bal.Acc: 90.6%  F1: 0.9020  |  Val Loss: 0.7557  Bal.Acc: 80.9%  F1: 0.8096  |  LR: 1.00e-04  (22.3s)
 Best checkpoint saved (val_loss: 0.7557)
Epoch [ 17/20]  Train Loss: 0.1927  Bal.Acc: 93.4%  F1: 0.9319  |  Val Loss: 0.7636  Bal.Acc: 81.2%  F1: 0.8108  |  LR: 1.00e-04  (21.3s)
Epoch [ 18/20]  Train Loss: 0.1629  Bal.Acc: 94.3%  F1: 0.9396  |  Val Loss: 0.7719  Bal.Acc: 82.6%  F1: 0.8275  |  LR: 1.00e-04  (17.1s)
Epoch [ 19/20]  Train Loss: 0.1678  Bal.Acc: 93.6%  F1: 0.9309  |  Val Loss: 0.7507  Bal.Acc: 80.6%  F1: 0.8079  |  LR: 1.00e-04  (17.5s)
 Best checkpoint saved (val_loss: 0.7507)
Epoch [ 20/20]  Train Loss: 0.1491  Bal.Acc: 94.5%  F1: 0.9443  |  Val Loss: 0.7493  Bal.Acc: 80.6%  F1: 0.8101  |  LR: 1.00e-04  (22.7s)
 Best checkpoint saved (val_loss: 0.7493)

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold2_best.pt
Log CSV: results/logs/efficientnet_b3_fold2_training_log.csv
Best weights loaded from epoch 20

Model evaluation: efficientnet_b3_fold2
----------------------------------------
  Balanced Accuracy:       80.57%
  F1 (macro):              0.8101
  Quadratic Cohen's Kappa: 0.8946
  ECE:                     0.1051
  Brier Score (mean):      0.0625

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.85      0.86        88
    Doubtful       0.69      0.75      0.72        81
        Mild       0.81      0.65      0.72        40
    Moderate       0.84      0.97      0.90        37
      Severe       0.90      0.80      0.85        35

    accuracy                           0.80       281
   macro avg       0.82      0.81      0.81       281
weighted avg       0.81      0.80      0.80       281

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
Epoch [  1/20]  Train Loss: 2.4360  Bal.Acc: 28.4%  F1: 0.2702  |  Val Loss: 1.9625  Bal.Acc: 33.5%  F1: 0.3237  |  LR: 1.00e-04  (22.3s)
 Best checkpoint saved (val_loss: 1.9625)
Epoch [  2/20]  Train Loss: 1.4918  Bal.Acc: 49.5%  F1: 0.4685  |  Val Loss: 1.6932  Bal.Acc: 37.7%  F1: 0.3640  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 1.6932)
Epoch [  3/20]  Train Loss: 1.1688  Bal.Acc: 59.9%  F1: 0.5813  |  Val Loss: 1.5083  Bal.Acc: 48.9%  F1: 0.4743  |  LR: 1.00e-04  (21.6s)
 Best checkpoint saved (val_loss: 1.5083)
Epoch [  4/20]  Train Loss: 0.9652  Bal.Acc: 64.9%  F1: 0.6309  |  Val Loss: 1.3270  Bal.Acc: 58.6%  F1: 0.5694  |  LR: 1.00e-04  (22.0s)
 Best checkpoint saved (val_loss: 1.3270)
Epoch [  5/20]  Train Loss: 0.7604  Bal.Acc: 71.4%  F1: 0.7047  |  Val Loss: 1.0825  Bal.Acc: 62.5%  F1: 0.6151  |  LR: 1.00e-04  (22.1s)
 Best checkpoint saved (val_loss: 1.0825)
Epoch [  6/20]  Train Loss: 0.6906  Bal.Acc: 75.1%  F1: 0.7339  |  Val Loss: 1.0517  Bal.Acc: 65.7%  F1: 0.6507  |  LR: 1.00e-04  (22.5s)
 Best checkpoint saved (val_loss: 1.0517)
Epoch [  7/20]  Train Loss: 0.6108  Bal.Acc: 77.2%  F1: 0.7630  |  Val Loss: 0.9107  Bal.Acc: 69.9%  F1: 0.6888  |  LR: 1.00e-04  (21.5s)
 Best checkpoint saved (val_loss: 0.9107)
Epoch [  8/20]  Train Loss: 0.5638  Bal.Acc: 78.6%  F1: 0.7730  |  Val Loss: 0.8767  Bal.Acc: 69.4%  F1: 0.6861  |  LR: 1.00e-04  (22.6s)
 Best checkpoint saved (val_loss: 0.8767)
Epoch [  9/20]  Train Loss: 0.4876  Bal.Acc: 81.9%  F1: 0.8077  |  Val Loss: 0.8611  Bal.Acc: 70.4%  F1: 0.6970  |  LR: 1.00e-04  (21.7s)
 Best checkpoint saved (val_loss: 0.8611)
Epoch [ 10/20]  Train Loss: 0.4674  Bal.Acc: 83.4%  F1: 0.8265  |  Val Loss: 0.7833  Bal.Acc: 71.6%  F1: 0.7061  |  LR: 1.00e-04  (21.5s)
 Best checkpoint saved (val_loss: 0.7833)
Epoch [ 11/20]  Train Loss: 0.4018  Bal.Acc: 85.2%  F1: 0.8407  |  Val Loss: 0.7302  Bal.Acc: 73.4%  F1: 0.7171  |  LR: 1.00e-04  (21.7s)
 Best checkpoint saved (val_loss: 0.7302)
Epoch [ 12/20]  Train Loss: 0.3256  Bal.Acc: 87.8%  F1: 0.8681  |  Val Loss: 0.7016  Bal.Acc: 72.3%  F1: 0.7210  |  LR: 1.00e-04  (20.9s)
 Best checkpoint saved (val_loss: 0.7016)
Epoch [ 13/20]  Train Loss: 0.3309  Bal.Acc: 87.5%  F1: 0.8730  |  Val Loss: 0.6584  Bal.Acc: 76.0%  F1: 0.7526  |  LR: 1.00e-04  (22.2s)
 Best checkpoint saved (val_loss: 0.6584)
Epoch [ 14/20]  Train Loss: 0.2800  Bal.Acc: 88.9%  F1: 0.8814  |  Val Loss: 0.6483  Bal.Acc: 76.4%  F1: 0.7596  |  LR: 1.00e-04  (21.7s)
 Best checkpoint saved (val_loss: 0.6483)
Epoch [ 15/20]  Train Loss: 0.2801  Bal.Acc: 89.4%  F1: 0.8908  |  Val Loss: 0.6515  Bal.Acc: 79.0%  F1: 0.7833  |  LR: 1.00e-04  (21.4s)
Epoch [ 16/20]  Train Loss: 0.2677  Bal.Acc: 90.7%  F1: 0.9011  |  Val Loss: 0.6346  Bal.Acc: 76.0%  F1: 0.7585  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.6346)
Epoch [ 17/20]  Train Loss: 0.2634  Bal.Acc: 90.6%  F1: 0.9026  |  Val Loss: 0.6408  Bal.Acc: 79.8%  F1: 0.7908  |  LR: 1.00e-04  (21.6s)
Epoch [ 18/20]  Train Loss: 0.2143  Bal.Acc: 91.9%  F1: 0.9092  |  Val Loss: 0.6540  Bal.Acc: 80.0%  F1: 0.7888  |  LR: 1.00e-04  (17.7s)
Epoch [ 19/20]  Train Loss: 0.1680  Bal.Acc: 94.3%  F1: 0.9393  |  Val Loss: 0.6545  Bal.Acc: 80.5%  F1: 0.7959  |  LR: 1.00e-04  (17.8s)
Epoch [ 20/20]  Train Loss: 0.1714  Bal.Acc: 94.0%  F1: 0.9382  |  Val Loss: 0.6810  Bal.Acc: 79.1%  F1: 0.7865  |  LR: 5.00e-05  (17.8s)

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold3_best.pt
Log CSV: results/logs/efficientnet_b3_fold3_training_log.csv
Best weights loaded from epoch 16

Model evaluation: efficientnet_b3_fold3
----------------------------------------
  Balanced Accuracy:       75.96%
  F1 (macro):              0.7585
  Quadratic Cohen's Kappa: 0.8917
  ECE:                     0.1136
  Brier Score (mean):      0.0726

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.83      0.83        87
    Doubtful       0.64      0.60      0.62        81
        Mild       0.56      0.69      0.62        39
    Moderate       0.86      0.82      0.84        38
      Severe       0.91      0.86      0.88        35

    accuracy                           0.75       280
   macro avg       0.76      0.76      0.76       280
weighted avg       0.75      0.75      0.75       280

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
Epoch [  1/20]  Train Loss: 3.0157  Bal.Acc: 27.7%  F1: 0.2532  |  Val Loss: 2.5965  Bal.Acc: 29.2%  F1: 0.2676  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 2.5965)
Epoch [  2/20]  Train Loss: 1.7021  Bal.Acc: 47.4%  F1: 0.4437  |  Val Loss: 1.9261  Bal.Acc: 41.6%  F1: 0.3995  |  LR: 1.00e-04  (18.7s)
 Best checkpoint saved (val_loss: 1.9261)
Epoch [  3/20]  Train Loss: 1.2471  Bal.Acc: 57.0%  F1: 0.5515  |  Val Loss: 1.5551  Bal.Acc: 46.8%  F1: 0.4690  |  LR: 1.00e-04  (21.1s)
 Best checkpoint saved (val_loss: 1.5551)
Epoch [  4/20]  Train Loss: 1.0299  Bal.Acc: 62.9%  F1: 0.6156  |  Val Loss: 1.2345  Bal.Acc: 56.9%  F1: 0.5720  |  LR: 1.00e-04  (22.5s)
 Best checkpoint saved (val_loss: 1.2345)
Epoch [  5/20]  Train Loss: 0.9128  Bal.Acc: 68.2%  F1: 0.6629  |  Val Loss: 1.1268  Bal.Acc: 62.5%  F1: 0.6092  |  LR: 1.00e-04  (21.5s)
 Best checkpoint saved (val_loss: 1.1268)
Epoch [  6/20]  Train Loss: 0.7336  Bal.Acc: 74.0%  F1: 0.7283  |  Val Loss: 13.2898  Bal.Acc: 65.1%  F1: 0.6520  |  LR: 1.00e-04  (21.2s)
Epoch [  7/20]  Train Loss: 0.6568  Bal.Acc: 76.2%  F1: 0.7520  |  Val Loss: 1.0052  Bal.Acc: 62.7%  F1: 0.6197  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 1.0052)
Epoch [  8/20]  Train Loss: 0.6049  Bal.Acc: 77.8%  F1: 0.7712  |  Val Loss: 15.8349  Bal.Acc: 65.7%  F1: 0.6525  |  LR: 1.00e-04  (21.9s)
Epoch [  9/20]  Train Loss: 0.4996  Bal.Acc: 81.6%  F1: 0.8025  |  Val Loss: 7.9385  Bal.Acc: 66.7%  F1: 0.6773  |  LR: 1.00e-04  (17.8s)
Epoch [ 10/20]  Train Loss: 0.5190  Bal.Acc: 80.7%  F1: 0.8012  |  Val Loss: 280.0585  Bal.Acc: 66.1%  F1: 0.6586  |  LR: 1.00e-04  (17.8s)
Epoch [ 11/20]  Train Loss: 0.3968  Bal.Acc: 85.7%  F1: 0.8431  |  Val Loss: 23.6755  Bal.Acc: 68.6%  F1: 0.6842  |  LR: 5.00e-05  (17.8s)
Epoch [ 12/20]  Train Loss: 0.3889  Bal.Acc: 85.2%  F1: 0.8435  |  Val Loss: 1251.9832  Bal.Acc: 69.6%  F1: 0.7022  |  LR: 5.00e-05  (17.8s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 1.0052

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold4_best.pt
Log CSV: results/logs/efficientnet_b3_fold4_training_log.csv
Best weights loaded from epoch 7

Model evaluation: efficientnet_b3_fold4
----------------------------------------
  Balanced Accuracy:       62.67%
  F1 (macro):              0.6197
  Quadratic Cohen's Kappa: 0.7613
  ECE:                     0.1972
  Brier Score (mean):      0.1034

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.73      0.70      0.71        87
    Doubtful       0.55      0.52      0.54        81
        Mild       0.38      0.41      0.40        39
    Moderate       0.77      0.79      0.78        38
      Severe       0.64      0.71      0.68        35

    accuracy                           0.62       280
   macro avg       0.61      0.63      0.62       280
weighted avg       0.62      0.62      0.62       280

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
Epoch [  1/20]  Train Loss: 2.4835  Bal.Acc: 30.9%  F1: 0.2965  |  Val Loss: 2.0764  Bal.Acc: 37.3%  F1: 0.3496  |  LR: 1.00e-04  (17.2s)
 Best checkpoint saved (val_loss: 2.0764)
Epoch [  2/20]  Train Loss: 1.6370  Bal.Acc: 48.8%  F1: 0.4577  |  Val Loss: 1.7392  Bal.Acc: 50.1%  F1: 0.4755  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 1.7392)
Epoch [  3/20]  Train Loss: 1.2526  Bal.Acc: 59.4%  F1: 0.5784  |  Val Loss: 1.5928  Bal.Acc: 49.4%  F1: 0.4702  |  LR: 1.00e-04  (21.1s)
 Best checkpoint saved (val_loss: 1.5928)
Epoch [  4/20]  Train Loss: 0.9927  Bal.Acc: 63.3%  F1: 0.6160  |  Val Loss: 1.2753  Bal.Acc: 59.4%  F1: 0.5649  |  LR: 1.00e-04  (22.8s)
 Best checkpoint saved (val_loss: 1.2753)
Epoch [  5/20]  Train Loss: 0.8208  Bal.Acc: 68.0%  F1: 0.6662  |  Val Loss: 1.2872  Bal.Acc: 60.9%  F1: 0.6049  |  LR: 1.00e-04  (21.5s)
Epoch [  6/20]  Train Loss: 0.7138  Bal.Acc: 73.0%  F1: 0.7163  |  Val Loss: 31.1439  Bal.Acc: 68.6%  F1: 0.6915  |  LR: 1.00e-04  (17.9s)
Epoch [  7/20]  Train Loss: 0.6116  Bal.Acc: 76.4%  F1: 0.7541  |  Val Loss: 0.8990  Bal.Acc: 68.8%  F1: 0.6836  |  LR: 1.00e-04  (28.6s)
 Best checkpoint saved (val_loss: 0.8990)
Epoch [  8/20]  Train Loss: 0.6114  Bal.Acc: 77.2%  F1: 0.7595  |  Val Loss: 53.0061  Bal.Acc: 72.7%  F1: 0.7220  |  LR: 1.00e-04  (22.0s)
Epoch [  9/20]  Train Loss: 0.5113  Bal.Acc: 81.3%  F1: 0.7985  |  Val Loss: 228.4405  Bal.Acc: 73.0%  F1: 0.7283  |  LR: 1.00e-04  (17.4s)
Epoch [ 10/20]  Train Loss: 0.4825  Bal.Acc: 82.1%  F1: 0.8100  |  Val Loss: 51.2544  Bal.Acc: 73.3%  F1: 0.7341  |  LR: 1.00e-04  (17.8s)
Epoch [ 11/20]  Train Loss: 0.4290  Bal.Acc: 83.1%  F1: 0.8237  |  Val Loss: 41.6489  Bal.Acc: 75.8%  F1: 0.7477  |  LR: 5.00e-05  (17.8s)
Epoch [ 12/20]  Train Loss: 0.3797  Bal.Acc: 85.1%  F1: 0.8403  |  Val Loss: 16.9747  Bal.Acc: 74.9%  F1: 0.7464  |  LR: 5.00e-05  (17.9s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.8990

 Training finished. Checkpoint: checkpoints/efficientnet_b3_fold5_best.pt
Log CSV: results/logs/efficientnet_b3_fold5_training_log.csv
Best weights loaded from epoch 7

Model evaluation: efficientnet_b3_fold5
----------------------------------------
  Balanced Accuracy:       68.75%
  F1 (macro):              0.6836
  Quadratic Cohen's Kappa: 0.8297
  ECE:                     0.1003
  Brier Score (mean):      0.0871

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.77      0.83      0.80        87
    Doubtful       0.66      0.58      0.62        81
        Mild       0.41      0.44      0.42        39
    Moderate       0.88      0.74      0.80        38
      Severe       0.71      0.86      0.78        35

    accuracy                           0.69       280
   macro avg       0.69      0.69      0.68       280
weighted avg       0.70      0.69      0.69       280

  Metrics saved to: results/individual_models/efficientnet_b3_fold5_metrics.json
  Probabilities saved to: results/individual_models/efficientnet_b3_fold5_test_probs.npz

 FINISHED: efficientnet_b3. Average kappa out of 5 folds: 0.8441 ±0.0487

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
model.safetensors: 100% 32.3M/32.3M [00:01<00:00, 26.5MB/s]
  Parameters: 6,958,981 total, 6,958,981 trainable

============================================================
TRAINING: densenet121_fold1
============================================================
Epoch [  1/20]  Train Loss: 1.5142  Bal.Acc: 34.2%  F1: 0.3020  |  Val Loss: 1.4621  Bal.Acc: 37.7%  F1: 0.3173  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 1.4621)
Epoch [  2/20]  Train Loss: 1.1997  Bal.Acc: 54.5%  F1: 0.4880  |  Val Loss: 1.1703  Bal.Acc: 53.0%  F1: 0.4869  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 1.1703)
Epoch [  3/20]  Train Loss: 1.0011  Bal.Acc: 64.1%  F1: 0.6033  |  Val Loss: 1.0126  Bal.Acc: 63.2%  F1: 0.6262  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 1.0126)
Epoch [  4/20]  Train Loss: 0.7737  Bal.Acc: 76.1%  F1: 0.7498  |  Val Loss: 0.8735  Bal.Acc: 69.2%  F1: 0.6913  |  LR: 1.00e-04  (20.9s)
 Best checkpoint saved (val_loss: 0.8735)
Epoch [  5/20]  Train Loss: 0.6524  Bal.Acc: 78.8%  F1: 0.7788  |  Val Loss: 0.8718  Bal.Acc: 66.1%  F1: 0.6662  |  LR: 1.00e-04  (20.4s)
 Best checkpoint saved (val_loss: 0.8718)
Epoch [  6/20]  Train Loss: 0.5520  Bal.Acc: 81.4%  F1: 0.8010  |  Val Loss: 0.8537  Bal.Acc: 67.6%  F1: 0.6719  |  LR: 1.00e-04  (20.7s)
 Best checkpoint saved (val_loss: 0.8537)
Epoch [  7/20]  Train Loss: 0.4716  Bal.Acc: 83.9%  F1: 0.8287  |  Val Loss: 0.7514  Bal.Acc: 71.3%  F1: 0.7235  |  LR: 1.00e-04  (21.0s)
 Best checkpoint saved (val_loss: 0.7514)
Epoch [  8/20]  Train Loss: 0.3918  Bal.Acc: 86.5%  F1: 0.8589  |  Val Loss: 0.7767  Bal.Acc: 72.2%  F1: 0.7189  |  LR: 1.00e-04  (20.3s)
Epoch [  9/20]  Train Loss: 0.3483  Bal.Acc: 88.2%  F1: 0.8757  |  Val Loss: 0.6771  Bal.Acc: 75.4%  F1: 0.7470  |  LR: 1.00e-04  (17.4s)
 Best checkpoint saved (val_loss: 0.6771)
Epoch [ 10/20]  Train Loss: 0.3124  Bal.Acc: 88.8%  F1: 0.8814  |  Val Loss: 0.7416  Bal.Acc: 74.4%  F1: 0.7426  |  LR: 1.00e-04  (20.3s)
Epoch [ 11/20]  Train Loss: 0.2760  Bal.Acc: 91.5%  F1: 0.9046  |  Val Loss: 0.7636  Bal.Acc: 72.6%  F1: 0.7331  |  LR: 1.00e-04  (18.4s)
Epoch [ 12/20]  Train Loss: 0.2638  Bal.Acc: 90.9%  F1: 0.9028  |  Val Loss: 0.6857  Bal.Acc: 77.7%  F1: 0.7702  |  LR: 1.00e-04  (17.5s)
Epoch [ 13/20]  Train Loss: 0.2371  Bal.Acc: 91.9%  F1: 0.9147  |  Val Loss: 0.7291  Bal.Acc: 73.6%  F1: 0.7448  |  LR: 5.00e-05  (17.2s)
Epoch [ 14/20]  Train Loss: 0.1941  Bal.Acc: 94.2%  F1: 0.9394  |  Val Loss: 0.6857  Bal.Acc: 78.6%  F1: 0.7826  |  LR: 5.00e-05  (17.5s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.6771

 Training finished. Checkpoint: checkpoints/densenet121_fold1_best.pt
Log CSV: results/logs/densenet121_fold1_training_log.csv
Best weights loaded from epoch 9

Model evaluation: densenet121_fold1
----------------------------------------
  Balanced Accuracy:       75.44%
  F1 (macro):              0.7470
  Quadratic Cohen's Kappa: 0.9011
  ECE:                     0.0237
  Brier Score (mean):      0.0672

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.92      0.87        88
    Doubtful       0.78      0.60      0.68        81
        Mild       0.48      0.55      0.51        40
    Moderate       0.81      0.81      0.81        37
      Severe       0.84      0.89      0.86        35

    accuracy                           0.76       281
   macro avg       0.75      0.75      0.75       281
weighted avg       0.76      0.76      0.76       281

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
Epoch [  1/20]  Train Loss: 1.5158  Bal.Acc: 32.9%  F1: 0.3000  |  Val Loss: 1.4481  Bal.Acc: 37.5%  F1: 0.3218  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 1.4481)
Epoch [  2/20]  Train Loss: 1.2455  Bal.Acc: 54.3%  F1: 0.4869  |  Val Loss: 1.2060  Bal.Acc: 48.2%  F1: 0.4324  |  LR: 1.00e-04  (18.7s)
 Best checkpoint saved (val_loss: 1.2060)
Epoch [  3/20]  Train Loss: 1.0008  Bal.Acc: 66.1%  F1: 0.6238  |  Val Loss: 1.0203  Bal.Acc: 61.1%  F1: 0.6040  |  LR: 1.00e-04  (20.5s)
 Best checkpoint saved (val_loss: 1.0203)
Epoch [  4/20]  Train Loss: 0.8079  Bal.Acc: 72.0%  F1: 0.7079  |  Val Loss: 0.8564  Bal.Acc: 65.6%  F1: 0.6519  |  LR: 1.00e-04  (20.1s)
 Best checkpoint saved (val_loss: 0.8564)
Epoch [  5/20]  Train Loss: 0.6656  Bal.Acc: 77.8%  F1: 0.7701  |  Val Loss: 0.7708  Bal.Acc: 68.4%  F1: 0.6757  |  LR: 1.00e-04  (20.4s)
 Best checkpoint saved (val_loss: 0.7708)
Epoch [  6/20]  Train Loss: 0.5606  Bal.Acc: 81.0%  F1: 0.8018  |  Val Loss: 0.7121  Bal.Acc: 70.6%  F1: 0.7069  |  LR: 1.00e-04  (20.4s)
 Best checkpoint saved (val_loss: 0.7121)
Epoch [  7/20]  Train Loss: 0.4818  Bal.Acc: 83.2%  F1: 0.8216  |  Val Loss: 0.6918  Bal.Acc: 70.0%  F1: 0.6999  |  LR: 1.00e-04  (20.1s)
 Best checkpoint saved (val_loss: 0.6918)
Epoch [  8/20]  Train Loss: 0.4072  Bal.Acc: 86.2%  F1: 0.8607  |  Val Loss: 0.6148  Bal.Acc: 77.1%  F1: 0.7593  |  LR: 1.00e-04  (20.2s)
 Best checkpoint saved (val_loss: 0.6148)
Epoch [  9/20]  Train Loss: 0.3700  Bal.Acc: 87.5%  F1: 0.8679  |  Val Loss: 0.5328  Bal.Acc: 77.4%  F1: 0.7579  |  LR: 1.00e-04  (20.8s)
 Best checkpoint saved (val_loss: 0.5328)
Epoch [ 10/20]  Train Loss: 0.3118  Bal.Acc: 89.9%  F1: 0.8887  |  Val Loss: 0.5198  Bal.Acc: 80.7%  F1: 0.8024  |  LR: 1.00e-04  (19.8s)
 Best checkpoint saved (val_loss: 0.5198)
Epoch [ 11/20]  Train Loss: 0.2867  Bal.Acc: 90.7%  F1: 0.8998  |  Val Loss: 0.5078  Bal.Acc: 81.0%  F1: 0.7933  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 0.5078)
Epoch [ 12/20]  Train Loss: 0.2712  Bal.Acc: 90.8%  F1: 0.9038  |  Val Loss: 0.5256  Bal.Acc: 79.8%  F1: 0.7870  |  LR: 1.00e-04  (21.0s)
Epoch [ 13/20]  Train Loss: 0.2376  Bal.Acc: 92.2%  F1: 0.9193  |  Val Loss: 0.6902  Bal.Acc: 73.7%  F1: 0.7130  |  LR: 1.00e-04  (16.7s)
Epoch [ 14/20]  Train Loss: 0.2420  Bal.Acc: 92.1%  F1: 0.9114  |  Val Loss: 0.5971  Bal.Acc: 78.3%  F1: 0.7822  |  LR: 1.00e-04  (17.4s)
Epoch [ 15/20]  Train Loss: 0.1985  Bal.Acc: 93.7%  F1: 0.9329  |  Val Loss: 0.5312  Bal.Acc: 82.0%  F1: 0.8180  |  LR: 5.00e-05  (17.9s)
Epoch [ 16/20]  Train Loss: 0.1615  Bal.Acc: 94.7%  F1: 0.9436  |  Val Loss: 0.5176  Bal.Acc: 80.7%  F1: 0.8047  |  LR: 5.00e-05  (17.6s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.5078

 Training finished. Checkpoint: checkpoints/densenet121_fold2_best.pt
Log CSV: results/logs/densenet121_fold2_training_log.csv
Best weights loaded from epoch 11

Model evaluation: densenet121_fold2
----------------------------------------
  Balanced Accuracy:       81.04%
  F1 (macro):              0.7933
  Quadratic Cohen's Kappa: 0.9059
  ECE:                     0.0202
  Brier Score (mean):      0.0584

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.88      0.88      0.88        88
    Doubtful       0.80      0.63      0.70        81
        Mild       0.62      0.80      0.70        40
    Moderate       0.77      0.92      0.84        37
      Severe       0.88      0.83      0.85        35

    accuracy                           0.79       281
   macro avg       0.79      0.81      0.79       281
weighted avg       0.80      0.79      0.79       281

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
  Parameters: 6,958,981 total, 6,958,981 trainable

============================================================
TRAINING: densenet121_fold3
============================================================
Epoch [  1/20]  Train Loss: 1.4668  Bal.Acc: 38.8%  F1: 0.3652  |  Val Loss: 1.4676  Bal.Acc: 31.6%  F1: 0.3170  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 1.4676)
Epoch [  2/20]  Train Loss: 1.1892  Bal.Acc: 55.7%  F1: 0.5124  |  Val Loss: 1.1649  Bal.Acc: 52.8%  F1: 0.5430  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 1.1649)
Epoch [  3/20]  Train Loss: 0.9493  Bal.Acc: 66.8%  F1: 0.6447  |  Val Loss: 1.0524  Bal.Acc: 55.1%  F1: 0.5595  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 1.0524)
Epoch [  4/20]  Train Loss: 0.7795  Bal.Acc: 72.6%  F1: 0.7127  |  Val Loss: 0.9440  Bal.Acc: 61.3%  F1: 0.6186  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 0.9440)
Epoch [  5/20]  Train Loss: 0.6322  Bal.Acc: 77.2%  F1: 0.7689  |  Val Loss: 0.7241  Bal.Acc: 67.1%  F1: 0.6701  |  LR: 1.00e-04  (21.1s)
 Best checkpoint saved (val_loss: 0.7241)
Epoch [  6/20]  Train Loss: 0.5530  Bal.Acc: 80.5%  F1: 0.7947  |  Val Loss: 0.6412  Bal.Acc: 74.4%  F1: 0.7430  |  LR: 1.00e-04  (20.1s)
 Best checkpoint saved (val_loss: 0.6412)
Epoch [  7/20]  Train Loss: 0.4758  Bal.Acc: 82.6%  F1: 0.8153  |  Val Loss: 0.6218  Bal.Acc: 72.8%  F1: 0.7226  |  LR: 1.00e-04  (19.7s)
 Best checkpoint saved (val_loss: 0.6218)
Epoch [  8/20]  Train Loss: 0.4165  Bal.Acc: 85.2%  F1: 0.8458  |  Val Loss: 0.5981  Bal.Acc: 75.6%  F1: 0.7614  |  LR: 1.00e-04  (21.3s)
 Best checkpoint saved (val_loss: 0.5981)
Epoch [  9/20]  Train Loss: 0.3549  Bal.Acc: 88.3%  F1: 0.8763  |  Val Loss: 0.5100  Bal.Acc: 79.0%  F1: 0.7748  |  LR: 1.00e-04  (20.3s)
 Best checkpoint saved (val_loss: 0.5100)
Epoch [ 10/20]  Train Loss: 0.3003  Bal.Acc: 89.4%  F1: 0.8858  |  Val Loss: 0.5620  Bal.Acc: 75.1%  F1: 0.7495  |  LR: 1.00e-04  (20.0s)
Epoch [ 11/20]  Train Loss: 0.2915  Bal.Acc: 89.7%  F1: 0.8902  |  Val Loss: 0.4807  Bal.Acc: 80.8%  F1: 0.8080  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 0.4807)
Epoch [ 12/20]  Train Loss: 0.2363  Bal.Acc: 92.9%  F1: 0.9263  |  Val Loss: 0.5334  Bal.Acc: 77.6%  F1: 0.7785  |  LR: 1.00e-04  (20.8s)
Epoch [ 13/20]  Train Loss: 0.2351  Bal.Acc: 91.5%  F1: 0.9113  |  Val Loss: 0.5146  Bal.Acc: 79.2%  F1: 0.7958  |  LR: 1.00e-04  (17.0s)
Epoch [ 14/20]  Train Loss: 0.2032  Bal.Acc: 93.0%  F1: 0.9247  |  Val Loss: 0.5152  Bal.Acc: 81.8%  F1: 0.7977  |  LR: 1.00e-04  (17.8s)
Epoch [ 15/20]  Train Loss: 0.1850  Bal.Acc: 94.5%  F1: 0.9400  |  Val Loss: 0.5117  Bal.Acc: 80.1%  F1: 0.8099  |  LR: 5.00e-05  (17.3s)
Epoch [ 16/20]  Train Loss: 0.1579  Bal.Acc: 95.4%  F1: 0.9525  |  Val Loss: 0.4938  Bal.Acc: 81.7%  F1: 0.8068  |  LR: 5.00e-05  (17.6s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.4807

 Training finished. Checkpoint: checkpoints/densenet121_fold3_best.pt
Log CSV: results/logs/densenet121_fold3_training_log.csv
Best weights loaded from epoch 11

Model evaluation: densenet121_fold3
----------------------------------------
  Balanced Accuracy:       80.85%
  F1 (macro):              0.8080
  Quadratic Cohen's Kappa: 0.9144
  ECE:                     0.0463
  Brier Score (mean):      0.0590

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.82      0.86      0.84        87
    Doubtful       0.69      0.65      0.67        81
        Mild       0.67      0.67      0.67        39
    Moderate       0.95      0.97      0.96        38
      Severe       0.91      0.89      0.90        35

    accuracy                           0.79       280
   macro avg       0.81      0.81      0.81       280
weighted avg       0.79      0.79      0.79       280

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
Epoch [  1/20]  Train Loss: 1.5050  Bal.Acc: 34.0%  F1: 0.3113  |  Val Loss: 1.4142  Bal.Acc: 45.5%  F1: 0.3986  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 1.4142)
Epoch [  2/20]  Train Loss: 1.2330  Bal.Acc: 54.2%  F1: 0.4990  |  Val Loss: 1.2304  Bal.Acc: 51.7%  F1: 0.5045  |  LR: 1.00e-04  (17.6s)
 Best checkpoint saved (val_loss: 1.2304)
Epoch [  3/20]  Train Loss: 0.9934  Bal.Acc: 65.0%  F1: 0.6235  |  Val Loss: 1.0729  Bal.Acc: 53.7%  F1: 0.5543  |  LR: 1.00e-04  (20.3s)
 Best checkpoint saved (val_loss: 1.0729)
Epoch [  4/20]  Train Loss: 0.8077  Bal.Acc: 72.2%  F1: 0.7063  |  Val Loss: 0.8182  Bal.Acc: 70.3%  F1: 0.7149  |  LR: 1.00e-04  (20.8s)
 Best checkpoint saved (val_loss: 0.8182)
Epoch [  5/20]  Train Loss: 0.6595  Bal.Acc: 78.7%  F1: 0.7780  |  Val Loss: 0.7469  Bal.Acc: 70.5%  F1: 0.7087  |  LR: 1.00e-04  (20.2s)
 Best checkpoint saved (val_loss: 0.7469)
Epoch [  6/20]  Train Loss: 0.5617  Bal.Acc: 81.2%  F1: 0.8041  |  Val Loss: 0.5925  Bal.Acc: 77.9%  F1: 0.7809  |  LR: 1.00e-04  (20.4s)
 Best checkpoint saved (val_loss: 0.5925)
Epoch [  7/20]  Train Loss: 0.4757  Bal.Acc: 83.4%  F1: 0.8225  |  Val Loss: 0.5517  Bal.Acc: 80.8%  F1: 0.8165  |  LR: 1.00e-04  (21.1s)
 Best checkpoint saved (val_loss: 0.5517)
Epoch [  8/20]  Train Loss: 0.4044  Bal.Acc: 85.7%  F1: 0.8492  |  Val Loss: 0.5494  Bal.Acc: 81.0%  F1: 0.8194  |  LR: 1.00e-04  (20.4s)
 Best checkpoint saved (val_loss: 0.5494)
Epoch [  9/20]  Train Loss: 0.3673  Bal.Acc: 88.3%  F1: 0.8805  |  Val Loss: 0.4535  Bal.Acc: 83.7%  F1: 0.8383  |  LR: 1.00e-04  (20.1s)
 Best checkpoint saved (val_loss: 0.4535)
Epoch [ 10/20]  Train Loss: 0.3113  Bal.Acc: 90.6%  F1: 0.9000  |  Val Loss: 0.6007  Bal.Acc: 74.7%  F1: 0.7618  |  LR: 1.00e-04  (21.2s)
Epoch [ 11/20]  Train Loss: 0.3031  Bal.Acc: 90.0%  F1: 0.8957  |  Val Loss: 0.4484  Bal.Acc: 84.3%  F1: 0.8474  |  LR: 1.00e-04  (17.4s)
 Best checkpoint saved (val_loss: 0.4484)
Epoch [ 12/20]  Train Loss: 0.2496  Bal.Acc: 91.6%  F1: 0.9123  |  Val Loss: 0.4312  Bal.Acc: 83.5%  F1: 0.8354  |  LR: 1.00e-04  (20.1s)
 Best checkpoint saved (val_loss: 0.4312)
Epoch [ 13/20]  Train Loss: 0.2332  Bal.Acc: 92.5%  F1: 0.9205  |  Val Loss: 0.4812  Bal.Acc: 81.2%  F1: 0.8300  |  LR: 1.00e-04  (19.9s)
Epoch [ 14/20]  Train Loss: 0.2318  Bal.Acc: 91.9%  F1: 0.9129  |  Val Loss: 0.4104  Bal.Acc: 83.7%  F1: 0.8406  |  LR: 1.00e-04  (17.6s)
 Best checkpoint saved (val_loss: 0.4104)
Epoch [ 15/20]  Train Loss: 0.2155  Bal.Acc: 92.9%  F1: 0.9237  |  Val Loss: 0.4270  Bal.Acc: 83.8%  F1: 0.8431  |  LR: 1.00e-04  (21.4s)
Epoch [ 16/20]  Train Loss: 0.1982  Bal.Acc: 93.8%  F1: 0.9377  |  Val Loss: 0.4725  Bal.Acc: 82.9%  F1: 0.8252  |  LR: 1.00e-04  (16.9s)
Epoch [ 17/20]  Train Loss: 0.1692  Bal.Acc: 94.3%  F1: 0.9355  |  Val Loss: 0.4709  Bal.Acc: 83.9%  F1: 0.8431  |  LR: 1.00e-04  (17.5s)
Epoch [ 18/20]  Train Loss: 0.1643  Bal.Acc: 94.5%  F1: 0.9446  |  Val Loss: 0.5110  Bal.Acc: 83.6%  F1: 0.8342  |  LR: 5.00e-05  (18.0s)
Epoch [ 19/20]  Train Loss: 0.1216  Bal.Acc: 96.4%  F1: 0.9609  |  Val Loss: 0.4609  Bal.Acc: 84.8%  F1: 0.8574  |  LR: 5.00e-05  (17.7s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.4104

 Training finished. Checkpoint: checkpoints/densenet121_fold4_best.pt
Log CSV: results/logs/densenet121_fold4_training_log.csv
Best weights loaded from epoch 14

Model evaluation: densenet121_fold4
----------------------------------------
  Balanced Accuracy:       83.74%
  F1 (macro):              0.8406
  Quadratic Cohen's Kappa: 0.9132
  ECE:                     0.0389
  Brier Score (mean):      0.0425

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.86      0.91      0.88        87
    Doubtful       0.79      0.80      0.80        81
        Mild       0.83      0.64      0.72        39
    Moderate       0.90      0.92      0.91        38
      Severe       0.86      0.91      0.89        35

    accuracy                           0.84       280
   macro avg       0.85      0.84      0.84       280
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
Epoch [  1/20]  Train Loss: 1.5329  Bal.Acc: 32.1%  F1: 0.3078  |  Val Loss: 1.4610  Bal.Acc: 33.2%  F1: 0.2964  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 1.4610)
Epoch [  2/20]  Train Loss: 1.2797  Bal.Acc: 50.4%  F1: 0.4663  |  Val Loss: 1.1769  Bal.Acc: 51.2%  F1: 0.4787  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 1.1769)
Epoch [  3/20]  Train Loss: 1.0654  Bal.Acc: 61.5%  F1: 0.5833  |  Val Loss: 1.0038  Bal.Acc: 61.6%  F1: 0.6122  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 1.0038)
Epoch [  4/20]  Train Loss: 0.8927  Bal.Acc: 68.7%  F1: 0.6685  |  Val Loss: 0.8763  Bal.Acc: 67.0%  F1: 0.6699  |  LR: 1.00e-04  (20.6s)
 Best checkpoint saved (val_loss: 0.8763)
Epoch [  5/20]  Train Loss: 0.7334  Bal.Acc: 73.2%  F1: 0.7254  |  Val Loss: 0.7070  Bal.Acc: 74.0%  F1: 0.7392  |  LR: 1.00e-04  (19.9s)
 Best checkpoint saved (val_loss: 0.7070)
Epoch [  6/20]  Train Loss: 0.6089  Bal.Acc: 80.5%  F1: 0.7985  |  Val Loss: 0.6428  Bal.Acc: 76.1%  F1: 0.7518  |  LR: 1.00e-04  (20.4s)
 Best checkpoint saved (val_loss: 0.6428)
Epoch [  7/20]  Train Loss: 0.5270  Bal.Acc: 82.9%  F1: 0.8195  |  Val Loss: 0.5828  Bal.Acc: 77.8%  F1: 0.7662  |  LR: 1.00e-04  (20.2s)
 Best checkpoint saved (val_loss: 0.5828)
Epoch [  8/20]  Train Loss: 0.4722  Bal.Acc: 83.7%  F1: 0.8359  |  Val Loss: 0.5017  Bal.Acc: 80.7%  F1: 0.7951  |  LR: 1.00e-04  (20.9s)
 Best checkpoint saved (val_loss: 0.5017)
Epoch [  9/20]  Train Loss: 0.4086  Bal.Acc: 86.4%  F1: 0.8608  |  Val Loss: 0.5491  Bal.Acc: 80.8%  F1: 0.8024  |  LR: 1.00e-04  (20.0s)
Epoch [ 10/20]  Train Loss: 0.3595  Bal.Acc: 88.1%  F1: 0.8757  |  Val Loss: 0.4591  Bal.Acc: 82.4%  F1: 0.8127  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 0.4591)
Epoch [ 11/20]  Train Loss: 0.3399  Bal.Acc: 88.5%  F1: 0.8787  |  Val Loss: 0.4886  Bal.Acc: 81.2%  F1: 0.7901  |  LR: 1.00e-04  (21.4s)
Epoch [ 12/20]  Train Loss: 0.2819  Bal.Acc: 91.4%  F1: 0.9077  |  Val Loss: 0.5878  Bal.Acc: 77.5%  F1: 0.7724  |  LR: 1.00e-04  (17.7s)
Epoch [ 13/20]  Train Loss: 0.2736  Bal.Acc: 91.0%  F1: 0.9074  |  Val Loss: 0.4949  Bal.Acc: 83.3%  F1: 0.8290  |  LR: 1.00e-04  (17.4s)
Epoch [ 14/20]  Train Loss: 0.2311  Bal.Acc: 93.3%  F1: 0.9283  |  Val Loss: 0.4477  Bal.Acc: 83.4%  F1: 0.8279  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 0.4477)
Epoch [ 15/20]  Train Loss: 0.2056  Bal.Acc: 93.4%  F1: 0.9302  |  Val Loss: 0.5332  Bal.Acc: 80.5%  F1: 0.8062  |  LR: 1.00e-04  (20.1s)
Epoch [ 16/20]  Train Loss: 0.1795  Bal.Acc: 94.6%  F1: 0.9421  |  Val Loss: 0.4384  Bal.Acc: 85.1%  F1: 0.8352  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 0.4384)
Epoch [ 17/20]  Train Loss: 0.1745  Bal.Acc: 94.9%  F1: 0.9441  |  Val Loss: 0.6661  Bal.Acc: 75.5%  F1: 0.7625  |  LR: 1.00e-04  (21.1s)
Epoch [ 18/20]  Train Loss: 0.1669  Bal.Acc: 94.2%  F1: 0.9381  |  Val Loss: 0.4165  Bal.Acc: 86.7%  F1: 0.8555  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 0.4165)
Epoch [ 19/20]  Train Loss: 0.1409  Bal.Acc: 95.7%  F1: 0.9559  |  Val Loss: 0.4308  Bal.Acc: 85.3%  F1: 0.8417  |  LR: 1.00e-04  (20.1s)
Epoch [ 20/20]  Train Loss: 0.1377  Bal.Acc: 95.5%  F1: 0.9514  |  Val Loss: 0.4923  Bal.Acc: 82.9%  F1: 0.8275  |  LR: 1.00e-04  (17.7s)

 Training finished. Checkpoint: checkpoints/densenet121_fold5_best.pt
Log CSV: results/logs/densenet121_fold5_training_log.csv
Best weights loaded from epoch 18

Model evaluation: densenet121_fold5
----------------------------------------
  Balanced Accuracy:       86.68%
  F1 (macro):              0.8555
  Quadratic Cohen's Kappa: 0.9373
  ECE:                     0.0378
  Brier Score (mean):      0.0473

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.92      0.93      0.93        87
    Doubtful       0.82      0.69      0.75        81
        Mild       0.62      0.79      0.70        39
    Moderate       0.97      0.97      0.97        38
      Severe       0.92      0.94      0.93        35

    accuracy                           0.85       280
   macro avg       0.85      0.87      0.86       280
weighted avg       0.86      0.85      0.85       280

  Metrics saved to: results/individual_models/densenet121_fold5_metrics.json
  Probabilities saved to: results/individual_models/densenet121_fold5_test_probs.npz

 FINISHED: densenet121. Average kappa out of 5 folds: 0.9144 ±0.0124

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
model.safetensors: 100% 22.1M/22.1M [00:00<00:00, 27.0MB/s]
  Parameters: 4,208,437 total, 4,208,437 trainable

============================================================
TRAINING: mobilenetv3_large_fold1
============================================================
Epoch [  1/20]  Train Loss: 2.0845  Bal.Acc: 38.2%  F1: 0.3633  |  Val Loss: 2.5635  Bal.Acc: 30.1%  F1: 0.3019  |  LR: 1.00e-04  (14.2s)
 Best checkpoint saved (val_loss: 2.5635)
Epoch [  2/20]  Train Loss: 1.3576  Bal.Acc: 53.9%  F1: 0.5221  |  Val Loss: 2.0597  Bal.Acc: 42.1%  F1: 0.4289  |  LR: 1.00e-04  (14.8s)
 Best checkpoint saved (val_loss: 2.0597)
Epoch [  3/20]  Train Loss: 1.0450  Bal.Acc: 62.1%  F1: 0.6049  |  Val Loss: 1.7165  Bal.Acc: 52.3%  F1: 0.5305  |  LR: 1.00e-04  (17.2s)
 Best checkpoint saved (val_loss: 1.7165)
Epoch [  4/20]  Train Loss: 0.8995  Bal.Acc: 66.9%  F1: 0.6597  |  Val Loss: 1.2226  Bal.Acc: 64.7%  F1: 0.6235  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 1.2226)
Epoch [  5/20]  Train Loss: 0.7153  Bal.Acc: 72.5%  F1: 0.7136  |  Val Loss: 1.0825  Bal.Acc: 66.7%  F1: 0.6652  |  LR: 1.00e-04  (17.3s)
 Best checkpoint saved (val_loss: 1.0825)
Epoch [  6/20]  Train Loss: 0.6969  Bal.Acc: 75.6%  F1: 0.7418  |  Val Loss: 0.9818  Bal.Acc: 69.1%  F1: 0.6872  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 0.9818)
Epoch [  7/20]  Train Loss: 0.5748  Bal.Acc: 78.5%  F1: 0.7753  |  Val Loss: 0.9947  Bal.Acc: 71.1%  F1: 0.7015  |  LR: 1.00e-04  (16.8s)
Epoch [  8/20]  Train Loss: 0.5291  Bal.Acc: 79.7%  F1: 0.7879  |  Val Loss: 0.9092  Bal.Acc: 71.2%  F1: 0.7068  |  LR: 1.00e-04  (14.7s)
 Best checkpoint saved (val_loss: 0.9092)
Epoch [  9/20]  Train Loss: 0.4641  Bal.Acc: 82.7%  F1: 0.8141  |  Val Loss: 0.9558  Bal.Acc: 71.8%  F1: 0.7301  |  LR: 1.00e-04  (16.7s)
Epoch [ 10/20]  Train Loss: 0.4844  Bal.Acc: 82.1%  F1: 0.8074  |  Val Loss: 0.9062  Bal.Acc: 69.6%  F1: 0.7001  |  LR: 1.00e-04  (14.6s)
 Best checkpoint saved (val_loss: 0.9062)
Epoch [ 11/20]  Train Loss: 0.3876  Bal.Acc: 85.1%  F1: 0.8406  |  Val Loss: 0.8215  Bal.Acc: 73.8%  F1: 0.7419  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 0.8215)
Epoch [ 12/20]  Train Loss: 0.3549  Bal.Acc: 86.5%  F1: 0.8622  |  Val Loss: 0.7530  Bal.Acc: 74.5%  F1: 0.7380  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 0.7530)
Epoch [ 13/20]  Train Loss: 0.3248  Bal.Acc: 87.7%  F1: 0.8673  |  Val Loss: 0.8663  Bal.Acc: 75.0%  F1: 0.7587  |  LR: 1.00e-04  (16.8s)
Epoch [ 14/20]  Train Loss: 0.3068  Bal.Acc: 89.0%  F1: 0.8828  |  Val Loss: 0.8008  Bal.Acc: 78.0%  F1: 0.7817  |  LR: 1.00e-04  (14.4s)
Epoch [ 15/20]  Train Loss: 0.2923  Bal.Acc: 88.7%  F1: 0.8804  |  Val Loss: 0.8077  Bal.Acc: 76.8%  F1: 0.7653  |  LR: 1.00e-04  (14.5s)
Epoch [ 16/20]  Train Loss: 0.2830  Bal.Acc: 90.0%  F1: 0.8950  |  Val Loss: 0.7651  Bal.Acc: 75.4%  F1: 0.7567  |  LR: 5.00e-05  (14.9s)
Epoch [ 17/20]  Train Loss: 0.2275  Bal.Acc: 91.3%  F1: 0.9063  |  Val Loss: 0.8837  Bal.Acc: 74.0%  F1: 0.7365  |  LR: 5.00e-05  (14.5s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.7530

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold1_best.pt
Log CSV: results/logs/mobilenetv3_large_fold1_training_log.csv
Best weights loaded from epoch 12

Model evaluation: mobilenetv3_large_fold1
----------------------------------------
  Balanced Accuracy:       74.45%
  F1 (macro):              0.7380
  Quadratic Cohen's Kappa: 0.8658
  ECE:                     0.1073
  Brier Score (mean):      0.0740

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.81      0.80      0.80        88
    Doubtful       0.71      0.70      0.71        81
        Mild       0.54      0.50      0.52        40
    Moderate       0.74      0.84      0.78        37
      Severe       0.86      0.89      0.87        35

    accuracy                           0.74       281
   macro avg       0.73      0.74      0.74       281
weighted avg       0.74      0.74      0.74       281

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
Epoch [  1/20]  Train Loss: 2.2800  Bal.Acc: 35.3%  F1: 0.3368  |  Val Loss: 2.9868  Bal.Acc: 31.7%  F1: 0.2962  |  LR: 1.00e-04  (14.7s)
 Best checkpoint saved (val_loss: 2.9868)
Epoch [  2/20]  Train Loss: 1.3603  Bal.Acc: 52.2%  F1: 0.5062  |  Val Loss: 2.2089  Bal.Acc: 40.2%  F1: 0.3967  |  LR: 1.00e-04  (15.3s)
 Best checkpoint saved (val_loss: 2.2089)
Epoch [  3/20]  Train Loss: 1.0182  Bal.Acc: 62.1%  F1: 0.6089  |  Val Loss: 1.8447  Bal.Acc: 48.5%  F1: 0.4884  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 1.8447)
Epoch [  4/20]  Train Loss: 0.8740  Bal.Acc: 68.1%  F1: 0.6679  |  Val Loss: 1.1257  Bal.Acc: 61.1%  F1: 0.6076  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 1.1257)
Epoch [  5/20]  Train Loss: 0.6961  Bal.Acc: 72.7%  F1: 0.7140  |  Val Loss: 0.9401  Bal.Acc: 65.7%  F1: 0.6463  |  LR: 1.00e-04  (17.2s)
 Best checkpoint saved (val_loss: 0.9401)
Epoch [  6/20]  Train Loss: 0.6196  Bal.Acc: 77.1%  F1: 0.7621  |  Val Loss: 0.8305  Bal.Acc: 70.3%  F1: 0.6822  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 0.8305)
Epoch [  7/20]  Train Loss: 0.5736  Bal.Acc: 79.8%  F1: 0.7840  |  Val Loss: 0.7413  Bal.Acc: 72.1%  F1: 0.7141  |  LR: 1.00e-04  (17.1s)
 Best checkpoint saved (val_loss: 0.7413)
Epoch [  8/20]  Train Loss: 0.5228  Bal.Acc: 79.7%  F1: 0.7849  |  Val Loss: 0.7200  Bal.Acc: 74.4%  F1: 0.7434  |  LR: 1.00e-04  (17.4s)
 Best checkpoint saved (val_loss: 0.7200)
Epoch [  9/20]  Train Loss: 0.5071  Bal.Acc: 79.3%  F1: 0.7891  |  Val Loss: 0.7206  Bal.Acc: 72.3%  F1: 0.7065  |  LR: 1.00e-04  (17.1s)
Epoch [ 10/20]  Train Loss: 0.4207  Bal.Acc: 84.6%  F1: 0.8358  |  Val Loss: 0.7327  Bal.Acc: 73.8%  F1: 0.7406  |  LR: 1.00e-04  (14.1s)
Epoch [ 11/20]  Train Loss: 0.3885  Bal.Acc: 85.4%  F1: 0.8444  |  Val Loss: 0.6667  Bal.Acc: 75.9%  F1: 0.7574  |  LR: 1.00e-04  (14.0s)
 Best checkpoint saved (val_loss: 0.6667)
Epoch [ 12/20]  Train Loss: 0.3325  Bal.Acc: 87.9%  F1: 0.8747  |  Val Loss: 0.6517  Bal.Acc: 76.1%  F1: 0.7497  |  LR: 1.00e-04  (17.1s)
 Best checkpoint saved (val_loss: 0.6517)
Epoch [ 13/20]  Train Loss: 0.3294  Bal.Acc: 87.2%  F1: 0.8621  |  Val Loss: 0.6611  Bal.Acc: 77.4%  F1: 0.7727  |  LR: 1.00e-04  (17.2s)
Epoch [ 14/20]  Train Loss: 0.3154  Bal.Acc: 88.4%  F1: 0.8789  |  Val Loss: 0.6670  Bal.Acc: 76.8%  F1: 0.7595  |  LR: 1.00e-04  (13.7s)
Epoch [ 15/20]  Train Loss: 0.2590  Bal.Acc: 90.3%  F1: 0.8975  |  Val Loss: 0.6493  Bal.Acc: 76.0%  F1: 0.7584  |  LR: 1.00e-04  (13.9s)
 Best checkpoint saved (val_loss: 0.6493)
Epoch [ 16/20]  Train Loss: 0.2266  Bal.Acc: 91.2%  F1: 0.9087  |  Val Loss: 0.6617  Bal.Acc: 78.1%  F1: 0.7801  |  LR: 1.00e-04  (17.0s)
Epoch [ 17/20]  Train Loss: 0.2103  Bal.Acc: 91.8%  F1: 0.9137  |  Val Loss: 0.6327  Bal.Acc: 79.8%  F1: 0.7926  |  LR: 1.00e-04  (13.9s)
 Best checkpoint saved (val_loss: 0.6327)
Epoch [ 18/20]  Train Loss: 0.2224  Bal.Acc: 91.2%  F1: 0.9058  |  Val Loss: 0.6465  Bal.Acc: 82.1%  F1: 0.8198  |  LR: 1.00e-04  (17.1s)
Epoch [ 19/20]  Train Loss: 0.2201  Bal.Acc: 92.1%  F1: 0.9164  |  Val Loss: 0.6980  Bal.Acc: 79.9%  F1: 0.8059  |  LR: 1.00e-04  (14.0s)
Epoch [ 20/20]  Train Loss: 0.1954  Bal.Acc: 92.9%  F1: 0.9232  |  Val Loss: 0.6372  Bal.Acc: 80.9%  F1: 0.8047  |  LR: 1.00e-04  (13.9s)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold2_best.pt
Log CSV: results/logs/mobilenetv3_large_fold2_training_log.csv
Best weights loaded from epoch 17

Model evaluation: mobilenetv3_large_fold2
----------------------------------------
  Balanced Accuracy:       79.77%
  F1 (macro):              0.7926
  Quadratic Cohen's Kappa: 0.8833
  ECE:                     0.0984
  Brier Score (mean):      0.0636

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.84      0.90      0.87        88
    Doubtful       0.76      0.67      0.71        81
        Mild       0.64      0.68      0.66        40
    Moderate       0.92      0.89      0.90        37
      Severe       0.79      0.86      0.82        35

    accuracy                           0.79       281
   macro avg       0.79      0.80      0.79       281
weighted avg       0.79      0.79      0.79       281

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
Epoch [  1/20]  Train Loss: 2.4737  Bal.Acc: 34.9%  F1: 0.3229  |  Val Loss: 2.6486  Bal.Acc: 34.2%  F1: 0.3181  |  LR: 1.00e-04  (14.3s)
 Best checkpoint saved (val_loss: 2.6486)
Epoch [  2/20]  Train Loss: 1.4821  Bal.Acc: 54.2%  F1: 0.5230  |  Val Loss: 2.1213  Bal.Acc: 38.3%  F1: 0.3161  |  LR: 1.00e-04  (14.7s)
 Best checkpoint saved (val_loss: 2.1213)
Epoch [  3/20]  Train Loss: 1.0999  Bal.Acc: 60.0%  F1: 0.5911  |  Val Loss: 1.6351  Bal.Acc: 47.4%  F1: 0.4528  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 1.6351)
Epoch [  4/20]  Train Loss: 0.8522  Bal.Acc: 67.6%  F1: 0.6628  |  Val Loss: 1.2069  Bal.Acc: 54.6%  F1: 0.5541  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 1.2069)
Epoch [  5/20]  Train Loss: 0.7707  Bal.Acc: 72.2%  F1: 0.7105  |  Val Loss: 1.0024  Bal.Acc: 62.7%  F1: 0.6264  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 1.0024)
Epoch [  6/20]  Train Loss: 0.7429  Bal.Acc: 71.5%  F1: 0.7043  |  Val Loss: 0.9822  Bal.Acc: 61.4%  F1: 0.6207  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 0.9822)
Epoch [  7/20]  Train Loss: 0.5318  Bal.Acc: 81.0%  F1: 0.7969  |  Val Loss: 0.8610  Bal.Acc: 67.7%  F1: 0.6810  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 0.8610)
Epoch [  8/20]  Train Loss: 0.4867  Bal.Acc: 81.8%  F1: 0.8127  |  Val Loss: 0.8101  Bal.Acc: 69.7%  F1: 0.6992  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 0.8101)
Epoch [  9/20]  Train Loss: 0.5002  Bal.Acc: 81.5%  F1: 0.8043  |  Val Loss: 0.8423  Bal.Acc: 70.0%  F1: 0.6988  |  LR: 1.00e-04  (16.5s)
Epoch [ 10/20]  Train Loss: 0.4563  Bal.Acc: 83.9%  F1: 0.8311  |  Val Loss: 0.7727  Bal.Acc: 71.9%  F1: 0.7165  |  LR: 1.00e-04  (14.5s)
 Best checkpoint saved (val_loss: 0.7727)
Epoch [ 11/20]  Train Loss: 0.4010  Bal.Acc: 85.7%  F1: 0.8484  |  Val Loss: 0.7638  Bal.Acc: 71.3%  F1: 0.7127  |  LR: 1.00e-04  (16.6s)
 Best checkpoint saved (val_loss: 0.7638)
Epoch [ 12/20]  Train Loss: 0.3496  Bal.Acc: 87.2%  F1: 0.8637  |  Val Loss: 0.7098  Bal.Acc: 76.4%  F1: 0.7546  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 0.7098)
Epoch [ 13/20]  Train Loss: 0.3514  Bal.Acc: 86.4%  F1: 0.8522  |  Val Loss: 0.7061  Bal.Acc: 75.1%  F1: 0.7472  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 0.7061)
Epoch [ 14/20]  Train Loss: 0.3528  Bal.Acc: 87.5%  F1: 0.8719  |  Val Loss: 0.6764  Bal.Acc: 74.7%  F1: 0.7404  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 0.6764)
Epoch [ 15/20]  Train Loss: 0.2607  Bal.Acc: 91.1%  F1: 0.9091  |  Val Loss: 0.7002  Bal.Acc: 77.8%  F1: 0.7631  |  LR: 1.00e-04  (17.3s)
Epoch [ 16/20]  Train Loss: 0.2858  Bal.Acc: 88.9%  F1: 0.8822  |  Val Loss: 0.6896  Bal.Acc: 76.6%  F1: 0.7639  |  LR: 1.00e-04  (14.1s)
Epoch [ 17/20]  Train Loss: 0.2610  Bal.Acc: 90.4%  F1: 0.8972  |  Val Loss: 0.6886  Bal.Acc: 77.4%  F1: 0.7670  |  LR: 1.00e-04  (14.3s)
Epoch [ 18/20]  Train Loss: 0.2338  Bal.Acc: 92.1%  F1: 0.9172  |  Val Loss: 0.6539  Bal.Acc: 75.9%  F1: 0.7531  |  LR: 1.00e-04  (14.4s)
 Best checkpoint saved (val_loss: 0.6539)
Epoch [ 19/20]  Train Loss: 0.2406  Bal.Acc: 90.8%  F1: 0.9055  |  Val Loss: 0.6821  Bal.Acc: 76.5%  F1: 0.7654  |  LR: 1.00e-04  (16.7s)
Epoch [ 20/20]  Train Loss: 0.2406  Bal.Acc: 91.6%  F1: 0.9104  |  Val Loss: 0.6467  Bal.Acc: 76.5%  F1: 0.7642  |  LR: 1.00e-04  (14.4s)
 Best checkpoint saved (val_loss: 0.6467)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold3_best.pt
Log CSV: results/logs/mobilenetv3_large_fold3_training_log.csv
Best weights loaded from epoch 20

Model evaluation: mobilenetv3_large_fold3
----------------------------------------
  Balanced Accuracy:       76.54%
  F1 (macro):              0.7642
  Quadratic Cohen's Kappa: 0.8836
  ECE:                     0.1189
  Brier Score (mean):      0.0686

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.82      0.86      0.84        87
    Doubtful       0.73      0.68      0.71        81
        Mild       0.60      0.62      0.61        39
    Moderate       0.86      0.84      0.85        38
      Severe       0.81      0.83      0.82        35

    accuracy                           0.77       280
   macro avg       0.76      0.77      0.76       280
weighted avg       0.77      0.77      0.77       280

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
Epoch [  1/20]  Train Loss: 2.3093  Bal.Acc: 35.9%  F1: 0.3424  |  Val Loss: 2.1084  Bal.Acc: 41.9%  F1: 0.3835  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 2.1084)
Epoch [  2/20]  Train Loss: 1.3583  Bal.Acc: 52.8%  F1: 0.5123  |  Val Loss: 1.7085  Bal.Acc: 48.9%  F1: 0.4383  |  LR: 1.00e-04  (14.3s)
 Best checkpoint saved (val_loss: 1.7085)
Epoch [  3/20]  Train Loss: 1.0930  Bal.Acc: 61.6%  F1: 0.5988  |  Val Loss: 1.4617  Bal.Acc: 54.1%  F1: 0.5371  |  LR: 1.00e-04  (17.0s)
 Best checkpoint saved (val_loss: 1.4617)
Epoch [  4/20]  Train Loss: 0.9052  Bal.Acc: 66.7%  F1: 0.6563  |  Val Loss: 1.1292  Bal.Acc: 62.5%  F1: 0.6366  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 1.1292)
Epoch [  5/20]  Train Loss: 0.7975  Bal.Acc: 71.5%  F1: 0.7032  |  Val Loss: 0.9844  Bal.Acc: 67.1%  F1: 0.6729  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 0.9844)
Epoch [  6/20]  Train Loss: 0.7072  Bal.Acc: 74.1%  F1: 0.7284  |  Val Loss: 0.9430  Bal.Acc: 70.2%  F1: 0.7043  |  LR: 1.00e-04  (16.8s)
 Best checkpoint saved (val_loss: 0.9430)
Epoch [  7/20]  Train Loss: 0.6276  Bal.Acc: 75.6%  F1: 0.7463  |  Val Loss: 1.0650  Bal.Acc: 66.1%  F1: 0.6880  |  LR: 1.00e-04  (16.9s)
Epoch [  8/20]  Train Loss: 0.5659  Bal.Acc: 78.2%  F1: 0.7702  |  Val Loss: 0.9087  Bal.Acc: 70.0%  F1: 0.7177  |  LR: 1.00e-04  (14.7s)
 Best checkpoint saved (val_loss: 0.9087)
Epoch [  9/20]  Train Loss: 0.5055  Bal.Acc: 81.7%  F1: 0.8075  |  Val Loss: 0.8215  Bal.Acc: 74.0%  F1: 0.7473  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 0.8215)
Epoch [ 10/20]  Train Loss: 0.4552  Bal.Acc: 83.6%  F1: 0.8295  |  Val Loss: 1.0854  Bal.Acc: 74.5%  F1: 0.7603  |  LR: 1.00e-04  (16.6s)
Epoch [ 11/20]  Train Loss: 0.4184  Bal.Acc: 84.8%  F1: 0.8373  |  Val Loss: 0.7648  Bal.Acc: 74.4%  F1: 0.7563  |  LR: 1.00e-04  (14.8s)
 Best checkpoint saved (val_loss: 0.7648)
Epoch [ 12/20]  Train Loss: 0.4061  Bal.Acc: 84.5%  F1: 0.8327  |  Val Loss: 0.8993  Bal.Acc: 73.4%  F1: 0.7498  |  LR: 1.00e-04  (17.0s)
Epoch [ 13/20]  Train Loss: 0.3638  Bal.Acc: 86.8%  F1: 0.8628  |  Val Loss: 0.7819  Bal.Acc: 75.5%  F1: 0.7715  |  LR: 1.00e-04  (14.6s)
Epoch [ 14/20]  Train Loss: 0.3246  Bal.Acc: 87.8%  F1: 0.8663  |  Val Loss: 0.7952  Bal.Acc: 74.9%  F1: 0.7701  |  LR: 1.00e-04  (14.8s)
Epoch [ 15/20]  Train Loss: 0.2993  Bal.Acc: 88.3%  F1: 0.8795  |  Val Loss: 0.8053  Bal.Acc: 75.8%  F1: 0.7628  |  LR: 5.00e-05  (14.7s)
Epoch [ 16/20]  Train Loss: 0.3006  Bal.Acc: 89.0%  F1: 0.8822  |  Val Loss: 0.7184  Bal.Acc: 79.0%  F1: 0.7979  |  LR: 5.00e-05  (14.6s)
 Best checkpoint saved (val_loss: 0.7184)
Epoch [ 17/20]  Train Loss: 0.2535  Bal.Acc: 91.7%  F1: 0.9104  |  Val Loss: 0.8013  Bal.Acc: 76.4%  F1: 0.7801  |  LR: 5.00e-05  (17.0s)
Epoch [ 18/20]  Train Loss: 0.2460  Bal.Acc: 90.5%  F1: 0.8970  |  Val Loss: 0.7148  Bal.Acc: 76.3%  F1: 0.7788  |  LR: 5.00e-05  (14.9s)
 Best checkpoint saved (val_loss: 0.7148)
Epoch [ 19/20]  Train Loss: 0.2400  Bal.Acc: 91.2%  F1: 0.9102  |  Val Loss: 0.7787  Bal.Acc: 75.6%  F1: 0.7774  |  LR: 5.00e-05  (16.8s)
Epoch [ 20/20]  Train Loss: 0.2300  Bal.Acc: 91.4%  F1: 0.9086  |  Val Loss: 0.7549  Bal.Acc: 75.3%  F1: 0.7718  |  LR: 5.00e-05  (15.0s)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold4_best.pt
Log CSV: results/logs/mobilenetv3_large_fold4_training_log.csv
Best weights loaded from epoch 18

Model evaluation: mobilenetv3_large_fold4
----------------------------------------
  Balanced Accuracy:       76.25%
  F1 (macro):              0.7788
  Quadratic Cohen's Kappa: 0.8559
  ECE:                     0.0941
  Brier Score (mean):      0.0601

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.82      0.85      0.84        87
    Doubtful       0.69      0.77      0.73        81
        Mild       0.68      0.67      0.68        39
    Moderate       0.86      0.82      0.84        38
      Severe       0.96      0.71      0.82        35

    accuracy                           0.78       280
   macro avg       0.80      0.76      0.78       280
weighted avg       0.79      0.78      0.78       280

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
Epoch [  1/20]  Train Loss: 2.3962  Bal.Acc: 32.1%  F1: 0.3085  |  Val Loss: 2.2679  Bal.Acc: 36.6%  F1: 0.3698  |  LR: 1.00e-04  (15.0s)
 Best checkpoint saved (val_loss: 2.2679)
Epoch [  2/20]  Train Loss: 1.4857  Bal.Acc: 51.6%  F1: 0.4936  |  Val Loss: 1.5206  Bal.Acc: 48.4%  F1: 0.4828  |  LR: 1.00e-04  (14.9s)
 Best checkpoint saved (val_loss: 1.5206)
Epoch [  3/20]  Train Loss: 1.1117  Bal.Acc: 60.6%  F1: 0.5935  |  Val Loss: 1.4591  Bal.Acc: 47.1%  F1: 0.4664  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 1.4591)
Epoch [  4/20]  Train Loss: 1.0064  Bal.Acc: 64.0%  F1: 0.6291  |  Val Loss: 1.0547  Bal.Acc: 58.1%  F1: 0.5648  |  LR: 1.00e-04  (17.1s)
 Best checkpoint saved (val_loss: 1.0547)
Epoch [  5/20]  Train Loss: 0.7789  Bal.Acc: 70.3%  F1: 0.6870  |  Val Loss: 1.1069  Bal.Acc: 58.8%  F1: 0.5912  |  LR: 1.00e-04  (17.2s)
Epoch [  6/20]  Train Loss: 0.7858  Bal.Acc: 71.4%  F1: 0.7061  |  Val Loss: 0.9570  Bal.Acc: 65.8%  F1: 0.6651  |  LR: 1.00e-04  (14.3s)
 Best checkpoint saved (val_loss: 0.9570)
Epoch [  7/20]  Train Loss: 0.6480  Bal.Acc: 76.2%  F1: 0.7503  |  Val Loss: 0.9324  Bal.Acc: 69.1%  F1: 0.6910  |  LR: 1.00e-04  (17.4s)
 Best checkpoint saved (val_loss: 0.9324)
Epoch [  8/20]  Train Loss: 0.5518  Bal.Acc: 80.2%  F1: 0.7891  |  Val Loss: 0.7901  Bal.Acc: 73.8%  F1: 0.7364  |  LR: 1.00e-04  (17.2s)
 Best checkpoint saved (val_loss: 0.7901)
Epoch [  9/20]  Train Loss: 0.4992  Bal.Acc: 80.9%  F1: 0.7993  |  Val Loss: 0.7745  Bal.Acc: 69.6%  F1: 0.6918  |  LR: 1.00e-04  (16.7s)
 Best checkpoint saved (val_loss: 0.7745)
Epoch [ 10/20]  Train Loss: 0.4689  Bal.Acc: 82.5%  F1: 0.8187  |  Val Loss: 0.7579  Bal.Acc: 72.5%  F1: 0.7241  |  LR: 1.00e-04  (17.1s)
 Best checkpoint saved (val_loss: 0.7579)
Epoch [ 11/20]  Train Loss: 0.3957  Bal.Acc: 85.1%  F1: 0.8423  |  Val Loss: 0.7317  Bal.Acc: 74.1%  F1: 0.7356  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 0.7317)
Epoch [ 12/20]  Train Loss: 0.3466  Bal.Acc: 86.7%  F1: 0.8611  |  Val Loss: 0.7279  Bal.Acc: 74.0%  F1: 0.7345  |  LR: 1.00e-04  (17.2s)
 Best checkpoint saved (val_loss: 0.7279)
Epoch [ 13/20]  Train Loss: 0.3272  Bal.Acc: 87.9%  F1: 0.8777  |  Val Loss: 0.7014  Bal.Acc: 78.4%  F1: 0.7756  |  LR: 1.00e-04  (16.9s)
 Best checkpoint saved (val_loss: 0.7014)
Epoch [ 14/20]  Train Loss: 0.3283  Bal.Acc: 88.1%  F1: 0.8726  |  Val Loss: 0.6648  Bal.Acc: 78.5%  F1: 0.7823  |  LR: 1.00e-04  (17.2s)
 Best checkpoint saved (val_loss: 0.6648)
Epoch [ 15/20]  Train Loss: 0.3124  Bal.Acc: 87.8%  F1: 0.8732  |  Val Loss: 0.7066  Bal.Acc: 77.3%  F1: 0.7625  |  LR: 1.00e-04  (17.3s)
Epoch [ 16/20]  Train Loss: 0.3008  Bal.Acc: 89.1%  F1: 0.8835  |  Val Loss: 0.6929  Bal.Acc: 75.0%  F1: 0.7562  |  LR: 1.00e-04  (14.4s)
Epoch [ 17/20]  Train Loss: 0.2695  Bal.Acc: 90.5%  F1: 0.8955  |  Val Loss: 0.7234  Bal.Acc: 77.7%  F1: 0.7853  |  LR: 1.00e-04  (14.7s)
Epoch [ 18/20]  Train Loss: 0.2471  Bal.Acc: 91.5%  F1: 0.9097  |  Val Loss: 0.6779  Bal.Acc: 78.9%  F1: 0.7912  |  LR: 5.00e-05  (14.6s)
Epoch [ 19/20]  Train Loss: 0.2162  Bal.Acc: 92.0%  F1: 0.9155  |  Val Loss: 0.6417  Bal.Acc: 78.2%  F1: 0.7833  |  LR: 5.00e-05  (14.6s)
 Best checkpoint saved (val_loss: 0.6417)
Epoch [ 20/20]  Train Loss: 0.2127  Bal.Acc: 92.2%  F1: 0.9162  |  Val Loss: 0.6744  Bal.Acc: 78.1%  F1: 0.7833  |  LR: 5.00e-05  (16.9s)

 Training finished. Checkpoint: checkpoints/mobilenetv3_large_fold5_best.pt
Log CSV: results/logs/mobilenetv3_large_fold5_training_log.csv
Best weights loaded from epoch 19

Model evaluation: mobilenetv3_large_fold5
----------------------------------------
  Balanced Accuracy:       78.23%
  F1 (macro):              0.7833
  Quadratic Cohen's Kappa: 0.8990
  ECE:                     0.0919
  Brier Score (mean):      0.0611

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.90      0.86        87
    Doubtful       0.74      0.68      0.71        81
        Mild       0.65      0.72      0.68        39
    Moderate       0.91      0.79      0.85        38
      Severe       0.81      0.83      0.82        35

    accuracy                           0.79       280
   macro avg       0.79      0.78      0.78       280
weighted avg       0.79      0.79      0.79       280

  Metrics saved to: results/individual_models/mobilenetv3_large_fold5_metrics.json
  Probabilities saved to: results/individual_models/mobilenetv3_large_fold5_test_probs.npz

 FINISHED: mobilenetv3_large. Average kappa out of 5 folds: 0.8775 ±0.0151

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
model.safetensors: 100% 114M/114M [00:01<00:00, 62.9MB/s]
  Parameters: 27,823,973 total, 27,823,973 trainable

============================================================
TRAINING: convnext_tiny_fold1
============================================================
Epoch [  1/20]  Train Loss: 1.7375  Bal.Acc: 23.1%  F1: 0.2166  |  Val Loss: 1.4715  Bal.Acc: 41.8%  F1: 0.3296  |  LR: 1.00e-04  (37.8s)
 Best checkpoint saved (val_loss: 1.4715)
Epoch [  2/20]  Train Loss: 1.3668  Bal.Acc: 41.4%  F1: 0.3574  |  Val Loss: 1.2868  Bal.Acc: 48.3%  F1: 0.4921  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 1.2868)
Epoch [  3/20]  Train Loss: 1.0099  Bal.Acc: 58.7%  F1: 0.5632  |  Val Loss: 1.2213  Bal.Acc: 54.4%  F1: 0.5333  |  LR: 1.00e-04  (30.8s)
 Best checkpoint saved (val_loss: 1.2213)
Epoch [  4/20]  Train Loss: 0.7487  Bal.Acc: 68.6%  F1: 0.6789  |  Val Loss: 0.9239  Bal.Acc: 63.9%  F1: 0.6189  |  LR: 1.00e-04  (32.8s)
 Best checkpoint saved (val_loss: 0.9239)
Epoch [  5/20]  Train Loss: 0.6158  Bal.Acc: 73.7%  F1: 0.7280  |  Val Loss: 0.8868  Bal.Acc: 68.6%  F1: 0.6981  |  LR: 1.00e-04  (30.7s)
 Best checkpoint saved (val_loss: 0.8868)
Epoch [  6/20]  Train Loss: 0.5106  Bal.Acc: 79.8%  F1: 0.7897  |  Val Loss: 1.0831  Bal.Acc: 68.8%  F1: 0.6955  |  LR: 1.00e-04  (32.3s)
Epoch [  7/20]  Train Loss: 0.4623  Bal.Acc: 81.6%  F1: 0.8062  |  Val Loss: 0.8701  Bal.Acc: 73.1%  F1: 0.7222  |  LR: 1.00e-04  (24.5s)
 Best checkpoint saved (val_loss: 0.8701)
Epoch [  8/20]  Train Loss: 0.3816  Bal.Acc: 84.5%  F1: 0.8365  |  Val Loss: 0.8320  Bal.Acc: 77.1%  F1: 0.7714  |  LR: 1.00e-04  (30.9s)
 Best checkpoint saved (val_loss: 0.8320)
Epoch [  9/20]  Train Loss: 0.3283  Bal.Acc: 87.5%  F1: 0.8658  |  Val Loss: 0.7893  Bal.Acc: 73.9%  F1: 0.7402  |  LR: 1.00e-04  (31.6s)
 Best checkpoint saved (val_loss: 0.7893)
Epoch [ 10/20]  Train Loss: 0.3593  Bal.Acc: 86.1%  F1: 0.8490  |  Val Loss: 1.1421  Bal.Acc: 69.0%  F1: 0.6957  |  LR: 1.00e-04  (30.3s)
Epoch [ 11/20]  Train Loss: 0.2543  Bal.Acc: 91.2%  F1: 0.9075  |  Val Loss: 0.9501  Bal.Acc: 72.4%  F1: 0.7221  |  LR: 1.00e-04  (24.3s)
Epoch [ 12/20]  Train Loss: 0.2107  Bal.Acc: 92.6%  F1: 0.9200  |  Val Loss: 1.0163  Bal.Acc: 71.1%  F1: 0.6976  |  LR: 1.00e-04  (24.2s)
Epoch [ 13/20]  Train Loss: 0.1816  Bal.Acc: 92.9%  F1: 0.9233  |  Val Loss: 1.1155  Bal.Acc: 74.5%  F1: 0.7512  |  LR: 5.00e-05  (24.3s)
Epoch [ 14/20]  Train Loss: 0.1546  Bal.Acc: 94.6%  F1: 0.9415  |  Val Loss: 1.1986  Bal.Acc: 73.7%  F1: 0.7478  |  LR: 5.00e-05  (24.3s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.7893

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold1_best.pt
Log CSV: results/logs/convnext_tiny_fold1_training_log.csv
Best weights loaded from epoch 9

Model evaluation: convnext_tiny_fold1
----------------------------------------
  Balanced Accuracy:       73.90%
  F1 (macro):              0.7402
  Quadratic Cohen's Kappa: 0.9232
  ECE:                     0.0793
  Brier Score (mean):      0.0718

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.90      0.85      0.88        88
    Doubtful       0.74      0.72      0.73        81
        Mild       0.49      0.60      0.54        40
    Moderate       0.66      0.78      0.72        37
      Severe       0.96      0.74      0.84        35

    accuracy                           0.75       281
   macro avg       0.75      0.74      0.74       281
weighted avg       0.77      0.75      0.76       281

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
Epoch [  1/20]  Train Loss: 1.7408  Bal.Acc: 19.8%  F1: 0.1862  |  Val Loss: 1.5454  Bal.Acc: 31.2%  F1: 0.1359  |  LR: 1.00e-04  (23.4s)
 Best checkpoint saved (val_loss: 1.5454)
Epoch [  2/20]  Train Loss: 1.5452  Bal.Acc: 27.1%  F1: 0.2389  |  Val Loss: 1.4910  Bal.Acc: 33.5%  F1: 0.2904  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 1.4910)
Epoch [  3/20]  Train Loss: 1.3649  Bal.Acc: 40.1%  F1: 0.3805  |  Val Loss: 1.2555  Bal.Acc: 47.5%  F1: 0.4445  |  LR: 1.00e-04  (31.0s)
 Best checkpoint saved (val_loss: 1.2555)
Epoch [  4/20]  Train Loss: 1.0464  Bal.Acc: 54.1%  F1: 0.5158  |  Val Loss: 0.8858  Bal.Acc: 62.9%  F1: 0.6459  |  LR: 1.00e-04  (30.4s)
 Best checkpoint saved (val_loss: 0.8858)
Epoch [  5/20]  Train Loss: 0.7984  Bal.Acc: 66.2%  F1: 0.6502  |  Val Loss: 0.8461  Bal.Acc: 61.4%  F1: 0.6298  |  LR: 1.00e-04  (31.2s)
 Best checkpoint saved (val_loss: 0.8461)
Epoch [  6/20]  Train Loss: 0.6993  Bal.Acc: 68.1%  F1: 0.6738  |  Val Loss: 0.7829  Bal.Acc: 68.9%  F1: 0.6760  |  LR: 1.00e-04  (30.8s)
 Best checkpoint saved (val_loss: 0.7829)
Epoch [  7/20]  Train Loss: 0.5621  Bal.Acc: 78.3%  F1: 0.7714  |  Val Loss: 0.7727  Bal.Acc: 70.7%  F1: 0.7288  |  LR: 1.00e-04  (32.2s)
 Best checkpoint saved (val_loss: 0.7727)
Epoch [  8/20]  Train Loss: 0.5442  Bal.Acc: 77.0%  F1: 0.7583  |  Val Loss: 0.5664  Bal.Acc: 78.2%  F1: 0.7790  |  LR: 1.00e-04  (31.3s)
 Best checkpoint saved (val_loss: 0.5664)
Epoch [  9/20]  Train Loss: 0.4428  Bal.Acc: 82.7%  F1: 0.8219  |  Val Loss: 0.5891  Bal.Acc: 76.5%  F1: 0.7471  |  LR: 1.00e-04  (30.5s)
Epoch [ 10/20]  Train Loss: 0.4148  Bal.Acc: 83.6%  F1: 0.8300  |  Val Loss: 0.6069  Bal.Acc: 75.8%  F1: 0.7754  |  LR: 1.00e-04  (24.8s)
Epoch [ 11/20]  Train Loss: 0.3239  Bal.Acc: 86.9%  F1: 0.8604  |  Val Loss: 0.6017  Bal.Acc: 82.2%  F1: 0.8181  |  LR: 1.00e-04  (23.4s)
Epoch [ 12/20]  Train Loss: 0.3338  Bal.Acc: 86.3%  F1: 0.8605  |  Val Loss: 0.5377  Bal.Acc: 81.1%  F1: 0.7912  |  LR: 1.00e-04  (24.1s)
 Best checkpoint saved (val_loss: 0.5377)
Epoch [ 13/20]  Train Loss: 0.3378  Bal.Acc: 86.0%  F1: 0.8540  |  Val Loss: 0.7845  Bal.Acc: 76.6%  F1: 0.7403  |  LR: 1.00e-04  (30.7s)
Epoch [ 14/20]  Train Loss: 0.2960  Bal.Acc: 88.8%  F1: 0.8826  |  Val Loss: 0.4969  Bal.Acc: 83.1%  F1: 0.8173  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 0.4969)
Epoch [ 15/20]  Train Loss: 0.2197  Bal.Acc: 91.6%  F1: 0.9108  |  Val Loss: 0.4952  Bal.Acc: 84.1%  F1: 0.8265  |  LR: 1.00e-04  (30.9s)
 Best checkpoint saved (val_loss: 0.4952)
Epoch [ 16/20]  Train Loss: 0.2181  Bal.Acc: 91.6%  F1: 0.9079  |  Val Loss: 0.5302  Bal.Acc: 82.2%  F1: 0.8216  |  LR: 1.00e-04  (30.8s)
Epoch [ 17/20]  Train Loss: 0.1673  Bal.Acc: 93.5%  F1: 0.9342  |  Val Loss: 0.7978  Bal.Acc: 78.3%  F1: 0.7572  |  LR: 1.00e-04  (24.7s)
Epoch [ 18/20]  Train Loss: 0.2194  Bal.Acc: 91.8%  F1: 0.9141  |  Val Loss: 0.5141  Bal.Acc: 82.9%  F1: 0.8057  |  LR: 1.00e-04  (23.1s)
Epoch [ 19/20]  Train Loss: 0.2475  Bal.Acc: 90.3%  F1: 0.8970  |  Val Loss: 0.4830  Bal.Acc: 84.0%  F1: 0.8381  |  LR: 1.00e-04  (24.0s)
 Best checkpoint saved (val_loss: 0.4830)
Epoch [ 20/20]  Train Loss: 0.1651  Bal.Acc: 93.6%  F1: 0.9307  |  Val Loss: 0.5147  Bal.Acc: 85.1%  F1: 0.8477  |  LR: 1.00e-04  (29.9s)

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold2_best.pt
Log CSV: results/logs/convnext_tiny_fold2_training_log.csv
Best weights loaded from epoch 19

Model evaluation: convnext_tiny_fold2
----------------------------------------
  Balanced Accuracy:       84.05%
  F1 (macro):              0.8381
  Quadratic Cohen's Kappa: 0.9285
  ECE:                     0.0468
  Brier Score (mean):      0.0502

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.81      0.98      0.89        88
    Doubtful       0.86      0.69      0.77        81
        Mild       0.82      0.68      0.74        40
    Moderate       0.86      0.97      0.91        37
      Severe       0.89      0.89      0.89        35

    accuracy                           0.84       281
   macro avg       0.85      0.84      0.84       281
weighted avg       0.84      0.84      0.83       281

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
Epoch [  1/20]  Train Loss: 1.6985  Bal.Acc: 22.2%  F1: 0.2156  |  Val Loss: 1.5173  Bal.Acc: 30.4%  F1: 0.1389  |  LR: 1.00e-04  (31.4s)
 Best checkpoint saved (val_loss: 1.5173)
Epoch [  2/20]  Train Loss: 1.4748  Bal.Acc: 33.7%  F1: 0.3036  |  Val Loss: 1.2985  Bal.Acc: 46.4%  F1: 0.4121  |  LR: 1.00e-04  (23.5s)
 Best checkpoint saved (val_loss: 1.2985)
Epoch [  3/20]  Train Loss: 1.0791  Bal.Acc: 57.3%  F1: 0.5494  |  Val Loss: 0.8089  Bal.Acc: 66.1%  F1: 0.6413  |  LR: 1.00e-04  (31.8s)
 Best checkpoint saved (val_loss: 0.8089)
Epoch [  4/20]  Train Loss: 0.8959  Bal.Acc: 63.0%  F1: 0.6155  |  Val Loss: 0.7290  Bal.Acc: 71.9%  F1: 0.7064  |  LR: 1.00e-04  (32.4s)
 Best checkpoint saved (val_loss: 0.7290)
Epoch [  5/20]  Train Loss: 0.6730  Bal.Acc: 72.9%  F1: 0.7173  |  Val Loss: 0.6724  Bal.Acc: 75.7%  F1: 0.7347  |  LR: 1.00e-04  (32.7s)
 Best checkpoint saved (val_loss: 0.6724)
Epoch [  6/20]  Train Loss: 0.5738  Bal.Acc: 75.3%  F1: 0.7458  |  Val Loss: 0.7239  Bal.Acc: 68.1%  F1: 0.6734  |  LR: 1.00e-04  (31.6s)
Epoch [  7/20]  Train Loss: 0.5351  Bal.Acc: 79.0%  F1: 0.7849  |  Val Loss: 0.5434  Bal.Acc: 79.1%  F1: 0.7734  |  LR: 1.00e-04  (24.5s)
 Best checkpoint saved (val_loss: 0.5434)
Epoch [  8/20]  Train Loss: 0.4248  Bal.Acc: 83.5%  F1: 0.8262  |  Val Loss: 0.5913  Bal.Acc: 77.7%  F1: 0.7631  |  LR: 1.00e-04  (30.8s)
Epoch [  9/20]  Train Loss: 0.3566  Bal.Acc: 86.0%  F1: 0.8557  |  Val Loss: 0.6709  Bal.Acc: 74.1%  F1: 0.7504  |  LR: 1.00e-04  (23.9s)
Epoch [ 10/20]  Train Loss: 0.3390  Bal.Acc: 87.6%  F1: 0.8729  |  Val Loss: 0.5272  Bal.Acc: 79.2%  F1: 0.7841  |  LR: 1.00e-04  (24.2s)
 Best checkpoint saved (val_loss: 0.5272)
Epoch [ 11/20]  Train Loss: 0.2562  Bal.Acc: 90.6%  F1: 0.8991  |  Val Loss: 0.9288  Bal.Acc: 69.8%  F1: 0.6925  |  LR: 1.00e-04  (30.7s)
Epoch [ 12/20]  Train Loss: 0.4237  Bal.Acc: 83.9%  F1: 0.8348  |  Val Loss: 0.6410  Bal.Acc: 73.6%  F1: 0.7151  |  LR: 1.00e-04  (24.7s)
Epoch [ 13/20]  Train Loss: 0.2550  Bal.Acc: 90.0%  F1: 0.8944  |  Val Loss: 0.7424  Bal.Acc: 76.6%  F1: 0.7432  |  LR: 1.00e-04  (24.3s)
Epoch [ 14/20]  Train Loss: 0.2928  Bal.Acc: 89.3%  F1: 0.8895  |  Val Loss: 0.6504  Bal.Acc: 78.6%  F1: 0.7879  |  LR: 5.00e-05  (23.5s)
Epoch [ 15/20]  Train Loss: 0.1639  Bal.Acc: 94.2%  F1: 0.9402  |  Val Loss: 0.5887  Bal.Acc: 81.0%  F1: 0.7951  |  LR: 5.00e-05  (24.2s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.5272

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold3_best.pt
Log CSV: results/logs/convnext_tiny_fold3_training_log.csv
Best weights loaded from epoch 10

Model evaluation: convnext_tiny_fold3
----------------------------------------
  Balanced Accuracy:       79.17%
  F1 (macro):              0.7841
  Quadratic Cohen's Kappa: 0.9185
  ECE:                     0.0733
  Brier Score (mean):      0.0615

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.83      0.83        87
    Doubtful       0.69      0.65      0.67        81
        Mild       0.68      0.64      0.66        39
    Moderate       0.83      0.92      0.88        38
      Severe       0.86      0.91      0.89        35

    accuracy                           0.78       280
   macro avg       0.78      0.79      0.78       280
weighted avg       0.77      0.78      0.77       280

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
Epoch [  1/20]  Train Loss: 1.6636  Bal.Acc: 23.1%  F1: 0.2182  |  Val Loss: 1.6195  Bal.Acc: 29.5%  F1: 0.1219  |  LR: 1.00e-04  (23.7s)
 Best checkpoint saved (val_loss: 1.6195)
Epoch [  2/20]  Train Loss: 1.5995  Bal.Acc: 27.6%  F1: 0.2464  |  Val Loss: 1.5116  Bal.Acc: 25.4%  F1: 0.1286  |  LR: 1.00e-04  (24.2s)
 Best checkpoint saved (val_loss: 1.5116)
Epoch [  3/20]  Train Loss: 1.3419  Bal.Acc: 41.1%  F1: 0.3808  |  Val Loss: 1.1293  Bal.Acc: 53.9%  F1: 0.4903  |  LR: 1.00e-04  (30.2s)
 Best checkpoint saved (val_loss: 1.1293)
Epoch [  4/20]  Train Loss: 0.9926  Bal.Acc: 58.8%  F1: 0.5708  |  Val Loss: 0.8786  Bal.Acc: 63.5%  F1: 0.5761  |  LR: 1.00e-04  (30.2s)
 Best checkpoint saved (val_loss: 0.8786)
Epoch [  5/20]  Train Loss: 0.8214  Bal.Acc: 66.9%  F1: 0.6463  |  Val Loss: 0.7252  Bal.Acc: 68.9%  F1: 0.6528  |  LR: 1.00e-04  (32.9s)
 Best checkpoint saved (val_loss: 0.7252)
Epoch [  6/20]  Train Loss: 0.6136  Bal.Acc: 76.2%  F1: 0.7466  |  Val Loss: 0.5396  Bal.Acc: 81.4%  F1: 0.8084  |  LR: 1.00e-04  (32.0s)
 Best checkpoint saved (val_loss: 0.5396)
Epoch [  7/20]  Train Loss: 0.6107  Bal.Acc: 74.7%  F1: 0.7369  |  Val Loss: 0.6305  Bal.Acc: 74.5%  F1: 0.7171  |  LR: 1.00e-04  (32.2s)
Epoch [  8/20]  Train Loss: 0.5107  Bal.Acc: 79.8%  F1: 0.7866  |  Val Loss: 0.7935  Bal.Acc: 67.5%  F1: 0.6801  |  LR: 1.00e-04  (23.7s)
Epoch [  9/20]  Train Loss: 0.4583  Bal.Acc: 81.9%  F1: 0.8123  |  Val Loss: 0.5384  Bal.Acc: 81.2%  F1: 0.8198  |  LR: 1.00e-04  (24.1s)
 Best checkpoint saved (val_loss: 0.5384)
Epoch [ 10/20]  Train Loss: 0.4216  Bal.Acc: 83.6%  F1: 0.8271  |  Val Loss: 0.5871  Bal.Acc: 77.5%  F1: 0.7869  |  LR: 1.00e-04  (30.2s)
Epoch [ 11/20]  Train Loss: 0.3934  Bal.Acc: 84.4%  F1: 0.8405  |  Val Loss: 0.6062  Bal.Acc: 79.0%  F1: 0.7753  |  LR: 1.00e-04  (25.1s)
Epoch [ 12/20]  Train Loss: 0.4224  Bal.Acc: 83.6%  F1: 0.8319  |  Val Loss: 0.5438  Bal.Acc: 76.9%  F1: 0.7637  |  LR: 1.00e-04  (23.5s)
Epoch [ 13/20]  Train Loss: 0.3458  Bal.Acc: 87.1%  F1: 0.8632  |  Val Loss: 0.7569  Bal.Acc: 71.8%  F1: 0.7031  |  LR: 5.00e-05  (24.2s)
Epoch [ 14/20]  Train Loss: 0.2904  Bal.Acc: 88.9%  F1: 0.8857  |  Val Loss: 0.5524  Bal.Acc: 78.8%  F1: 0.7953  |  LR: 5.00e-05  (24.2s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.5384

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold4_best.pt
Log CSV: results/logs/convnext_tiny_fold4_training_log.csv
Best weights loaded from epoch 9

Model evaluation: convnext_tiny_fold4
----------------------------------------
  Balanced Accuracy:       81.24%
  F1 (macro):              0.8198
  Quadratic Cohen's Kappa: 0.9354
  ECE:                     0.0645
  Brier Score (mean):      0.0573

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.81      0.94      0.87        87
    Doubtful       0.79      0.70      0.75        81
        Mild       0.69      0.69      0.69        39
    Moderate       0.87      0.89      0.88        38
      Severe       1.00      0.83      0.91        35

    accuracy                           0.82       280
   macro avg       0.83      0.81      0.82       280
weighted avg       0.82      0.82      0.82       280

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
Epoch [  1/20]  Train Loss: 1.5530  Bal.Acc: 32.1%  F1: 0.2988  |  Val Loss: 1.3271  Bal.Acc: 40.1%  F1: 0.3370  |  LR: 1.00e-04  (23.5s)
 Best checkpoint saved (val_loss: 1.3271)
Epoch [  2/20]  Train Loss: 1.0617  Bal.Acc: 54.8%  F1: 0.5302  |  Val Loss: 0.7683  Bal.Acc: 69.5%  F1: 0.6651  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 0.7683)
Epoch [  3/20]  Train Loss: 0.8244  Bal.Acc: 66.1%  F1: 0.6473  |  Val Loss: 0.6286  Bal.Acc: 73.1%  F1: 0.7094  |  LR: 1.00e-04  (31.3s)
 Best checkpoint saved (val_loss: 0.6286)
Epoch [  4/20]  Train Loss: 0.6627  Bal.Acc: 73.6%  F1: 0.7311  |  Val Loss: 0.6889  Bal.Acc: 76.5%  F1: 0.7082  |  LR: 1.00e-04  (30.1s)
Epoch [  5/20]  Train Loss: 0.6181  Bal.Acc: 73.9%  F1: 0.7265  |  Val Loss: 0.5950  Bal.Acc: 78.0%  F1: 0.7349  |  LR: 1.00e-04  (24.5s)
 Best checkpoint saved (val_loss: 0.5950)
Epoch [  6/20]  Train Loss: 0.4965  Bal.Acc: 80.9%  F1: 0.8015  |  Val Loss: 0.4862  Bal.Acc: 80.2%  F1: 0.7686  |  LR: 1.00e-04  (29.9s)
 Best checkpoint saved (val_loss: 0.4862)
Epoch [  7/20]  Train Loss: 0.4481  Bal.Acc: 82.2%  F1: 0.8106  |  Val Loss: 0.5982  Bal.Acc: 76.4%  F1: 0.7348  |  LR: 1.00e-04  (33.0s)
Epoch [  8/20]  Train Loss: 0.4227  Bal.Acc: 82.7%  F1: 0.8218  |  Val Loss: 0.5486  Bal.Acc: 77.3%  F1: 0.7379  |  LR: 1.00e-04  (24.5s)
Epoch [  9/20]  Train Loss: 0.3843  Bal.Acc: 85.9%  F1: 0.8473  |  Val Loss: 0.4281  Bal.Acc: 86.5%  F1: 0.8494  |  LR: 1.00e-04  (24.2s)
 Best checkpoint saved (val_loss: 0.4281)
Epoch [ 10/20]  Train Loss: 0.3018  Bal.Acc: 89.0%  F1: 0.8856  |  Val Loss: 0.7073  Bal.Acc: 74.1%  F1: 0.7393  |  LR: 1.00e-04  (29.9s)
Epoch [ 11/20]  Train Loss: 0.2748  Bal.Acc: 89.0%  F1: 0.8851  |  Val Loss: 0.5256  Bal.Acc: 83.4%  F1: 0.8013  |  LR: 1.00e-04  (25.2s)
Epoch [ 12/20]  Train Loss: 0.2614  Bal.Acc: 90.4%  F1: 0.8986  |  Val Loss: 0.4169  Bal.Acc: 84.1%  F1: 0.8274  |  LR: 1.00e-04  (23.6s)
 Best checkpoint saved (val_loss: 0.4169)
Epoch [ 13/20]  Train Loss: 0.2981  Bal.Acc: 89.2%  F1: 0.8886  |  Val Loss: 0.6173  Bal.Acc: 79.6%  F1: 0.8023  |  LR: 1.00e-04  (32.6s)
Epoch [ 14/20]  Train Loss: 0.2029  Bal.Acc: 92.4%  F1: 0.9184  |  Val Loss: 0.5402  Bal.Acc: 77.6%  F1: 0.7836  |  LR: 1.00e-04  (24.1s)
Epoch [ 15/20]  Train Loss: 0.1757  Bal.Acc: 93.3%  F1: 0.9281  |  Val Loss: 0.5105  Bal.Acc: 83.3%  F1: 0.8217  |  LR: 1.00e-04  (24.0s)
Epoch [ 16/20]  Train Loss: 0.1634  Bal.Acc: 93.9%  F1: 0.9367  |  Val Loss: 0.6779  Bal.Acc: 78.9%  F1: 0.7686  |  LR: 5.00e-05  (23.1s)
Epoch [ 17/20]  Train Loss: 0.1267  Bal.Acc: 95.7%  F1: 0.9567  |  Val Loss: 0.4644  Bal.Acc: 85.8%  F1: 0.8438  |  LR: 5.00e-05  (24.1s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.4169

 Training finished. Checkpoint: checkpoints/convnext_tiny_fold5_best.pt
Log CSV: results/logs/convnext_tiny_fold5_training_log.csv
Best weights loaded from epoch 12

Model evaluation: convnext_tiny_fold5
----------------------------------------
  Balanced Accuracy:       84.12%
  F1 (macro):              0.8274
  Quadratic Cohen's Kappa: 0.9316
  ECE:                     0.0574
  Brier Score (mean):      0.0520

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.94      0.88        87
    Doubtful       0.84      0.60      0.71        81
        Mild       0.65      0.79      0.71        39
    Moderate       0.92      0.92      0.92        38
      Severe       0.89      0.94      0.92        35

    accuracy                           0.82       280
   macro avg       0.83      0.84      0.83       280
weighted avg       0.83      0.82      0.82       280

  Metrics saved to: results/individual_models/convnext_tiny_fold5_metrics.json
  Probabilities saved to: results/individual_models/convnext_tiny_fold5_test_probs.npz

 FINISHED: convnext_tiny. Average kappa out of 5 folds: 0.9274 ±0.0060

========================================================================================================================
Single Fold Summary:
========================================================================================================================
Model                        Kappa   F1-Mac     ECE   Brier |     KL0     KL1     KL2     KL3     KL4
------------------------------------------------------------------------------------------------------------------------
densenet121_fold5           0.9373   0.8555  0.0378  0.0473 |  0.9257  0.7517  0.6966  0.9737  0.9296
convnext_tiny_fold4         0.9354   0.8198  0.0645  0.0573 |  0.8723  0.7451  0.6923  0.8831  0.9062
convnext_tiny_fold5         0.9316   0.8274  0.0574  0.0520 |  0.8817  0.7050  0.7126  0.9211  0.9167
convnext_tiny_fold2         0.9285   0.8381  0.0468  0.0502 |  0.8866  0.7671  0.7397  0.9114  0.8857
convnext_tiny_fold1         0.9232   0.7402  0.0793  0.0718 |  0.8772  0.7296  0.5393  0.7160  0.8387
convnext_tiny_fold3         0.9185   0.7841  0.0733  0.0615 |  0.8276  0.6709  0.6579  0.8750  0.8889
densenet121_fold3           0.9144   0.8080  0.0463  0.0590 |  0.8427  0.6709  0.6667  0.9610  0.8986
densenet121_fold4           0.9132   0.8406  0.0389  0.0425 |  0.8827  0.7975  0.7246  0.9091  0.8889
resnet50_fold1              0.9082   0.7175  0.0626  0.0748 |  0.8454  0.6897  0.5455  0.6849  0.8219
densenet121_fold2           0.9059   0.7933  0.0202  0.0584 |  0.8750  0.7034  0.6957  0.8395  0.8529
densenet121_fold1           0.9011   0.7470  0.0237  0.0672 |  0.8710  0.6806  0.5116  0.8108  0.8611
mobilenetv3_large_fold5     0.8990   0.7833  0.0919  0.0611 |  0.8619  0.7097  0.6829  0.8451  0.8169
efficientnet_b3_fold2       0.8946   0.8101  0.1051  0.0625 |  0.8621  0.7176  0.7222  0.9000  0.8485
efficientnet_b3_fold3       0.8917   0.7585  0.1136  0.0726 |  0.8276  0.6242  0.6207  0.8378  0.8824
resnet50_fold2              0.8890   0.7404  0.0898  0.0773 |  0.8171  0.6667  0.5349  0.8101  0.8732
resnet50_fold5              0.8877   0.7749  0.0687  0.0701 |  0.8706  0.6928  0.5926  0.8684  0.8500
mobilenetv3_large_fold3     0.8836   0.7642  0.1189  0.0686 |  0.8380  0.7051  0.6076  0.8533  0.8169
mobilenetv3_large_fold2     0.8833   0.7926  0.0984  0.0636 |  0.8681  0.7105  0.6585  0.9041  0.8219
resnet50_fold4              0.8820   0.7488  0.0745  0.0769 |  0.8121  0.6710  0.5347  0.8649  0.8615
resnet50_fold3              0.8752   0.7005  0.0814  0.0808 |  0.7709  0.5103  0.5227  0.8750  0.8235
mobilenetv3_large_fold1     0.8658   0.7380  0.1073  0.0740 |  0.8046  0.7081  0.5195  0.7848  0.8732
mobilenetv3_large_fold4     0.8559   0.7788  0.0941  0.0601 |  0.8362  0.7251  0.6753  0.8378  0.8197
efficientnet_b3_fold1       0.8431   0.7128  0.1197  0.0856 |  0.7937  0.6331  0.5714  0.7945  0.7714
efficientnet_b3_fold5       0.8297   0.6836  0.1003  0.0871 |  0.7956  0.6184  0.4250  0.8000  0.7792
efficientnet_b3_fold4       0.7613   0.6197  0.1972  0.1034 |  0.7135  0.5350  0.3951  0.7792  0.6757
========================================================================================================================
Sorted by Cohen's Kappa

===================================================================================================================
 CROSS-VALIDATION SUMMARY — AVERAGE OUT OF 5 FOLDs
===================================================================================================================
Model                         Kappa         F1-Mac |      KL0      KL1      KL2      KL3      KL4
-------------------------------------------------------------------------------------------------------------------
resnet50              0.8884 ±0.0110  0.7364 ±0.0257 |   0.8232   0.6461   0.5461   0.8207   0.8460
efficientnet_b3       0.8441 ±0.0487  0.7169 ±0.0648 |   0.7985   0.6257   0.5469   0.8223   0.7914
densenet121           0.9144 ±0.0124  0.8089 ±0.0381 |   0.8794   0.7208   0.6590   0.8988   0.8862
mobilenetv3_large     0.8775 ±0.0151  0.7714 ±0.0190 |   0.8418   0.7117   0.6288   0.8450   0.8297
convnext_tiny         0.9274 ±0.0060  0.8019 ±0.0358 |   0.8691   0.7235   0.6684   0.8613   0.8872
===================================================================================================================
Results saved to: results
