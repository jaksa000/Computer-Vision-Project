Urządzenie: cuda
Modele do uruchomienia: ['resnet50', 'efficientnet_b3', 'densenet121', 'mobilenetv3_large', 'convnext_tiny']
Liczba foldów: 5
Ładowanie danych z: data/MedicalExpert-I
  Klasa 0 (0Normal): 514 obrazów
  Klasa 1 (1Doubtful): 477 obrazów
  Klasa 2 (2Mild): 232 obrazów
  Klasa 3 (3Moderate): 221 obrazów
  Klasa 4 (4Severe): 206 obrazów

  Łącznie: 1650 obrazów
  Łącznie: 1650 obrazów, 5 foldów CV

============================================================
PODZIAŁ NA ZBIÓR CV ORAZ HOLD-OUT (SEJF)
============================================================
  Dane do K-Fold CV (85%): 1402 obrazów
  Dane Testowe / Sejf (15%): 248 obrazów

================================================================================
🚀 ROZPOCZĘCIE TRENINGU MODELU: resnet50
================================================================================

--- resnet50 | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 1):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold1
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.5998  Bal.Acc: 26.8%  F1: 0.2279  |  Val Loss: 1.5929  Bal.Acc: 29.5%  F1: 0.2531  |  LR: 1.00e-04  (24.9s)
 Best checkpoint saved (val_loss: 1.5929)
Epoch [  2/20]  Train Loss: 1.5733  Bal.Acc: 38.0%  F1: 0.3450  |  Val Loss: 1.5594  Bal.Acc: 34.8%  F1: 0.2925  |  LR: 1.00e-04  (29.7s)
 Best checkpoint saved (val_loss: 1.5594)
Epoch [  3/20]  Train Loss: 1.5363  Bal.Acc: 46.6%  F1: 0.4323  |  Val Loss: 1.5066  Bal.Acc: 47.9%  F1: 0.4516  |  LR: 1.00e-04  (29.7s)
 Best checkpoint saved (val_loss: 1.5066)
Epoch [  4/20]  Train Loss: 1.4746  Bal.Acc: 48.5%  F1: 0.4198  |  Val Loss: 1.4255  Bal.Acc: 44.5%  F1: 0.3978  |  LR: 1.00e-04  (30.1s)
 Best checkpoint saved (val_loss: 1.4255)
Epoch [  5/20]  Train Loss: 1.3606  Bal.Acc: 54.6%  F1: 0.5077  |  Val Loss: 1.3016  Bal.Acc: 48.7%  F1: 0.4582  |  LR: 1.00e-04  (31.1s)
 Best checkpoint saved (val_loss: 1.3016)
Epoch [  6/20]  Train Loss: 1.1966  Bal.Acc: 57.0%  F1: 0.5070  |  Val Loss: 1.1938  Bal.Acc: 54.0%  F1: 0.5355  |  LR: 1.00e-04  (31.6s)
 Best checkpoint saved (val_loss: 1.1938)
Epoch [  7/20]  Train Loss: 1.0158  Bal.Acc: 63.3%  F1: 0.5956  |  Val Loss: 0.9181  Bal.Acc: 66.3%  F1: 0.6220  |  LR: 1.00e-04  (31.1s)
 Best checkpoint saved (val_loss: 0.9181)
Epoch [  8/20]  Train Loss: 0.9024  Bal.Acc: 65.8%  F1: 0.6321  |  Val Loss: 0.9420  Bal.Acc: 65.0%  F1: 0.6683  |  LR: 1.00e-04  (30.9s)
Epoch [  9/20]  Train Loss: 0.8192  Bal.Acc: 70.1%  F1: 0.6885  |  Val Loss: 0.8383  Bal.Acc: 65.9%  F1: 0.6466  |  LR: 1.00e-04  (23.8s)
 Best checkpoint saved (val_loss: 0.8383)
Epoch [ 10/20]  Train Loss: 0.7677  Bal.Acc: 70.6%  F1: 0.6991  |  Val Loss: 0.7806  Bal.Acc: 72.5%  F1: 0.7135  |  LR: 1.00e-04  (32.2s)
 Best checkpoint saved (val_loss: 0.7806)
Epoch [ 11/20]  Train Loss: 0.7067  Bal.Acc: 73.5%  F1: 0.7325  |  Val Loss: 0.7155  Bal.Acc: 72.7%  F1: 0.7093  |  LR: 1.00e-04  (31.1s)
 Best checkpoint saved (val_loss: 0.7155)
Epoch [ 12/20]  Train Loss: 0.6746  Bal.Acc: 74.6%  F1: 0.7365  |  Val Loss: 0.6588  Bal.Acc: 74.3%  F1: 0.7403  |  LR: 1.00e-04  (31.0s)
 Best checkpoint saved (val_loss: 0.6588)
Epoch [ 13/20]  Train Loss: 0.6413  Bal.Acc: 76.4%  F1: 0.7580  |  Val Loss: 0.7723  Bal.Acc: 70.1%  F1: 0.6993  |  LR: 1.00e-04  (31.4s)
Epoch [ 14/20]  Train Loss: 0.5846  Bal.Acc: 78.2%  F1: 0.7717  |  Val Loss: 0.7485  Bal.Acc: 71.0%  F1: 0.7234  |  LR: 1.00e-04  (23.6s)
Epoch [ 15/20]  Train Loss: 0.5391  Bal.Acc: 79.0%  F1: 0.7834  |  Val Loss: 0.6340  Bal.Acc: 72.1%  F1: 0.7222  |  LR: 1.00e-04  (24.1s)
 Best checkpoint saved (val_loss: 0.6340)
Epoch [ 16/20]  Train Loss: 0.5215  Bal.Acc: 81.0%  F1: 0.8017  |  Val Loss: 0.8452  Bal.Acc: 65.7%  F1: 0.6667  |  LR: 1.00e-04  (30.2s)
Epoch [ 17/20]  Train Loss: 0.4871  Bal.Acc: 82.8%  F1: 0.8209  |  Val Loss: 0.7250  Bal.Acc: 70.8%  F1: 0.6952  |  LR: 1.00e-04  (24.3s)
Epoch [ 18/20]  Train Loss: 0.4945  Bal.Acc: 81.6%  F1: 0.8132  |  Val Loss: 0.6251  Bal.Acc: 73.1%  F1: 0.7104  |  LR: 1.00e-04  (24.2s)
 Best checkpoint saved (val_loss: 0.6251)
Epoch [ 19/20]  Train Loss: 0.4671  Bal.Acc: 82.4%  F1: 0.8212  |  Val Loss: 0.5845  Bal.Acc: 75.4%  F1: 0.7446  |  LR: 1.00e-04  (31.4s)
 Best checkpoint saved (val_loss: 0.5845)
Epoch [ 20/20]  Train Loss: 0.4662  Bal.Acc: 82.6%  F1: 0.8194  |  Val Loss: 0.7151  Bal.Acc: 72.8%  F1: 0.7260  |  LR: 1.00e-04  (30.7s)

 Trening finished. Checkpoint: checkpoints/resnet50_fold1_best.pt
Log CSV: results/resnet50_fold1_training_log.csv
Best weights loaded from  19

Ewaluacja modelu: resnet50_fold1
----------------------------------------
  Balanced Accuracy:       75.44%
  F1 (macro):              0.7446
  Quadratic Cohen's Kappa: 0.9008
  ECE:                     0.0174
  Brier Score (mean):      0.0678

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.84      0.84        88
    Doubtful       0.70      0.64      0.67        81
        Mild       0.62      0.59      0.61        39
    Moderate       0.76      0.84      0.80        38
      Severe       0.77      0.86      0.81        35

    accuracy                           0.75       281
   macro avg       0.74      0.75      0.74       281
weighted avg       0.75      0.75      0.75       281

  Metryki zapisane: results/resnet50_fold1_metrics.json
  Prawdopodobieństwa zapisane: results/resnet50_fold1_test_probs.npz

--- resnet50 | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 2):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold2
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.6057  Bal.Acc: 21.7%  F1: 0.1342  |  Val Loss: 1.6000  Bal.Acc: 22.2%  F1: 0.2179  |  LR: 1.00e-04  (23.9s)
 Best checkpoint saved (val_loss: 1.6000)
Epoch [  2/20]  Train Loss: 1.5754  Bal.Acc: 33.1%  F1: 0.3352  |  Val Loss: 1.5744  Bal.Acc: 27.3%  F1: 0.2516  |  LR: 1.00e-04  (25.4s)
 Best checkpoint saved (val_loss: 1.5744)
Epoch [  3/20]  Train Loss: 1.5426  Bal.Acc: 36.7%  F1: 0.3664  |  Val Loss: 1.5394  Bal.Acc: 29.5%  F1: 0.2601  |  LR: 1.00e-04  (30.4s)
 Best checkpoint saved (val_loss: 1.5394)
Epoch [  4/20]  Train Loss: 1.4867  Bal.Acc: 45.1%  F1: 0.4014  |  Val Loss: 1.4346  Bal.Acc: 49.1%  F1: 0.4738  |  LR: 1.00e-04  (31.4s)
 Best checkpoint saved (val_loss: 1.4346)
Epoch [  5/20]  Train Loss: 1.3753  Bal.Acc: 55.6%  F1: 0.5083  |  Val Loss: 1.3570  Bal.Acc: 45.6%  F1: 0.4326  |  LR: 1.00e-04  (30.8s)
 Best checkpoint saved (val_loss: 1.3570)
Epoch [  6/20]  Train Loss: 1.2209  Bal.Acc: 57.0%  F1: 0.4997  |  Val Loss: 1.1470  Bal.Acc: 59.8%  F1: 0.5575  |  LR: 1.00e-04  (30.9s)
 Best checkpoint saved (val_loss: 1.1470)
Epoch [  7/20]  Train Loss: 1.0528  Bal.Acc: 59.1%  F1: 0.5413  |  Val Loss: 1.0299  Bal.Acc: 59.7%  F1: 0.5833  |  LR: 1.00e-04  (30.3s)
 Best checkpoint saved (val_loss: 1.0299)
Epoch [  8/20]  Train Loss: 0.9247  Bal.Acc: 65.8%  F1: 0.6227  |  Val Loss: 0.8992  Bal.Acc: 68.0%  F1: 0.6804  |  LR: 1.00e-04  (30.3s)
 Best checkpoint saved (val_loss: 0.8992)
Epoch [  9/20]  Train Loss: 0.8143  Bal.Acc: 69.9%  F1: 0.6859  |  Val Loss: 0.8401  Bal.Acc: 63.7%  F1: 0.6322  |  LR: 1.00e-04  (30.5s)
 Best checkpoint saved (val_loss: 0.8401)
Epoch [ 10/20]  Train Loss: 0.7593  Bal.Acc: 70.8%  F1: 0.6912  |  Val Loss: 0.8131  Bal.Acc: 68.6%  F1: 0.6876  |  LR: 1.00e-04  (30.8s)
 Best checkpoint saved (val_loss: 0.8131)
Epoch [ 11/20]  Train Loss: 0.6940  Bal.Acc: 72.9%  F1: 0.7175  |  Val Loss: 0.7096  Bal.Acc: 75.4%  F1: 0.7556  |  LR: 1.00e-04  (30.3s)
 Best checkpoint saved (val_loss: 0.7096)
Epoch [ 12/20]  Train Loss: 0.6663  Bal.Acc: 75.3%  F1: 0.7475  |  Val Loss: 0.6868  Bal.Acc: 75.8%  F1: 0.7604  |  LR: 1.00e-04  (31.0s)
 Best checkpoint saved (val_loss: 0.6868)
Epoch [ 13/20]  Train Loss: 0.6194  Bal.Acc: 77.1%  F1: 0.7657  |  Val Loss: 0.7057  Bal.Acc: 71.5%  F1: 0.7086  |  LR: 1.00e-04  (31.1s)
Epoch [ 14/20]  Train Loss: 0.5940  Bal.Acc: 77.5%  F1: 0.7605  |  Val Loss: 0.6396  Bal.Acc: 74.3%  F1: 0.7493  |  LR: 1.00e-04  (23.8s)
 Best checkpoint saved (val_loss: 0.6396)
Epoch [ 15/20]  Train Loss: 0.5733  Bal.Acc: 78.1%  F1: 0.7794  |  Val Loss: 0.6649  Bal.Acc: 74.1%  F1: 0.7325  |  LR: 1.00e-04  (30.9s)
Epoch [ 16/20]  Train Loss: 0.5202  Bal.Acc: 79.6%  F1: 0.7857  |  Val Loss: 0.9016  Bal.Acc: 61.5%  F1: 0.6170  |  LR: 1.00e-04  (23.7s)
Epoch [ 17/20]  Train Loss: 0.5114  Bal.Acc: 80.3%  F1: 0.8009  |  Val Loss: 0.7234  Bal.Acc: 73.0%  F1: 0.7208  |  LR: 1.00e-04  (24.1s)
Epoch [ 18/20]  Train Loss: 0.4687  Bal.Acc: 81.6%  F1: 0.8127  |  Val Loss: 0.5774  Bal.Acc: 77.0%  F1: 0.7582  |  LR: 1.00e-04  (24.2s)
 Best checkpoint saved (val_loss: 0.5774)
Epoch [ 19/20]  Train Loss: 0.4677  Bal.Acc: 82.3%  F1: 0.8179  |  Val Loss: 0.6255  Bal.Acc: 74.7%  F1: 0.7536  |  LR: 1.00e-04  (30.8s)
Epoch [ 20/20]  Train Loss: 0.4651  Bal.Acc: 82.8%  F1: 0.8242  |  Val Loss: 0.6228  Bal.Acc: 74.3%  F1: 0.7538  |  LR: 1.00e-04  (23.9s)

 Trening finished. Checkpoint: checkpoints/resnet50_fold2_best.pt
Log CSV: results/resnet50_fold2_training_log.csv
Best weights loaded from  18

Ewaluacja modelu: resnet50_fold2
----------------------------------------
  Balanced Accuracy:       77.01%
  F1 (macro):              0.7582
  Quadratic Cohen's Kappa: 0.9142
  ECE:                     0.0279
  Brier Score (mean):      0.0659

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.82      0.89      0.85        88
    Doubtful       0.79      0.60      0.69        81
        Mild       0.57      0.67      0.61        39
    Moderate       0.74      0.92      0.82        38
      Severe       0.87      0.77      0.82        35

    accuracy                           0.77       281
   macro avg       0.76      0.77      0.76       281
weighted avg       0.77      0.77      0.76       281

  Metryki zapisane: results/resnet50_fold2_metrics.json
  Prawdopodobieństwa zapisane: results/resnet50_fold2_test_probs.npz

--- resnet50 | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 3):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold3
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.6054  Bal.Acc: 24.6%  F1: 0.2269  |  Val Loss: 1.6008  Bal.Acc: 23.4%  F1: 0.1959  |  LR: 1.00e-04  (24.1s)
 Best checkpoint saved (val_loss: 1.6008)
Epoch [  2/20]  Train Loss: 1.5830  Bal.Acc: 30.0%  F1: 0.2551  |  Val Loss: 1.5765  Bal.Acc: 28.7%  F1: 0.2441  |  LR: 1.00e-04  (25.0s)
 Best checkpoint saved (val_loss: 1.5765)
Epoch [  3/20]  Train Loss: 1.5572  Bal.Acc: 36.8%  F1: 0.3442  |  Val Loss: 1.5269  Bal.Acc: 43.0%  F1: 0.4183  |  LR: 1.00e-04  (30.6s)
 Best checkpoint saved (val_loss: 1.5269)
Epoch [  4/20]  Train Loss: 1.5067  Bal.Acc: 47.5%  F1: 0.4669  |  Val Loss: 1.4683  Bal.Acc: 46.0%  F1: 0.4685  |  LR: 1.00e-04  (31.0s)
 Best checkpoint saved (val_loss: 1.4683)
Epoch [  5/20]  Train Loss: 1.4201  Bal.Acc: 52.1%  F1: 0.5023  |  Val Loss: 1.3500  Bal.Acc: 44.7%  F1: 0.4218  |  LR: 1.00e-04  (30.2s)
 Best checkpoint saved (val_loss: 1.3500)
Epoch [  6/20]  Train Loss: 1.2651  Bal.Acc: 57.5%  F1: 0.5226  |  Val Loss: 1.1713  Bal.Acc: 58.6%  F1: 0.5658  |  LR: 1.00e-04  (31.7s)
 Best checkpoint saved (val_loss: 1.1713)
Epoch [  7/20]  Train Loss: 1.0942  Bal.Acc: 62.1%  F1: 0.5760  |  Val Loss: 1.0872  Bal.Acc: 57.1%  F1: 0.5605  |  LR: 1.00e-04  (32.1s)
 Best checkpoint saved (val_loss: 1.0872)
Epoch [  8/20]  Train Loss: 0.9423  Bal.Acc: 65.7%  F1: 0.6314  |  Val Loss: 1.0939  Bal.Acc: 52.9%  F1: 0.5204  |  LR: 1.00e-04  (30.2s)
Epoch [  9/20]  Train Loss: 0.8384  Bal.Acc: 70.1%  F1: 0.6807  |  Val Loss: 0.8739  Bal.Acc: 70.8%  F1: 0.7179  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 0.8739)
Epoch [ 10/20]  Train Loss: 0.7700  Bal.Acc: 70.7%  F1: 0.6960  |  Val Loss: 0.8199  Bal.Acc: 67.6%  F1: 0.6731  |  LR: 1.00e-04  (31.8s)
 Best checkpoint saved (val_loss: 0.8199)
Epoch [ 11/20]  Train Loss: 0.7126  Bal.Acc: 73.2%  F1: 0.7273  |  Val Loss: 0.7606  Bal.Acc: 72.2%  F1: 0.7158  |  LR: 1.00e-04  (30.3s)
 Best checkpoint saved (val_loss: 0.7606)
Epoch [ 12/20]  Train Loss: 0.6309  Bal.Acc: 76.8%  F1: 0.7602  |  Val Loss: 0.7518  Bal.Acc: 70.6%  F1: 0.7094  |  LR: 1.00e-04  (30.3s)
 Best checkpoint saved (val_loss: 0.7518)
Epoch [ 13/20]  Train Loss: 0.5914  Bal.Acc: 76.5%  F1: 0.7600  |  Val Loss: 0.7128  Bal.Acc: 74.0%  F1: 0.7411  |  LR: 1.00e-04  (29.9s)
 Best checkpoint saved (val_loss: 0.7128)
Epoch [ 14/20]  Train Loss: 0.5553  Bal.Acc: 78.9%  F1: 0.7795  |  Val Loss: 0.7431  Bal.Acc: 71.7%  F1: 0.7059  |  LR: 1.00e-04  (30.5s)
Epoch [ 15/20]  Train Loss: 0.5530  Bal.Acc: 79.3%  F1: 0.7832  |  Val Loss: 0.7466  Bal.Acc: 71.8%  F1: 0.7188  |  LR: 1.00e-04  (23.7s)
Epoch [ 16/20]  Train Loss: 0.5445  Bal.Acc: 78.3%  F1: 0.7733  |  Val Loss: 0.7575  Bal.Acc: 69.5%  F1: 0.6901  |  LR: 1.00e-04  (24.0s)
Epoch [ 17/20]  Train Loss: 0.5317  Bal.Acc: 79.1%  F1: 0.7808  |  Val Loss: 0.6807  Bal.Acc: 74.6%  F1: 0.7484  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 0.6807)
Epoch [ 18/20]  Train Loss: 0.4874  Bal.Acc: 81.8%  F1: 0.8109  |  Val Loss: 0.6844  Bal.Acc: 74.0%  F1: 0.7410  |  LR: 1.00e-04  (31.3s)
Epoch [ 19/20]  Train Loss: 0.4280  Bal.Acc: 84.3%  F1: 0.8404  |  Val Loss: 0.6978  Bal.Acc: 75.2%  F1: 0.7424  |  LR: 1.00e-04  (24.0s)
Epoch [ 20/20]  Train Loss: 0.4043  Bal.Acc: 84.9%  F1: 0.8425  |  Val Loss: 0.6213  Bal.Acc: 77.9%  F1: 0.7736  |  LR: 1.00e-04  (23.9s)
 Best checkpoint saved (val_loss: 0.6213)

 Trening finished. Checkpoint: checkpoints/resnet50_fold3_best.pt
Log CSV: results/resnet50_fold3_training_log.csv
Best weights loaded from  20

Ewaluacja modelu: resnet50_fold3
----------------------------------------
  Balanced Accuracy:       77.91%
  F1 (macro):              0.7736
  Quadratic Cohen's Kappa: 0.9234
  ECE:                     0.0691
  Brier Score (mean):      0.0667

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.78      0.91      0.84        87
    Doubtful       0.79      0.62      0.69        81
        Mild       0.64      0.68      0.66        40
    Moderate       0.78      0.84      0.81        37
      Severe       0.88      0.86      0.87        35

    accuracy                           0.78       280
   macro avg       0.78      0.78      0.77       280
weighted avg       0.78      0.78      0.77       280

  Metryki zapisane: results/resnet50_fold3_metrics.json
  Prawdopodobieństwa zapisane: results/resnet50_fold3_test_probs.npz

--- resnet50 | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 4):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold4
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.6020  Bal.Acc: 22.8%  F1: 0.1891  |  Val Loss: 1.5937  Bal.Acc: 20.6%  F1: 0.1203  |  LR: 1.00e-04  (30.0s)
 Best checkpoint saved (val_loss: 1.5937)
Epoch [  2/20]  Train Loss: 1.5841  Bal.Acc: 29.8%  F1: 0.2815  |  Val Loss: 1.5723  Bal.Acc: 28.2%  F1: 0.2511  |  LR: 1.00e-04  (24.8s)
 Best checkpoint saved (val_loss: 1.5723)
Epoch [  3/20]  Train Loss: 1.5566  Bal.Acc: 37.5%  F1: 0.3867  |  Val Loss: 1.5358  Bal.Acc: 34.7%  F1: 0.3442  |  LR: 1.00e-04  (31.7s)
 Best checkpoint saved (val_loss: 1.5358)
Epoch [  4/20]  Train Loss: 1.5212  Bal.Acc: 40.6%  F1: 0.4133  |  Val Loss: 1.4914  Bal.Acc: 42.4%  F1: 0.4317  |  LR: 1.00e-04  (30.5s)
 Best checkpoint saved (val_loss: 1.4914)
Epoch [  5/20]  Train Loss: 1.4675  Bal.Acc: 46.9%  F1: 0.4696  |  Val Loss: 1.4394  Bal.Acc: 40.5%  F1: 0.3862  |  LR: 1.00e-04  (31.4s)
 Best checkpoint saved (val_loss: 1.4394)
Epoch [  6/20]  Train Loss: 1.3661  Bal.Acc: 55.2%  F1: 0.5049  |  Val Loss: 1.3426  Bal.Acc: 41.8%  F1: 0.3890  |  LR: 1.00e-04  (30.9s)
 Best checkpoint saved (val_loss: 1.3426)
Epoch [  7/20]  Train Loss: 1.2204  Bal.Acc: 56.7%  F1: 0.5136  |  Val Loss: 1.1748  Bal.Acc: 52.1%  F1: 0.4904  |  LR: 1.00e-04  (31.9s)
 Best checkpoint saved (val_loss: 1.1748)
Epoch [  8/20]  Train Loss: 1.0690  Bal.Acc: 61.2%  F1: 0.5558  |  Val Loss: 1.0208  Bal.Acc: 57.9%  F1: 0.5279  |  LR: 1.00e-04  (30.1s)
 Best checkpoint saved (val_loss: 1.0208)
Epoch [  9/20]  Train Loss: 0.9514  Bal.Acc: 62.9%  F1: 0.5870  |  Val Loss: 0.9292  Bal.Acc: 62.6%  F1: 0.5977  |  LR: 1.00e-04  (30.8s)
 Best checkpoint saved (val_loss: 0.9292)
Epoch [ 10/20]  Train Loss: 0.8605  Bal.Acc: 68.1%  F1: 0.6573  |  Val Loss: 0.8577  Bal.Acc: 66.5%  F1: 0.6673  |  LR: 1.00e-04  (31.1s)
 Best checkpoint saved (val_loss: 0.8577)
Epoch [ 11/20]  Train Loss: 0.7893  Bal.Acc: 70.9%  F1: 0.6985  |  Val Loss: 0.7426  Bal.Acc: 72.4%  F1: 0.7130  |  LR: 1.00e-04  (31.9s)
 Best checkpoint saved (val_loss: 0.7426)
Epoch [ 12/20]  Train Loss: 0.7184  Bal.Acc: 70.2%  F1: 0.6915  |  Val Loss: 0.7116  Bal.Acc: 69.9%  F1: 0.6881  |  LR: 1.00e-04  (30.4s)
 Best checkpoint saved (val_loss: 0.7116)
Epoch [ 13/20]  Train Loss: 0.6825  Bal.Acc: 72.8%  F1: 0.7146  |  Val Loss: 0.7543  Bal.Acc: 70.3%  F1: 0.7107  |  LR: 1.00e-04  (29.9s)
Epoch [ 14/20]  Train Loss: 0.6652  Bal.Acc: 74.4%  F1: 0.7335  |  Val Loss: 0.7858  Bal.Acc: 65.9%  F1: 0.6651  |  LR: 1.00e-04  (24.1s)
Epoch [ 15/20]  Train Loss: 0.5911  Bal.Acc: 78.0%  F1: 0.7731  |  Val Loss: 0.6498  Bal.Acc: 73.6%  F1: 0.7338  |  LR: 1.00e-04  (24.3s)
 Best checkpoint saved (val_loss: 0.6498)
Epoch [ 16/20]  Train Loss: 0.5779  Bal.Acc: 79.4%  F1: 0.7859  |  Val Loss: 0.5933  Bal.Acc: 77.5%  F1: 0.7671  |  LR: 1.00e-04  (30.9s)
 Best checkpoint saved (val_loss: 0.5933)
Epoch [ 17/20]  Train Loss: 0.5270  Bal.Acc: 80.4%  F1: 0.7992  |  Val Loss: 0.5906  Bal.Acc: 75.8%  F1: 0.7400  |  LR: 1.00e-04  (32.1s)
 Best checkpoint saved (val_loss: 0.5906)
Epoch [ 18/20]  Train Loss: 0.4757  Bal.Acc: 82.3%  F1: 0.8173  |  Val Loss: 0.6061  Bal.Acc: 75.3%  F1: 0.7391  |  LR: 1.00e-04  (30.2s)
Epoch [ 19/20]  Train Loss: 0.4974  Bal.Acc: 79.7%  F1: 0.7921  |  Val Loss: 0.6059  Bal.Acc: 76.2%  F1: 0.7479  |  LR: 1.00e-04  (24.0s)
Epoch [ 20/20]  Train Loss: 0.4704  Bal.Acc: 82.4%  F1: 0.8211  |  Val Loss: 0.6253  Bal.Acc: 76.9%  F1: 0.7521  |  LR: 1.00e-04  (23.9s)

 Trening finished. Checkpoint: checkpoints/resnet50_fold4_best.pt
Log CSV: results/resnet50_fold4_training_log.csv
Best weights loaded from  17

Ewaluacja modelu: resnet50_fold4
----------------------------------------
  Balanced Accuracy:       75.81%
  F1 (macro):              0.7400
  Quadratic Cohen's Kappa: 0.8897
  ECE:                     0.0375
  Brier Score (mean):      0.0676

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.86      0.89      0.87        87
    Doubtful       0.75      0.60      0.67        81
        Mild       0.54      0.55      0.54        40
    Moderate       0.76      0.86      0.81        37
      Severe       0.74      0.89      0.81        35

    accuracy                           0.75       280
   macro avg       0.73      0.76      0.74       280
weighted avg       0.75      0.75      0.75       280

  Metryki zapisane: results/resnet50_fold4_metrics.json
  Prawdopodobieństwa zapisane: results/resnet50_fold4_test_probs.npz

--- resnet50 | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 5):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.420  (count = 158)
      Klasa 3 (Moderate): waga = 1.496  (count = 150)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold5
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.6059  Bal.Acc: 20.7%  F1: 0.1802  |  Val Loss: 1.5979  Bal.Acc: 28.4%  F1: 0.2738  |  LR: 1.00e-04  (24.1s)
 Best checkpoint saved (val_loss: 1.5979)
Epoch [  2/20]  Train Loss: 1.5769  Bal.Acc: 35.5%  F1: 0.3043  |  Val Loss: 1.5640  Bal.Acc: 37.0%  F1: 0.3297  |  LR: 1.00e-04  (25.2s)
 Best checkpoint saved (val_loss: 1.5640)
Epoch [  3/20]  Train Loss: 1.5453  Bal.Acc: 39.3%  F1: 0.3577  |  Val Loss: 1.5328  Bal.Acc: 42.3%  F1: 0.4139  |  LR: 1.00e-04  (30.7s)
 Best checkpoint saved (val_loss: 1.5328)
Epoch [  4/20]  Train Loss: 1.4972  Bal.Acc: 44.7%  F1: 0.3884  |  Val Loss: 1.4587  Bal.Acc: 48.0%  F1: 0.4332  |  LR: 1.00e-04  (31.8s)
 Best checkpoint saved (val_loss: 1.4587)
Epoch [  5/20]  Train Loss: 1.4197  Bal.Acc: 48.2%  F1: 0.4000  |  Val Loss: 1.3754  Bal.Acc: 46.7%  F1: 0.4308  |  LR: 1.00e-04  (32.1s)
 Best checkpoint saved (val_loss: 1.3754)
Epoch [  6/20]  Train Loss: 1.2838  Bal.Acc: 55.8%  F1: 0.5077  |  Val Loss: 1.2768  Bal.Acc: 45.9%  F1: 0.4298  |  LR: 1.00e-04  (30.5s)
 Best checkpoint saved (val_loss: 1.2768)
Epoch [  7/20]  Train Loss: 1.1262  Bal.Acc: 58.5%  F1: 0.5325  |  Val Loss: 1.1570  Bal.Acc: 53.0%  F1: 0.5082  |  LR: 1.00e-04  (31.4s)
 Best checkpoint saved (val_loss: 1.1570)
Epoch [  8/20]  Train Loss: 0.9989  Bal.Acc: 60.8%  F1: 0.5603  |  Val Loss: 0.9785  Bal.Acc: 64.1%  F1: 0.6168  |  LR: 1.00e-04  (29.6s)
 Best checkpoint saved (val_loss: 0.9785)
Epoch [  9/20]  Train Loss: 0.8972  Bal.Acc: 67.3%  F1: 0.6466  |  Val Loss: 1.0360  Bal.Acc: 59.3%  F1: 0.6039  |  LR: 1.00e-04  (30.3s)
Epoch [ 10/20]  Train Loss: 0.7888  Bal.Acc: 70.5%  F1: 0.6958  |  Val Loss: 0.9248  Bal.Acc: 63.3%  F1: 0.6166  |  LR: 1.00e-04  (24.2s)
 Best checkpoint saved (val_loss: 0.9248)
Epoch [ 11/20]  Train Loss: 0.7341  Bal.Acc: 72.2%  F1: 0.7122  |  Val Loss: 0.7551  Bal.Acc: 74.4%  F1: 0.7448  |  LR: 1.00e-04  (30.5s)
 Best checkpoint saved (val_loss: 0.7551)
Epoch [ 12/20]  Train Loss: 0.7009  Bal.Acc: 73.4%  F1: 0.7295  |  Val Loss: 0.7446  Bal.Acc: 68.7%  F1: 0.6875  |  LR: 1.00e-04  (30.6s)
 Best checkpoint saved (val_loss: 0.7446)
Epoch [ 13/20]  Train Loss: 0.6427  Bal.Acc: 76.3%  F1: 0.7515  |  Val Loss: 0.6964  Bal.Acc: 72.0%  F1: 0.7229  |  LR: 1.00e-04  (30.3s)
 Best checkpoint saved (val_loss: 0.6964)
Epoch [ 14/20]  Train Loss: 0.5903  Bal.Acc: 78.2%  F1: 0.7756  |  Val Loss: 0.6832  Bal.Acc: 74.6%  F1: 0.7494  |  LR: 1.00e-04  (31.2s)
 Best checkpoint saved (val_loss: 0.6832)
Epoch [ 15/20]  Train Loss: 0.5570  Bal.Acc: 79.3%  F1: 0.7859  |  Val Loss: 0.7317  Bal.Acc: 73.3%  F1: 0.7413  |  LR: 1.00e-04  (31.7s)
Epoch [ 16/20]  Train Loss: 0.5326  Bal.Acc: 79.9%  F1: 0.7918  |  Val Loss: 0.6758  Bal.Acc: 73.1%  F1: 0.7332  |  LR: 1.00e-04  (23.7s)
 Best checkpoint saved (val_loss: 0.6758)
Epoch [ 17/20]  Train Loss: 0.5234  Bal.Acc: 80.2%  F1: 0.7936  |  Val Loss: 0.6609  Bal.Acc: 75.2%  F1: 0.7476  |  LR: 1.00e-04  (31.8s)
 Best checkpoint saved (val_loss: 0.6609)
Epoch [ 18/20]  Train Loss: 0.4647  Bal.Acc: 82.0%  F1: 0.8142  |  Val Loss: 0.6177  Bal.Acc: 76.4%  F1: 0.7591  |  LR: 1.00e-04  (32.0s)
 Best checkpoint saved (val_loss: 0.6177)
Epoch [ 19/20]  Train Loss: 0.4790  Bal.Acc: 80.9%  F1: 0.8001  |  Val Loss: 0.6134  Bal.Acc: 76.0%  F1: 0.7573  |  LR: 1.00e-04  (30.4s)
 Best checkpoint saved (val_loss: 0.6134)
Epoch [ 20/20]  Train Loss: 0.4540  Bal.Acc: 83.0%  F1: 0.8189  |  Val Loss: 0.7861  Bal.Acc: 70.1%  F1: 0.7147  |  LR: 1.00e-04  (30.3s)

 Trening finished. Checkpoint: checkpoints/resnet50_fold5_best.pt
Log CSV: results/resnet50_fold5_training_log.csv
Best weights loaded from  19

Ewaluacja modelu: resnet50_fold5
----------------------------------------
  Balanced Accuracy:       76.02%
  F1 (macro):              0.7573
  Quadratic Cohen's Kappa: 0.8994
  ECE:                     0.0627
  Brier Score (mean):      0.0616

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.79      0.98      0.87        87
    Doubtful       0.75      0.63      0.68        81
        Mild       0.59      0.44      0.50        39
    Moderate       0.94      0.82      0.87        38
      Severe       0.79      0.94      0.86        35

    accuracy                           0.78       280
   macro avg       0.77      0.76      0.76       280
weighted avg       0.77      0.78      0.76       280

  Metryki zapisane: results/resnet50_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/resnet50_fold5_test_probs.npz

✅ ZAKOŃCZONO: resnet50. Średnia Kappa z 5 foldów: 0.9055 ±0.0119

================================================================================
🚀 ROZPOCZĘCIE TRENINGU MODELU: efficientnet_b3
================================================================================

--- efficientnet_b3 | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 1):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
model.safetensors: 100% 49.3M/49.3M [00:01<00:00, 48.3MB/s]
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold1
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 2.4515  Bal.Acc: 35.0%  F1: 0.3309  |  Val Loss: 1.9510  Bal.Acc: 38.5%  F1: 0.3737  |  LR: 1.00e-04  (24.5s)
 Best checkpoint saved (val_loss: 1.9510)
Epoch [  2/20]  Train Loss: 1.4677  Bal.Acc: 51.5%  F1: 0.4936  |  Val Loss: 3.5326  Bal.Acc: 47.1%  F1: 0.4760  |  LR: 1.00e-04  (24.6s)
Epoch [  3/20]  Train Loss: 1.0058  Bal.Acc: 63.3%  F1: 0.6194  |  Val Loss: 4192.0622  Bal.Acc: 53.6%  F1: 0.5432  |  LR: 1.00e-04  (24.5s)
Epoch [  4/20]  Train Loss: 0.9326  Bal.Acc: 66.0%  F1: 0.6403  |  Val Loss: 7448.2257  Bal.Acc: 59.0%  F1: 0.5664  |  LR: 1.00e-04  (24.5s)
Epoch [  5/20]  Train Loss: 0.7409  Bal.Acc: 72.1%  F1: 0.7111  |  Val Loss: 436.6288  Bal.Acc: 66.0%  F1: 0.6532  |  LR: 5.00e-05  (24.2s)
Epoch [  6/20]  Train Loss: 0.6943  Bal.Acc: 75.5%  F1: 0.7469  |  Val Loss: 11290.9251  Bal.Acc: 67.1%  F1: 0.6560  |  LR: 5.00e-05  (24.4s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 1.9510

 Trening finished. Checkpoint: checkpoints/efficientnet_b3_fold1_best.pt
Log CSV: results/efficientnet_b3_fold1_training_log.csv
Best weights loaded from  1

Ewaluacja modelu: efficientnet_b3_fold1
----------------------------------------
  Balanced Accuracy:       38.46%
  F1 (macro):              0.3737
  Quadratic Cohen's Kappa: 0.3850
  ECE:                     0.2839
  Brier Score (mean):      0.1691

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.71      0.28      0.41        88
    Doubtful       0.37      0.62      0.46        81
        Mild       0.30      0.21      0.24        39
    Moderate       0.37      0.47      0.41        38
      Severe       0.34      0.34      0.34        35

    accuracy                           0.40       281
   macro avg       0.42      0.38      0.37       281
weighted avg       0.46      0.40      0.39       281

  Metryki zapisane: results/efficientnet_b3_fold1_metrics.json
  Prawdopodobieństwa zapisane: results/efficientnet_b3_fold1_test_probs.npz

--- efficientnet_b3 | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 2):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold2
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 2.1588  Bal.Acc: 34.9%  F1: 0.3301  |  Val Loss: 1.9378  Bal.Acc: 37.9%  F1: 0.3413  |  LR: 1.00e-04  (24.6s)
 Best checkpoint saved (val_loss: 1.9378)
Epoch [  2/20]  Train Loss: 1.2918  Bal.Acc: 55.5%  F1: 0.5390  |  Val Loss: 2.2343  Bal.Acc: 53.9%  F1: 0.5401  |  LR: 1.00e-04  (25.6s)
Epoch [  3/20]  Train Loss: 1.1065  Bal.Acc: 60.6%  F1: 0.5869  |  Val Loss: 1.1450  Bal.Acc: 57.7%  F1: 0.5817  |  LR: 1.00e-04  (24.6s)
 Best checkpoint saved (val_loss: 1.1450)
Epoch [  4/20]  Train Loss: 0.8107  Bal.Acc: 70.3%  F1: 0.6937  |  Val Loss: 1.0680  Bal.Acc: 66.3%  F1: 0.6553  |  LR: 1.00e-04  (29.3s)
 Best checkpoint saved (val_loss: 1.0680)
Epoch [  5/20]  Train Loss: 0.7502  Bal.Acc: 73.6%  F1: 0.7226  |  Val Loss: 3.0123  Bal.Acc: 65.8%  F1: 0.6605  |  LR: 1.00e-04  (28.8s)
Epoch [  6/20]  Train Loss: 0.6457  Bal.Acc: 76.4%  F1: 0.7533  |  Val Loss: 0.8244  Bal.Acc: 73.3%  F1: 0.7343  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 0.8244)
Epoch [  7/20]  Train Loss: 0.5552  Bal.Acc: 78.8%  F1: 0.7703  |  Val Loss: 1.3841  Bal.Acc: 73.1%  F1: 0.7201  |  LR: 1.00e-04  (28.8s)
Epoch [  8/20]  Train Loss: 0.5067  Bal.Acc: 80.9%  F1: 0.8061  |  Val Loss: 0.8001  Bal.Acc: 72.8%  F1: 0.7315  |  LR: 1.00e-04  (24.2s)
 Best checkpoint saved (val_loss: 0.8001)
Epoch [  9/20]  Train Loss: 0.4416  Bal.Acc: 84.5%  F1: 0.8379  |  Val Loss: 0.7117  Bal.Acc: 76.3%  F1: 0.7515  |  LR: 1.00e-04  (27.4s)
 Best checkpoint saved (val_loss: 0.7117)
Epoch [ 10/20]  Train Loss: 0.4208  Bal.Acc: 83.8%  F1: 0.8296  |  Val Loss: 0.7866  Bal.Acc: 76.8%  F1: 0.7691  |  LR: 1.00e-04  (27.8s)
Epoch [ 11/20]  Train Loss: 0.3354  Bal.Acc: 87.7%  F1: 0.8655  |  Val Loss: 0.7753  Bal.Acc: 73.3%  F1: 0.7291  |  LR: 1.00e-04  (24.4s)
Epoch [ 12/20]  Train Loss: 0.3405  Bal.Acc: 87.2%  F1: 0.8618  |  Val Loss: 0.6909  Bal.Acc: 75.0%  F1: 0.7439  |  LR: 1.00e-04  (24.5s)
 Best checkpoint saved (val_loss: 0.6909)
Epoch [ 13/20]  Train Loss: 0.2775  Bal.Acc: 89.2%  F1: 0.8880  |  Val Loss: 0.7272  Bal.Acc: 74.4%  F1: 0.7456  |  LR: 1.00e-04  (28.4s)
Epoch [ 14/20]  Train Loss: 0.2842  Bal.Acc: 88.7%  F1: 0.8814  |  Val Loss: 0.6412  Bal.Acc: 76.7%  F1: 0.7706  |  LR: 1.00e-04  (24.8s)
 Best checkpoint saved (val_loss: 0.6412)
Epoch [ 15/20]  Train Loss: 0.2430  Bal.Acc: 91.6%  F1: 0.9105  |  Val Loss: 0.6167  Bal.Acc: 79.0%  F1: 0.7939  |  LR: 1.00e-04  (28.9s)
 Best checkpoint saved (val_loss: 0.6167)
Epoch [ 16/20]  Train Loss: 0.2251  Bal.Acc: 91.2%  F1: 0.9033  |  Val Loss: 0.7178  Bal.Acc: 77.1%  F1: 0.7699  |  LR: 1.00e-04  (28.9s)
Epoch [ 17/20]  Train Loss: 0.2447  Bal.Acc: 90.2%  F1: 0.9014  |  Val Loss: 0.6558  Bal.Acc: 80.6%  F1: 0.8097  |  LR: 1.00e-04  (24.8s)
Epoch [ 18/20]  Train Loss: 0.2402  Bal.Acc: 91.4%  F1: 0.9119  |  Val Loss: 0.6392  Bal.Acc: 81.1%  F1: 0.8079  |  LR: 1.00e-04  (24.6s)
Epoch [ 19/20]  Train Loss: 0.2115  Bal.Acc: 92.9%  F1: 0.9280  |  Val Loss: 0.6988  Bal.Acc: 78.6%  F1: 0.7892  |  LR: 5.00e-05  (24.4s)
Epoch [ 20/20]  Train Loss: 0.1878  Bal.Acc: 93.8%  F1: 0.9351  |  Val Loss: 0.6598  Bal.Acc: 78.8%  F1: 0.7900  |  LR: 5.00e-05  (24.6s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.6167

 Trening finished. Checkpoint: checkpoints/efficientnet_b3_fold2_best.pt
Log CSV: results/efficientnet_b3_fold2_training_log.csv
Best weights loaded from  15

Ewaluacja modelu: efficientnet_b3_fold2
----------------------------------------
  Balanced Accuracy:       78.99%
  F1 (macro):              0.7939
  Quadratic Cohen's Kappa: 0.9059
  ECE:                     0.1006
  Brier Score (mean):      0.0663

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.79      0.88      0.83        88
    Doubtful       0.68      0.63      0.65        81
        Mild       0.67      0.67      0.67        39
    Moderate       0.92      0.92      0.92        38
      Severe       0.94      0.86      0.90        35

    accuracy                           0.78       281
   macro avg       0.80      0.79      0.79       281
weighted avg       0.78      0.78      0.78       281

  Metryki zapisane: results/efficientnet_b3_fold2_metrics.json
  Prawdopodobieństwa zapisane: results/efficientnet_b3_fold2_test_probs.npz

--- efficientnet_b3 | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 3):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold3
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 2.4316  Bal.Acc: 33.3%  F1: 0.3197  |  Val Loss: 1.7119  Bal.Acc: 41.1%  F1: 0.3685  |  LR: 1.00e-04  (24.4s)
 Best checkpoint saved (val_loss: 1.7119)
Epoch [  2/20]  Train Loss: 1.3505  Bal.Acc: 53.9%  F1: 0.5194  |  Val Loss: 1.5149  Bal.Acc: 51.5%  F1: 0.4869  |  LR: 1.00e-04  (24.5s)
 Best checkpoint saved (val_loss: 1.5149)
Epoch [  3/20]  Train Loss: 0.9896  Bal.Acc: 65.5%  F1: 0.6366  |  Val Loss: 1.1118  Bal.Acc: 65.1%  F1: 0.6200  |  LR: 1.00e-04  (27.7s)
 Best checkpoint saved (val_loss: 1.1118)
Epoch [  4/20]  Train Loss: 0.8524  Bal.Acc: 68.8%  F1: 0.6751  |  Val Loss: 1.1031  Bal.Acc: 61.8%  F1: 0.6008  |  LR: 1.00e-04  (27.6s)
 Best checkpoint saved (val_loss: 1.1031)
Epoch [  5/20]  Train Loss: 0.7676  Bal.Acc: 71.5%  F1: 0.7020  |  Val Loss: 1.0044  Bal.Acc: 67.3%  F1: 0.6665  |  LR: 1.00e-04  (28.0s)
 Best checkpoint saved (val_loss: 1.0044)
Epoch [  6/20]  Train Loss: 0.6082  Bal.Acc: 78.3%  F1: 0.7688  |  Val Loss: 0.9059  Bal.Acc: 67.5%  F1: 0.6650  |  LR: 1.00e-04  (27.4s)
 Best checkpoint saved (val_loss: 0.9059)
Epoch [  7/20]  Train Loss: 0.5429  Bal.Acc: 79.0%  F1: 0.7785  |  Val Loss: 0.9019  Bal.Acc: 71.6%  F1: 0.7032  |  LR: 1.00e-04  (27.3s)
 Best checkpoint saved (val_loss: 0.9019)
Epoch [  8/20]  Train Loss: 0.4966  Bal.Acc: 80.7%  F1: 0.8006  |  Val Loss: 0.8051  Bal.Acc: 73.0%  F1: 0.7174  |  LR: 1.00e-04  (27.2s)
 Best checkpoint saved (val_loss: 0.8051)
Epoch [  9/20]  Train Loss: 0.3868  Bal.Acc: 85.1%  F1: 0.8435  |  Val Loss: 0.8131  Bal.Acc: 72.4%  F1: 0.7228  |  LR: 1.00e-04  (27.4s)
Epoch [ 10/20]  Train Loss: 0.4423  Bal.Acc: 84.3%  F1: 0.8390  |  Val Loss: 0.8416  Bal.Acc: 73.0%  F1: 0.7212  |  LR: 1.00e-04  (24.0s)
Epoch [ 11/20]  Train Loss: 0.3254  Bal.Acc: 89.0%  F1: 0.8800  |  Val Loss: 0.7862  Bal.Acc: 77.7%  F1: 0.7761  |  LR: 1.00e-04  (23.9s)
 Best checkpoint saved (val_loss: 0.7862)
Epoch [ 12/20]  Train Loss: 0.3519  Bal.Acc: 86.5%  F1: 0.8619  |  Val Loss: 0.7807  Bal.Acc: 77.8%  F1: 0.7786  |  LR: 1.00e-04  (28.4s)
 Best checkpoint saved (val_loss: 0.7807)
Epoch [ 13/20]  Train Loss: 0.3256  Bal.Acc: 88.9%  F1: 0.8859  |  Val Loss: 0.7288  Bal.Acc: 77.8%  F1: 0.7816  |  LR: 1.00e-04  (28.0s)
 Best checkpoint saved (val_loss: 0.7288)
Epoch [ 14/20]  Train Loss: 0.3354  Bal.Acc: 87.9%  F1: 0.8769  |  Val Loss: 0.7974  Bal.Acc: 75.4%  F1: 0.7457  |  LR: 1.00e-04  (30.2s)
Epoch [ 15/20]  Train Loss: 0.2839  Bal.Acc: 88.9%  F1: 0.8844  |  Val Loss: 0.9830  Bal.Acc: 74.2%  F1: 0.7558  |  LR: 1.00e-04  (24.1s)
Epoch [ 16/20]  Train Loss: 0.2575  Bal.Acc: 89.8%  F1: 0.8886  |  Val Loss: 1.5580  Bal.Acc: 74.1%  F1: 0.7478  |  LR: 1.00e-04  (24.0s)
Epoch [ 17/20]  Train Loss: 0.2322  Bal.Acc: 91.0%  F1: 0.9055  |  Val Loss: 2.1627  Bal.Acc: 74.8%  F1: 0.7557  |  LR: 5.00e-05  (24.1s)
Epoch [ 18/20]  Train Loss: 0.2255  Bal.Acc: 91.8%  F1: 0.9151  |  Val Loss: 1.0118  Bal.Acc: 76.3%  F1: 0.7686  |  LR: 5.00e-05  (24.3s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.7288

 Trening finished. Checkpoint: checkpoints/efficientnet_b3_fold3_best.pt
Log CSV: results/efficientnet_b3_fold3_training_log.csv
Best weights loaded from  13

Ewaluacja modelu: efficientnet_b3_fold3
----------------------------------------
  Balanced Accuracy:       77.77%
  F1 (macro):              0.7816
  Quadratic Cohen's Kappa: 0.8996
  ECE:                     0.1209
  Brier Score (mean):      0.0682

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.79      0.82        87
    Doubtful       0.69      0.75      0.72        81
        Mild       0.64      0.70      0.67        40
    Moderate       0.97      0.76      0.85        37
      Severe       0.82      0.89      0.85        35

    accuracy                           0.78       280
   macro avg       0.79      0.78      0.78       280
weighted avg       0.79      0.78      0.78       280

  Metryki zapisane: results/efficientnet_b3_fold3_metrics.json
  Prawdopodobieństwa zapisane: results/efficientnet_b3_fold3_test_probs.npz

--- efficientnet_b3 | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 4):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold4
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 2.2385  Bal.Acc: 36.4%  F1: 0.3435  |  Val Loss: 1.7822  Bal.Acc: 41.2%  F1: 0.3607  |  LR: 1.00e-04  (24.1s)
 Best checkpoint saved (val_loss: 1.7822)
Epoch [  2/20]  Train Loss: 1.3544  Bal.Acc: 53.5%  F1: 0.5182  |  Val Loss: 1.4649  Bal.Acc: 54.9%  F1: 0.5432  |  LR: 1.00e-04  (25.1s)
 Best checkpoint saved (val_loss: 1.4649)
Epoch [  3/20]  Train Loss: 1.0239  Bal.Acc: 63.8%  F1: 0.6324  |  Val Loss: 1.1706  Bal.Acc: 61.6%  F1: 0.6136  |  LR: 1.00e-04  (28.4s)
 Best checkpoint saved (val_loss: 1.1706)
Epoch [  4/20]  Train Loss: 0.8252  Bal.Acc: 70.7%  F1: 0.6926  |  Val Loss: 1.0039  Bal.Acc: 66.8%  F1: 0.6602  |  LR: 1.00e-04  (28.2s)
 Best checkpoint saved (val_loss: 1.0039)
Epoch [  5/20]  Train Loss: 0.7179  Bal.Acc: 73.1%  F1: 0.7227  |  Val Loss: 0.9953  Bal.Acc: 64.9%  F1: 0.6551  |  LR: 1.00e-04  (27.2s)
 Best checkpoint saved (val_loss: 0.9953)
Epoch [  6/20]  Train Loss: 0.6171  Bal.Acc: 76.9%  F1: 0.7524  |  Val Loss: 0.7848  Bal.Acc: 69.7%  F1: 0.6957  |  LR: 1.00e-04  (28.6s)
 Best checkpoint saved (val_loss: 0.7848)
Epoch [  7/20]  Train Loss: 0.5526  Bal.Acc: 76.9%  F1: 0.7606  |  Val Loss: 0.8534  Bal.Acc: 69.5%  F1: 0.6933  |  LR: 1.00e-04  (28.2s)
Epoch [  8/20]  Train Loss: 0.4366  Bal.Acc: 83.2%  F1: 0.8185  |  Val Loss: 0.6715  Bal.Acc: 76.8%  F1: 0.7653  |  LR: 1.00e-04  (23.6s)
 Best checkpoint saved (val_loss: 0.6715)
Epoch [  9/20]  Train Loss: 0.4104  Bal.Acc: 84.5%  F1: 0.8357  |  Val Loss: 0.7684  Bal.Acc: 72.2%  F1: 0.7050  |  LR: 1.00e-04  (27.2s)
Epoch [ 10/20]  Train Loss: 0.3890  Bal.Acc: 85.4%  F1: 0.8521  |  Val Loss: 0.6482  Bal.Acc: 79.1%  F1: 0.7804  |  LR: 1.00e-04  (23.9s)
 Best checkpoint saved (val_loss: 0.6482)
Epoch [ 11/20]  Train Loss: 0.3605  Bal.Acc: 86.0%  F1: 0.8524  |  Val Loss: 0.6921  Bal.Acc: 76.9%  F1: 0.7530  |  LR: 1.00e-04  (27.3s)
Epoch [ 12/20]  Train Loss: 0.3230  Bal.Acc: 88.1%  F1: 0.8734  |  Val Loss: 0.6262  Bal.Acc: 81.0%  F1: 0.7956  |  LR: 1.00e-04  (24.0s)
 Best checkpoint saved (val_loss: 0.6262)
Epoch [ 13/20]  Train Loss: 0.2754  Bal.Acc: 89.2%  F1: 0.8889  |  Val Loss: 0.6085  Bal.Acc: 80.3%  F1: 0.7935  |  LR: 1.00e-04  (28.4s)
 Best checkpoint saved (val_loss: 0.6085)
Epoch [ 14/20]  Train Loss: 0.2408  Bal.Acc: 90.9%  F1: 0.9041  |  Val Loss: 0.6113  Bal.Acc: 81.9%  F1: 0.8150  |  LR: 1.00e-04  (28.8s)
Epoch [ 15/20]  Train Loss: 0.3009  Bal.Acc: 89.3%  F1: 0.8939  |  Val Loss: 0.5296  Bal.Acc: 85.0%  F1: 0.8393  |  LR: 1.00e-04  (25.0s)
 Best checkpoint saved (val_loss: 0.5296)
Epoch [ 16/20]  Train Loss: 0.2306  Bal.Acc: 91.8%  F1: 0.9120  |  Val Loss: 0.6364  Bal.Acc: 82.7%  F1: 0.8216  |  LR: 1.00e-04  (29.7s)
Epoch [ 17/20]  Train Loss: 0.2240  Bal.Acc: 90.9%  F1: 0.9011  |  Val Loss: 0.6291  Bal.Acc: 82.5%  F1: 0.8202  |  LR: 1.00e-04  (24.6s)
Epoch [ 18/20]  Train Loss: 0.2445  Bal.Acc: 90.9%  F1: 0.9075  |  Val Loss: 0.7026  Bal.Acc: 79.5%  F1: 0.7917  |  LR: 1.00e-04  (25.8s)
Epoch [ 19/20]  Train Loss: 0.2496  Bal.Acc: 91.0%  F1: 0.9000  |  Val Loss: 0.6108  Bal.Acc: 81.9%  F1: 0.8124  |  LR: 5.00e-05  (25.6s)
Epoch [ 20/20]  Train Loss: 0.1904  Bal.Acc: 93.8%  F1: 0.9320  |  Val Loss: 0.5703  Bal.Acc: 83.4%  F1: 0.8227  |  LR: 5.00e-05  (25.0s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.5296

 Trening finished. Checkpoint: checkpoints/efficientnet_b3_fold4_best.pt
Log CSV: results/efficientnet_b3_fold4_training_log.csv
Best weights loaded from  15

Ewaluacja modelu: efficientnet_b3_fold4
----------------------------------------
  Balanced Accuracy:       84.96%
  F1 (macro):              0.8393
  Quadratic Cohen's Kappa: 0.9110
  ECE:                     0.0536
  Brier Score (mean):      0.0488

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.91      0.89        87
    Doubtful       0.89      0.73      0.80        81
        Mild       0.62      0.75      0.68        40
    Moderate       0.97      0.92      0.94        37
      Severe       0.82      0.94      0.88        35

    accuracy                           0.84       280
   macro avg       0.84      0.85      0.84       280
weighted avg       0.85      0.84      0.84       280

  Metryki zapisane: results/efficientnet_b3_fold4_metrics.json
  Prawdopodobieństwa zapisane: results/efficientnet_b3_fold4_test_probs.npz

--- efficientnet_b3 | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 5):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.420  (count = 158)
      Klasa 3 (Moderate): waga = 1.496  (count = 150)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold5
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 2.2267  Bal.Acc: 38.3%  F1: 0.3566  |  Val Loss: 1.9092  Bal.Acc: 42.4%  F1: 0.4018  |  LR: 1.00e-04  (25.7s)
 Best checkpoint saved (val_loss: 1.9092)
Epoch [  2/20]  Train Loss: 1.4032  Bal.Acc: 52.6%  F1: 0.5126  |  Val Loss: 1.6775  Bal.Acc: 47.8%  F1: 0.4894  |  LR: 1.00e-04  (26.8s)
 Best checkpoint saved (val_loss: 1.6775)
Epoch [  3/20]  Train Loss: 1.1154  Bal.Acc: 60.9%  F1: 0.5973  |  Val Loss: 1.4569  Bal.Acc: 50.8%  F1: 0.5105  |  LR: 1.00e-04  (30.5s)
 Best checkpoint saved (val_loss: 1.4569)
Epoch [  4/20]  Train Loss: 0.8458  Bal.Acc: 70.7%  F1: 0.6943  |  Val Loss: 1.2358  Bal.Acc: 57.8%  F1: 0.5890  |  LR: 1.00e-04  (30.6s)
 Best checkpoint saved (val_loss: 1.2358)
Epoch [  5/20]  Train Loss: 0.6981  Bal.Acc: 73.5%  F1: 0.7305  |  Val Loss: 1.2576  Bal.Acc: 59.4%  F1: 0.5945  |  LR: 1.00e-04  (28.5s)
Epoch [  6/20]  Train Loss: 0.6134  Bal.Acc: 76.1%  F1: 0.7489  |  Val Loss: 1.1517  Bal.Acc: 64.3%  F1: 0.6422  |  LR: 1.00e-04  (25.5s)
 Best checkpoint saved (val_loss: 1.1517)
Epoch [  7/20]  Train Loss: 0.5397  Bal.Acc: 80.3%  F1: 0.7918  |  Val Loss: 1.0344  Bal.Acc: 71.1%  F1: 0.7065  |  LR: 1.00e-04  (28.8s)
 Best checkpoint saved (val_loss: 1.0344)
Epoch [  8/20]  Train Loss: 0.4868  Bal.Acc: 81.8%  F1: 0.8112  |  Val Loss: 0.9663  Bal.Acc: 69.9%  F1: 0.7044  |  LR: 1.00e-04  (28.8s)
 Best checkpoint saved (val_loss: 0.9663)
Epoch [  9/20]  Train Loss: 0.4873  Bal.Acc: 82.2%  F1: 0.8216  |  Val Loss: 0.9725  Bal.Acc: 72.2%  F1: 0.7200  |  LR: 1.00e-04  (28.8s)
Epoch [ 10/20]  Train Loss: 0.3948  Bal.Acc: 85.4%  F1: 0.8448  |  Val Loss: 0.9669  Bal.Acc: 72.8%  F1: 0.7325  |  LR: 1.00e-04  (25.5s)
Epoch [ 11/20]  Train Loss: 0.3483  Bal.Acc: 85.4%  F1: 0.8444  |  Val Loss: 1.0044  Bal.Acc: 71.5%  F1: 0.7133  |  LR: 1.00e-04  (25.4s)
Epoch [ 12/20]  Train Loss: 0.3213  Bal.Acc: 88.1%  F1: 0.8719  |  Val Loss: 1.0452  Bal.Acc: 67.4%  F1: 0.6668  |  LR: 5.00e-05  (25.2s)
Epoch [ 13/20]  Train Loss: 0.2713  Bal.Acc: 89.7%  F1: 0.8858  |  Val Loss: 0.9943  Bal.Acc: 74.3%  F1: 0.7453  |  LR: 5.00e-05  (25.1s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.9663

 Trening finished. Checkpoint: checkpoints/efficientnet_b3_fold5_best.pt
Log CSV: results/efficientnet_b3_fold5_training_log.csv
Best weights loaded from  8

Ewaluacja modelu: efficientnet_b3_fold5
----------------------------------------
  Balanced Accuracy:       69.92%
  F1 (macro):              0.7044
  Quadratic Cohen's Kappa: 0.8315
  ECE:                     0.1179
  Brier Score (mean):      0.0828

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.73      0.90      0.80        87
    Doubtful       0.67      0.60      0.64        81
        Mild       0.59      0.44      0.50        39
    Moderate       0.89      0.82      0.85        38
      Severe       0.72      0.74      0.73        35

    accuracy                           0.72       280
   macro avg       0.72      0.70      0.70       280
weighted avg       0.71      0.72      0.71       280

  Metryki zapisane: results/efficientnet_b3_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/efficientnet_b3_fold5_test_probs.npz

✅ ZAKOŃCZONO: efficientnet_b3. Średnia Kappa z 5 foldów: 0.7866 ±0.2029

================================================================================
🚀 ROZPOCZĘCIE TRENINGU MODELU: densenet121
================================================================================

--- densenet121 | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 1):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
model.safetensors: 100% 32.3M/32.3M [00:02<00:00, 11.5MB/s]
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold1
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.4515  Bal.Acc: 38.4%  F1: 0.3383  |  Val Loss: 1.2597  Bal.Acc: 50.5%  F1: 0.4506  |  LR: 1.00e-04  (25.5s)
 Best checkpoint saved (val_loss: 1.2597)
Epoch [  2/20]  Train Loss: 1.0869  Bal.Acc: 60.9%  F1: 0.5738  |  Val Loss: 1.0131  Bal.Acc: 63.4%  F1: 0.6421  |  LR: 1.00e-04  (25.4s)
 Best checkpoint saved (val_loss: 1.0131)
Epoch [  3/20]  Train Loss: 0.8538  Bal.Acc: 69.5%  F1: 0.6801  |  Val Loss: 0.8527  Bal.Acc: 68.3%  F1: 0.6989  |  LR: 1.00e-04  (26.9s)
 Best checkpoint saved (val_loss: 0.8527)
Epoch [  4/20]  Train Loss: 0.6793  Bal.Acc: 77.2%  F1: 0.7632  |  Val Loss: 0.8548  Bal.Acc: 63.6%  F1: 0.6513  |  LR: 1.00e-04  (27.0s)
Epoch [  5/20]  Train Loss: 0.5818  Bal.Acc: 79.6%  F1: 0.7840  |  Val Loss: 0.6294  Bal.Acc: 76.9%  F1: 0.7682  |  LR: 1.00e-04  (25.1s)
 Best checkpoint saved (val_loss: 0.6294)
Epoch [  6/20]  Train Loss: 0.4961  Bal.Acc: 83.7%  F1: 0.8302  |  Val Loss: 0.5571  Bal.Acc: 78.9%  F1: 0.7938  |  LR: 1.00e-04  (26.9s)
 Best checkpoint saved (val_loss: 0.5571)
Epoch [  7/20]  Train Loss: 0.4338  Bal.Acc: 84.2%  F1: 0.8359  |  Val Loss: 0.5568  Bal.Acc: 81.0%  F1: 0.8034  |  LR: 1.00e-04  (27.1s)
 Best checkpoint saved (val_loss: 0.5568)
Epoch [  8/20]  Train Loss: 0.4065  Bal.Acc: 85.9%  F1: 0.8544  |  Val Loss: 0.5562  Bal.Acc: 78.1%  F1: 0.7648  |  LR: 1.00e-04  (27.1s)
 Best checkpoint saved (val_loss: 0.5562)
Epoch [  9/20]  Train Loss: 0.3529  Bal.Acc: 87.1%  F1: 0.8677  |  Val Loss: 0.5299  Bal.Acc: 78.2%  F1: 0.7655  |  LR: 1.00e-04  (27.0s)
 Best checkpoint saved (val_loss: 0.5299)
Epoch [ 10/20]  Train Loss: 0.3102  Bal.Acc: 89.0%  F1: 0.8875  |  Val Loss: 0.5430  Bal.Acc: 79.2%  F1: 0.7928  |  LR: 1.00e-04  (27.5s)
Epoch [ 11/20]  Train Loss: 0.2975  Bal.Acc: 90.3%  F1: 0.8969  |  Val Loss: 0.4664  Bal.Acc: 82.0%  F1: 0.8226  |  LR: 1.00e-04  (25.8s)
 Best checkpoint saved (val_loss: 0.4664)
Epoch [ 12/20]  Train Loss: 0.2745  Bal.Acc: 91.8%  F1: 0.9130  |  Val Loss: 0.4881  Bal.Acc: 81.6%  F1: 0.8059  |  LR: 1.00e-04  (27.9s)
Epoch [ 13/20]  Train Loss: 0.2218  Bal.Acc: 92.1%  F1: 0.9201  |  Val Loss: 0.4618  Bal.Acc: 84.8%  F1: 0.8436  |  LR: 1.00e-04  (25.2s)
 Best checkpoint saved (val_loss: 0.4618)
Epoch [ 14/20]  Train Loss: 0.2259  Bal.Acc: 93.3%  F1: 0.9274  |  Val Loss: 0.4347  Bal.Acc: 83.8%  F1: 0.8343  |  LR: 1.00e-04  (27.5s)
 Best checkpoint saved (val_loss: 0.4347)
Epoch [ 15/20]  Train Loss: 0.2149  Bal.Acc: 93.5%  F1: 0.9308  |  Val Loss: 0.4553  Bal.Acc: 83.4%  F1: 0.8334  |  LR: 1.00e-04  (26.9s)
Epoch [ 16/20]  Train Loss: 0.2014  Bal.Acc: 93.2%  F1: 0.9295  |  Val Loss: 0.5130  Bal.Acc: 80.4%  F1: 0.8043  |  LR: 1.00e-04  (25.0s)
Epoch [ 17/20]  Train Loss: 0.1754  Bal.Acc: 95.2%  F1: 0.9491  |  Val Loss: 0.5483  Bal.Acc: 82.4%  F1: 0.8221  |  LR: 1.00e-04  (25.0s)
Epoch [ 18/20]  Train Loss: 0.1917  Bal.Acc: 93.4%  F1: 0.9307  |  Val Loss: 0.4342  Bal.Acc: 84.6%  F1: 0.8383  |  LR: 1.00e-04  (25.1s)
 Best checkpoint saved (val_loss: 0.4342)
Epoch [ 19/20]  Train Loss: 0.1625  Bal.Acc: 95.4%  F1: 0.9525  |  Val Loss: 0.4727  Bal.Acc: 82.3%  F1: 0.8129  |  LR: 1.00e-04  (27.9s)
Epoch [ 20/20]  Train Loss: 0.1663  Bal.Acc: 94.4%  F1: 0.9436  |  Val Loss: 0.5568  Bal.Acc: 79.5%  F1: 0.7811  |  LR: 1.00e-04  (25.0s)

 Trening finished. Checkpoint: checkpoints/densenet121_fold1_best.pt
Log CSV: results/densenet121_fold1_training_log.csv
Best weights loaded from  18

Ewaluacja modelu: densenet121_fold1
----------------------------------------
  Balanced Accuracy:       84.59%
  F1 (macro):              0.8383
  Quadratic Cohen's Kappa: 0.9359
  ECE:                     0.0589
  Brier Score (mean):      0.0507

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.88      0.88      0.88        88
    Doubtful       0.81      0.75      0.78        81
        Mild       0.76      0.79      0.78        39
    Moderate       0.85      0.92      0.89        38
      Severe       0.86      0.89      0.87        35

    accuracy                           0.84       281
   macro avg       0.83      0.85      0.84       281
weighted avg       0.84      0.84      0.84       281

  Metryki zapisane: results/densenet121_fold1_metrics.json
  Prawdopodobieństwa zapisane: results/densenet121_fold1_test_probs.npz

--- densenet121 | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 2):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold2
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.3959  Bal.Acc: 44.0%  F1: 0.3939  |  Val Loss: 1.3193  Bal.Acc: 45.4%  F1: 0.4342  |  LR: 1.00e-04  (25.1s)
 Best checkpoint saved (val_loss: 1.3193)
Epoch [  2/20]  Train Loss: 1.0547  Bal.Acc: 62.6%  F1: 0.5933  |  Val Loss: 1.0388  Bal.Acc: 59.6%  F1: 0.5969  |  LR: 1.00e-04  (25.1s)
 Best checkpoint saved (val_loss: 1.0388)
Epoch [  3/20]  Train Loss: 0.8029  Bal.Acc: 70.7%  F1: 0.6980  |  Val Loss: 0.7404  Bal.Acc: 73.5%  F1: 0.7294  |  LR: 1.00e-04  (26.6s)
 Best checkpoint saved (val_loss: 0.7404)
Epoch [  4/20]  Train Loss: 0.6655  Bal.Acc: 76.6%  F1: 0.7598  |  Val Loss: 0.7589  Bal.Acc: 68.5%  F1: 0.6933  |  LR: 1.00e-04  (27.1s)
Epoch [  5/20]  Train Loss: 0.5491  Bal.Acc: 81.0%  F1: 0.8023  |  Val Loss: 0.7411  Bal.Acc: 70.3%  F1: 0.6731  |  LR: 1.00e-04  (25.1s)
Epoch [  6/20]  Train Loss: 0.4733  Bal.Acc: 84.0%  F1: 0.8364  |  Val Loss: 0.5433  Bal.Acc: 79.4%  F1: 0.7763  |  LR: 1.00e-04  (24.8s)
 Best checkpoint saved (val_loss: 0.5433)
Epoch [  7/20]  Train Loss: 0.4377  Bal.Acc: 85.1%  F1: 0.8430  |  Val Loss: 0.6456  Bal.Acc: 75.4%  F1: 0.7588  |  LR: 1.00e-04  (28.1s)
Epoch [  8/20]  Train Loss: 0.4226  Bal.Acc: 85.4%  F1: 0.8434  |  Val Loss: 0.7845  Bal.Acc: 67.5%  F1: 0.6852  |  LR: 1.00e-04  (25.2s)
Epoch [  9/20]  Train Loss: 0.3584  Bal.Acc: 87.6%  F1: 0.8736  |  Val Loss: 0.5214  Bal.Acc: 78.7%  F1: 0.7737  |  LR: 1.00e-04  (24.9s)
 Best checkpoint saved (val_loss: 0.5214)
Epoch [ 10/20]  Train Loss: 0.3198  Bal.Acc: 88.9%  F1: 0.8873  |  Val Loss: 0.5624  Bal.Acc: 76.0%  F1: 0.7413  |  LR: 1.00e-04  (27.2s)
Epoch [ 11/20]  Train Loss: 0.3011  Bal.Acc: 89.4%  F1: 0.8920  |  Val Loss: 0.5619  Bal.Acc: 79.2%  F1: 0.7776  |  LR: 1.00e-04  (25.2s)
Epoch [ 12/20]  Train Loss: 0.2672  Bal.Acc: 90.0%  F1: 0.8934  |  Val Loss: 0.5207  Bal.Acc: 78.9%  F1: 0.7970  |  LR: 1.00e-04  (25.0s)
 Best checkpoint saved (val_loss: 0.5207)
Epoch [ 13/20]  Train Loss: 0.2608  Bal.Acc: 91.8%  F1: 0.9132  |  Val Loss: 0.5050  Bal.Acc: 82.4%  F1: 0.8222  |  LR: 1.00e-04  (26.8s)
 Best checkpoint saved (val_loss: 0.5050)
Epoch [ 14/20]  Train Loss: 0.2086  Bal.Acc: 92.9%  F1: 0.9241  |  Val Loss: 0.5024  Bal.Acc: 80.6%  F1: 0.8082  |  LR: 1.00e-04  (27.3s)
 Best checkpoint saved (val_loss: 0.5024)
Epoch [ 15/20]  Train Loss: 0.2096  Bal.Acc: 93.0%  F1: 0.9290  |  Val Loss: 0.4955  Bal.Acc: 82.6%  F1: 0.8304  |  LR: 1.00e-04  (26.7s)
 Best checkpoint saved (val_loss: 0.4955)
Epoch [ 16/20]  Train Loss: 0.1847  Bal.Acc: 93.6%  F1: 0.9320  |  Val Loss: 0.4897  Bal.Acc: 80.6%  F1: 0.7934  |  LR: 1.00e-04  (27.6s)
 Best checkpoint saved (val_loss: 0.4897)
Epoch [ 17/20]  Train Loss: 0.1921  Bal.Acc: 93.6%  F1: 0.9335  |  Val Loss: 0.4778  Bal.Acc: 81.4%  F1: 0.8184  |  LR: 1.00e-04  (27.3s)
 Best checkpoint saved (val_loss: 0.4778)
Epoch [ 18/20]  Train Loss: 0.1744  Bal.Acc: 94.0%  F1: 0.9384  |  Val Loss: 0.4825  Bal.Acc: 84.0%  F1: 0.8271  |  LR: 1.00e-04  (27.5s)
Epoch [ 19/20]  Train Loss: 0.1737  Bal.Acc: 94.2%  F1: 0.9399  |  Val Loss: 0.6157  Bal.Acc: 78.6%  F1: 0.7921  |  LR: 1.00e-04  (24.8s)
Epoch [ 20/20]  Train Loss: 0.2017  Bal.Acc: 93.8%  F1: 0.9372  |  Val Loss: 0.4373  Bal.Acc: 83.8%  F1: 0.8219  |  LR: 1.00e-04  (24.6s)
 Best checkpoint saved (val_loss: 0.4373)

 Trening finished. Checkpoint: checkpoints/densenet121_fold2_best.pt
Log CSV: results/densenet121_fold2_training_log.csv
Best weights loaded from  20

Ewaluacja modelu: densenet121_fold2
----------------------------------------
  Balanced Accuracy:       83.83%
  F1 (macro):              0.8219
  Quadratic Cohen's Kappa: 0.9359
  ECE:                     0.0693
  Brier Score (mean):      0.0562

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.84      0.89      0.86        88
    Doubtful       0.75      0.57      0.65        81
        Mild       0.60      0.82      0.70        39
    Moderate       0.95      0.97      0.96        38
      Severe       0.94      0.94      0.94        35

    accuracy                           0.80       281
   macro avg       0.82      0.84      0.82       281
weighted avg       0.81      0.80      0.80       281

  Metryki zapisane: results/densenet121_fold2_metrics.json
  Prawdopodobieństwa zapisane: results/densenet121_fold2_test_probs.npz

--- densenet121 | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 3):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold3
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.4233  Bal.Acc: 40.8%  F1: 0.3774  |  Val Loss: 1.3009  Bal.Acc: 49.5%  F1: 0.4449  |  LR: 1.00e-04  (26.9s)
 Best checkpoint saved (val_loss: 1.3009)
Epoch [  2/20]  Train Loss: 1.0392  Bal.Acc: 61.4%  F1: 0.5755  |  Val Loss: 0.9477  Bal.Acc: 69.5%  F1: 0.6827  |  LR: 1.00e-04  (25.1s)
 Best checkpoint saved (val_loss: 0.9477)
Epoch [  3/20]  Train Loss: 0.8156  Bal.Acc: 71.3%  F1: 0.6952  |  Val Loss: 0.7297  Bal.Acc: 73.3%  F1: 0.7345  |  LR: 1.00e-04  (26.4s)
 Best checkpoint saved (val_loss: 0.7297)
Epoch [  4/20]  Train Loss: 0.6801  Bal.Acc: 75.9%  F1: 0.7562  |  Val Loss: 0.6863  Bal.Acc: 74.3%  F1: 0.7251  |  LR: 1.00e-04  (27.0s)
 Best checkpoint saved (val_loss: 0.6863)
Epoch [  5/20]  Train Loss: 0.5437  Bal.Acc: 81.9%  F1: 0.8122  |  Val Loss: 0.6289  Bal.Acc: 77.6%  F1: 0.7775  |  LR: 1.00e-04  (26.6s)
 Best checkpoint saved (val_loss: 0.6289)
Epoch [  6/20]  Train Loss: 0.4681  Bal.Acc: 83.9%  F1: 0.8290  |  Val Loss: 0.5625  Bal.Acc: 80.6%  F1: 0.7937  |  LR: 1.00e-04  (26.5s)
 Best checkpoint saved (val_loss: 0.5625)
Epoch [  7/20]  Train Loss: 0.4137  Bal.Acc: 85.3%  F1: 0.8442  |  Val Loss: 0.5682  Bal.Acc: 80.7%  F1: 0.8092  |  LR: 1.00e-04  (27.0s)
Epoch [  8/20]  Train Loss: 0.3523  Bal.Acc: 87.8%  F1: 0.8775  |  Val Loss: 0.5385  Bal.Acc: 80.3%  F1: 0.7829  |  LR: 1.00e-04  (25.0s)
 Best checkpoint saved (val_loss: 0.5385)
Epoch [  9/20]  Train Loss: 0.3333  Bal.Acc: 88.0%  F1: 0.8733  |  Val Loss: 0.5475  Bal.Acc: 82.1%  F1: 0.8260  |  LR: 1.00e-04  (26.3s)
Epoch [ 10/20]  Train Loss: 0.3067  Bal.Acc: 89.9%  F1: 0.8953  |  Val Loss: 0.5410  Bal.Acc: 79.1%  F1: 0.7750  |  LR: 1.00e-04  (25.6s)
Epoch [ 11/20]  Train Loss: 0.2611  Bal.Acc: 91.7%  F1: 0.9116  |  Val Loss: 0.6155  Bal.Acc: 76.4%  F1: 0.7540  |  LR: 1.00e-04  (25.0s)
Epoch [ 12/20]  Train Loss: 0.2175  Bal.Acc: 92.6%  F1: 0.9200  |  Val Loss: 0.5664  Bal.Acc: 79.4%  F1: 0.7959  |  LR: 5.00e-05  (25.0s)
Epoch [ 13/20]  Train Loss: 0.2241  Bal.Acc: 92.7%  F1: 0.9232  |  Val Loss: 0.5165  Bal.Acc: 83.5%  F1: 0.8294  |  LR: 5.00e-05  (25.5s)
 Best checkpoint saved (val_loss: 0.5165)
Epoch [ 14/20]  Train Loss: 0.1894  Bal.Acc: 93.9%  F1: 0.9390  |  Val Loss: 0.5089  Bal.Acc: 84.5%  F1: 0.8464  |  LR: 5.00e-05  (27.9s)
 Best checkpoint saved (val_loss: 0.5089)
Epoch [ 15/20]  Train Loss: 0.1780  Bal.Acc: 94.2%  F1: 0.9377  |  Val Loss: 0.4863  Bal.Acc: 84.7%  F1: 0.8520  |  LR: 5.00e-05  (27.4s)
 Best checkpoint saved (val_loss: 0.4863)
Epoch [ 16/20]  Train Loss: 0.1644  Bal.Acc: 94.4%  F1: 0.9389  |  Val Loss: 0.4905  Bal.Acc: 84.3%  F1: 0.8419  |  LR: 5.00e-05  (28.2s)
Epoch [ 17/20]  Train Loss: 0.1469  Bal.Acc: 95.7%  F1: 0.9567  |  Val Loss: 0.4378  Bal.Acc: 85.3%  F1: 0.8458  |  LR: 5.00e-05  (25.1s)
 Best checkpoint saved (val_loss: 0.4378)
Epoch [ 18/20]  Train Loss: 0.1552  Bal.Acc: 95.6%  F1: 0.9526  |  Val Loss: 0.4576  Bal.Acc: 85.5%  F1: 0.8516  |  LR: 5.00e-05  (26.9s)
Epoch [ 19/20]  Train Loss: 0.1822  Bal.Acc: 93.6%  F1: 0.9363  |  Val Loss: 0.4828  Bal.Acc: 84.8%  F1: 0.8416  |  LR: 5.00e-05  (25.0s)
Epoch [ 20/20]  Train Loss: 0.1469  Bal.Acc: 95.5%  F1: 0.9533  |  Val Loss: 0.4848  Bal.Acc: 83.8%  F1: 0.8271  |  LR: 5.00e-05  (24.6s)

 Trening finished. Checkpoint: checkpoints/densenet121_fold3_best.pt
Log CSV: results/densenet121_fold3_training_log.csv
Best weights loaded from  17

Ewaluacja modelu: densenet121_fold3
----------------------------------------
  Balanced Accuracy:       85.34%
  F1 (macro):              0.8458
  Quadratic Cohen's Kappa: 0.9329
  ECE:                     0.0644
  Brier Score (mean):      0.0481

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.91      0.91      0.91        87
    Doubtful       0.84      0.78      0.81        81
        Mild       0.74      0.78      0.76        40
    Moderate       0.85      0.89      0.87        37
      Severe       0.86      0.91      0.89        35

    accuracy                           0.85       280
   macro avg       0.84      0.85      0.85       280
weighted avg       0.85      0.85      0.85       280

  Metryki zapisane: results/densenet121_fold3_metrics.json
  Prawdopodobieństwa zapisane: results/densenet121_fold3_test_probs.npz

--- densenet121 | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 4):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold4
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.4482  Bal.Acc: 41.4%  F1: 0.3597  |  Val Loss: 1.2837  Bal.Acc: 50.4%  F1: 0.4958  |  LR: 1.00e-04  (24.8s)
 Best checkpoint saved (val_loss: 1.2837)
Epoch [  2/20]  Train Loss: 1.0628  Bal.Acc: 61.6%  F1: 0.5857  |  Val Loss: 1.0188  Bal.Acc: 59.2%  F1: 0.5854  |  LR: 1.00e-04  (25.2s)
 Best checkpoint saved (val_loss: 1.0188)
Epoch [  3/20]  Train Loss: 0.8089  Bal.Acc: 70.4%  F1: 0.6971  |  Val Loss: 0.8545  Bal.Acc: 63.0%  F1: 0.6313  |  LR: 1.00e-04  (27.6s)
 Best checkpoint saved (val_loss: 0.8545)
Epoch [  4/20]  Train Loss: 0.6630  Bal.Acc: 78.0%  F1: 0.7728  |  Val Loss: 0.6136  Bal.Acc: 80.4%  F1: 0.7921  |  LR: 1.00e-04  (27.2s)
 Best checkpoint saved (val_loss: 0.6136)
Epoch [  5/20]  Train Loss: 0.5369  Bal.Acc: 81.5%  F1: 0.8092  |  Val Loss: 0.5670  Bal.Acc: 81.0%  F1: 0.8048  |  LR: 1.00e-04  (26.9s)
 Best checkpoint saved (val_loss: 0.5670)
Epoch [  6/20]  Train Loss: 0.4992  Bal.Acc: 82.2%  F1: 0.8152  |  Val Loss: 0.6184  Bal.Acc: 74.3%  F1: 0.7204  |  LR: 1.00e-04  (27.7s)
Epoch [  7/20]  Train Loss: 0.4448  Bal.Acc: 84.2%  F1: 0.8351  |  Val Loss: 0.5539  Bal.Acc: 77.6%  F1: 0.7651  |  LR: 1.00e-04  (24.5s)
 Best checkpoint saved (val_loss: 0.5539)
Epoch [  8/20]  Train Loss: 0.3993  Bal.Acc: 87.4%  F1: 0.8667  |  Val Loss: 0.5104  Bal.Acc: 83.4%  F1: 0.8252  |  LR: 1.00e-04  (27.4s)
 Best checkpoint saved (val_loss: 0.5104)
Epoch [  9/20]  Train Loss: 0.3356  Bal.Acc: 88.1%  F1: 0.8763  |  Val Loss: 0.5076  Bal.Acc: 80.6%  F1: 0.8021  |  LR: 1.00e-04  (26.6s)
 Best checkpoint saved (val_loss: 0.5076)
Epoch [ 10/20]  Train Loss: 0.3136  Bal.Acc: 90.0%  F1: 0.8941  |  Val Loss: 0.5098  Bal.Acc: 82.2%  F1: 0.8072  |  LR: 1.00e-04  (26.4s)
Epoch [ 11/20]  Train Loss: 0.2667  Bal.Acc: 91.6%  F1: 0.9119  |  Val Loss: 0.5240  Bal.Acc: 83.0%  F1: 0.8174  |  LR: 1.00e-04  (24.5s)
Epoch [ 12/20]  Train Loss: 0.2809  Bal.Acc: 90.9%  F1: 0.9051  |  Val Loss: 0.6575  Bal.Acc: 77.2%  F1: 0.7565  |  LR: 1.00e-04  (24.9s)
Epoch [ 13/20]  Train Loss: 0.2555  Bal.Acc: 90.7%  F1: 0.9053  |  Val Loss: 0.5470  Bal.Acc: 81.2%  F1: 0.7878  |  LR: 5.00e-05  (25.0s)
Epoch [ 14/20]  Train Loss: 0.2156  Bal.Acc: 92.4%  F1: 0.9190  |  Val Loss: 0.4700  Bal.Acc: 84.1%  F1: 0.8386  |  LR: 5.00e-05  (25.2s)
 Best checkpoint saved (val_loss: 0.4700)
Epoch [ 15/20]  Train Loss: 0.1981  Bal.Acc: 94.1%  F1: 0.9376  |  Val Loss: 0.4653  Bal.Acc: 82.2%  F1: 0.8137  |  LR: 5.00e-05  (27.5s)
 Best checkpoint saved (val_loss: 0.4653)
Epoch [ 16/20]  Train Loss: 0.1673  Bal.Acc: 94.4%  F1: 0.9391  |  Val Loss: 0.4465  Bal.Acc: 83.9%  F1: 0.8354  |  LR: 5.00e-05  (28.1s)
 Best checkpoint saved (val_loss: 0.4465)
Epoch [ 17/20]  Train Loss: 0.1611  Bal.Acc: 94.4%  F1: 0.9412  |  Val Loss: 0.4507  Bal.Acc: 85.6%  F1: 0.8486  |  LR: 5.00e-05  (27.5s)
Epoch [ 18/20]  Train Loss: 0.1667  Bal.Acc: 94.8%  F1: 0.9429  |  Val Loss: 0.4815  Bal.Acc: 83.7%  F1: 0.8293  |  LR: 5.00e-05  (25.5s)
Epoch [ 19/20]  Train Loss: 0.1327  Bal.Acc: 96.1%  F1: 0.9588  |  Val Loss: 0.4443  Bal.Acc: 83.5%  F1: 0.8281  |  LR: 5.00e-05  (26.2s)
 Best checkpoint saved (val_loss: 0.4443)
Epoch [ 20/20]  Train Loss: 0.1289  Bal.Acc: 95.9%  F1: 0.9587  |  Val Loss: 0.4694  Bal.Acc: 85.4%  F1: 0.8520  |  LR: 5.00e-05  (28.3s)

 Trening finished. Checkpoint: checkpoints/densenet121_fold4_best.pt
Log CSV: results/densenet121_fold4_training_log.csv
Best weights loaded from  19

Ewaluacja modelu: densenet121_fold4
----------------------------------------
  Balanced Accuracy:       83.54%
  F1 (macro):              0.8281
  Quadratic Cohen's Kappa: 0.9292
  ECE:                     0.0592
  Brier Score (mean):      0.0475

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.95      0.89        87
    Doubtful       0.86      0.69      0.77        81
        Mild       0.76      0.70      0.73        40
    Moderate       0.85      0.95      0.90        37
      Severe       0.84      0.89      0.86        35

    accuracy                           0.83       280
   macro avg       0.83      0.84      0.83       280
weighted avg       0.83      0.83      0.83       280

  Metryki zapisane: results/densenet121_fold4_metrics.json
  Prawdopodobieństwa zapisane: results/densenet121_fold4_test_probs.npz

--- densenet121 | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 5):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.420  (count = 158)
      Klasa 3 (Moderate): waga = 1.496  (count = 150)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold5
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.4498  Bal.Acc: 37.6%  F1: 0.3557  |  Val Loss: 1.3777  Bal.Acc: 42.9%  F1: 0.4243  |  LR: 1.00e-04  (25.9s)
 Best checkpoint saved (val_loss: 1.3777)
Epoch [  2/20]  Train Loss: 1.0894  Bal.Acc: 59.3%  F1: 0.5694  |  Val Loss: 1.0395  Bal.Acc: 58.1%  F1: 0.6010  |  LR: 1.00e-04  (26.0s)
 Best checkpoint saved (val_loss: 1.0395)
Epoch [  3/20]  Train Loss: 0.8522  Bal.Acc: 69.9%  F1: 0.6890  |  Val Loss: 0.8759  Bal.Acc: 65.7%  F1: 0.6568  |  LR: 1.00e-04  (27.2s)
 Best checkpoint saved (val_loss: 0.8759)
Epoch [  4/20]  Train Loss: 0.6801  Bal.Acc: 75.4%  F1: 0.7434  |  Val Loss: 0.7211  Bal.Acc: 76.7%  F1: 0.7712  |  LR: 1.00e-04  (27.3s)
 Best checkpoint saved (val_loss: 0.7211)
Epoch [  5/20]  Train Loss: 0.5815  Bal.Acc: 78.8%  F1: 0.7832  |  Val Loss: 0.6720  Bal.Acc: 76.5%  F1: 0.7658  |  LR: 1.00e-04  (27.9s)
 Best checkpoint saved (val_loss: 0.6720)
Epoch [  6/20]  Train Loss: 0.5228  Bal.Acc: 80.4%  F1: 0.7967  |  Val Loss: 0.7372  Bal.Acc: 72.3%  F1: 0.7282  |  LR: 1.00e-04  (27.1s)
Epoch [  7/20]  Train Loss: 0.4368  Bal.Acc: 84.1%  F1: 0.8353  |  Val Loss: 0.6964  Bal.Acc: 76.0%  F1: 0.7759  |  LR: 1.00e-04  (25.2s)
Epoch [  8/20]  Train Loss: 0.4142  Bal.Acc: 85.0%  F1: 0.8460  |  Val Loss: 0.5662  Bal.Acc: 81.7%  F1: 0.8113  |  LR: 1.00e-04  (25.0s)
 Best checkpoint saved (val_loss: 0.5662)
Epoch [  9/20]  Train Loss: 0.3797  Bal.Acc: 86.2%  F1: 0.8530  |  Val Loss: 0.6136  Bal.Acc: 77.3%  F1: 0.7741  |  LR: 1.00e-04  (27.1s)
Epoch [ 10/20]  Train Loss: 0.3170  Bal.Acc: 88.5%  F1: 0.8798  |  Val Loss: 0.5677  Bal.Acc: 81.0%  F1: 0.8193  |  LR: 1.00e-04  (24.7s)
Epoch [ 11/20]  Train Loss: 0.2838  Bal.Acc: 90.1%  F1: 0.8967  |  Val Loss: 0.5798  Bal.Acc: 78.8%  F1: 0.7909  |  LR: 1.00e-04  (24.9s)
Epoch [ 12/20]  Train Loss: 0.2997  Bal.Acc: 89.8%  F1: 0.8938  |  Val Loss: 0.5934  Bal.Acc: 79.5%  F1: 0.8039  |  LR: 5.00e-05  (25.0s)
Epoch [ 13/20]  Train Loss: 0.2430  Bal.Acc: 92.1%  F1: 0.9197  |  Val Loss: 0.5238  Bal.Acc: 84.4%  F1: 0.8484  |  LR: 5.00e-05  (25.1s)
 Best checkpoint saved (val_loss: 0.5238)
Epoch [ 14/20]  Train Loss: 0.2053  Bal.Acc: 93.9%  F1: 0.9346  |  Val Loss: 0.5985  Bal.Acc: 81.6%  F1: 0.8234  |  LR: 5.00e-05  (26.6s)
Epoch [ 15/20]  Train Loss: 0.2070  Bal.Acc: 93.3%  F1: 0.9305  |  Val Loss: 0.6117  Bal.Acc: 81.6%  F1: 0.8232  |  LR: 5.00e-05  (25.0s)
Epoch [ 16/20]  Train Loss: 0.1886  Bal.Acc: 93.6%  F1: 0.9347  |  Val Loss: 0.6064  Bal.Acc: 81.1%  F1: 0.8249  |  LR: 5.00e-05  (25.0s)
Epoch [ 17/20]  Train Loss: 0.1758  Bal.Acc: 94.1%  F1: 0.9362  |  Val Loss: 0.5625  Bal.Acc: 83.9%  F1: 0.8483  |  LR: 2.50e-05  (24.9s)
Epoch [ 18/20]  Train Loss: 0.1447  Bal.Acc: 95.4%  F1: 0.9526  |  Val Loss: 0.5813  Bal.Acc: 83.3%  F1: 0.8431  |  LR: 2.50e-05  (25.3s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.5238

 Trening finished. Checkpoint: checkpoints/densenet121_fold5_best.pt
Log CSV: results/densenet121_fold5_training_log.csv
Best weights loaded from  13

Ewaluacja modelu: densenet121_fold5
----------------------------------------
  Balanced Accuracy:       84.43%
  F1 (macro):              0.8484
  Quadratic Cohen's Kappa: 0.9344
  ECE:                     0.0199
  Brier Score (mean):      0.0485

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.94      0.92        87
    Doubtful       0.80      0.78      0.79        81
        Mild       0.69      0.69      0.69        39
    Moderate       1.00      0.89      0.94        38
      Severe       0.89      0.91      0.90        35

    accuracy                           0.85       280
   macro avg       0.85      0.84      0.85       280
weighted avg       0.85      0.85      0.85       280

  Metryki zapisane: results/densenet121_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/densenet121_fold5_test_probs.npz

✅ ZAKOŃCZONO: densenet121. Średnia Kappa z 5 foldów: 0.9337 ±0.0025

================================================================================
🚀 ROZPOCZĘCIE TRENINGU MODELU: mobilenetv3_large
================================================================================

--- mobilenetv3_large | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 1):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
model.safetensors: 100% 22.1M/22.1M [00:00<00:00, 36.0MB/s]
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold1
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.9061  Bal.Acc: 40.6%  F1: 0.3921  |  Val Loss: 1.8494  Bal.Acc: 39.5%  F1: 0.3635  |  LR: 1.00e-04  (15.5s)
 Best checkpoint saved (val_loss: 1.8494)
Epoch [  2/20]  Train Loss: 1.2702  Bal.Acc: 55.6%  F1: 0.5325  |  Val Loss: 1.2623  Bal.Acc: 54.0%  F1: 0.5329  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 1.2623)
Epoch [  3/20]  Train Loss: 0.9562  Bal.Acc: 66.2%  F1: 0.6487  |  Val Loss: 1.1066  Bal.Acc: 65.0%  F1: 0.6074  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 1.1066)
Epoch [  4/20]  Train Loss: 0.7668  Bal.Acc: 72.1%  F1: 0.7138  |  Val Loss: 0.9068  Bal.Acc: 69.5%  F1: 0.6947  |  LR: 1.00e-04  (18.4s)
 Best checkpoint saved (val_loss: 0.9068)
Epoch [  5/20]  Train Loss: 0.7220  Bal.Acc: 73.7%  F1: 0.7202  |  Val Loss: 0.8381  Bal.Acc: 71.3%  F1: 0.7030  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 0.8381)
Epoch [  6/20]  Train Loss: 0.6100  Bal.Acc: 78.2%  F1: 0.7719  |  Val Loss: 0.7970  Bal.Acc: 72.4%  F1: 0.7322  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.7970)
Epoch [  7/20]  Train Loss: 0.5273  Bal.Acc: 80.4%  F1: 0.7959  |  Val Loss: 0.7622  Bal.Acc: 72.6%  F1: 0.7156  |  LR: 1.00e-04  (17.6s)
 Best checkpoint saved (val_loss: 0.7622)
Epoch [  8/20]  Train Loss: 0.5513  Bal.Acc: 81.1%  F1: 0.8067  |  Val Loss: 0.7705  Bal.Acc: 75.3%  F1: 0.7363  |  LR: 1.00e-04  (18.1s)
Epoch [  9/20]  Train Loss: 0.4181  Bal.Acc: 84.8%  F1: 0.8408  |  Val Loss: 0.7420  Bal.Acc: 76.2%  F1: 0.7635  |  LR: 1.00e-04  (15.7s)
 Best checkpoint saved (val_loss: 0.7420)
Epoch [ 10/20]  Train Loss: 0.4117  Bal.Acc: 85.7%  F1: 0.8470  |  Val Loss: 0.7152  Bal.Acc: 77.8%  F1: 0.7752  |  LR: 1.00e-04  (18.5s)
 Best checkpoint saved (val_loss: 0.7152)
Epoch [ 11/20]  Train Loss: 0.3929  Bal.Acc: 82.9%  F1: 0.8230  |  Val Loss: 0.6575  Bal.Acc: 78.1%  F1: 0.7769  |  LR: 1.00e-04  (17.6s)
 Best checkpoint saved (val_loss: 0.6575)
Epoch [ 12/20]  Train Loss: 0.3505  Bal.Acc: 87.0%  F1: 0.8635  |  Val Loss: 0.6240  Bal.Acc: 78.3%  F1: 0.7764  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.6240)
Epoch [ 13/20]  Train Loss: 0.3625  Bal.Acc: 87.2%  F1: 0.8693  |  Val Loss: 0.6624  Bal.Acc: 80.7%  F1: 0.8085  |  LR: 1.00e-04  (18.3s)
Epoch [ 14/20]  Train Loss: 0.2771  Bal.Acc: 89.3%  F1: 0.8864  |  Val Loss: 0.7425  Bal.Acc: 78.5%  F1: 0.7843  |  LR: 1.00e-04  (17.3s)
Epoch [ 15/20]  Train Loss: 0.2654  Bal.Acc: 90.8%  F1: 0.9063  |  Val Loss: 0.6632  Bal.Acc: 77.1%  F1: 0.7632  |  LR: 1.00e-04  (15.8s)
Epoch [ 16/20]  Train Loss: 0.3447  Bal.Acc: 87.3%  F1: 0.8644  |  Val Loss: 0.7843  Bal.Acc: 78.3%  F1: 0.7718  |  LR: 5.00e-05  (16.0s)
Epoch [ 17/20]  Train Loss: 0.2899  Bal.Acc: 89.2%  F1: 0.8871  |  Val Loss: 0.6540  Bal.Acc: 79.5%  F1: 0.7923  |  LR: 5.00e-05  (16.0s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.6240

 Trening finished. Checkpoint: checkpoints/mobilenetv3_large_fold1_best.pt
Log CSV: results/mobilenetv3_large_fold1_training_log.csv
Best weights loaded from  12

Ewaluacja modelu: mobilenetv3_large_fold1
----------------------------------------
  Balanced Accuracy:       78.29%
  F1 (macro):              0.7764
  Quadratic Cohen's Kappa: 0.9047
  ECE:                     0.1008
  Brier Score (mean):      0.0661

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.78      0.89      0.83        88
    Doubtful       0.72      0.58      0.64        81
        Mild       0.61      0.64      0.62        39
    Moderate       0.88      0.92      0.90        38
      Severe       0.89      0.89      0.89        35

    accuracy                           0.77       281
   macro avg       0.77      0.78      0.78       281
weighted avg       0.77      0.77      0.76       281

  Metryki zapisane: results/mobilenetv3_large_fold1_metrics.json
  Prawdopodobieństwa zapisane: results/mobilenetv3_large_fold1_test_probs.npz

--- mobilenetv3_large | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 2):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold2
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 2.0500  Bal.Acc: 39.9%  F1: 0.3808  |  Val Loss: 2.2441  Bal.Acc: 38.7%  F1: 0.3935  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 2.2441)
Epoch [  2/20]  Train Loss: 1.2755  Bal.Acc: 56.1%  F1: 0.5444  |  Val Loss: 1.5054  Bal.Acc: 51.3%  F1: 0.5029  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 1.5054)
Epoch [  3/20]  Train Loss: 1.0055  Bal.Acc: 64.4%  F1: 0.6308  |  Val Loss: 1.1565  Bal.Acc: 60.8%  F1: 0.5998  |  LR: 1.00e-04  (17.5s)
 Best checkpoint saved (val_loss: 1.1565)
Epoch [  4/20]  Train Loss: 0.8530  Bal.Acc: 69.1%  F1: 0.6707  |  Val Loss: 1.2317  Bal.Acc: 61.5%  F1: 0.6089  |  LR: 1.00e-04  (18.6s)
Epoch [  5/20]  Train Loss: 0.7560  Bal.Acc: 72.2%  F1: 0.7121  |  Val Loss: 0.9909  Bal.Acc: 64.9%  F1: 0.6519  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 0.9909)
Epoch [  6/20]  Train Loss: 0.6036  Bal.Acc: 78.3%  F1: 0.7750  |  Val Loss: 0.9097  Bal.Acc: 70.8%  F1: 0.7009  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.9097)
Epoch [  7/20]  Train Loss: 0.5576  Bal.Acc: 79.7%  F1: 0.7875  |  Val Loss: 0.8685  Bal.Acc: 71.5%  F1: 0.7003  |  LR: 1.00e-04  (17.7s)
 Best checkpoint saved (val_loss: 0.8685)
Epoch [  8/20]  Train Loss: 0.5326  Bal.Acc: 81.4%  F1: 0.8072  |  Val Loss: 0.8811  Bal.Acc: 71.1%  F1: 0.7090  |  LR: 1.00e-04  (17.7s)
Epoch [  9/20]  Train Loss: 0.4372  Bal.Acc: 83.5%  F1: 0.8240  |  Val Loss: 0.9079  Bal.Acc: 70.2%  F1: 0.7040  |  LR: 1.00e-04  (15.7s)
Epoch [ 10/20]  Train Loss: 0.4185  Bal.Acc: 83.9%  F1: 0.8352  |  Val Loss: 0.8097  Bal.Acc: 73.1%  F1: 0.7444  |  LR: 1.00e-04  (15.4s)
 Best checkpoint saved (val_loss: 0.8097)
Epoch [ 11/20]  Train Loss: 0.4051  Bal.Acc: 85.0%  F1: 0.8419  |  Val Loss: 0.7775  Bal.Acc: 72.1%  F1: 0.7224  |  LR: 1.00e-04  (17.4s)
 Best checkpoint saved (val_loss: 0.7775)
Epoch [ 12/20]  Train Loss: 0.4223  Bal.Acc: 85.1%  F1: 0.8394  |  Val Loss: 0.7380  Bal.Acc: 74.3%  F1: 0.7426  |  LR: 1.00e-04  (17.5s)
 Best checkpoint saved (val_loss: 0.7380)
Epoch [ 13/20]  Train Loss: 0.2661  Bal.Acc: 90.3%  F1: 0.8959  |  Val Loss: 0.6414  Bal.Acc: 76.8%  F1: 0.7727  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 0.6414)
Epoch [ 14/20]  Train Loss: 0.3218  Bal.Acc: 88.3%  F1: 0.8805  |  Val Loss: 0.7237  Bal.Acc: 73.7%  F1: 0.7273  |  LR: 1.00e-04  (17.9s)
Epoch [ 15/20]  Train Loss: 0.2984  Bal.Acc: 88.2%  F1: 0.8789  |  Val Loss: 0.6796  Bal.Acc: 76.8%  F1: 0.7654  |  LR: 1.00e-04  (15.9s)
Epoch [ 16/20]  Train Loss: 0.2926  Bal.Acc: 90.7%  F1: 0.9031  |  Val Loss: 0.6071  Bal.Acc: 80.2%  F1: 0.8010  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 0.6071)
Epoch [ 17/20]  Train Loss: 0.2465  Bal.Acc: 91.1%  F1: 0.9038  |  Val Loss: 0.6981  Bal.Acc: 75.1%  F1: 0.7588  |  LR: 1.00e-04  (18.1s)
Epoch [ 18/20]  Train Loss: 0.2162  Bal.Acc: 92.0%  F1: 0.9188  |  Val Loss: 0.5988  Bal.Acc: 76.7%  F1: 0.7726  |  LR: 1.00e-04  (16.4s)
 Best checkpoint saved (val_loss: 0.5988)
Epoch [ 19/20]  Train Loss: 0.2183  Bal.Acc: 92.2%  F1: 0.9220  |  Val Loss: 0.6043  Bal.Acc: 79.6%  F1: 0.7969  |  LR: 1.00e-04  (18.2s)
Epoch [ 20/20]  Train Loss: 0.3058  Bal.Acc: 89.2%  F1: 0.8817  |  Val Loss: 0.6952  Bal.Acc: 78.7%  F1: 0.7815  |  LR: 1.00e-04  (16.0s)

 Trening finished. Checkpoint: checkpoints/mobilenetv3_large_fold2_best.pt
Log CSV: results/mobilenetv3_large_fold2_training_log.csv
Best weights loaded from  18

Ewaluacja modelu: mobilenetv3_large_fold2
----------------------------------------
  Balanced Accuracy:       76.72%
  F1 (macro):              0.7726
  Quadratic Cohen's Kappa: 0.8525
  ECE:                     0.0920
  Brier Score (mean):      0.0612

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.86      0.84        88
    Doubtful       0.73      0.75      0.74        81
        Mild       0.68      0.64      0.66        39
    Moderate       0.83      0.92      0.88        38
      Severe       0.85      0.66      0.74        35

    accuracy                           0.78       281
   macro avg       0.78      0.77      0.77       281
weighted avg       0.78      0.78      0.78       281

  Metryki zapisane: results/mobilenetv3_large_fold2_metrics.json
  Prawdopodobieństwa zapisane: results/mobilenetv3_large_fold2_test_probs.npz

--- mobilenetv3_large | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 3):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold3
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 2.1200  Bal.Acc: 38.2%  F1: 0.3614  |  Val Loss: 2.3422  Bal.Acc: 29.3%  F1: 0.2814  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 2.3422)
Epoch [  2/20]  Train Loss: 1.2435  Bal.Acc: 56.8%  F1: 0.5550  |  Val Loss: 1.2024  Bal.Acc: 55.3%  F1: 0.5519  |  LR: 1.00e-04  (16.5s)
 Best checkpoint saved (val_loss: 1.2024)
Epoch [  3/20]  Train Loss: 0.9903  Bal.Acc: 64.2%  F1: 0.6286  |  Val Loss: 0.9376  Bal.Acc: 62.4%  F1: 0.6188  |  LR: 1.00e-04  (18.1s)
 Best checkpoint saved (val_loss: 0.9376)
Epoch [  4/20]  Train Loss: 0.8617  Bal.Acc: 68.4%  F1: 0.6674  |  Val Loss: 0.9611  Bal.Acc: 67.4%  F1: 0.6808  |  LR: 1.00e-04  (18.5s)
Epoch [  5/20]  Train Loss: 0.6535  Bal.Acc: 76.6%  F1: 0.7540  |  Val Loss: 0.8715  Bal.Acc: 71.2%  F1: 0.7208  |  LR: 1.00e-04  (15.6s)
 Best checkpoint saved (val_loss: 0.8715)
Epoch [  6/20]  Train Loss: 0.6275  Bal.Acc: 77.8%  F1: 0.7735  |  Val Loss: 0.7731  Bal.Acc: 71.1%  F1: 0.7034  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 0.7731)
Epoch [  7/20]  Train Loss: 0.5305  Bal.Acc: 79.8%  F1: 0.7875  |  Val Loss: 0.7839  Bal.Acc: 72.3%  F1: 0.7302  |  LR: 1.00e-04  (18.2s)
Epoch [  8/20]  Train Loss: 0.5122  Bal.Acc: 81.2%  F1: 0.8022  |  Val Loss: 0.7302  Bal.Acc: 75.8%  F1: 0.7684  |  LR: 1.00e-04  (16.1s)
 Best checkpoint saved (val_loss: 0.7302)
Epoch [  9/20]  Train Loss: 0.4489  Bal.Acc: 82.0%  F1: 0.8139  |  Val Loss: 0.7425  Bal.Acc: 72.5%  F1: 0.7259  |  LR: 1.00e-04  (18.4s)
Epoch [ 10/20]  Train Loss: 0.4013  Bal.Acc: 85.7%  F1: 0.8447  |  Val Loss: 0.6712  Bal.Acc: 75.4%  F1: 0.7464  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 0.6712)
Epoch [ 11/20]  Train Loss: 0.3444  Bal.Acc: 87.1%  F1: 0.8656  |  Val Loss: 0.6471  Bal.Acc: 77.6%  F1: 0.7742  |  LR: 1.00e-04  (18.3s)
 Best checkpoint saved (val_loss: 0.6471)
Epoch [ 12/20]  Train Loss: 0.3406  Bal.Acc: 88.1%  F1: 0.8741  |  Val Loss: 0.6590  Bal.Acc: 76.1%  F1: 0.7728  |  LR: 1.00e-04  (17.9s)
Epoch [ 13/20]  Train Loss: 0.2677  Bal.Acc: 90.7%  F1: 0.9053  |  Val Loss: 0.6843  Bal.Acc: 77.9%  F1: 0.7790  |  LR: 1.00e-04  (15.9s)
Epoch [ 14/20]  Train Loss: 0.2917  Bal.Acc: 89.6%  F1: 0.8924  |  Val Loss: 0.7174  Bal.Acc: 76.6%  F1: 0.7782  |  LR: 1.00e-04  (15.8s)
Epoch [ 15/20]  Train Loss: 0.2624  Bal.Acc: 90.9%  F1: 0.9025  |  Val Loss: 0.6531  Bal.Acc: 78.5%  F1: 0.7958  |  LR: 5.00e-05  (16.1s)
Epoch [ 16/20]  Train Loss: 0.2589  Bal.Acc: 90.6%  F1: 0.9022  |  Val Loss: 0.5780  Bal.Acc: 82.2%  F1: 0.8231  |  LR: 5.00e-05  (15.8s)
 Best checkpoint saved (val_loss: 0.5780)
Epoch [ 17/20]  Train Loss: 0.2067  Bal.Acc: 92.4%  F1: 0.9196  |  Val Loss: 0.5903  Bal.Acc: 80.6%  F1: 0.8122  |  LR: 5.00e-05  (18.1s)
Epoch [ 18/20]  Train Loss: 0.2255  Bal.Acc: 91.6%  F1: 0.9074  |  Val Loss: 0.6034  Bal.Acc: 80.1%  F1: 0.8072  |  LR: 5.00e-05  (15.8s)
Epoch [ 19/20]  Train Loss: 0.1994  Bal.Acc: 92.6%  F1: 0.9286  |  Val Loss: 0.5641  Bal.Acc: 82.7%  F1: 0.8242  |  LR: 5.00e-05  (15.9s)
 Best checkpoint saved (val_loss: 0.5641)
Epoch [ 20/20]  Train Loss: 0.2263  Bal.Acc: 91.7%  F1: 0.9136  |  Val Loss: 0.5826  Bal.Acc: 82.5%  F1: 0.8264  |  LR: 5.00e-05  (18.4s)

 Trening finished. Checkpoint: checkpoints/mobilenetv3_large_fold3_best.pt
Log CSV: results/mobilenetv3_large_fold3_training_log.csv
Best weights loaded from  19

Ewaluacja modelu: mobilenetv3_large_fold3
----------------------------------------
  Balanced Accuracy:       82.68%
  F1 (macro):              0.8242
  Quadratic Cohen's Kappa: 0.9273
  ECE:                     0.0805
  Brier Score (mean):      0.0557

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.88      0.80      0.84        87
    Doubtful       0.79      0.80      0.80        81
        Mild       0.63      0.78      0.70        40
    Moderate       0.89      0.84      0.86        37
      Severe       0.94      0.91      0.93        35

    accuracy                           0.82       280
   macro avg       0.83      0.83      0.82       280
weighted avg       0.83      0.82      0.82       280

  Metryki zapisane: results/mobilenetv3_large_fold3_metrics.json
  Prawdopodobieństwa zapisane: results/mobilenetv3_large_fold3_test_probs.npz

--- mobilenetv3_large | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 4):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold4
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 2.2253  Bal.Acc: 38.4%  F1: 0.3672  |  Val Loss: 1.8877  Bal.Acc: 42.4%  F1: 0.4021  |  LR: 1.00e-04  (16.3s)
 Best checkpoint saved (val_loss: 1.8877)
Epoch [  2/20]  Train Loss: 1.2768  Bal.Acc: 55.3%  F1: 0.5443  |  Val Loss: 1.2291  Bal.Acc: 61.9%  F1: 0.5937  |  LR: 1.00e-04  (16.2s)
 Best checkpoint saved (val_loss: 1.2291)
Epoch [  3/20]  Train Loss: 0.9817  Bal.Acc: 65.3%  F1: 0.6352  |  Val Loss: 1.0872  Bal.Acc: 62.4%  F1: 0.6092  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 1.0872)
Epoch [  4/20]  Train Loss: 0.9493  Bal.Acc: 66.6%  F1: 0.6459  |  Val Loss: 0.9210  Bal.Acc: 66.6%  F1: 0.6470  |  LR: 1.00e-04  (17.6s)
 Best checkpoint saved (val_loss: 0.9210)
Epoch [  5/20]  Train Loss: 0.6693  Bal.Acc: 73.3%  F1: 0.7208  |  Val Loss: 0.7989  Bal.Acc: 73.4%  F1: 0.7271  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 0.7989)
Epoch [  6/20]  Train Loss: 0.5869  Bal.Acc: 77.8%  F1: 0.7666  |  Val Loss: 0.7358  Bal.Acc: 73.1%  F1: 0.7120  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.7358)
Epoch [  7/20]  Train Loss: 0.5803  Bal.Acc: 78.9%  F1: 0.7769  |  Val Loss: 0.7039  Bal.Acc: 75.3%  F1: 0.7317  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 0.7039)
Epoch [  8/20]  Train Loss: 0.4751  Bal.Acc: 83.2%  F1: 0.8191  |  Val Loss: 0.6972  Bal.Acc: 76.3%  F1: 0.7434  |  LR: 1.00e-04  (18.2s)
 Best checkpoint saved (val_loss: 0.6972)
Epoch [  9/20]  Train Loss: 0.5046  Bal.Acc: 81.1%  F1: 0.8035  |  Val Loss: 0.6494  Bal.Acc: 76.7%  F1: 0.7544  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 0.6494)
Epoch [ 10/20]  Train Loss: 0.4086  Bal.Acc: 85.5%  F1: 0.8449  |  Val Loss: 0.6996  Bal.Acc: 74.8%  F1: 0.7560  |  LR: 1.00e-04  (17.9s)
Epoch [ 11/20]  Train Loss: 0.3995  Bal.Acc: 83.9%  F1: 0.8314  |  Val Loss: 0.6977  Bal.Acc: 75.4%  F1: 0.7581  |  LR: 1.00e-04  (15.7s)
Epoch [ 12/20]  Train Loss: 0.3177  Bal.Acc: 87.7%  F1: 0.8735  |  Val Loss: 0.6778  Bal.Acc: 77.3%  F1: 0.7602  |  LR: 1.00e-04  (15.8s)
Epoch [ 13/20]  Train Loss: 0.3604  Bal.Acc: 85.8%  F1: 0.8459  |  Val Loss: 0.7265  Bal.Acc: 75.6%  F1: 0.7575  |  LR: 5.00e-05  (15.6s)
Epoch [ 14/20]  Train Loss: 0.2784  Bal.Acc: 89.3%  F1: 0.8862  |  Val Loss: 0.6071  Bal.Acc: 79.3%  F1: 0.7898  |  LR: 5.00e-05  (15.4s)
 Best checkpoint saved (val_loss: 0.6071)
Epoch [ 15/20]  Train Loss: 0.2486  Bal.Acc: 90.8%  F1: 0.9017  |  Val Loss: 0.5677  Bal.Acc: 79.5%  F1: 0.7956  |  LR: 5.00e-05  (17.6s)
 Best checkpoint saved (val_loss: 0.5677)
Epoch [ 16/20]  Train Loss: 0.2167  Bal.Acc: 91.8%  F1: 0.9162  |  Val Loss: 0.5577  Bal.Acc: 80.1%  F1: 0.8039  |  LR: 5.00e-05  (18.0s)
 Best checkpoint saved (val_loss: 0.5577)
Epoch [ 17/20]  Train Loss: 0.2262  Bal.Acc: 91.2%  F1: 0.9042  |  Val Loss: 0.5518  Bal.Acc: 82.9%  F1: 0.8297  |  LR: 5.00e-05  (17.9s)
 Best checkpoint saved (val_loss: 0.5518)
Epoch [ 18/20]  Train Loss: 0.2356  Bal.Acc: 90.9%  F1: 0.9049  |  Val Loss: 0.5186  Bal.Acc: 83.9%  F1: 0.8385  |  LR: 5.00e-05  (18.0s)
 Best checkpoint saved (val_loss: 0.5186)
Epoch [ 19/20]  Train Loss: 0.2065  Bal.Acc: 92.6%  F1: 0.9212  |  Val Loss: 0.5529  Bal.Acc: 80.1%  F1: 0.7982  |  LR: 5.00e-05  (17.7s)
Epoch [ 20/20]  Train Loss: 0.1949  Bal.Acc: 93.7%  F1: 0.9356  |  Val Loss: 0.5157  Bal.Acc: 82.3%  F1: 0.8164  |  LR: 5.00e-05  (15.6s)
 Best checkpoint saved (val_loss: 0.5157)

 Trening finished. Checkpoint: checkpoints/mobilenetv3_large_fold4_best.pt
Log CSV: results/mobilenetv3_large_fold4_training_log.csv
Best weights loaded from  20

Ewaluacja modelu: mobilenetv3_large_fold4
----------------------------------------
  Balanced Accuracy:       82.34%
  F1 (macro):              0.8164
  Quadratic Cohen's Kappa: 0.9264
  ECE:                     0.0821
  Brier Score (mean):      0.0568

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.86      0.87      0.87        87
    Doubtful       0.81      0.72      0.76        81
        Mild       0.64      0.75      0.69        40
    Moderate       0.87      0.89      0.88        37
      Severe       0.89      0.89      0.89        35

    accuracy                           0.81       280
   macro avg       0.81      0.82      0.82       280
weighted avg       0.82      0.81      0.81       280

  Metryki zapisane: results/mobilenetv3_large_fold4_metrics.json
  Prawdopodobieństwa zapisane: results/mobilenetv3_large_fold4_test_probs.npz

--- mobilenetv3_large | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 5):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.420  (count = 158)
      Klasa 3 (Moderate): waga = 1.496  (count = 150)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold5
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.9723  Bal.Acc: 40.9%  F1: 0.3900  |  Val Loss: 1.8875  Bal.Acc: 39.9%  F1: 0.3729  |  LR: 1.00e-04  (17.4s)
 Best checkpoint saved (val_loss: 1.8875)
Epoch [  2/20]  Train Loss: 1.1770  Bal.Acc: 58.6%  F1: 0.5742  |  Val Loss: 1.3192  Bal.Acc: 57.8%  F1: 0.5700  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 1.3192)
Epoch [  3/20]  Train Loss: 0.9393  Bal.Acc: 65.9%  F1: 0.6482  |  Val Loss: 1.2712  Bal.Acc: 57.9%  F1: 0.5845  |  LR: 1.00e-04  (18.0s)
 Best checkpoint saved (val_loss: 1.2712)
Epoch [  4/20]  Train Loss: 0.8696  Bal.Acc: 69.3%  F1: 0.6766  |  Val Loss: 1.1348  Bal.Acc: 65.3%  F1: 0.6617  |  LR: 1.00e-04  (17.9s)
 Best checkpoint saved (val_loss: 1.1348)
Epoch [  5/20]  Train Loss: 0.6906  Bal.Acc: 74.7%  F1: 0.7328  |  Val Loss: 0.9879  Bal.Acc: 67.4%  F1: 0.6833  |  LR: 1.00e-04  (17.6s)
 Best checkpoint saved (val_loss: 0.9879)
Epoch [  6/20]  Train Loss: 0.5969  Bal.Acc: 77.6%  F1: 0.7619  |  Val Loss: 0.9905  Bal.Acc: 66.8%  F1: 0.6737  |  LR: 1.00e-04  (18.1s)
Epoch [  7/20]  Train Loss: 0.5500  Bal.Acc: 79.3%  F1: 0.7872  |  Val Loss: 0.8618  Bal.Acc: 74.4%  F1: 0.7474  |  LR: 1.00e-04  (15.4s)
 Best checkpoint saved (val_loss: 0.8618)
Epoch [  8/20]  Train Loss: 0.4956  Bal.Acc: 81.1%  F1: 0.8039  |  Val Loss: 0.8813  Bal.Acc: 71.9%  F1: 0.7259  |  LR: 1.00e-04  (17.6s)
Epoch [  9/20]  Train Loss: 0.4476  Bal.Acc: 82.7%  F1: 0.8197  |  Val Loss: 0.8235  Bal.Acc: 76.7%  F1: 0.7670  |  LR: 1.00e-04  (15.8s)
 Best checkpoint saved (val_loss: 0.8235)
Epoch [ 10/20]  Train Loss: 0.3808  Bal.Acc: 85.3%  F1: 0.8472  |  Val Loss: 0.8284  Bal.Acc: 72.6%  F1: 0.7256  |  LR: 1.00e-04  (17.2s)
Epoch [ 11/20]  Train Loss: 0.3914  Bal.Acc: 84.7%  F1: 0.8405  |  Val Loss: 0.8087  Bal.Acc: 74.1%  F1: 0.7470  |  LR: 1.00e-04  (15.2s)
 Best checkpoint saved (val_loss: 0.8087)
Epoch [ 12/20]  Train Loss: 0.3765  Bal.Acc: 85.7%  F1: 0.8506  |  Val Loss: 0.7569  Bal.Acc: 78.5%  F1: 0.7935  |  LR: 1.00e-04  (17.8s)
 Best checkpoint saved (val_loss: 0.7569)
Epoch [ 13/20]  Train Loss: 0.3136  Bal.Acc: 88.1%  F1: 0.8775  |  Val Loss: 0.7799  Bal.Acc: 77.4%  F1: 0.7758  |  LR: 1.00e-04  (17.6s)
Epoch [ 14/20]  Train Loss: 0.3322  Bal.Acc: 87.0%  F1: 0.8632  |  Val Loss: 0.8490  Bal.Acc: 76.7%  F1: 0.7695  |  LR: 1.00e-04  (15.5s)
Epoch [ 15/20]  Train Loss: 0.2798  Bal.Acc: 89.7%  F1: 0.8889  |  Val Loss: 0.9217  Bal.Acc: 76.7%  F1: 0.7718  |  LR: 1.00e-04  (15.5s)
Epoch [ 16/20]  Train Loss: 0.2857  Bal.Acc: 89.0%  F1: 0.8852  |  Val Loss: 0.8525  Bal.Acc: 74.2%  F1: 0.7502  |  LR: 5.00e-05  (15.9s)
Epoch [ 17/20]  Train Loss: 0.2561  Bal.Acc: 90.5%  F1: 0.8962  |  Val Loss: 0.8601  Bal.Acc: 74.3%  F1: 0.7583  |  LR: 5.00e-05  (15.7s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.7569

 Trening finished. Checkpoint: checkpoints/mobilenetv3_large_fold5_best.pt
Log CSV: results/mobilenetv3_large_fold5_training_log.csv
Best weights loaded from  12

Ewaluacja modelu: mobilenetv3_large_fold5
----------------------------------------
  Balanced Accuracy:       78.48%
  F1 (macro):              0.7935
  Quadratic Cohen's Kappa: 0.9137
  ECE:                     0.1007
  Brier Score (mean):      0.0650

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.77      0.97      0.86        87
    Doubtful       0.81      0.72      0.76        81
        Mild       0.72      0.54      0.62        39
    Moderate       0.97      0.79      0.87        38
      Severe       0.82      0.91      0.86        35

    accuracy                           0.80       280
   macro avg       0.82      0.78      0.79       280
weighted avg       0.81      0.80      0.80       280

  Metryki zapisane: results/mobilenetv3_large_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/mobilenetv3_large_fold5_test_probs.npz

✅ ZAKOŃCZONO: mobilenetv3_large. Średnia Kappa z 5 foldów: 0.9049 ±0.0275

================================================================================
🚀 ROZPOCZĘCIE TRENINGU MODELU: convnext_tiny
================================================================================

--- convnext_tiny | FOLD 1/5 ---

  Fold 1/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 1):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
model.safetensors: 100% 114M/114M [00:02<00:00, 51.7MB/s]
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold1
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.5386  Bal.Acc: 32.0%  F1: 0.3004  |  Val Loss: 1.5595  Bal.Acc: 42.2%  F1: 0.3254  |  LR: 1.00e-04  (46.8s)
 Best checkpoint saved (val_loss: 1.5595)
Epoch [  2/20]  Train Loss: 1.0655  Bal.Acc: 54.8%  F1: 0.5310  |  Val Loss: 0.9406  Bal.Acc: 61.2%  F1: 0.6073  |  LR: 1.00e-04  (34.4s)
 Best checkpoint saved (val_loss: 0.9406)
Epoch [  3/20]  Train Loss: 0.7354  Bal.Acc: 70.3%  F1: 0.6932  |  Val Loss: 0.7918  Bal.Acc: 69.8%  F1: 0.7058  |  LR: 1.00e-04  (41.2s)
 Best checkpoint saved (val_loss: 0.7918)
Epoch [  4/20]  Train Loss: 0.6285  Bal.Acc: 76.6%  F1: 0.7581  |  Val Loss: 1.1594  Bal.Acc: 48.6%  F1: 0.4268  |  LR: 1.00e-04  (41.3s)
Epoch [  5/20]  Train Loss: 0.7261  Bal.Acc: 68.5%  F1: 0.6773  |  Val Loss: 0.8470  Bal.Acc: 65.1%  F1: 0.5792  |  LR: 1.00e-04  (34.2s)
Epoch [  6/20]  Train Loss: 0.5270  Bal.Acc: 80.4%  F1: 0.7948  |  Val Loss: 0.7527  Bal.Acc: 71.3%  F1: 0.6986  |  LR: 1.00e-04  (33.6s)
 Best checkpoint saved (val_loss: 0.7527)
Epoch [  7/20]  Train Loss: 0.4260  Bal.Acc: 84.4%  F1: 0.8398  |  Val Loss: 1.4409  Bal.Acc: 46.2%  F1: 0.4524  |  LR: 1.00e-04  (41.7s)
Epoch [  8/20]  Train Loss: 0.4502  Bal.Acc: 81.6%  F1: 0.8115  |  Val Loss: 0.7377  Bal.Acc: 72.4%  F1: 0.7011  |  LR: 1.00e-04  (34.1s)
 Best checkpoint saved (val_loss: 0.7377)
Epoch [  9/20]  Train Loss: 0.3621  Bal.Acc: 87.1%  F1: 0.8671  |  Val Loss: 0.7156  Bal.Acc: 75.9%  F1: 0.7464  |  LR: 1.00e-04  (40.9s)
 Best checkpoint saved (val_loss: 0.7156)
Epoch [ 10/20]  Train Loss: 0.3501  Bal.Acc: 87.2%  F1: 0.8665  |  Val Loss: 0.9315  Bal.Acc: 70.4%  F1: 0.7139  |  LR: 1.00e-04  (40.6s)
Epoch [ 11/20]  Train Loss: 0.6474  Bal.Acc: 74.8%  F1: 0.7442  |  Val Loss: 0.8731  Bal.Acc: 67.3%  F1: 0.6250  |  LR: 1.00e-04  (34.1s)
Epoch [ 12/20]  Train Loss: 0.3641  Bal.Acc: 86.3%  F1: 0.8592  |  Val Loss: 0.5293  Bal.Acc: 80.6%  F1: 0.8086  |  LR: 1.00e-04  (34.0s)
 Best checkpoint saved (val_loss: 0.5293)
Epoch [ 13/20]  Train Loss: 0.2833  Bal.Acc: 88.9%  F1: 0.8868  |  Val Loss: 0.6060  Bal.Acc: 76.6%  F1: 0.7558  |  LR: 1.00e-04  (41.8s)
Epoch [ 14/20]  Train Loss: 0.2500  Bal.Acc: 89.1%  F1: 0.8868  |  Val Loss: 0.5489  Bal.Acc: 78.4%  F1: 0.7858  |  LR: 1.00e-04  (33.9s)
Epoch [ 15/20]  Train Loss: 0.2450  Bal.Acc: 90.4%  F1: 0.9009  |  Val Loss: 0.7648  Bal.Acc: 76.9%  F1: 0.7763  |  LR: 1.00e-04  (33.7s)
Epoch [ 16/20]  Train Loss: 0.3953  Bal.Acc: 84.9%  F1: 0.8409  |  Val Loss: 0.7391  Bal.Acc: 77.0%  F1: 0.7732  |  LR: 5.00e-05  (35.0s)
Epoch [ 17/20]  Train Loss: 0.2548  Bal.Acc: 90.3%  F1: 0.9018  |  Val Loss: 0.4800  Bal.Acc: 82.1%  F1: 0.8253  |  LR: 5.00e-05  (34.4s)
 Best checkpoint saved (val_loss: 0.4800)
Epoch [ 18/20]  Train Loss: 0.1566  Bal.Acc: 93.8%  F1: 0.9356  |  Val Loss: 0.5508  Bal.Acc: 81.1%  F1: 0.8108  |  LR: 5.00e-05  (41.9s)
Epoch [ 19/20]  Train Loss: 0.1325  Bal.Acc: 95.0%  F1: 0.9479  |  Val Loss: 0.6462  Bal.Acc: 81.4%  F1: 0.8104  |  LR: 5.00e-05  (34.1s)
Epoch [ 20/20]  Train Loss: 0.1091  Bal.Acc: 95.8%  F1: 0.9571  |  Val Loss: 0.6146  Bal.Acc: 82.1%  F1: 0.8201  |  LR: 5.00e-05  (33.8s)

 Trening finished. Checkpoint: checkpoints/convnext_tiny_fold1_best.pt
Log CSV: results/convnext_tiny_fold1_training_log.csv
Best weights loaded from  17

Ewaluacja modelu: convnext_tiny_fold1
----------------------------------------
  Balanced Accuracy:       82.06%
  F1 (macro):              0.8253
  Quadratic Cohen's Kappa: 0.9477
  ECE:                     0.0568
  Brier Score (mean):      0.0500

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.89      0.87        88
    Doubtful       0.76      0.77      0.76        81
        Mild       0.75      0.69      0.72        39
    Moderate       0.89      0.82      0.85        38
      Severe       0.92      0.94      0.93        35

    accuracy                           0.82       281
   macro avg       0.83      0.82      0.83       281
weighted avg       0.82      0.82      0.82       281

  Metryki zapisane: results/convnext_tiny_fold1_metrics.json
  Prawdopodobieństwa zapisane: results/convnext_tiny_fold1_test_probs.npz

--- convnext_tiny | FOLD 2/5 ---

  Fold 2/5:
    Train: 1121 obrazów
    Val:   281 obrazów

    Wagi klas (fold 2):
      Klasa 0 (Normal): waga = 0.642  (count = 349)
      Klasa 1 (Doubtful): waga = 0.692  (count = 324)
      Klasa 2 (Mild): waga = 1.419  (count = 158)
      Klasa 3 (Moderate): waga = 1.495  (count = 150)
      Klasa 4 (Severe): waga = 1.601  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold2
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.6332  Bal.Acc: 26.5%  F1: 0.2457  |  Val Loss: 1.6743  Bal.Acc: 24.6%  F1: 0.1832  |  LR: 1.00e-04  (33.9s)
 Best checkpoint saved (val_loss: 1.6743)
Epoch [  2/20]  Train Loss: 1.4721  Bal.Acc: 34.5%  F1: 0.3193  |  Val Loss: 1.5106  Bal.Acc: 40.2%  F1: 0.3611  |  LR: 1.00e-04  (35.1s)
 Best checkpoint saved (val_loss: 1.5106)
Epoch [  3/20]  Train Loss: 1.2622  Bal.Acc: 45.3%  F1: 0.4421  |  Val Loss: 1.7351  Bal.Acc: 40.3%  F1: 0.2669  |  LR: 1.00e-04  (41.4s)
Epoch [  4/20]  Train Loss: 1.0110  Bal.Acc: 58.1%  F1: 0.5648  |  Val Loss: 0.8565  Bal.Acc: 64.4%  F1: 0.6108  |  LR: 1.00e-04  (34.2s)
 Best checkpoint saved (val_loss: 0.8565)
Epoch [  5/20]  Train Loss: 0.7984  Bal.Acc: 67.8%  F1: 0.6733  |  Val Loss: 0.7986  Bal.Acc: 61.8%  F1: 0.5691  |  LR: 1.00e-04  (41.5s)
 Best checkpoint saved (val_loss: 0.7986)
Epoch [  6/20]  Train Loss: 0.6868  Bal.Acc: 73.1%  F1: 0.7227  |  Val Loss: 0.6894  Bal.Acc: 74.3%  F1: 0.7123  |  LR: 1.00e-04  (42.9s)
 Best checkpoint saved (val_loss: 0.6894)
Epoch [  7/20]  Train Loss: 0.5987  Bal.Acc: 76.0%  F1: 0.7460  |  Val Loss: 1.1938  Bal.Acc: 54.2%  F1: 0.5473  |  LR: 1.00e-04  (43.2s)
Epoch [  8/20]  Train Loss: 0.6223  Bal.Acc: 76.2%  F1: 0.7591  |  Val Loss: 0.6800  Bal.Acc: 71.0%  F1: 0.6938  |  LR: 1.00e-04  (34.8s)
 Best checkpoint saved (val_loss: 0.6800)
Epoch [  9/20]  Train Loss: 0.7539  Bal.Acc: 68.1%  F1: 0.6694  |  Val Loss: 0.6131  Bal.Acc: 77.8%  F1: 0.7571  |  LR: 1.00e-04  (41.9s)
 Best checkpoint saved (val_loss: 0.6131)
Epoch [ 10/20]  Train Loss: 0.5954  Bal.Acc: 78.1%  F1: 0.7795  |  Val Loss: 0.6199  Bal.Acc: 75.4%  F1: 0.7254  |  LR: 1.00e-04  (42.2s)
Epoch [ 11/20]  Train Loss: 0.4657  Bal.Acc: 81.6%  F1: 0.8078  |  Val Loss: 0.6564  Bal.Acc: 73.9%  F1: 0.7466  |  LR: 1.00e-04  (35.3s)
Epoch [ 12/20]  Train Loss: 0.4291  Bal.Acc: 83.2%  F1: 0.8318  |  Val Loss: 0.4828  Bal.Acc: 81.4%  F1: 0.8222  |  LR: 1.00e-04  (34.8s)
 Best checkpoint saved (val_loss: 0.4828)
Epoch [ 13/20]  Train Loss: 0.3820  Bal.Acc: 86.6%  F1: 0.8622  |  Val Loss: 0.9093  Bal.Acc: 68.3%  F1: 0.6852  |  LR: 1.00e-04  (42.4s)
Epoch [ 14/20]  Train Loss: 0.4555  Bal.Acc: 82.9%  F1: 0.8238  |  Val Loss: 0.5357  Bal.Acc: 79.5%  F1: 0.7725  |  LR: 1.00e-04  (34.8s)
Epoch [ 15/20]  Train Loss: 0.3021  Bal.Acc: 88.5%  F1: 0.8810  |  Val Loss: 0.4194  Bal.Acc: 83.9%  F1: 0.8349  |  LR: 1.00e-04  (34.3s)
 Best checkpoint saved (val_loss: 0.4194)
Epoch [ 16/20]  Train Loss: 0.2107  Bal.Acc: 91.3%  F1: 0.9100  |  Val Loss: 0.6899  Bal.Acc: 77.0%  F1: 0.7721  |  LR: 1.00e-04  (41.6s)
Epoch [ 17/20]  Train Loss: 0.2521  Bal.Acc: 92.3%  F1: 0.9209  |  Val Loss: 0.8149  Bal.Acc: 71.0%  F1: 0.7017  |  LR: 1.00e-04  (34.3s)
Epoch [ 18/20]  Train Loss: 0.2573  Bal.Acc: 91.0%  F1: 0.9071  |  Val Loss: 0.8220  Bal.Acc: 72.0%  F1: 0.6945  |  LR: 1.00e-04  (33.9s)
Epoch [ 19/20]  Train Loss: 0.5605  Bal.Acc: 79.7%  F1: 0.7928  |  Val Loss: 0.6158  Bal.Acc: 72.7%  F1: 0.7298  |  LR: 5.00e-05  (34.0s)
Epoch [ 20/20]  Train Loss: 0.3301  Bal.Acc: 87.5%  F1: 0.8691  |  Val Loss: 0.6789  Bal.Acc: 72.4%  F1: 0.7272  |  LR: 5.00e-05  (34.1s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.4194

 Trening finished. Checkpoint: checkpoints/convnext_tiny_fold2_best.pt
Log CSV: results/convnext_tiny_fold2_training_log.csv
Best weights loaded from  15

Ewaluacja modelu: convnext_tiny_fold2
----------------------------------------
  Balanced Accuracy:       83.95%
  F1 (macro):              0.8349
  Quadratic Cohen's Kappa: 0.9397
  ECE:                     0.0626
  Brier Score (mean):      0.0514

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.85      0.86        88
    Doubtful       0.77      0.77      0.77        81
        Mild       0.78      0.72      0.75        39
    Moderate       0.88      0.95      0.91        38
      Severe       0.86      0.91      0.89        35

    accuracy                           0.83       281
   macro avg       0.83      0.84      0.83       281
weighted avg       0.83      0.83      0.83       281

  Metryki zapisane: results/convnext_tiny_fold2_metrics.json
  Prawdopodobieństwa zapisane: results/convnext_tiny_fold2_test_probs.npz

--- convnext_tiny | FOLD 3/5 ---

  Fold 3/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 3):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold3
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.6981  Bal.Acc: 22.1%  F1: 0.1938  |  Val Loss: 1.6315  Bal.Acc: 23.2%  F1: 0.1631  |  LR: 1.00e-04  (41.0s)
 Best checkpoint saved (val_loss: 1.6315)
Epoch [  2/20]  Train Loss: 1.5107  Bal.Acc: 32.6%  F1: 0.2937  |  Val Loss: 1.5284  Bal.Acc: 28.1%  F1: 0.2376  |  LR: 1.00e-04  (35.5s)
 Best checkpoint saved (val_loss: 1.5284)
Epoch [  3/20]  Train Loss: 1.2579  Bal.Acc: 47.9%  F1: 0.4534  |  Val Loss: 1.0219  Bal.Acc: 57.5%  F1: 0.4932  |  LR: 1.00e-04  (41.7s)
 Best checkpoint saved (val_loss: 1.0219)
Epoch [  4/20]  Train Loss: 0.8327  Bal.Acc: 66.1%  F1: 0.6509  |  Val Loss: 0.9813  Bal.Acc: 56.5%  F1: 0.5361  |  LR: 1.00e-04  (42.5s)
 Best checkpoint saved (val_loss: 0.9813)
Epoch [  5/20]  Train Loss: 0.7164  Bal.Acc: 69.1%  F1: 0.6809  |  Val Loss: 0.7364  Bal.Acc: 69.2%  F1: 0.6713  |  LR: 1.00e-04  (42.3s)
 Best checkpoint saved (val_loss: 0.7364)
Epoch [  6/20]  Train Loss: 0.5621  Bal.Acc: 77.8%  F1: 0.7689  |  Val Loss: 0.7529  Bal.Acc: 71.0%  F1: 0.7081  |  LR: 1.00e-04  (43.2s)
Epoch [  7/20]  Train Loss: 0.5030  Bal.Acc: 79.6%  F1: 0.7918  |  Val Loss: 0.6355  Bal.Acc: 75.7%  F1: 0.7539  |  LR: 1.00e-04  (34.1s)
 Best checkpoint saved (val_loss: 0.6355)
Epoch [  8/20]  Train Loss: 0.5226  Bal.Acc: 80.3%  F1: 0.8001  |  Val Loss: 0.6190  Bal.Acc: 75.0%  F1: 0.7471  |  LR: 1.00e-04  (42.0s)
 Best checkpoint saved (val_loss: 0.6190)
Epoch [  9/20]  Train Loss: 0.4460  Bal.Acc: 82.9%  F1: 0.8183  |  Val Loss: 0.6311  Bal.Acc: 75.3%  F1: 0.7515  |  LR: 1.00e-04  (39.9s)
Epoch [ 10/20]  Train Loss: 0.4677  Bal.Acc: 81.3%  F1: 0.8063  |  Val Loss: 1.1193  Bal.Acc: 65.3%  F1: 0.6460  |  LR: 1.00e-04  (34.7s)
Epoch [ 11/20]  Train Loss: 0.5730  Bal.Acc: 80.0%  F1: 0.7898  |  Val Loss: 1.2659  Bal.Acc: 59.8%  F1: 0.6230  |  LR: 1.00e-04  (32.9s)
Epoch [ 12/20]  Train Loss: 0.4237  Bal.Acc: 84.3%  F1: 0.8401  |  Val Loss: 0.6310  Bal.Acc: 74.0%  F1: 0.7321  |  LR: 5.00e-05  (34.3s)
Epoch [ 13/20]  Train Loss: 0.2541  Bal.Acc: 90.5%  F1: 0.9013  |  Val Loss: 0.8112  Bal.Acc: 74.8%  F1: 0.7585  |  LR: 5.00e-05  (33.9s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.6190

 Trening finished. Checkpoint: checkpoints/convnext_tiny_fold3_best.pt
Log CSV: results/convnext_tiny_fold3_training_log.csv
Best weights loaded from  8

Ewaluacja modelu: convnext_tiny_fold3
----------------------------------------
  Balanced Accuracy:       74.98%
  F1 (macro):              0.7471
  Quadratic Cohen's Kappa: 0.9040
  ECE:                     0.0596
  Brier Score (mean):      0.0671

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.81      0.90      0.85        87
    Doubtful       0.78      0.67      0.72        81
        Mild       0.54      0.65      0.59        40
    Moderate       0.85      0.62      0.72        37
      Severe       0.80      0.91      0.85        35

    accuracy                           0.76       280
   macro avg       0.76      0.75      0.75       280
weighted avg       0.77      0.76      0.76       280

  Metryki zapisane: results/convnext_tiny_fold3_metrics.json
  Prawdopodobieństwa zapisane: results/convnext_tiny_fold3_test_probs.npz

--- convnext_tiny | FOLD 4/5 ---

  Fold 4/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 4):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.429  (count = 157)
      Klasa 3 (Moderate): waga = 1.486  (count = 151)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold4
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.6647  Bal.Acc: 22.9%  F1: 0.2167  |  Val Loss: 1.4875  Bal.Acc: 37.1%  F1: 0.2282  |  LR: 1.00e-04  (34.2s)
 Best checkpoint saved (val_loss: 1.4875)
Epoch [  2/20]  Train Loss: 1.3296  Bal.Acc: 42.5%  F1: 0.3913  |  Val Loss: 1.1208  Bal.Acc: 51.0%  F1: 0.4947  |  LR: 1.00e-04  (34.1s)
 Best checkpoint saved (val_loss: 1.1208)
Epoch [  3/20]  Train Loss: 0.9637  Bal.Acc: 57.6%  F1: 0.5674  |  Val Loss: 0.8110  Bal.Acc: 64.6%  F1: 0.6221  |  LR: 1.00e-04  (42.1s)
 Best checkpoint saved (val_loss: 0.8110)
Epoch [  4/20]  Train Loss: 0.7284  Bal.Acc: 70.0%  F1: 0.6893  |  Val Loss: 0.6894  Bal.Acc: 71.5%  F1: 0.6897  |  LR: 1.00e-04  (40.2s)
 Best checkpoint saved (val_loss: 0.6894)
Epoch [  5/20]  Train Loss: 0.6701  Bal.Acc: 73.2%  F1: 0.7217  |  Val Loss: 1.0434  Bal.Acc: 62.6%  F1: 0.6369  |  LR: 1.00e-04  (42.2s)
Epoch [  6/20]  Train Loss: 0.6349  Bal.Acc: 74.9%  F1: 0.7379  |  Val Loss: 0.6574  Bal.Acc: 73.8%  F1: 0.7325  |  LR: 1.00e-04  (34.1s)
 Best checkpoint saved (val_loss: 0.6574)
Epoch [  7/20]  Train Loss: 0.5530  Bal.Acc: 78.2%  F1: 0.7760  |  Val Loss: 0.6292  Bal.Acc: 71.5%  F1: 0.7053  |  LR: 1.00e-04  (41.8s)
 Best checkpoint saved (val_loss: 0.6292)
Epoch [  8/20]  Train Loss: 0.4443  Bal.Acc: 83.1%  F1: 0.8251  |  Val Loss: 0.7068  Bal.Acc: 69.0%  F1: 0.6496  |  LR: 1.00e-04  (42.0s)
Epoch [  9/20]  Train Loss: 0.4347  Bal.Acc: 83.3%  F1: 0.8235  |  Val Loss: 1.5690  Bal.Acc: 53.7%  F1: 0.5203  |  LR: 1.00e-04  (34.1s)
Epoch [ 10/20]  Train Loss: 0.7632  Bal.Acc: 73.8%  F1: 0.7307  |  Val Loss: 0.8384  Bal.Acc: 71.0%  F1: 0.7253  |  LR: 1.00e-04  (34.0s)
Epoch [ 11/20]  Train Loss: 0.5041  Bal.Acc: 81.2%  F1: 0.8093  |  Val Loss: 0.6253  Bal.Acc: 78.8%  F1: 0.7896  |  LR: 1.00e-04  (34.4s)
 Best checkpoint saved (val_loss: 0.6253)
Epoch [ 12/20]  Train Loss: 0.3596  Bal.Acc: 86.0%  F1: 0.8534  |  Val Loss: 1.0336  Bal.Acc: 65.7%  F1: 0.5920  |  LR: 1.00e-04  (42.9s)
Epoch [ 13/20]  Train Loss: 0.3790  Bal.Acc: 85.6%  F1: 0.8520  |  Val Loss: 0.5108  Bal.Acc: 80.1%  F1: 0.8034  |  LR: 1.00e-04  (34.3s)
 Best checkpoint saved (val_loss: 0.5108)
Epoch [ 14/20]  Train Loss: 0.2646  Bal.Acc: 89.6%  F1: 0.8912  |  Val Loss: 0.6408  Bal.Acc: 78.1%  F1: 0.7734  |  LR: 1.00e-04  (40.7s)
Epoch [ 15/20]  Train Loss: 0.2731  Bal.Acc: 90.5%  F1: 0.9020  |  Val Loss: 0.5522  Bal.Acc: 82.1%  F1: 0.8162  |  LR: 1.00e-04  (34.6s)
Epoch [ 16/20]  Train Loss: 0.2623  Bal.Acc: 89.4%  F1: 0.8915  |  Val Loss: 0.6124  Bal.Acc: 80.0%  F1: 0.7750  |  LR: 1.00e-04  (33.2s)
Epoch [ 17/20]  Train Loss: 0.2707  Bal.Acc: 90.6%  F1: 0.9010  |  Val Loss: 0.5285  Bal.Acc: 83.2%  F1: 0.8181  |  LR: 5.00e-05  (34.4s)
Epoch [ 18/20]  Train Loss: 0.1593  Bal.Acc: 94.5%  F1: 0.9406  |  Val Loss: 0.6299  Bal.Acc: 81.7%  F1: 0.8070  |  LR: 5.00e-05  (33.4s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.5108

 Trening finished. Checkpoint: checkpoints/convnext_tiny_fold4_best.pt
Log CSV: results/convnext_tiny_fold4_training_log.csv
Best weights loaded from  13

Ewaluacja modelu: convnext_tiny_fold4
----------------------------------------
  Balanced Accuracy:       80.07%
  F1 (macro):              0.8034
  Quadratic Cohen's Kappa: 0.9109
  ECE:                     0.0652
  Brier Score (mean):      0.0535

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.84      0.93      0.88        87
    Doubtful       0.78      0.73      0.75        81
        Mild       0.71      0.62      0.67        40
    Moderate       0.87      0.92      0.89        37
      Severe       0.85      0.80      0.82        35

    accuracy                           0.81       280
   macro avg       0.81      0.80      0.80       280
weighted avg       0.81      0.81      0.81       280

  Metryki zapisane: results/convnext_tiny_fold4_metrics.json
  Prawdopodobieństwa zapisane: results/convnext_tiny_fold4_test_probs.npz

--- convnext_tiny | FOLD 5/5 ---

  Fold 5/5:
    Train: 1122 obrazów
    Val:   280 obrazów

    Wagi klas (fold 5):
      Klasa 0 (Normal): waga = 0.641  (count = 350)
      Klasa 1 (Doubtful): waga = 0.693  (count = 324)
      Klasa 2 (Mild): waga = 1.420  (count = 158)
      Klasa 3 (Moderate): waga = 1.496  (count = 150)
      Klasa 4 (Severe): waga = 1.603  (count = 140)
/content/drive/MyDrive/Knee_Project/dataset.py:270: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold5
Device: cuda
============================================================
Epoch [  1/20]  Train Loss: 1.5661  Bal.Acc: 32.8%  F1: 0.3002  |  Val Loss: 2.0440  Bal.Acc: 21.0%  F1: 0.1183  |  LR: 1.00e-04  (34.8s)
 Best checkpoint saved (val_loss: 2.0440)
Epoch [  2/20]  Train Loss: 1.2926  Bal.Acc: 46.1%  F1: 0.4290  |  Val Loss: 0.9617  Bal.Acc: 62.5%  F1: 0.5581  |  LR: 1.00e-04  (34.7s)
 Best checkpoint saved (val_loss: 0.9617)
Epoch [  3/20]  Train Loss: 0.9456  Bal.Acc: 62.2%  F1: 0.6016  |  Val Loss: 0.8919  Bal.Acc: 65.6%  F1: 0.6143  |  LR: 1.00e-04  (41.3s)
 Best checkpoint saved (val_loss: 0.8919)
Epoch [  4/20]  Train Loss: 0.8040  Bal.Acc: 67.0%  F1: 0.6629  |  Val Loss: 0.9114  Bal.Acc: 67.9%  F1: 0.6549  |  LR: 1.00e-04  (40.7s)
Epoch [  5/20]  Train Loss: 0.7243  Bal.Acc: 71.2%  F1: 0.7029  |  Val Loss: 0.7092  Bal.Acc: 72.2%  F1: 0.7032  |  LR: 1.00e-04  (34.6s)
 Best checkpoint saved (val_loss: 0.7092)
Epoch [  6/20]  Train Loss: 0.5702  Bal.Acc: 77.7%  F1: 0.7659  |  Val Loss: 0.8762  Bal.Acc: 66.6%  F1: 0.6416  |  LR: 1.00e-04  (41.7s)
Epoch [  7/20]  Train Loss: 0.5043  Bal.Acc: 80.8%  F1: 0.7994  |  Val Loss: 0.7049  Bal.Acc: 74.5%  F1: 0.7487  |  LR: 1.00e-04  (34.6s)
 Best checkpoint saved (val_loss: 0.7049)
Epoch [  8/20]  Train Loss: 0.5129  Bal.Acc: 80.0%  F1: 0.7926  |  Val Loss: 0.5609  Bal.Acc: 80.1%  F1: 0.8018  |  LR: 1.00e-04  (40.5s)
 Best checkpoint saved (val_loss: 0.5609)
Epoch [  9/20]  Train Loss: 0.4343  Bal.Acc: 83.0%  F1: 0.8232  |  Val Loss: 0.6184  Bal.Acc: 77.4%  F1: 0.7773  |  LR: 1.00e-04  (41.3s)
Epoch [ 10/20]  Train Loss: 0.3761  Bal.Acc: 85.8%  F1: 0.8506  |  Val Loss: 0.7698  Bal.Acc: 74.1%  F1: 0.7261  |  LR: 1.00e-04  (33.5s)
Epoch [ 11/20]  Train Loss: 0.3273  Bal.Acc: 88.3%  F1: 0.8790  |  Val Loss: 1.1845  Bal.Acc: 67.6%  F1: 0.7036  |  LR: 1.00e-04  (34.0s)
Epoch [ 12/20]  Train Loss: 0.3157  Bal.Acc: 87.2%  F1: 0.8698  |  Val Loss: 0.8095  Bal.Acc: 76.1%  F1: 0.7537  |  LR: 5.00e-05  (33.7s)
Epoch [ 13/20]  Train Loss: 0.1958  Bal.Acc: 93.0%  F1: 0.9277  |  Val Loss: 0.6624  Bal.Acc: 78.4%  F1: 0.7714  |  LR: 5.00e-05  (33.6s)

  Early stopping due to lack of improvement 5 epoch.
  Best val_loss: 0.5609

 Trening finished. Checkpoint: checkpoints/convnext_tiny_fold5_best.pt
Log CSV: results/convnext_tiny_fold5_training_log.csv
Best weights loaded from  8

Ewaluacja modelu: convnext_tiny_fold5
----------------------------------------
  Balanced Accuracy:       80.09%
  F1 (macro):              0.8018
  Quadratic Cohen's Kappa: 0.9166
  ECE:                     0.0190
  Brier Score (mean):      0.0588

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.89      0.88        87
    Doubtful       0.73      0.70      0.72        81
        Mild       0.64      0.69      0.67        39
    Moderate       0.85      0.89      0.87        38
      Severe       0.94      0.83      0.88        35

    accuracy                           0.80       280
   macro avg       0.80      0.80      0.80       280
weighted avg       0.80      0.80      0.80       280

  Metryki zapisane: results/convnext_tiny_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/convnext_tiny_fold5_test_probs.npz

✅ ZAKOŃCZONO: convnext_tiny. Średnia Kappa z 5 foldów: 0.9238 ±0.0169

========================================================================================================================
PODSUMOWANIE POJEDYNCZYCH FOLDÓW
========================================================================================================================
Model                        Kappa   F1-Mac     ECE   Brier |     KL0     KL1     KL2     KL3     KL4
------------------------------------------------------------------------------------------------------------------------
convnext_tiny_fold1         0.9477   0.8253  0.0568  0.0500 |  0.8667  0.7607  0.7200  0.8493  0.9296
convnext_tiny_fold2         0.9397   0.8349  0.0626  0.0514 |  0.8621  0.7654  0.7467  0.9114  0.8889
densenet121_fold1           0.9359   0.8383  0.0589  0.0507 |  0.8750  0.7821  0.7750  0.8861  0.8732
densenet121_fold2           0.9359   0.8219  0.0693  0.0562 |  0.8619  0.6479  0.6957  0.9610  0.9429
densenet121_fold5           0.9344   0.8484  0.0199  0.0485 |  0.9162  0.7875  0.6923  0.9444  0.9014
densenet121_fold3           0.9329   0.8458  0.0644  0.0481 |  0.9080  0.8077  0.7561  0.8684  0.8889
densenet121_fold4           0.9292   0.8281  0.0592  0.0475 |  0.8877  0.7671  0.7273  0.8974  0.8611
mobilenetv3_large_fold3     0.9273   0.8242  0.0805  0.0557 |  0.8383  0.7975  0.6966  0.8611  0.9275
mobilenetv3_large_fold4     0.9264   0.8164  0.0821  0.0568 |  0.8686  0.7582  0.6897  0.8800  0.8857
resnet50_fold3              0.9234   0.7736  0.0691  0.0667 |  0.8404  0.6944  0.6585  0.8052  0.8696
convnext_tiny_fold5         0.9166   0.8018  0.0190  0.0588 |  0.8750  0.7170  0.6667  0.8718  0.8788
resnet50_fold2              0.9142   0.7582  0.0279  0.0659 |  0.8525  0.6853  0.6118  0.8235  0.8182
mobilenetv3_large_fold5     0.9137   0.7935  0.1007  0.0650 |  0.8571  0.7582  0.6176  0.8696  0.8649
efficientnet_b3_fold4       0.9110   0.8393  0.0536  0.0488 |  0.8876  0.8027  0.6818  0.9444  0.8800
convnext_tiny_fold4         0.9109   0.8034  0.0652  0.0535 |  0.8804  0.7516  0.6667  0.8947  0.8235
efficientnet_b3_fold2       0.9059   0.7939  0.1006  0.0663 |  0.8324  0.6538  0.6667  0.9211  0.8955
mobilenetv3_large_fold1     0.9047   0.7764  0.1008  0.0661 |  0.8298  0.6438  0.6250  0.8974  0.8857
convnext_tiny_fold3         0.9040   0.7471  0.0596  0.0671 |  0.8525  0.7200  0.5909  0.7188  0.8533
resnet50_fold1              0.9008   0.7446  0.0174  0.0678 |  0.8362  0.6710  0.6053  0.8000  0.8108
efficientnet_b3_fold3       0.8996   0.7816  0.1209  0.0682 |  0.8214  0.7219  0.6667  0.8485  0.8493
resnet50_fold5              0.8994   0.7573  0.0627  0.0616 |  0.8718  0.6846  0.5000  0.8732  0.8571
resnet50_fold4              0.8897   0.7400  0.0375  0.0676 |  0.8701  0.6712  0.5432  0.8101  0.8052
mobilenetv3_large_fold2     0.8525   0.7726  0.0920  0.0612 |  0.8444  0.7439  0.6579  0.8750  0.7419
efficientnet_b3_fold5       0.8315   0.7044  0.1179  0.0828 |  0.8041  0.6364  0.5000  0.8493  0.7324
efficientnet_b3_fold1       0.3850   0.3737  0.2839  0.1691 |  0.4065  0.4630  0.2424  0.4138  0.3429
========================================================================================================================
Posortowane wg Cohen's Kappa

===================================================================================================================
PODSUMOWANIE CROSS-VALIDATION — ŚREDNIA Z 5 FOLDÓW
===================================================================================================================
Model                         Kappa         F1-Mac |      KL0      KL1      KL2      KL3      KL4
-------------------------------------------------------------------------------------------------------------------
resnet50              0.9055 ±0.0119  0.7547 ±0.0118 |   0.8542   0.6813   0.5838   0.8224   0.8322
efficientnet_b3       0.7866 ±0.2029  0.6986 ±0.1681 |   0.7504   0.6556   0.5515   0.7954   0.7400
densenet121           0.9337 ±0.0025  0.8365 ±0.0101 |   0.8898   0.7585   0.7293   0.9115   0.8935
mobilenetv3_large     0.9049 ±0.0275  0.7966 ±0.0207 |   0.8476   0.7403   0.6574   0.8766   0.8611
convnext_tiny         0.9238 ±0.0169  0.8025 ±0.0305 |   0.8673   0.7429   0.6782   0.8492   0.8748
===================================================================================================================
Wyniki w folderze: results
