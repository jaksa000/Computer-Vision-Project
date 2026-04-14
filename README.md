Dataset is too big to upload here, downolad it from here: https://www.kaggle.com/datasets/tommyngx/digital-knee-xray/data?select=MedicalExpert-I
unzip and rename to data and put in project directory, should work from here

/content/drive/MyDrive/Knee_Project
Urządzenie: cuda
Modele do uruchomienia: ['resnet50', 'efficientnet_b3', 'densenet121', 'mobilenetv3_large', 'convnext_tiny']
Liczba foldów: 5
Ładowanie danych z: /content/data/MedicalExpert-I
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
 ROZPOCZĘCIE TRENINGU MODELU: resnet50
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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold1
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.6039  Bal.Acc: 23.4%  F1: 0.2009  |  Val Loss: 1.5856  Bal.Acc: 26.3%  F1: 0.2353  |  LR: 1.00e-04  (17.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5856)
Epoka [  2/20]  Train Loss: 1.5805  Bal.Acc: 33.4%  F1: 0.3125  |  Val Loss: 1.5662  Bal.Acc: 34.5%  F1: 0.3156  |  LR: 1.00e-04  (17.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5662)
Epoka [  3/20]  Train Loss: 1.5515  Bal.Acc: 39.0%  F1: 0.3699  |  Val Loss: 1.5214  Bal.Acc: 31.0%  F1: 0.2715  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5214)
Epoka [  4/20]  Train Loss: 1.5049  Bal.Acc: 43.0%  F1: 0.3701  |  Val Loss: 1.4611  Bal.Acc: 40.2%  F1: 0.3488  |  LR: 1.00e-04  (20.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4611)
Epoka [  5/20]  Train Loss: 1.4209  Bal.Acc: 50.5%  F1: 0.4605  |  Val Loss: 1.3722  Bal.Acc: 38.6%  F1: 0.3553  |  LR: 1.00e-04  (19.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3722)
Epoka [  6/20]  Train Loss: 1.2924  Bal.Acc: 52.7%  F1: 0.4562  |  Val Loss: 1.2687  Bal.Acc: 47.5%  F1: 0.4469  |  LR: 1.00e-04  (18.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2687)
Epoka [  7/20]  Train Loss: 1.1528  Bal.Acc: 58.0%  F1: 0.5298  |  Val Loss: 1.0289  Bal.Acc: 62.6%  F1: 0.5783  |  LR: 1.00e-04  (20.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0289)
Epoka [  8/20]  Train Loss: 1.0152  Bal.Acc: 62.1%  F1: 0.5892  |  Val Loss: 0.9251  Bal.Acc: 62.3%  F1: 0.5949  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9251)
Epoka [  9/20]  Train Loss: 0.9302  Bal.Acc: 65.9%  F1: 0.6346  |  Val Loss: 0.9406  Bal.Acc: 61.9%  F1: 0.6200  |  LR: 1.00e-04  (19.0s)
Epoka [ 10/20]  Train Loss: 0.8467  Bal.Acc: 68.4%  F1: 0.6715  |  Val Loss: 0.7880  Bal.Acc: 69.6%  F1: 0.6669  |  LR: 1.00e-04  (19.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7880)
Epoka [ 11/20]  Train Loss: 0.7664  Bal.Acc: 72.0%  F1: 0.6986  |  Val Loss: 0.7384  Bal.Acc: 71.1%  F1: 0.6974  |  LR: 1.00e-04  (19.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7384)
Epoka [ 12/20]  Train Loss: 0.6948  Bal.Acc: 74.6%  F1: 0.7422  |  Val Loss: 0.7277  Bal.Acc: 71.3%  F1: 0.6902  |  LR: 1.00e-04  (18.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7277)
Epoka [ 13/20]  Train Loss: 0.6621  Bal.Acc: 74.5%  F1: 0.7330  |  Val Loss: 0.6740  Bal.Acc: 73.7%  F1: 0.7315  |  LR: 1.00e-04  (22.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6740)
Epoka [ 14/20]  Train Loss: 0.6269  Bal.Acc: 76.0%  F1: 0.7558  |  Val Loss: 0.6383  Bal.Acc: 76.5%  F1: 0.7589  |  LR: 1.00e-04  (18.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6383)
Epoka [ 15/20]  Train Loss: 0.5745  Bal.Acc: 77.4%  F1: 0.7670  |  Val Loss: 0.6678  Bal.Acc: 74.3%  F1: 0.7260  |  LR: 1.00e-04  (20.2s)
Epoka [ 16/20]  Train Loss: 0.5321  Bal.Acc: 79.2%  F1: 0.7842  |  Val Loss: 0.8054  Bal.Acc: 65.8%  F1: 0.6775  |  LR: 1.00e-04  (18.9s)
Epoka [ 17/20]  Train Loss: 0.5028  Bal.Acc: 82.2%  F1: 0.8132  |  Val Loss: 0.5944  Bal.Acc: 78.1%  F1: 0.7585  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5944)
Epoka [ 18/20]  Train Loss: 0.4706  Bal.Acc: 82.3%  F1: 0.8183  |  Val Loss: 0.5632  Bal.Acc: 78.3%  F1: 0.7754  |  LR: 1.00e-04  (19.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5632)
Epoka [ 19/20]  Train Loss: 0.4684  Bal.Acc: 83.5%  F1: 0.8309  |  Val Loss: 0.5387  Bal.Acc: 77.2%  F1: 0.7400  |  LR: 1.00e-04  (18.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5387)
Epoka [ 20/20]  Train Loss: 0.4508  Bal.Acc: 83.6%  F1: 0.8296  |  Val Loss: 0.5132  Bal.Acc: 79.0%  F1: 0.7717  |  LR: 1.00e-04  (20.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5132)

Trening zakończony. Checkpoint: checkpoints/resnet50_fold1_best.pt
Log CSV: results/resnet50_fold1_training_log.csv
Załadowano najlepsze wagi z epoki 20

Ewaluacja modelu: resnet50_fold1
----------------------------------------
  Balanced Accuracy: 78.97%
  F1 (macro):        0.7717
  Quadratic Cohen's Kappa: 0.9142

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.82      0.83        88
    Doubtful       0.74      0.60      0.67        81
        Mild       0.55      0.69      0.61        39
    Moderate       0.77      0.95      0.85        38
      Severe       0.91      0.89      0.90        35

    accuracy                           0.77       281
   macro avg       0.76      0.79      0.77       281
weighted avg       0.77      0.77      0.76       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold2
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.6007  Bal.Acc: 26.8%  F1: 0.2568  |  Val Loss: 1.5925  Bal.Acc: 31.6%  F1: 0.2948  |  LR: 1.00e-04  (20.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5925)
Epoka [  2/20]  Train Loss: 1.5749  Bal.Acc: 34.6%  F1: 0.3587  |  Val Loss: 1.5601  Bal.Acc: 31.7%  F1: 0.3077  |  LR: 1.00e-04  (20.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5601)
Epoka [  3/20]  Train Loss: 1.5505  Bal.Acc: 37.9%  F1: 0.3789  |  Val Loss: 1.5103  Bal.Acc: 45.5%  F1: 0.4521  |  LR: 1.00e-04  (19.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5103)
Epoka [  4/20]  Train Loss: 1.5030  Bal.Acc: 46.3%  F1: 0.4277  |  Val Loss: 1.4618  Bal.Acc: 45.2%  F1: 0.4220  |  LR: 1.00e-04  (19.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4618)
Epoka [  5/20]  Train Loss: 1.4214  Bal.Acc: 51.0%  F1: 0.4583  |  Val Loss: 1.3934  Bal.Acc: 43.5%  F1: 0.4094  |  LR: 1.00e-04  (20.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3934)
Epoka [  6/20]  Train Loss: 1.2864  Bal.Acc: 56.8%  F1: 0.5195  |  Val Loss: 1.2698  Bal.Acc: 50.1%  F1: 0.4697  |  LR: 1.00e-04  (20.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2698)
Epoka [  7/20]  Train Loss: 1.1279  Bal.Acc: 60.2%  F1: 0.5422  |  Val Loss: 1.1141  Bal.Acc: 57.2%  F1: 0.5340  |  LR: 1.00e-04  (20.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.1141)
Epoka [  8/20]  Train Loss: 0.9941  Bal.Acc: 61.7%  F1: 0.5755  |  Val Loss: 1.0288  Bal.Acc: 60.5%  F1: 0.5972  |  LR: 1.00e-04  (18.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0288)
Epoka [  9/20]  Train Loss: 0.9067  Bal.Acc: 66.2%  F1: 0.6442  |  Val Loss: 0.8648  Bal.Acc: 67.4%  F1: 0.6669  |  LR: 1.00e-04  (18.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8648)
Epoka [ 10/20]  Train Loss: 0.8473  Bal.Acc: 67.6%  F1: 0.6580  |  Val Loss: 0.8570  Bal.Acc: 68.5%  F1: 0.6838  |  LR: 1.00e-04  (20.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8570)
Epoka [ 11/20]  Train Loss: 0.7623  Bal.Acc: 70.3%  F1: 0.6941  |  Val Loss: 0.8188  Bal.Acc: 67.7%  F1: 0.6852  |  LR: 1.00e-04  (20.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8188)
Epoka [ 12/20]  Train Loss: 0.7473  Bal.Acc: 69.1%  F1: 0.6808  |  Val Loss: 0.7713  Bal.Acc: 68.3%  F1: 0.6641  |  LR: 1.00e-04  (19.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7713)
Epoka [ 13/20]  Train Loss: 0.6754  Bal.Acc: 74.2%  F1: 0.7307  |  Val Loss: 0.7440  Bal.Acc: 70.0%  F1: 0.6970  |  LR: 1.00e-04  (19.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7440)
Epoka [ 14/20]  Train Loss: 0.6569  Bal.Acc: 74.7%  F1: 0.7381  |  Val Loss: 0.6920  Bal.Acc: 69.7%  F1: 0.6898  |  LR: 1.00e-04  (19.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6920)
Epoka [ 15/20]  Train Loss: 0.5982  Bal.Acc: 78.5%  F1: 0.7746  |  Val Loss: 0.6059  Bal.Acc: 76.1%  F1: 0.7537  |  LR: 1.00e-04  (20.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6059)
Epoka [ 16/20]  Train Loss: 0.5878  Bal.Acc: 76.9%  F1: 0.7646  |  Val Loss: 0.6274  Bal.Acc: 76.5%  F1: 0.7396  |  LR: 1.00e-04  (19.5s)
Epoka [ 17/20]  Train Loss: 0.5503  Bal.Acc: 80.5%  F1: 0.7927  |  Val Loss: 0.7426  Bal.Acc: 70.0%  F1: 0.7026  |  LR: 1.00e-04  (20.8s)
Epoka [ 18/20]  Train Loss: 0.5136  Bal.Acc: 80.0%  F1: 0.7887  |  Val Loss: 0.5780  Bal.Acc: 75.8%  F1: 0.7540  |  LR: 1.00e-04  (18.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5780)
Epoka [ 19/20]  Train Loss: 0.4958  Bal.Acc: 79.7%  F1: 0.7869  |  Val Loss: 0.5719  Bal.Acc: 76.1%  F1: 0.7485  |  LR: 1.00e-04  (19.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5719)
Epoka [ 20/20]  Train Loss: 0.4775  Bal.Acc: 81.0%  F1: 0.8006  |  Val Loss: 0.6447  Bal.Acc: 73.1%  F1: 0.7252  |  LR: 1.00e-04  (18.4s)

Trening zakończony. Checkpoint: checkpoints/resnet50_fold2_best.pt
Log CSV: results/resnet50_fold2_training_log.csv
Załadowano najlepsze wagi z epoki 19

Ewaluacja modelu: resnet50_fold2
----------------------------------------
  Balanced Accuracy: 76.12%
  F1 (macro):        0.7485
  Quadratic Cohen's Kappa: 0.9095

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.91      0.88        88
    Doubtful       0.80      0.56      0.66        81
        Mild       0.46      0.69      0.55        39
    Moderate       0.83      0.76      0.79        38
      Severe       0.84      0.89      0.86        35

    accuracy                           0.75       281
   macro avg       0.76      0.76      0.75       281
weighted avg       0.78      0.75      0.76       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold3
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.6066  Bal.Acc: 22.7%  F1: 0.2242  |  Val Loss: 1.5962  Bal.Acc: 24.8%  F1: 0.2062  |  LR: 1.00e-04  (19.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5962)
Epoka [  2/20]  Train Loss: 1.5901  Bal.Acc: 30.6%  F1: 0.2999  |  Val Loss: 1.5783  Bal.Acc: 31.7%  F1: 0.3309  |  LR: 1.00e-04  (18.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5783)
Epoka [  3/20]  Train Loss: 1.5633  Bal.Acc: 39.6%  F1: 0.3879  |  Val Loss: 1.6520  Bal.Acc: 43.5%  F1: 0.4494  |  LR: 1.00e-04  (18.9s)
Epoka [  4/20]  Train Loss: 1.5235  Bal.Acc: 44.4%  F1: 0.4461  |  Val Loss: 1.4977  Bal.Acc: 37.8%  F1: 0.3683  |  LR: 1.00e-04  (21.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4977)
Epoka [  5/20]  Train Loss: 1.4641  Bal.Acc: 48.2%  F1: 0.4729  |  Val Loss: 1.4105  Bal.Acc: 47.0%  F1: 0.4630  |  LR: 1.00e-04  (19.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4105)
Epoka [  6/20]  Train Loss: 1.3379  Bal.Acc: 57.7%  F1: 0.5521  |  Val Loss: 1.2625  Bal.Acc: 55.1%  F1: 0.5144  |  LR: 1.00e-04  (20.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2625)
Epoka [  7/20]  Train Loss: 1.1763  Bal.Acc: 59.7%  F1: 0.5555  |  Val Loss: 1.2674  Bal.Acc: 57.8%  F1: 0.5486  |  LR: 1.00e-04  (19.2s)
Epoka [  8/20]  Train Loss: 1.0272  Bal.Acc: 61.6%  F1: 0.5847  |  Val Loss: 1.0404  Bal.Acc: 60.2%  F1: 0.5938  |  LR: 1.00e-04  (18.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0404)
Epoka [  9/20]  Train Loss: 0.9366  Bal.Acc: 64.8%  F1: 0.6291  |  Val Loss: 0.8929  Bal.Acc: 65.0%  F1: 0.6282  |  LR: 1.00e-04  (20.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8929)
Epoka [ 10/20]  Train Loss: 0.8263  Bal.Acc: 70.2%  F1: 0.6921  |  Val Loss: 0.8468  Bal.Acc: 64.9%  F1: 0.6343  |  LR: 1.00e-04  (20.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8468)
Epoka [ 11/20]  Train Loss: 0.7667  Bal.Acc: 71.9%  F1: 0.7095  |  Val Loss: 0.8109  Bal.Acc: 67.2%  F1: 0.6764  |  LR: 1.00e-04  (19.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8109)
Epoka [ 12/20]  Train Loss: 0.7246  Bal.Acc: 73.4%  F1: 0.7252  |  Val Loss: 0.8038  Bal.Acc: 68.0%  F1: 0.6884  |  LR: 1.00e-04  (19.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8038)
Epoka [ 13/20]  Train Loss: 0.6593  Bal.Acc: 75.7%  F1: 0.7487  |  Val Loss: 0.7964  Bal.Acc: 67.8%  F1: 0.6683  |  LR: 1.00e-04  (19.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7964)
Epoka [ 14/20]  Train Loss: 0.6500  Bal.Acc: 74.6%  F1: 0.7408  |  Val Loss: 0.7192  Bal.Acc: 73.2%  F1: 0.7397  |  LR: 1.00e-04  (18.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7192)
Epoka [ 15/20]  Train Loss: 0.6155  Bal.Acc: 75.3%  F1: 0.7407  |  Val Loss: 0.6900  Bal.Acc: 73.9%  F1: 0.7192  |  LR: 1.00e-04  (22.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6900)
Epoka [ 16/20]  Train Loss: 0.5482  Bal.Acc: 79.1%  F1: 0.7800  |  Val Loss: 0.7097  Bal.Acc: 72.3%  F1: 0.7196  |  LR: 1.00e-04  (18.9s)
Epoka [ 17/20]  Train Loss: 0.5268  Bal.Acc: 80.3%  F1: 0.7980  |  Val Loss: 0.7125  Bal.Acc: 71.0%  F1: 0.7070  |  LR: 1.00e-04  (20.7s)
Epoka [ 18/20]  Train Loss: 0.4810  Bal.Acc: 82.9%  F1: 0.8223  |  Val Loss: 0.6510  Bal.Acc: 74.9%  F1: 0.7433  |  LR: 1.00e-04  (18.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6510)
Epoka [ 19/20]  Train Loss: 0.4989  Bal.Acc: 80.4%  F1: 0.8013  |  Val Loss: 0.6749  Bal.Acc: 73.1%  F1: 0.7217  |  LR: 1.00e-04  (19.8s)
Epoka [ 20/20]  Train Loss: 0.4907  Bal.Acc: 80.8%  F1: 0.8006  |  Val Loss: 0.7544  Bal.Acc: 69.7%  F1: 0.7034  |  LR: 1.00e-04  (20.5s)

Trening zakończony. Checkpoint: checkpoints/resnet50_fold3_best.pt
Log CSV: results/resnet50_fold3_training_log.csv
Załadowano najlepsze wagi z epoki 18

Ewaluacja modelu: resnet50_fold3
----------------------------------------
  Balanced Accuracy: 74.87%
  F1 (macro):        0.7433
  Quadratic Cohen's Kappa: 0.8720

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.89      0.90      0.89        87
    Doubtful       0.73      0.64      0.68        81
        Mild       0.53      0.65      0.58        40
    Moderate       0.85      0.78      0.82        37
      Severe       0.71      0.77      0.74        35

    accuracy                           0.76       280
   macro avg       0.74      0.75      0.74       280
weighted avg       0.76      0.76      0.76       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold4
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.5994  Bal.Acc: 23.7%  F1: 0.2149  |  Val Loss: 1.5919  Bal.Acc: 26.4%  F1: 0.2416  |  LR: 1.00e-04  (18.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5919)
Epoka [  2/20]  Train Loss: 1.5805  Bal.Acc: 28.2%  F1: 0.2549  |  Val Loss: 1.5798  Bal.Acc: 22.0%  F1: 0.1320  |  LR: 1.00e-04  (18.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5798)
Epoka [  3/20]  Train Loss: 1.5518  Bal.Acc: 36.0%  F1: 0.3351  |  Val Loss: 1.5370  Bal.Acc: 32.3%  F1: 0.2976  |  LR: 1.00e-04  (18.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5370)
Epoka [  4/20]  Train Loss: 1.5109  Bal.Acc: 46.1%  F1: 0.4241  |  Val Loss: 1.4922  Bal.Acc: 36.7%  F1: 0.3424  |  LR: 1.00e-04  (18.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4922)
Epoka [  5/20]  Train Loss: 1.4532  Bal.Acc: 46.8%  F1: 0.4329  |  Val Loss: 1.4350  Bal.Acc: 39.2%  F1: 0.3542  |  LR: 1.00e-04  (19.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4350)
Epoka [  6/20]  Train Loss: 1.3326  Bal.Acc: 53.6%  F1: 0.4646  |  Val Loss: 1.2666  Bal.Acc: 54.3%  F1: 0.4983  |  LR: 1.00e-04  (20.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2666)
Epoka [  7/20]  Train Loss: 1.2123  Bal.Acc: 57.1%  F1: 0.5136  |  Val Loss: 1.1342  Bal.Acc: 58.1%  F1: 0.5237  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.1342)
Epoka [  8/20]  Train Loss: 1.0720  Bal.Acc: 62.3%  F1: 0.5790  |  Val Loss: 1.0548  Bal.Acc: 58.2%  F1: 0.5727  |  LR: 1.00e-04  (18.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0548)
Epoka [  9/20]  Train Loss: 0.9387  Bal.Acc: 65.4%  F1: 0.6282  |  Val Loss: 0.9480  Bal.Acc: 60.4%  F1: 0.5980  |  LR: 1.00e-04  (18.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9480)
Epoka [ 10/20]  Train Loss: 0.8447  Bal.Acc: 67.1%  F1: 0.6550  |  Val Loss: 0.8284  Bal.Acc: 66.4%  F1: 0.6377  |  LR: 1.00e-04  (19.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8284)
Epoka [ 11/20]  Train Loss: 0.7684  Bal.Acc: 71.0%  F1: 0.6932  |  Val Loss: 0.7897  Bal.Acc: 71.9%  F1: 0.7170  |  LR: 1.00e-04  (20.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7897)
Epoka [ 12/20]  Train Loss: 0.6984  Bal.Acc: 75.0%  F1: 0.7453  |  Val Loss: 0.7591  Bal.Acc: 68.0%  F1: 0.6774  |  LR: 1.00e-04  (19.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7591)
Epoka [ 13/20]  Train Loss: 0.6875  Bal.Acc: 73.7%  F1: 0.7239  |  Val Loss: 0.8340  Bal.Acc: 67.6%  F1: 0.6815  |  LR: 1.00e-04  (19.5s)
Epoka [ 14/20]  Train Loss: 0.6246  Bal.Acc: 76.1%  F1: 0.7589  |  Val Loss: 0.7193  Bal.Acc: 71.9%  F1: 0.6990  |  LR: 1.00e-04  (18.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7193)
Epoka [ 15/20]  Train Loss: 0.5932  Bal.Acc: 77.5%  F1: 0.7624  |  Val Loss: 1.1034  Bal.Acc: 49.8%  F1: 0.5022  |  LR: 1.00e-04  (19.9s)
Epoka [ 16/20]  Train Loss: 0.5676  Bal.Acc: 76.9%  F1: 0.7589  |  Val Loss: 0.7995  Bal.Acc: 69.9%  F1: 0.6984  |  LR: 1.00e-04  (20.0s)
Epoka [ 17/20]  Train Loss: 0.5370  Bal.Acc: 78.5%  F1: 0.7731  |  Val Loss: 0.7274  Bal.Acc: 73.9%  F1: 0.7370  |  LR: 1.00e-04  (18.5s)
Epoka [ 18/20]  Train Loss: 0.5267  Bal.Acc: 81.2%  F1: 0.8037  |  Val Loss: 0.7029  Bal.Acc: 76.1%  F1: 0.7679  |  LR: 1.00e-04  (18.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7029)
Epoka [ 19/20]  Train Loss: 0.4617  Bal.Acc: 81.8%  F1: 0.8091  |  Val Loss: 0.7547  Bal.Acc: 71.2%  F1: 0.7105  |  LR: 1.00e-04  (20.4s)
Epoka [ 20/20]  Train Loss: 0.4674  Bal.Acc: 83.1%  F1: 0.8209  |  Val Loss: 0.6841  Bal.Acc: 77.1%  F1: 0.7679  |  LR: 1.00e-04  (18.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6841)

Trening zakończony. Checkpoint: checkpoints/resnet50_fold4_best.pt
Log CSV: results/resnet50_fold4_training_log.csv
Załadowano najlepsze wagi z epoki 20

Ewaluacja modelu: resnet50_fold4
----------------------------------------
  Balanced Accuracy: 77.06%
  F1 (macro):        0.7679
  Quadratic Cohen's Kappa: 0.9030

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.83      0.83      0.83        87
    Doubtful       0.75      0.72      0.73        81
        Mild       0.65      0.70      0.67        40
    Moderate       0.74      0.84      0.78        37
      Severe       0.87      0.77      0.82        35

    accuracy                           0.77       280
   macro avg       0.77      0.77      0.77       280
weighted avg       0.77      0.77      0.77       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: resnet50
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50_fold5
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.5988  Bal.Acc: 24.2%  F1: 0.2215  |  Val Loss: 1.6010  Bal.Acc: 22.6%  F1: 0.1675  |  LR: 1.00e-04  (24.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.6010)
Epoka [  2/20]  Train Loss: 1.5714  Bal.Acc: 31.8%  F1: 0.2968  |  Val Loss: 1.5640  Bal.Acc: 29.7%  F1: 0.2493  |  LR: 1.00e-04  (19.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5640)
Epoka [  3/20]  Train Loss: 1.5335  Bal.Acc: 37.6%  F1: 0.3281  |  Val Loss: 1.6486  Bal.Acc: 35.6%  F1: 0.3393  |  LR: 1.00e-04  (21.4s)
Epoka [  4/20]  Train Loss: 1.4661  Bal.Acc: 41.7%  F1: 0.3428  |  Val Loss: 1.9009  Bal.Acc: 28.6%  F1: 0.2441  |  LR: 1.00e-04  (18.8s)
Epoka [  5/20]  Train Loss: 1.3510  Bal.Acc: 49.9%  F1: 0.4495  |  Val Loss: 1.7349  Bal.Acc: 38.8%  F1: 0.3748  |  LR: 1.00e-04  (18.3s)
Epoka [  6/20]  Train Loss: 1.2224  Bal.Acc: 54.7%  F1: 0.4892  |  Val Loss: 1.4143  Bal.Acc: 54.5%  F1: 0.5377  |  LR: 1.00e-04  (18.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4143)
Epoka [  7/20]  Train Loss: 1.1026  Bal.Acc: 58.3%  F1: 0.5425  |  Val Loss: 1.3985  Bal.Acc: 42.2%  F1: 0.4139  |  LR: 1.00e-04  (20.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3985)
Epoka [  8/20]  Train Loss: 0.9876  Bal.Acc: 62.5%  F1: 0.5907  |  Val Loss: 1.3914  Bal.Acc: 47.6%  F1: 0.4844  |  LR: 1.00e-04  (20.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3914)
Epoka [  9/20]  Train Loss: 0.9001  Bal.Acc: 65.7%  F1: 0.6304  |  Val Loss: 1.3676  Bal.Acc: 55.4%  F1: 0.5606  |  LR: 1.00e-04  (19.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3676)
Epoka [ 10/20]  Train Loss: 0.8151  Bal.Acc: 66.2%  F1: 0.6499  |  Val Loss: 1.4286  Bal.Acc: 71.4%  F1: 0.7030  |  LR: 1.00e-04  (20.7s)
Epoka [ 11/20]  Train Loss: 0.7718  Bal.Acc: 69.9%  F1: 0.6852  |  Val Loss: 1.7357  Bal.Acc: 70.5%  F1: 0.7102  |  LR: 1.00e-04  (20.0s)
Epoka [ 12/20]  Train Loss: 0.6784  Bal.Acc: 73.6%  F1: 0.7290  |  Val Loss: 0.9827  Bal.Acc: 74.4%  F1: 0.7400  |  LR: 1.00e-04  (18.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9827)
Epoka [ 13/20]  Train Loss: 0.6325  Bal.Acc: 76.7%  F1: 0.7609  |  Val Loss: 0.8944  Bal.Acc: 71.4%  F1: 0.7099  |  LR: 1.00e-04  (20.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8944)
Epoka [ 14/20]  Train Loss: 0.6164  Bal.Acc: 77.4%  F1: 0.7633  |  Val Loss: 0.9925  Bal.Acc: 71.8%  F1: 0.7244  |  LR: 1.00e-04  (20.5s)
Epoka [ 15/20]  Train Loss: 0.5500  Bal.Acc: 79.5%  F1: 0.7915  |  Val Loss: 0.8247  Bal.Acc: 74.4%  F1: 0.7347  |  LR: 1.00e-04  (20.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8247)
Epoka [ 16/20]  Train Loss: 0.5630  Bal.Acc: 78.0%  F1: 0.7711  |  Val Loss: 0.8138  Bal.Acc: 70.1%  F1: 0.6998  |  LR: 1.00e-04  (20.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8138)
Epoka [ 17/20]  Train Loss: 0.5484  Bal.Acc: 79.7%  F1: 0.7925  |  Val Loss: 0.8593  Bal.Acc: 70.9%  F1: 0.7145  |  LR: 1.00e-04  (19.9s)
Epoka [ 18/20]  Train Loss: 0.5258  Bal.Acc: 80.0%  F1: 0.7996  |  Val Loss: 0.6682  Bal.Acc: 76.7%  F1: 0.7495  |  LR: 1.00e-04  (19.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6682)
Epoka [ 19/20]  Train Loss: 0.4917  Bal.Acc: 82.0%  F1: 0.8059  |  Val Loss: 0.7175  Bal.Acc: 75.6%  F1: 0.7437  |  LR: 1.00e-04  (19.4s)
Epoka [ 20/20]  Train Loss: 0.4697  Bal.Acc: 81.4%  F1: 0.8055  |  Val Loss: 0.7169  Bal.Acc: 76.8%  F1: 0.7707  |  LR: 1.00e-04  (20.7s)

Trening zakończony. Checkpoint: checkpoints/resnet50_fold5_best.pt
Log CSV: results/resnet50_fold5_training_log.csv
Załadowano najlepsze wagi z epoki 18

Ewaluacja modelu: resnet50_fold5
----------------------------------------
  Balanced Accuracy: 76.72%
  F1 (macro):        0.7495
  Quadratic Cohen's Kappa: 0.8758

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.86      0.80      0.83        87
    Doubtful       0.71      0.57      0.63        81
        Mild       0.52      0.79      0.63        39
    Moderate       0.82      0.87      0.85        38
      Severe       0.82      0.80      0.81        35

    accuracy                           0.74       280
   macro avg       0.75      0.77      0.75       280
weighted avg       0.76      0.74      0.74       280

  Metryki zapisane: results/resnet50_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/resnet50_fold5_test_probs.npz

ZAKOŃCZONO: resnet50. Średnia Kappa z 5 foldów: 0.8949 ±0.0176

================================================================================
 ROZPOCZĘCIE TRENINGU MODELU: efficientnet_b3
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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
model.safetensors: 100% 49.3M/49.3M [00:01<00:00, 34.7MB/s]
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold1
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.2625  Bal.Acc: 34.3%  F1: 0.3215  |  Val Loss: 1.5820  Bal.Acc: 44.1%  F1: 0.3899  |  LR: 1.00e-04  (18.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5820)
Epoka [  2/20]  Train Loss: 1.3377  Bal.Acc: 55.1%  F1: 0.5383  |  Val Loss: 1.3422  Bal.Acc: 53.1%  F1: 0.5359  |  LR: 1.00e-04  (18.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3422)
Epoka [  3/20]  Train Loss: 1.0704  Bal.Acc: 63.3%  F1: 0.6206  |  Val Loss: 0.9358  Bal.Acc: 68.1%  F1: 0.6732  |  LR: 1.00e-04  (17.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9358)
Epoka [  4/20]  Train Loss: 0.8831  Bal.Acc: 67.7%  F1: 0.6676  |  Val Loss: 0.8005  Bal.Acc: 69.8%  F1: 0.6800  |  LR: 1.00e-04  (17.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8005)
Epoka [  5/20]  Train Loss: 0.7811  Bal.Acc: 70.6%  F1: 0.6943  |  Val Loss: 0.8911  Bal.Acc: 68.6%  F1: 0.6866  |  LR: 1.00e-04  (18.2s)
Epoka [  6/20]  Train Loss: 0.6516  Bal.Acc: 76.6%  F1: 0.7586  |  Val Loss: 0.9294  Bal.Acc: 70.0%  F1: 0.7081  |  LR: 1.00e-04  (18.2s)
Epoka [  7/20]  Train Loss: 0.5741  Bal.Acc: 77.1%  F1: 0.7568  |  Val Loss: 0.7218  Bal.Acc: 73.2%  F1: 0.7179  |  LR: 1.00e-04  (17.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7218)
Epoka [  8/20]  Train Loss: 0.4997  Bal.Acc: 81.8%  F1: 0.8106  |  Val Loss: 0.9759  Bal.Acc: 70.3%  F1: 0.7191  |  LR: 1.00e-04  (18.6s)
Epoka [  9/20]  Train Loss: 0.5238  Bal.Acc: 79.4%  F1: 0.7833  |  Val Loss: 0.8225  Bal.Acc: 67.6%  F1: 0.6841  |  LR: 1.00e-04  (17.8s)
Epoka [ 10/20]  Train Loss: 0.4310  Bal.Acc: 82.8%  F1: 0.8179  |  Val Loss: 0.7262  Bal.Acc: 77.9%  F1: 0.7769  |  LR: 1.00e-04  (18.3s)
Epoka [ 11/20]  Train Loss: 0.3568  Bal.Acc: 87.0%  F1: 0.8613  |  Val Loss: 0.7549  Bal.Acc: 75.5%  F1: 0.7498  |  LR: 5.00e-05  (17.8s)
Epoka [ 12/20]  Train Loss: 0.3192  Bal.Acc: 88.0%  F1: 0.8713  |  Val Loss: 0.8990  Bal.Acc: 71.5%  F1: 0.7263  |  LR: 5.00e-05  (17.8s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.7218

Trening zakończony. Checkpoint: checkpoints/efficientnet_b3_fold1_best.pt
Log CSV: results/efficientnet_b3_fold1_training_log.csv
Załadowano najlepsze wagi z epoki 7

Ewaluacja modelu: efficientnet_b3_fold1
----------------------------------------
  Balanced Accuracy: 73.23%
  F1 (macro):        0.7179
  Quadratic Cohen's Kappa: 0.8730

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.74      0.95      0.83        88
    Doubtful       0.84      0.52      0.64        81
        Mild       0.54      0.49      0.51        39
    Moderate       0.72      0.82      0.77        38
      Severe       0.79      0.89      0.84        35

    accuracy                           0.74       281
   macro avg       0.73      0.73      0.72       281
weighted avg       0.74      0.74      0.72       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold2
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.1770  Bal.Acc: 34.8%  F1: 0.3307  |  Val Loss: 1.7793  Bal.Acc: 40.2%  F1: 0.3739  |  LR: 1.00e-04  (18.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.7793)
Epoka [  2/20]  Train Loss: 1.3487  Bal.Acc: 54.9%  F1: 0.5236  |  Val Loss: 1.4427  Bal.Acc: 52.6%  F1: 0.4904  |  LR: 1.00e-04  (18.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4427)
Epoka [  3/20]  Train Loss: 1.0373  Bal.Acc: 63.8%  F1: 0.6241  |  Val Loss: 1.2891  Bal.Acc: 60.2%  F1: 0.6087  |  LR: 1.00e-04  (18.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2891)
Epoka [  4/20]  Train Loss: 0.8750  Bal.Acc: 68.7%  F1: 0.6689  |  Val Loss: 1.2183  Bal.Acc: 59.9%  F1: 0.5979  |  LR: 1.00e-04  (20.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2183)
Epoka [  5/20]  Train Loss: 0.7481  Bal.Acc: 71.7%  F1: 0.7063  |  Val Loss: 0.9791  Bal.Acc: 66.8%  F1: 0.6708  |  LR: 1.00e-04  (19.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9791)
Epoka [  6/20]  Train Loss: 0.6818  Bal.Acc: 74.5%  F1: 0.7294  |  Val Loss: 0.9402  Bal.Acc: 67.6%  F1: 0.6829  |  LR: 1.00e-04  (20.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9402)
Epoka [  7/20]  Train Loss: 0.6225  Bal.Acc: 77.8%  F1: 0.7694  |  Val Loss: 0.8194  Bal.Acc: 71.1%  F1: 0.7030  |  LR: 1.00e-04  (19.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8194)
Epoka [  8/20]  Train Loss: 0.5448  Bal.Acc: 79.7%  F1: 0.7895  |  Val Loss: 0.8140  Bal.Acc: 73.0%  F1: 0.7302  |  LR: 1.00e-04  (18.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8140)
Epoka [  9/20]  Train Loss: 0.4497  Bal.Acc: 84.1%  F1: 0.8331  |  Val Loss: 0.7298  Bal.Acc: 73.6%  F1: 0.7399  |  LR: 1.00e-04  (18.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7298)
Epoka [ 10/20]  Train Loss: 0.4385  Bal.Acc: 84.4%  F1: 0.8272  |  Val Loss: 0.7029  Bal.Acc: 75.5%  F1: 0.7505  |  LR: 1.00e-04  (18.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7029)
Epoka [ 11/20]  Train Loss: 0.4166  Bal.Acc: 85.1%  F1: 0.8431  |  Val Loss: 0.7921  Bal.Acc: 72.9%  F1: 0.7343  |  LR: 1.00e-04  (18.5s)
Epoka [ 12/20]  Train Loss: 0.3766  Bal.Acc: 85.9%  F1: 0.8489  |  Val Loss: 0.6809  Bal.Acc: 76.4%  F1: 0.7636  |  LR: 1.00e-04  (19.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6809)
Epoka [ 13/20]  Train Loss: 0.4096  Bal.Acc: 84.6%  F1: 0.8419  |  Val Loss: 0.6756  Bal.Acc: 76.0%  F1: 0.7582  |  LR: 1.00e-04  (19.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6756)
Epoka [ 14/20]  Train Loss: 0.3639  Bal.Acc: 86.9%  F1: 0.8625  |  Val Loss: 0.8216  Bal.Acc: 75.1%  F1: 0.7515  |  LR: 1.00e-04  (19.2s)
Epoka [ 15/20]  Train Loss: 0.3048  Bal.Acc: 88.0%  F1: 0.8690  |  Val Loss: 0.6902  Bal.Acc: 78.5%  F1: 0.7783  |  LR: 1.00e-04  (18.9s)
Epoka [ 16/20]  Train Loss: 0.3218  Bal.Acc: 87.6%  F1: 0.8713  |  Val Loss: 0.8113  Bal.Acc: 76.3%  F1: 0.7595  |  LR: 1.00e-04  (18.3s)
Epoka [ 17/20]  Train Loss: 0.2616  Bal.Acc: 89.7%  F1: 0.8892  |  Val Loss: 0.9195  Bal.Acc: 74.9%  F1: 0.7518  |  LR: 5.00e-05  (18.3s)
Epoka [ 18/20]  Train Loss: 0.3098  Bal.Acc: 88.0%  F1: 0.8650  |  Val Loss: 0.7279  Bal.Acc: 78.0%  F1: 0.7694  |  LR: 5.00e-05  (18.1s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.6756

Trening zakończony. Checkpoint: checkpoints/efficientnet_b3_fold2_best.pt
Log CSV: results/efficientnet_b3_fold2_training_log.csv
Załadowano najlepsze wagi z epoki 13

Ewaluacja modelu: efficientnet_b3_fold2
----------------------------------------
  Balanced Accuracy: 75.96%
  F1 (macro):        0.7582
  Quadratic Cohen's Kappa: 0.8992

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.75      0.88      0.81        88
    Doubtful       0.71      0.58      0.64        81
        Mild       0.62      0.64      0.63        39
    Moderate       0.84      0.82      0.83        38
      Severe       0.89      0.89      0.89        35

    accuracy                           0.75       281
   macro avg       0.76      0.76      0.76       281
weighted avg       0.75      0.75      0.75       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold3
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.3297  Bal.Acc: 35.8%  F1: 0.3401  |  Val Loss: 1.8342  Bal.Acc: 40.3%  F1: 0.3725  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.8342)
Epoka [  2/20]  Train Loss: 1.3869  Bal.Acc: 53.6%  F1: 0.5176  |  Val Loss: 1.4737  Bal.Acc: 56.1%  F1: 0.5578  |  LR: 1.00e-04  (18.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4737)
Epoka [  3/20]  Train Loss: 1.0980  Bal.Acc: 61.4%  F1: 0.5950  |  Val Loss: 1863.1332  Bal.Acc: 62.9%  F1: 0.6215  |  LR: 1.00e-04  (19.5s)
Epoka [  4/20]  Train Loss: 0.8936  Bal.Acc: 66.9%  F1: 0.6592  |  Val Loss: 1.0330  Bal.Acc: 65.3%  F1: 0.6510  |  LR: 1.00e-04  (18.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0330)
Epoka [  5/20]  Train Loss: 0.7476  Bal.Acc: 71.4%  F1: 0.6979  |  Val Loss: 118.2301  Bal.Acc: 70.8%  F1: 0.7074  |  LR: 1.00e-04  (19.4s)
Epoka [  6/20]  Train Loss: 0.6083  Bal.Acc: 76.6%  F1: 0.7529  |  Val Loss: 1.4202  Bal.Acc: 67.7%  F1: 0.6808  |  LR: 1.00e-04  (18.5s)
Epoka [  7/20]  Train Loss: 0.5986  Bal.Acc: 78.0%  F1: 0.7744  |  Val Loss: 364.5292  Bal.Acc: 67.9%  F1: 0.6722  |  LR: 1.00e-04  (18.2s)
Epoka [  8/20]  Train Loss: 0.5261  Bal.Acc: 79.4%  F1: 0.7896  |  Val Loss: 2268.9497  Bal.Acc: 68.1%  F1: 0.6853  |  LR: 5.00e-05  (18.1s)
Epoka [  9/20]  Train Loss: 0.4403  Bal.Acc: 83.2%  F1: 0.8157  |  Val Loss: 5.4875  Bal.Acc: 73.1%  F1: 0.7307  |  LR: 5.00e-05  (18.4s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 1.0330

Trening zakończony. Checkpoint: checkpoints/efficientnet_b3_fold3_best.pt
Log CSV: results/efficientnet_b3_fold3_training_log.csv
Załadowano najlepsze wagi z epoki 4

Ewaluacja modelu: efficientnet_b3_fold3
----------------------------------------
  Balanced Accuracy: 65.29%
  F1 (macro):        0.6510
  Quadratic Cohen's Kappa: 0.7878

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.74      0.75      0.74        87
    Doubtful       0.57      0.57      0.57        81
        Mild       0.47      0.45      0.46        40
    Moderate       0.76      0.76      0.76        37
      Severe       0.70      0.74      0.72        35

    accuracy                           0.65       280
   macro avg       0.65      0.65      0.65       280
weighted avg       0.65      0.65      0.65       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold4
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.2925  Bal.Acc: 33.1%  F1: 0.3092  |  Val Loss: 1.9409  Bal.Acc: 33.3%  F1: 0.3214  |  LR: 1.00e-04  (18.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.9409)
Epoka [  2/20]  Train Loss: 1.2599  Bal.Acc: 56.6%  F1: 0.5480  |  Val Loss: 1.3920  Bal.Acc: 54.1%  F1: 0.5415  |  LR: 1.00e-04  (18.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3920)
Epoka [  3/20]  Train Loss: 1.0573  Bal.Acc: 61.6%  F1: 0.6018  |  Val Loss: 1.0888  Bal.Acc: 62.9%  F1: 0.6242  |  LR: 1.00e-04  (19.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0888)
Epoka [  4/20]  Train Loss: 0.8751  Bal.Acc: 66.7%  F1: 0.6480  |  Val Loss: 0.9322  Bal.Acc: 67.4%  F1: 0.6739  |  LR: 1.00e-04  (19.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9322)
Epoka [  5/20]  Train Loss: 0.7616  Bal.Acc: 73.3%  F1: 0.7234  |  Val Loss: 0.9158  Bal.Acc: 67.4%  F1: 0.6799  |  LR: 1.00e-04  (19.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9158)
Epoka [  6/20]  Train Loss: 0.5969  Bal.Acc: 77.6%  F1: 0.7721  |  Val Loss: 0.8560  Bal.Acc: 69.4%  F1: 0.7019  |  LR: 1.00e-04  (19.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8560)
Epoka [  7/20]  Train Loss: 0.5532  Bal.Acc: 80.0%  F1: 0.7905  |  Val Loss: 0.7921  Bal.Acc: 69.7%  F1: 0.7073  |  LR: 1.00e-04  (19.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7921)
Epoka [  8/20]  Train Loss: 0.4659  Bal.Acc: 82.0%  F1: 0.8084  |  Val Loss: 0.7691  Bal.Acc: 72.1%  F1: 0.7217  |  LR: 1.00e-04  (19.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7691)
Epoka [  9/20]  Train Loss: 0.4254  Bal.Acc: 84.6%  F1: 0.8384  |  Val Loss: 0.7356  Bal.Acc: 77.1%  F1: 0.7773  |  LR: 1.00e-04  (19.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7356)
Epoka [ 10/20]  Train Loss: 0.4008  Bal.Acc: 85.2%  F1: 0.8521  |  Val Loss: 0.7139  Bal.Acc: 77.1%  F1: 0.7600  |  LR: 1.00e-04  (18.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7139)
Epoka [ 11/20]  Train Loss: 0.3860  Bal.Acc: 85.3%  F1: 0.8461  |  Val Loss: 0.7676  Bal.Acc: 75.7%  F1: 0.7677  |  LR: 1.00e-04  (18.1s)
Epoka [ 12/20]  Train Loss: 0.3495  Bal.Acc: 86.3%  F1: 0.8584  |  Val Loss: 0.7785  Bal.Acc: 75.7%  F1: 0.7636  |  LR: 1.00e-04  (19.3s)
Epoka [ 13/20]  Train Loss: 0.3121  Bal.Acc: 87.9%  F1: 0.8739  |  Val Loss: 0.7244  Bal.Acc: 76.4%  F1: 0.7733  |  LR: 1.00e-04  (17.9s)
Epoka [ 14/20]  Train Loss: 0.3468  Bal.Acc: 86.8%  F1: 0.8616  |  Val Loss: 0.7673  Bal.Acc: 77.2%  F1: 0.7732  |  LR: 5.00e-05  (17.7s)
Epoka [ 15/20]  Train Loss: 0.2783  Bal.Acc: 90.3%  F1: 0.8932  |  Val Loss: 0.7533  Bal.Acc: 79.1%  F1: 0.7880  |  LR: 5.00e-05  (18.1s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.7139

Trening zakończony. Checkpoint: checkpoints/efficientnet_b3_fold4_best.pt
Log CSV: results/efficientnet_b3_fold4_training_log.csv
Załadowano najlepsze wagi z epoki 10

Ewaluacja modelu: efficientnet_b3_fold4
----------------------------------------
  Balanced Accuracy: 77.07%
  F1 (macro):        0.7600
  Quadratic Cohen's Kappa: 0.8606

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.86      0.62      0.72        87
    Doubtful       0.62      0.74      0.68        81
        Mild       0.70      0.80      0.74        40
    Moderate       0.80      0.89      0.85        37
      Severe       0.82      0.80      0.81        35

    accuracy                           0.74       280
   macro avg       0.76      0.77      0.76       280
weighted avg       0.76      0.74      0.74       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: efficientnet_b3
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3_fold5
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.3411  Bal.Acc: 36.1%  F1: 0.3329  |  Val Loss: 2.1484  Bal.Acc: 33.7%  F1: 0.3253  |  LR: 1.00e-04  (17.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 2.1484)
Epoka [  2/20]  Train Loss: 1.3705  Bal.Acc: 53.6%  F1: 0.5280  |  Val Loss: 10.0290  Bal.Acc: 48.8%  F1: 0.4920  |  LR: 1.00e-04  (17.8s)
Epoka [  3/20]  Train Loss: 1.1263  Bal.Acc: 59.0%  F1: 0.5630  |  Val Loss: 1069.5533  Bal.Acc: 53.6%  F1: 0.5249  |  LR: 1.00e-04  (18.4s)
Epoka [  4/20]  Train Loss: 0.9026  Bal.Acc: 65.1%  F1: 0.6361  |  Val Loss: 1162.9531  Bal.Acc: 62.8%  F1: 0.6204  |  LR: 1.00e-04  (18.0s)
Epoka [  5/20]  Train Loss: 0.7276  Bal.Acc: 71.2%  F1: 0.7063  |  Val Loss: 0.9329  Bal.Acc: 65.6%  F1: 0.6416  |  LR: 1.00e-04  (17.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9329)
Epoka [  6/20]  Train Loss: 0.6387  Bal.Acc: 76.4%  F1: 0.7565  |  Val Loss: 3.7982  Bal.Acc: 66.0%  F1: 0.6588  |  LR: 1.00e-04  (18.5s)
Epoka [  7/20]  Train Loss: 0.5872  Bal.Acc: 77.7%  F1: 0.7644  |  Val Loss: 7.2328  Bal.Acc: 68.6%  F1: 0.6859  |  LR: 1.00e-04  (17.7s)
Epoka [  8/20]  Train Loss: 0.5017  Bal.Acc: 80.9%  F1: 0.8011  |  Val Loss: 579.0088  Bal.Acc: 70.2%  F1: 0.7011  |  LR: 1.00e-04  (18.0s)
Epoka [  9/20]  Train Loss: 0.4755  Bal.Acc: 82.0%  F1: 0.8068  |  Val Loss: 0.8528  Bal.Acc: 69.4%  F1: 0.6908  |  LR: 1.00e-04  (17.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8528)
Epoka [ 10/20]  Train Loss: 0.4118  Bal.Acc: 83.7%  F1: 0.8319  |  Val Loss: 0.9275  Bal.Acc: 72.0%  F1: 0.7250  |  LR: 1.00e-04  (18.6s)
Epoka [ 11/20]  Train Loss: 0.3951  Bal.Acc: 85.5%  F1: 0.8410  |  Val Loss: 5.3490  Bal.Acc: 73.3%  F1: 0.7317  |  LR: 1.00e-04  (18.0s)
Epoka [ 12/20]  Train Loss: 0.3444  Bal.Acc: 86.6%  F1: 0.8543  |  Val Loss: 0.8809  Bal.Acc: 74.6%  F1: 0.7492  |  LR: 1.00e-04  (17.7s)
Epoka [ 13/20]  Train Loss: 0.3462  Bal.Acc: 86.3%  F1: 0.8550  |  Val Loss: 0.8157  Bal.Acc: 75.4%  F1: 0.7609  |  LR: 1.00e-04  (18.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8157)
Epoka [ 14/20]  Train Loss: 0.2635  Bal.Acc: 89.9%  F1: 0.8921  |  Val Loss: 2.8388  Bal.Acc: 74.2%  F1: 0.7463  |  LR: 1.00e-04  (19.1s)
Epoka [ 15/20]  Train Loss: 0.2325  Bal.Acc: 91.1%  F1: 0.9102  |  Val Loss: 0.8272  Bal.Acc: 72.2%  F1: 0.7247  |  LR: 1.00e-04  (17.8s)
Epoka [ 16/20]  Train Loss: 0.2380  Bal.Acc: 90.3%  F1: 0.8953  |  Val Loss: 0.9240  Bal.Acc: 72.5%  F1: 0.7307  |  LR: 1.00e-04  (18.1s)
Epoka [ 17/20]  Train Loss: 0.2219  Bal.Acc: 91.6%  F1: 0.9117  |  Val Loss: 20.6904  Bal.Acc: 75.8%  F1: 0.7655  |  LR: 5.00e-05  (17.9s)
Epoka [ 18/20]  Train Loss: 0.2057  Bal.Acc: 92.2%  F1: 0.9155  |  Val Loss: 176.3063  Bal.Acc: 76.6%  F1: 0.7715  |  LR: 5.00e-05  (17.7s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.8157

Trening zakończony. Checkpoint: checkpoints/efficientnet_b3_fold5_best.pt
Log CSV: results/efficientnet_b3_fold5_training_log.csv
Załadowano najlepsze wagi z epoki 13

Ewaluacja modelu: efficientnet_b3_fold5
----------------------------------------
  Balanced Accuracy: 75.40%
  F1 (macro):        0.7609
  Quadratic Cohen's Kappa: 0.8820

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.73      0.89      0.80        87
    Doubtful       0.73      0.63      0.68        81
        Mild       0.73      0.62      0.67        39
    Moderate       0.82      0.87      0.85        38
      Severe       0.87      0.77      0.82        35

    accuracy                           0.76       280
   macro avg       0.78      0.75      0.76       280
weighted avg       0.76      0.76      0.75       280

  Metryki zapisane: results/efficientnet_b3_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/efficientnet_b3_fold5_test_probs.npz

ZAKOŃCZONO: efficientnet_b3. Średnia Kappa z 5 foldów: 0.8605 ±0.0385

================================================================================
 ROZPOCZĘCIE TRENINGU MODELU: densenet121
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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
model.safetensors: 100% 32.3M/32.3M [00:01<00:00, 17.9MB/s]
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold1
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.4756  Bal.Acc: 37.3%  F1: 0.3466  |  Val Loss: 1.2572  Bal.Acc: 49.3%  F1: 0.4086  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2572)
Epoka [  2/20]  Train Loss: 1.0909  Bal.Acc: 63.1%  F1: 0.5981  |  Val Loss: 0.9561  Bal.Acc: 66.6%  F1: 0.6580  |  LR: 1.00e-04  (17.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9561)
Epoka [  3/20]  Train Loss: 0.8650  Bal.Acc: 68.7%  F1: 0.6700  |  Val Loss: 0.8212  Bal.Acc: 68.0%  F1: 0.6796  |  LR: 1.00e-04  (18.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8212)
Epoka [  4/20]  Train Loss: 0.7072  Bal.Acc: 74.8%  F1: 0.7434  |  Val Loss: 0.6758  Bal.Acc: 69.5%  F1: 0.6771  |  LR: 1.00e-04  (18.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6758)
Epoka [  5/20]  Train Loss: 0.6085  Bal.Acc: 77.0%  F1: 0.7611  |  Val Loss: 0.5687  Bal.Acc: 79.9%  F1: 0.7934  |  LR: 1.00e-04  (18.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5687)
Epoka [  6/20]  Train Loss: 0.5291  Bal.Acc: 81.3%  F1: 0.8088  |  Val Loss: 0.5655  Bal.Acc: 78.3%  F1: 0.7707  |  LR: 1.00e-04  (18.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5655)
Epoka [  7/20]  Train Loss: 0.4577  Bal.Acc: 84.5%  F1: 0.8403  |  Val Loss: 0.4981  Bal.Acc: 83.3%  F1: 0.8176  |  LR: 1.00e-04  (18.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.4981)
Epoka [  8/20]  Train Loss: 0.4274  Bal.Acc: 83.5%  F1: 0.8306  |  Val Loss: 0.5145  Bal.Acc: 82.2%  F1: 0.8138  |  LR: 1.00e-04  (18.3s)
Epoka [  9/20]  Train Loss: 0.3937  Bal.Acc: 86.9%  F1: 0.8648  |  Val Loss: 0.5102  Bal.Acc: 77.9%  F1: 0.7827  |  LR: 1.00e-04  (18.7s)
Epoka [ 10/20]  Train Loss: 0.3317  Bal.Acc: 89.6%  F1: 0.8879  |  Val Loss: 0.4651  Bal.Acc: 84.1%  F1: 0.8233  |  LR: 1.00e-04  (17.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.4651)
Epoka [ 11/20]  Train Loss: 0.3522  Bal.Acc: 87.0%  F1: 0.8653  |  Val Loss: 0.5303  Bal.Acc: 82.5%  F1: 0.8055  |  LR: 1.00e-04  (18.1s)
Epoka [ 12/20]  Train Loss: 0.3022  Bal.Acc: 89.9%  F1: 0.8942  |  Val Loss: 0.4859  Bal.Acc: 82.1%  F1: 0.7906  |  LR: 1.00e-04  (18.1s)
Epoka [ 13/20]  Train Loss: 0.2570  Bal.Acc: 91.0%  F1: 0.9072  |  Val Loss: 0.5119  Bal.Acc: 82.8%  F1: 0.8119  |  LR: 1.00e-04  (17.9s)
Epoka [ 14/20]  Train Loss: 0.2576  Bal.Acc: 90.8%  F1: 0.9063  |  Val Loss: 0.4903  Bal.Acc: 84.4%  F1: 0.8243  |  LR: 5.00e-05  (17.7s)
Epoka [ 15/20]  Train Loss: 0.1923  Bal.Acc: 93.6%  F1: 0.9323  |  Val Loss: 0.4583  Bal.Acc: 86.2%  F1: 0.8447  |  LR: 5.00e-05  (17.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.4583)
Epoka [ 16/20]  Train Loss: 0.1754  Bal.Acc: 94.4%  F1: 0.9397  |  Val Loss: 0.4546  Bal.Acc: 86.9%  F1: 0.8506  |  LR: 5.00e-05  (18.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.4546)
Epoka [ 17/20]  Train Loss: 0.1797  Bal.Acc: 93.8%  F1: 0.9365  |  Val Loss: 0.4775  Bal.Acc: 84.2%  F1: 0.8208  |  LR: 5.00e-05  (19.0s)
Epoka [ 18/20]  Train Loss: 0.1543  Bal.Acc: 94.8%  F1: 0.9443  |  Val Loss: 0.4912  Bal.Acc: 84.9%  F1: 0.8370  |  LR: 5.00e-05  (17.7s)
Epoka [ 19/20]  Train Loss: 0.1528  Bal.Acc: 95.7%  F1: 0.9558  |  Val Loss: 0.4971  Bal.Acc: 84.4%  F1: 0.8385  |  LR: 5.00e-05  (17.7s)
Epoka [ 20/20]  Train Loss: 0.1549  Bal.Acc: 95.3%  F1: 0.9506  |  Val Loss: 0.5414  Bal.Acc: 84.4%  F1: 0.8380  |  LR: 2.50e-05  (18.0s)

Trening zakończony. Checkpoint: checkpoints/densenet121_fold1_best.pt
Log CSV: results/densenet121_fold1_training_log.csv
Załadowano najlepsze wagi z epoki 16

Ewaluacja modelu: densenet121_fold1
----------------------------------------
  Balanced Accuracy: 86.86%
  F1 (macro):        0.8506
  Quadratic Cohen's Kappa: 0.9313

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.88      0.89      0.88        88
    Doubtful       0.80      0.64      0.71        81
        Mild       0.67      0.87      0.76        39
    Moderate       0.95      1.00      0.97        38
      Severe       0.92      0.94      0.93        35

    accuracy                           0.84       281
   macro avg       0.84      0.87      0.85       281
weighted avg       0.84      0.84      0.83       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold2
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.3849  Bal.Acc: 43.7%  F1: 0.3972  |  Val Loss: 1.2406  Bal.Acc: 47.9%  F1: 0.4732  |  LR: 1.00e-04  (17.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2406)
Epoka [  2/20]  Train Loss: 0.9999  Bal.Acc: 62.5%  F1: 0.6032  |  Val Loss: 1.0913  Bal.Acc: 50.6%  F1: 0.5013  |  LR: 1.00e-04  (18.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0913)
Epoka [  3/20]  Train Loss: 0.8152  Bal.Acc: 68.0%  F1: 0.6728  |  Val Loss: 0.8550  Bal.Acc: 64.0%  F1: 0.6352  |  LR: 1.00e-04  (18.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8550)
Epoka [  4/20]  Train Loss: 0.6695  Bal.Acc: 74.8%  F1: 0.7461  |  Val Loss: 0.6701  Bal.Acc: 74.5%  F1: 0.7393  |  LR: 1.00e-04  (18.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6701)
Epoka [  5/20]  Train Loss: 0.5876  Bal.Acc: 77.2%  F1: 0.7656  |  Val Loss: 0.6122  Bal.Acc: 76.5%  F1: 0.7616  |  LR: 1.00e-04  (18.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6122)
Epoka [  6/20]  Train Loss: 0.4957  Bal.Acc: 81.7%  F1: 0.8107  |  Val Loss: 0.5794  Bal.Acc: 79.2%  F1: 0.7760  |  LR: 1.00e-04  (18.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5794)
Epoka [  7/20]  Train Loss: 0.4591  Bal.Acc: 83.3%  F1: 0.8281  |  Val Loss: 0.6143  Bal.Acc: 76.3%  F1: 0.7580  |  LR: 1.00e-04  (18.4s)
Epoka [  8/20]  Train Loss: 0.4087  Bal.Acc: 85.9%  F1: 0.8525  |  Val Loss: 0.5955  Bal.Acc: 76.9%  F1: 0.7727  |  LR: 1.00e-04  (17.7s)
Epoka [  9/20]  Train Loss: 0.3529  Bal.Acc: 88.1%  F1: 0.8772  |  Val Loss: 0.5662  Bal.Acc: 79.7%  F1: 0.7904  |  LR: 1.00e-04  (17.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5662)
Epoka [ 10/20]  Train Loss: 0.3171  Bal.Acc: 88.9%  F1: 0.8810  |  Val Loss: 0.5390  Bal.Acc: 81.3%  F1: 0.8091  |  LR: 1.00e-04  (18.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5390)
Epoka [ 11/20]  Train Loss: 0.2754  Bal.Acc: 89.9%  F1: 0.8988  |  Val Loss: 0.4432  Bal.Acc: 82.5%  F1: 0.8196  |  LR: 1.00e-04  (19.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.4432)
Epoka [ 12/20]  Train Loss: 0.2623  Bal.Acc: 91.3%  F1: 0.9085  |  Val Loss: 0.4579  Bal.Acc: 80.9%  F1: 0.8097  |  LR: 1.00e-04  (18.0s)
Epoka [ 13/20]  Train Loss: 0.2517  Bal.Acc: 90.5%  F1: 0.9030  |  Val Loss: 0.5077  Bal.Acc: 82.6%  F1: 0.8215  |  LR: 1.00e-04  (18.0s)
Epoka [ 14/20]  Train Loss: 0.2135  Bal.Acc: 91.9%  F1: 0.9139  |  Val Loss: 0.5598  Bal.Acc: 79.8%  F1: 0.7796  |  LR: 1.00e-04  (17.7s)
Epoka [ 15/20]  Train Loss: 0.2224  Bal.Acc: 92.2%  F1: 0.9204  |  Val Loss: 0.5029  Bal.Acc: 81.4%  F1: 0.8197  |  LR: 5.00e-05  (17.6s)
Epoka [ 16/20]  Train Loss: 0.1766  Bal.Acc: 94.3%  F1: 0.9377  |  Val Loss: 0.4949  Bal.Acc: 82.0%  F1: 0.8137  |  LR: 5.00e-05  (17.7s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.4432

Trening zakończony. Checkpoint: checkpoints/densenet121_fold2_best.pt
Log CSV: results/densenet121_fold2_training_log.csv
Załadowano najlepsze wagi z epoki 11

Ewaluacja modelu: densenet121_fold2
----------------------------------------
  Balanced Accuracy: 82.50%
  F1 (macro):        0.8196
  Quadratic Cohen's Kappa: 0.9295

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.87      0.92      0.90        88
    Doubtful       0.81      0.70      0.75        81
        Mild       0.62      0.72      0.67        39
    Moderate       0.89      0.87      0.88        38
      Severe       0.89      0.91      0.90        35

    accuracy                           0.82       281
   macro avg       0.82      0.82      0.82       281
weighted avg       0.83      0.82      0.82       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold3
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.5011  Bal.Acc: 35.3%  F1: 0.3285  |  Val Loss: 1.2901  Bal.Acc: 50.2%  F1: 0.4982  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2901)
Epoka [  2/20]  Train Loss: 1.1315  Bal.Acc: 56.9%  F1: 0.5316  |  Val Loss: 1.0516  Bal.Acc: 60.6%  F1: 0.6064  |  LR: 1.00e-04  (18.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0516)
Epoka [  3/20]  Train Loss: 0.8733  Bal.Acc: 69.5%  F1: 0.6855  |  Val Loss: 0.8813  Bal.Acc: 65.6%  F1: 0.6572  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8813)
Epoka [  4/20]  Train Loss: 0.7120  Bal.Acc: 73.6%  F1: 0.7224  |  Val Loss: 0.7750  Bal.Acc: 69.2%  F1: 0.6898  |  LR: 1.00e-04  (19.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7750)
Epoka [  5/20]  Train Loss: 0.6124  Bal.Acc: 77.3%  F1: 0.7672  |  Val Loss: 0.7355  Bal.Acc: 70.7%  F1: 0.7091  |  LR: 1.00e-04  (18.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7355)
Epoka [  6/20]  Train Loss: 0.5412  Bal.Acc: 81.0%  F1: 0.8084  |  Val Loss: 0.6523  Bal.Acc: 76.6%  F1: 0.7550  |  LR: 1.00e-04  (19.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6523)
Epoka [  7/20]  Train Loss: 0.4606  Bal.Acc: 81.9%  F1: 0.8110  |  Val Loss: 0.6025  Bal.Acc: 78.6%  F1: 0.7687  |  LR: 1.00e-04  (18.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6025)
Epoka [  8/20]  Train Loss: 0.4202  Bal.Acc: 84.0%  F1: 0.8348  |  Val Loss: 0.6084  Bal.Acc: 77.8%  F1: 0.7593  |  LR: 1.00e-04  (18.9s)
Epoka [  9/20]  Train Loss: 0.3751  Bal.Acc: 86.3%  F1: 0.8548  |  Val Loss: 0.6393  Bal.Acc: 75.1%  F1: 0.7532  |  LR: 1.00e-04  (17.8s)
Epoka [ 10/20]  Train Loss: 0.3529  Bal.Acc: 87.5%  F1: 0.8707  |  Val Loss: 0.6107  Bal.Acc: 76.8%  F1: 0.7537  |  LR: 1.00e-04  (17.8s)
Epoka [ 11/20]  Train Loss: 0.3376  Bal.Acc: 88.8%  F1: 0.8832  |  Val Loss: 0.5798  Bal.Acc: 78.5%  F1: 0.7852  |  LR: 1.00e-04  (17.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5798)
Epoka [ 12/20]  Train Loss: 0.2974  Bal.Acc: 90.2%  F1: 0.8993  |  Val Loss: 0.5523  Bal.Acc: 82.2%  F1: 0.8060  |  LR: 1.00e-04  (18.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5523)
Epoka [ 13/20]  Train Loss: 0.2627  Bal.Acc: 91.9%  F1: 0.9157  |  Val Loss: 0.5291  Bal.Acc: 82.9%  F1: 0.8227  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5291)
Epoka [ 14/20]  Train Loss: 0.2645  Bal.Acc: 90.4%  F1: 0.9021  |  Val Loss: 0.6023  Bal.Acc: 80.1%  F1: 0.8042  |  LR: 1.00e-04  (19.5s)
Epoka [ 15/20]  Train Loss: 0.2726  Bal.Acc: 90.1%  F1: 0.8967  |  Val Loss: 0.5211  Bal.Acc: 80.6%  F1: 0.7999  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5211)
Epoka [ 16/20]  Train Loss: 0.2183  Bal.Acc: 92.4%  F1: 0.9220  |  Val Loss: 0.4909  Bal.Acc: 81.4%  F1: 0.8046  |  LR: 1.00e-04  (18.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.4909)
Epoka [ 17/20]  Train Loss: 0.1890  Bal.Acc: 93.9%  F1: 0.9336  |  Val Loss: 0.5264  Bal.Acc: 79.5%  F1: 0.7881  |  LR: 1.00e-04  (18.5s)
Epoka [ 18/20]  Train Loss: 0.1925  Bal.Acc: 93.1%  F1: 0.9273  |  Val Loss: 0.5496  Bal.Acc: 80.3%  F1: 0.7920  |  LR: 1.00e-04  (18.2s)
Epoka [ 19/20]  Train Loss: 0.1946  Bal.Acc: 93.7%  F1: 0.9305  |  Val Loss: 0.5621  Bal.Acc: 81.7%  F1: 0.8241  |  LR: 1.00e-04  (17.7s)
Epoka [ 20/20]  Train Loss: 0.2031  Bal.Acc: 93.1%  F1: 0.9239  |  Val Loss: 0.5747  Bal.Acc: 81.5%  F1: 0.8206  |  LR: 5.00e-05  (17.7s)

Trening zakończony. Checkpoint: checkpoints/densenet121_fold3_best.pt
Log CSV: results/densenet121_fold3_training_log.csv
Załadowano najlepsze wagi z epoki 16

Ewaluacja modelu: densenet121_fold3
----------------------------------------
  Balanced Accuracy: 81.41%
  F1 (macro):        0.8046
  Quadratic Cohen's Kappa: 0.8964

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.92      0.87      0.89        87
    Doubtful       0.73      0.69      0.71        81
        Mild       0.66      0.72      0.69        40
    Moderate       0.89      0.84      0.86        37
      Severe       0.80      0.94      0.87        35

    accuracy                           0.80       280
   macro avg       0.80      0.81      0.80       280
weighted avg       0.81      0.80      0.80       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold4
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.4677  Bal.Acc: 38.2%  F1: 0.3472  |  Val Loss: 1.3339  Bal.Acc: 47.7%  F1: 0.4007  |  LR: 1.00e-04  (18.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3339)
Epoka [  2/20]  Train Loss: 1.1024  Bal.Acc: 59.5%  F1: 0.5616  |  Val Loss: 1.0807  Bal.Acc: 56.9%  F1: 0.5614  |  LR: 1.00e-04  (17.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0807)
Epoka [  3/20]  Train Loss: 0.8751  Bal.Acc: 70.4%  F1: 0.6947  |  Val Loss: 0.8808  Bal.Acc: 65.7%  F1: 0.6464  |  LR: 1.00e-04  (18.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8808)
Epoka [  4/20]  Train Loss: 0.6925  Bal.Acc: 74.4%  F1: 0.7326  |  Val Loss: 0.7390  Bal.Acc: 75.6%  F1: 0.7571  |  LR: 1.00e-04  (18.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7390)
Epoka [  5/20]  Train Loss: 0.6036  Bal.Acc: 79.0%  F1: 0.7785  |  Val Loss: 0.6824  Bal.Acc: 74.9%  F1: 0.7451  |  LR: 1.00e-04  (18.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6824)
Epoka [  6/20]  Train Loss: 0.5190  Bal.Acc: 81.9%  F1: 0.8116  |  Val Loss: 0.6274  Bal.Acc: 77.7%  F1: 0.7784  |  LR: 1.00e-04  (19.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6274)
Epoka [  7/20]  Train Loss: 0.4383  Bal.Acc: 85.1%  F1: 0.8466  |  Val Loss: 0.6008  Bal.Acc: 79.1%  F1: 0.7900  |  LR: 1.00e-04  (18.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6008)
Epoka [  8/20]  Train Loss: 0.4356  Bal.Acc: 85.0%  F1: 0.8421  |  Val Loss: 0.6305  Bal.Acc: 77.4%  F1: 0.7736  |  LR: 1.00e-04  (18.6s)
Epoka [  9/20]  Train Loss: 0.3701  Bal.Acc: 87.6%  F1: 0.8680  |  Val Loss: 0.6114  Bal.Acc: 77.7%  F1: 0.7733  |  LR: 1.00e-04  (17.7s)
Epoka [ 10/20]  Train Loss: 0.3251  Bal.Acc: 89.3%  F1: 0.8863  |  Val Loss: 0.6151  Bal.Acc: 78.2%  F1: 0.7835  |  LR: 1.00e-04  (17.5s)
Epoka [ 11/20]  Train Loss: 0.3220  Bal.Acc: 87.4%  F1: 0.8679  |  Val Loss: 0.6015  Bal.Acc: 82.5%  F1: 0.8295  |  LR: 5.00e-05  (17.8s)
Epoka [ 12/20]  Train Loss: 0.2672  Bal.Acc: 91.8%  F1: 0.9130  |  Val Loss: 0.6271  Bal.Acc: 78.2%  F1: 0.7884  |  LR: 5.00e-05  (17.9s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.6008

Trening zakończony. Checkpoint: checkpoints/densenet121_fold4_best.pt
Log CSV: results/densenet121_fold4_training_log.csv
Załadowano najlepsze wagi z epoki 7

Ewaluacja modelu: densenet121_fold4
----------------------------------------
  Balanced Accuracy: 79.08%
  F1 (macro):        0.7900
  Quadratic Cohen's Kappa: 0.8961

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.81      0.83      0.82        87
    Doubtful       0.75      0.75      0.75        81
        Mild       0.76      0.65      0.70        40
    Moderate       0.86      0.84      0.85        37
      Severe       0.78      0.89      0.83        35

    accuracy                           0.79       280
   macro avg       0.79      0.79      0.79       280
weighted avg       0.79      0.79      0.79       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: densenet121
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121_fold5
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.4095  Bal.Acc: 42.1%  F1: 0.4065  |  Val Loss: 1.2602  Bal.Acc: 50.8%  F1: 0.5022  |  LR: 1.00e-04  (17.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2602)
Epoka [  2/20]  Train Loss: 1.0148  Bal.Acc: 62.9%  F1: 0.6030  |  Val Loss: 0.9623  Bal.Acc: 63.1%  F1: 0.6110  |  LR: 1.00e-04  (18.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9623)
Epoka [  3/20]  Train Loss: 0.7809  Bal.Acc: 69.8%  F1: 0.6856  |  Val Loss: 0.8453  Bal.Acc: 69.0%  F1: 0.6954  |  LR: 1.00e-04  (18.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8453)
Epoka [  4/20]  Train Loss: 0.6589  Bal.Acc: 76.5%  F1: 0.7554  |  Val Loss: 0.7624  Bal.Acc: 73.2%  F1: 0.7198  |  LR: 1.00e-04  (18.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7624)
Epoka [  5/20]  Train Loss: 0.5962  Bal.Acc: 77.8%  F1: 0.7668  |  Val Loss: 0.6882  Bal.Acc: 74.5%  F1: 0.7442  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6882)
Epoka [  6/20]  Train Loss: 0.4660  Bal.Acc: 83.9%  F1: 0.8302  |  Val Loss: 0.6880  Bal.Acc: 73.1%  F1: 0.7422  |  LR: 1.00e-04  (18.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6880)
Epoka [  7/20]  Train Loss: 0.4729  Bal.Acc: 81.3%  F1: 0.8065  |  Val Loss: 0.6008  Bal.Acc: 76.7%  F1: 0.7649  |  LR: 1.00e-04  (18.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6008)
Epoka [  8/20]  Train Loss: 0.4270  Bal.Acc: 85.6%  F1: 0.8449  |  Val Loss: 0.5953  Bal.Acc: 76.2%  F1: 0.7474  |  LR: 1.00e-04  (19.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5953)
Epoka [  9/20]  Train Loss: 0.3741  Bal.Acc: 87.3%  F1: 0.8680  |  Val Loss: 0.5980  Bal.Acc: 78.7%  F1: 0.7764  |  LR: 1.00e-04  (18.8s)
Epoka [ 10/20]  Train Loss: 0.3130  Bal.Acc: 89.8%  F1: 0.8931  |  Val Loss: 0.5791  Bal.Acc: 80.9%  F1: 0.8111  |  LR: 1.00e-04  (18.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5791)
Epoka [ 11/20]  Train Loss: 0.2738  Bal.Acc: 90.1%  F1: 0.8966  |  Val Loss: 0.5613  Bal.Acc: 79.7%  F1: 0.7942  |  LR: 1.00e-04  (18.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5613)
Epoka [ 12/20]  Train Loss: 0.2713  Bal.Acc: 91.5%  F1: 0.9093  |  Val Loss: 0.5633  Bal.Acc: 80.5%  F1: 0.7979  |  LR: 1.00e-04  (18.6s)
Epoka [ 13/20]  Train Loss: 0.2613  Bal.Acc: 90.1%  F1: 0.8998  |  Val Loss: 0.7337  Bal.Acc: 74.5%  F1: 0.7293  |  LR: 1.00e-04  (18.3s)
Epoka [ 14/20]  Train Loss: 0.2410  Bal.Acc: 91.8%  F1: 0.9117  |  Val Loss: 0.5683  Bal.Acc: 82.6%  F1: 0.8313  |  LR: 1.00e-04  (17.9s)
Epoka [ 15/20]  Train Loss: 0.2501  Bal.Acc: 91.2%  F1: 0.9101  |  Val Loss: 0.5250  Bal.Acc: 82.5%  F1: 0.8163  |  LR: 1.00e-04  (17.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5250)
Epoka [ 16/20]  Train Loss: 0.2206  Bal.Acc: 93.1%  F1: 0.9272  |  Val Loss: 0.5232  Bal.Acc: 81.4%  F1: 0.8094  |  LR: 1.00e-04  (18.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5232)
Epoka [ 17/20]  Train Loss: 0.2146  Bal.Acc: 93.5%  F1: 0.9311  |  Val Loss: 0.6669  Bal.Acc: 78.3%  F1: 0.7883  |  LR: 1.00e-04  (18.9s)
Epoka [ 18/20]  Train Loss: 0.1717  Bal.Acc: 94.2%  F1: 0.9369  |  Val Loss: 0.6115  Bal.Acc: 79.8%  F1: 0.7980  |  LR: 1.00e-04  (17.8s)
Epoka [ 19/20]  Train Loss: 0.1605  Bal.Acc: 94.3%  F1: 0.9417  |  Val Loss: 0.5548  Bal.Acc: 80.4%  F1: 0.8185  |  LR: 1.00e-04  (17.6s)
Epoka [ 20/20]  Train Loss: 0.2015  Bal.Acc: 92.2%  F1: 0.9198  |  Val Loss: 0.5021  Bal.Acc: 82.9%  F1: 0.8281  |  LR: 1.00e-04  (18.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5021)

Trening zakończony. Checkpoint: checkpoints/densenet121_fold5_best.pt
Log CSV: results/densenet121_fold5_training_log.csv
Załadowano najlepsze wagi z epoki 20

Ewaluacja modelu: densenet121_fold5
----------------------------------------
  Balanced Accuracy: 82.86%
  F1 (macro):        0.8281
  Quadratic Cohen's Kappa: 0.9150

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.82      0.85      0.84        87
    Doubtful       0.79      0.73      0.76        81
        Mild       0.72      0.87      0.79        39
    Moderate       0.85      0.92      0.89        38
      Severe       1.00      0.77      0.87        35

    accuracy                           0.82       280
   macro avg       0.84      0.83      0.83       280
weighted avg       0.82      0.82      0.82       280

  Metryki zapisane: results/densenet121_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/densenet121_fold5_test_probs.npz

 ZAKOŃCZONO: densenet121. Średnia Kappa z 5 foldów: 0.9137 ±0.0153

================================================================================
 ROZPOCZĘCIE TRENINGU MODELU: mobilenetv3_large
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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
model.safetensors: 100% 22.1M/22.1M [00:01<00:00, 15.6MB/s]
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold1
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.2021  Bal.Acc: 36.9%  F1: 0.3526  |  Val Loss: 2.1375  Bal.Acc: 36.3%  F1: 0.3547  |  LR: 1.00e-04  (11.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 2.1375)
Epoka [  2/20]  Train Loss: 1.2914  Bal.Acc: 55.8%  F1: 0.5428  |  Val Loss: 1.5176  Bal.Acc: 52.3%  F1: 0.5340  |  LR: 1.00e-04  (8.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5176)
Epoka [  3/20]  Train Loss: 1.0115  Bal.Acc: 62.8%  F1: 0.6176  |  Val Loss: 1.2042  Bal.Acc: 61.1%  F1: 0.6275  |  LR: 1.00e-04  (9.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2042)
Epoka [  4/20]  Train Loss: 0.8682  Bal.Acc: 67.8%  F1: 0.6707  |  Val Loss: 1.0949  Bal.Acc: 65.1%  F1: 0.6403  |  LR: 1.00e-04  (8.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0949)
Epoka [  5/20]  Train Loss: 0.6769  Bal.Acc: 73.3%  F1: 0.7207  |  Val Loss: 1.2187  Bal.Acc: 63.8%  F1: 0.6270  |  LR: 1.00e-04  (8.6s)
Epoka [  6/20]  Train Loss: 0.6729  Bal.Acc: 75.7%  F1: 0.7450  |  Val Loss: 0.9949  Bal.Acc: 68.5%  F1: 0.6560  |  LR: 1.00e-04  (9.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9949)
Epoka [  7/20]  Train Loss: 0.6376  Bal.Acc: 77.2%  F1: 0.7687  |  Val Loss: 0.8309  Bal.Acc: 73.8%  F1: 0.7395  |  LR: 1.00e-04  (8.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8309)
Epoka [  8/20]  Train Loss: 0.5272  Bal.Acc: 79.9%  F1: 0.7909  |  Val Loss: 0.9207  Bal.Acc: 74.5%  F1: 0.7521  |  LR: 1.00e-04  (9.4s)
Epoka [  9/20]  Train Loss: 0.5689  Bal.Acc: 79.2%  F1: 0.7808  |  Val Loss: 0.7639  Bal.Acc: 77.9%  F1: 0.7737  |  LR: 1.00e-04  (8.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7639)
Epoka [ 10/20]  Train Loss: 0.5179  Bal.Acc: 81.7%  F1: 0.8052  |  Val Loss: 0.7151  Bal.Acc: 79.7%  F1: 0.7927  |  LR: 1.00e-04  (8.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7151)
Epoka [ 11/20]  Train Loss: 0.4347  Bal.Acc: 83.1%  F1: 0.8300  |  Val Loss: 0.8765  Bal.Acc: 72.0%  F1: 0.7168  |  LR: 1.00e-04  (9.6s)
Epoka [ 12/20]  Train Loss: 0.4063  Bal.Acc: 84.1%  F1: 0.8298  |  Val Loss: 0.6872  Bal.Acc: 81.3%  F1: 0.8070  |  LR: 1.00e-04  (8.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6872)
Epoka [ 13/20]  Train Loss: 0.3658  Bal.Acc: 86.3%  F1: 0.8581  |  Val Loss: 0.6027  Bal.Acc: 82.4%  F1: 0.8147  |  LR: 1.00e-04  (9.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6027)
Epoka [ 14/20]  Train Loss: 0.3831  Bal.Acc: 86.2%  F1: 0.8496  |  Val Loss: 0.6751  Bal.Acc: 79.1%  F1: 0.7880  |  LR: 1.00e-04  (9.5s)
Epoka [ 15/20]  Train Loss: 0.2859  Bal.Acc: 89.5%  F1: 0.8928  |  Val Loss: 0.5588  Bal.Acc: 83.6%  F1: 0.8301  |  LR: 1.00e-04  (10.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5588)
Epoka [ 16/20]  Train Loss: 0.2949  Bal.Acc: 88.9%  F1: 0.8877  |  Val Loss: 0.5958  Bal.Acc: 82.2%  F1: 0.8144  |  LR: 1.00e-04  (9.3s)
Epoka [ 17/20]  Train Loss: 0.2550  Bal.Acc: 90.2%  F1: 0.8978  |  Val Loss: 0.7026  Bal.Acc: 78.8%  F1: 0.7836  |  LR: 1.00e-04  (9.3s)
Epoka [ 18/20]  Train Loss: 0.3494  Bal.Acc: 88.0%  F1: 0.8726  |  Val Loss: 0.7128  Bal.Acc: 79.1%  F1: 0.7786  |  LR: 1.00e-04  (8.7s)
Epoka [ 19/20]  Train Loss: 0.2556  Bal.Acc: 90.0%  F1: 0.8985  |  Val Loss: 0.6714  Bal.Acc: 79.2%  F1: 0.7862  |  LR: 5.00e-05  (8.9s)
Epoka [ 20/20]  Train Loss: 0.2289  Bal.Acc: 91.8%  F1: 0.9060  |  Val Loss: 0.5795  Bal.Acc: 81.7%  F1: 0.8218  |  LR: 5.00e-05  (7.8s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.5588

Trening zakończony. Checkpoint: checkpoints/mobilenetv3_large_fold1_best.pt
Log CSV: results/mobilenetv3_large_fold1_training_log.csv
Załadowano najlepsze wagi z epoki 15

Ewaluacja modelu: mobilenetv3_large_fold1
----------------------------------------
  Balanced Accuracy: 83.59%
  F1 (macro):        0.8301
  Quadratic Cohen's Kappa: 0.9157

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.90      0.89      0.89        88
    Doubtful       0.79      0.77      0.78        81
        Mild       0.69      0.69      0.69        39
    Moderate       0.95      0.92      0.93        38
      Severe       0.80      0.91      0.85        35

    accuracy                           0.83       281
   macro avg       0.83      0.84      0.83       281
weighted avg       0.83      0.83      0.83       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold2
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.0498  Bal.Acc: 38.8%  F1: 0.3713  |  Val Loss: 1.6228  Bal.Acc: 44.9%  F1: 0.4436  |  LR: 1.00e-04  (8.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.6228)
Epoka [  2/20]  Train Loss: 1.1901  Bal.Acc: 59.2%  F1: 0.5775  |  Val Loss: 1.4205  Bal.Acc: 51.1%  F1: 0.5194  |  LR: 1.00e-04  (8.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4205)
Epoka [  3/20]  Train Loss: 1.0372  Bal.Acc: 62.0%  F1: 0.6016  |  Val Loss: 0.9893  Bal.Acc: 62.9%  F1: 0.6121  |  LR: 1.00e-04  (7.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9893)
Epoka [  4/20]  Train Loss: 0.8085  Bal.Acc: 72.1%  F1: 0.7095  |  Val Loss: 0.9614  Bal.Acc: 62.6%  F1: 0.6148  |  LR: 1.00e-04  (8.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9614)
Epoka [  5/20]  Train Loss: 0.6742  Bal.Acc: 74.1%  F1: 0.7331  |  Val Loss: 0.8836  Bal.Acc: 67.0%  F1: 0.6596  |  LR: 1.00e-04  (8.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8836)
Epoka [  6/20]  Train Loss: 0.6027  Bal.Acc: 77.5%  F1: 0.7647  |  Val Loss: 0.8979  Bal.Acc: 67.9%  F1: 0.6763  |  LR: 1.00e-04  (8.6s)
Epoka [  7/20]  Train Loss: 0.6273  Bal.Acc: 76.6%  F1: 0.7558  |  Val Loss: 0.8675  Bal.Acc: 68.9%  F1: 0.6703  |  LR: 1.00e-04  (9.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8675)
Epoka [  8/20]  Train Loss: 0.5122  Bal.Acc: 81.2%  F1: 0.7968  |  Val Loss: 0.7974  Bal.Acc: 74.4%  F1: 0.7491  |  LR: 1.00e-04  (8.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7974)
Epoka [  9/20]  Train Loss: 0.5015  Bal.Acc: 80.8%  F1: 0.8042  |  Val Loss: 0.6713  Bal.Acc: 75.9%  F1: 0.7557  |  LR: 1.00e-04  (9.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6713)
Epoka [ 10/20]  Train Loss: 0.4448  Bal.Acc: 82.6%  F1: 0.8163  |  Val Loss: 0.6792  Bal.Acc: 74.5%  F1: 0.7437  |  LR: 1.00e-04  (8.5s)
Epoka [ 11/20]  Train Loss: 0.3839  Bal.Acc: 86.8%  F1: 0.8643  |  Val Loss: 0.6688  Bal.Acc: 76.7%  F1: 0.7519  |  LR: 1.00e-04  (9.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6688)
Epoka [ 12/20]  Train Loss: 0.3609  Bal.Acc: 86.5%  F1: 0.8544  |  Val Loss: 0.6333  Bal.Acc: 75.8%  F1: 0.7581  |  LR: 1.00e-04  (9.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6333)
Epoka [ 13/20]  Train Loss: 0.3157  Bal.Acc: 87.8%  F1: 0.8672  |  Val Loss: 0.6437  Bal.Acc: 77.6%  F1: 0.7750  |  LR: 1.00e-04  (8.4s)
Epoka [ 14/20]  Train Loss: 0.3033  Bal.Acc: 89.2%  F1: 0.8925  |  Val Loss: 0.5942  Bal.Acc: 80.7%  F1: 0.7974  |  LR: 1.00e-04  (10.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5942)
Epoka [ 15/20]  Train Loss: 0.3532  Bal.Acc: 87.3%  F1: 0.8689  |  Val Loss: 0.6799  Bal.Acc: 79.2%  F1: 0.7994  |  LR: 1.00e-04  (8.9s)
Epoka [ 16/20]  Train Loss: 0.2873  Bal.Acc: 89.0%  F1: 0.8876  |  Val Loss: 0.6202  Bal.Acc: 80.7%  F1: 0.8055  |  LR: 1.00e-04  (9.2s)
Epoka [ 17/20]  Train Loss: 0.3054  Bal.Acc: 89.2%  F1: 0.8841  |  Val Loss: 0.5843  Bal.Acc: 81.3%  F1: 0.8139  |  LR: 1.00e-04  (8.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5843)
Epoka [ 18/20]  Train Loss: 0.3146  Bal.Acc: 88.5%  F1: 0.8783  |  Val Loss: 0.6065  Bal.Acc: 80.0%  F1: 0.7906  |  LR: 1.00e-04  (10.5s)
Epoka [ 19/20]  Train Loss: 0.2816  Bal.Acc: 90.5%  F1: 0.9053  |  Val Loss: 0.7517  Bal.Acc: 79.8%  F1: 0.8001  |  LR: 1.00e-04  (8.3s)
Epoka [ 20/20]  Train Loss: 0.1926  Bal.Acc: 93.1%  F1: 0.9277  |  Val Loss: 0.5503  Bal.Acc: 83.7%  F1: 0.8345  |  LR: 1.00e-04  (9.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5503)

Trening zakończony. Checkpoint: checkpoints/mobilenetv3_large_fold2_best.pt
Log CSV: results/mobilenetv3_large_fold2_training_log.csv
Załadowano najlepsze wagi z epoki 20

Ewaluacja modelu: mobilenetv3_large_fold2
----------------------------------------
  Balanced Accuracy: 83.68%
  F1 (macro):        0.8345
  Quadratic Cohen's Kappa: 0.9146

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.89      0.87        88
    Doubtful       0.81      0.75      0.78        81
        Mild       0.78      0.79      0.78        39
    Moderate       0.85      0.92      0.89        38
      Severe       0.88      0.83      0.85        35

    accuracy                           0.83       281
   macro avg       0.83      0.84      0.83       281
weighted avg       0.83      0.83      0.83       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold3
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.0359  Bal.Acc: 38.1%  F1: 0.3655  |  Val Loss: 1.6575  Bal.Acc: 43.2%  F1: 0.3877  |  LR: 1.00e-04  (11.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.6575)
Epoka [  2/20]  Train Loss: 1.2412  Bal.Acc: 56.9%  F1: 0.5545  |  Val Loss: 1.3747  Bal.Acc: 59.2%  F1: 0.5976  |  LR: 1.00e-04  (8.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3747)
Epoka [  3/20]  Train Loss: 0.9567  Bal.Acc: 65.7%  F1: 0.6485  |  Val Loss: 1.1544  Bal.Acc: 63.8%  F1: 0.6327  |  LR: 1.00e-04  (8.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.1544)
Epoka [  4/20]  Train Loss: 0.7776  Bal.Acc: 71.7%  F1: 0.7032  |  Val Loss: 1.0766  Bal.Acc: 64.2%  F1: 0.6292  |  LR: 1.00e-04  (8.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0766)
Epoka [  5/20]  Train Loss: 0.6867  Bal.Acc: 74.8%  F1: 0.7378  |  Val Loss: 0.9421  Bal.Acc: 70.2%  F1: 0.6727  |  LR: 1.00e-04  (8.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9421)
Epoka [  6/20]  Train Loss: 0.6205  Bal.Acc: 76.8%  F1: 0.7569  |  Val Loss: 0.9392  Bal.Acc: 66.1%  F1: 0.6710  |  LR: 1.00e-04  (9.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9392)
Epoka [  7/20]  Train Loss: 0.5234  Bal.Acc: 80.6%  F1: 0.7999  |  Val Loss: 0.8108  Bal.Acc: 70.3%  F1: 0.7015  |  LR: 1.00e-04  (8.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8108)
Epoka [  8/20]  Train Loss: 0.5122  Bal.Acc: 81.4%  F1: 0.8084  |  Val Loss: 0.8311  Bal.Acc: 74.1%  F1: 0.7321  |  LR: 1.00e-04  (9.4s)
Epoka [  9/20]  Train Loss: 0.4455  Bal.Acc: 82.3%  F1: 0.8157  |  Val Loss: 0.8594  Bal.Acc: 70.8%  F1: 0.6903  |  LR: 1.00e-04  (10.1s)
Epoka [ 10/20]  Train Loss: 0.3980  Bal.Acc: 84.7%  F1: 0.8371  |  Val Loss: 0.8339  Bal.Acc: 72.2%  F1: 0.7256  |  LR: 1.00e-04  (8.0s)
Epoka [ 11/20]  Train Loss: 0.4117  Bal.Acc: 84.1%  F1: 0.8374  |  Val Loss: 0.8591  Bal.Acc: 71.4%  F1: 0.7122  |  LR: 5.00e-05  (8.8s)
Epoka [ 12/20]  Train Loss: 0.3452  Bal.Acc: 87.4%  F1: 0.8669  |  Val Loss: 0.7463  Bal.Acc: 76.7%  F1: 0.7659  |  LR: 5.00e-05  (8.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7463)
Epoka [ 13/20]  Train Loss: 0.3135  Bal.Acc: 87.9%  F1: 0.8717  |  Val Loss: 0.7663  Bal.Acc: 74.7%  F1: 0.7496  |  LR: 5.00e-05  (9.1s)
Epoka [ 14/20]  Train Loss: 0.3058  Bal.Acc: 88.7%  F1: 0.8783  |  Val Loss: 0.7876  Bal.Acc: 73.1%  F1: 0.7258  |  LR: 5.00e-05  (9.2s)
Epoka [ 15/20]  Train Loss: 0.2768  Bal.Acc: 90.3%  F1: 0.8938  |  Val Loss: 0.7302  Bal.Acc: 75.7%  F1: 0.7622  |  LR: 5.00e-05  (7.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7302)
Epoka [ 16/20]  Train Loss: 0.2444  Bal.Acc: 91.3%  F1: 0.9078  |  Val Loss: 0.7564  Bal.Acc: 74.8%  F1: 0.7405  |  LR: 5.00e-05  (9.0s)
Epoka [ 17/20]  Train Loss: 0.2902  Bal.Acc: 89.1%  F1: 0.8885  |  Val Loss: 0.7265  Bal.Acc: 78.1%  F1: 0.7693  |  LR: 5.00e-05  (9.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7265)
Epoka [ 18/20]  Train Loss: 0.2581  Bal.Acc: 90.8%  F1: 0.9051  |  Val Loss: 0.6994  Bal.Acc: 78.5%  F1: 0.7751  |  LR: 5.00e-05  (8.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6994)
Epoka [ 19/20]  Train Loss: 0.2601  Bal.Acc: 89.6%  F1: 0.8915  |  Val Loss: 0.7115  Bal.Acc: 75.2%  F1: 0.7481  |  LR: 5.00e-05  (10.7s)
Epoka [ 20/20]  Train Loss: 0.2173  Bal.Acc: 93.1%  F1: 0.9264  |  Val Loss: 0.7425  Bal.Acc: 76.6%  F1: 0.7639  |  LR: 5.00e-05  (9.1s)

Trening zakończony. Checkpoint: checkpoints/mobilenetv3_large_fold3_best.pt
Log CSV: results/mobilenetv3_large_fold3_training_log.csv
Załadowano najlepsze wagi z epoki 18

Ewaluacja modelu: mobilenetv3_large_fold3
----------------------------------------
  Balanced Accuracy: 78.54%
  F1 (macro):        0.7751
  Quadratic Cohen's Kappa: 0.8722

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.81      0.93      0.87        87
    Doubtful       0.81      0.58      0.68        81
        Mild       0.61      0.78      0.68        40
    Moderate       0.91      0.78      0.84        37
      Severe       0.77      0.86      0.81        35

    accuracy                           0.78       280
   macro avg       0.78      0.79      0.78       280
weighted avg       0.79      0.78      0.77       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold4
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.0566  Bal.Acc: 39.3%  F1: 0.3767  |  Val Loss: 2.0010  Bal.Acc: 34.1%  F1: 0.3588  |  LR: 1.00e-04  (8.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 2.0010)
Epoka [  2/20]  Train Loss: 1.2489  Bal.Acc: 56.7%  F1: 0.5478  |  Val Loss: 1.4495  Bal.Acc: 52.9%  F1: 0.5041  |  LR: 1.00e-04  (8.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4495)
Epoka [  3/20]  Train Loss: 0.9841  Bal.Acc: 65.9%  F1: 0.6480  |  Val Loss: 1.1045  Bal.Acc: 61.1%  F1: 0.6223  |  LR: 1.00e-04  (8.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.1045)
Epoka [  4/20]  Train Loss: 0.8789  Bal.Acc: 68.4%  F1: 0.6691  |  Val Loss: 1.0132  Bal.Acc: 66.1%  F1: 0.6552  |  LR: 1.00e-04  (8.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0132)
Epoka [  5/20]  Train Loss: 0.6728  Bal.Acc: 74.7%  F1: 0.7378  |  Val Loss: 0.9967  Bal.Acc: 64.5%  F1: 0.6452  |  LR: 1.00e-04  (8.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9967)
Epoka [  6/20]  Train Loss: 0.6125  Bal.Acc: 76.6%  F1: 0.7567  |  Val Loss: 1.1079  Bal.Acc: 63.7%  F1: 0.6397  |  LR: 1.00e-04  (7.9s)
Epoka [  7/20]  Train Loss: 0.5969  Bal.Acc: 77.7%  F1: 0.7659  |  Val Loss: 0.8254  Bal.Acc: 72.0%  F1: 0.7202  |  LR: 1.00e-04  (9.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8254)
Epoka [  8/20]  Train Loss: 0.4711  Bal.Acc: 82.4%  F1: 0.8179  |  Val Loss: 0.8040  Bal.Acc: 72.9%  F1: 0.7218  |  LR: 1.00e-04  (8.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8040)
Epoka [  9/20]  Train Loss: 0.4571  Bal.Acc: 81.9%  F1: 0.8115  |  Val Loss: 0.9553  Bal.Acc: 68.6%  F1: 0.6835  |  LR: 1.00e-04  (8.2s)
Epoka [ 10/20]  Train Loss: 0.4006  Bal.Acc: 84.7%  F1: 0.8393  |  Val Loss: 0.8033  Bal.Acc: 73.5%  F1: 0.7407  |  LR: 1.00e-04  (10.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8033)
Epoka [ 11/20]  Train Loss: 0.4536  Bal.Acc: 82.9%  F1: 0.8194  |  Val Loss: 0.8422  Bal.Acc: 72.4%  F1: 0.7293  |  LR: 1.00e-04  (8.7s)
Epoka [ 12/20]  Train Loss: 0.3903  Bal.Acc: 86.0%  F1: 0.8504  |  Val Loss: 0.8264  Bal.Acc: 74.3%  F1: 0.7407  |  LR: 1.00e-04  (9.1s)
Epoka [ 13/20]  Train Loss: 0.3340  Bal.Acc: 87.9%  F1: 0.8734  |  Val Loss: 0.8341  Bal.Acc: 75.8%  F1: 0.7647  |  LR: 1.00e-04  (10.0s)
Epoka [ 14/20]  Train Loss: 0.3078  Bal.Acc: 88.6%  F1: 0.8773  |  Val Loss: 0.7863  Bal.Acc: 76.7%  F1: 0.7684  |  LR: 1.00e-04  (8.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7863)
Epoka [ 15/20]  Train Loss: 0.3048  Bal.Acc: 88.0%  F1: 0.8748  |  Val Loss: 0.7767  Bal.Acc: 76.4%  F1: 0.7700  |  LR: 1.00e-04  (9.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7767)
Epoka [ 16/20]  Train Loss: 0.2572  Bal.Acc: 90.8%  F1: 0.9023  |  Val Loss: 0.7388  Bal.Acc: 76.8%  F1: 0.7719  |  LR: 1.00e-04  (9.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7388)
Epoka [ 17/20]  Train Loss: 0.2392  Bal.Acc: 91.1%  F1: 0.9047  |  Val Loss: 0.7638  Bal.Acc: 73.6%  F1: 0.7386  |  LR: 1.00e-04  (7.9s)
Epoka [ 18/20]  Train Loss: 0.1826  Bal.Acc: 93.7%  F1: 0.9313  |  Val Loss: 0.7207  Bal.Acc: 78.7%  F1: 0.7855  |  LR: 1.00e-04  (11.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7207)
Epoka [ 19/20]  Train Loss: 0.2513  Bal.Acc: 90.6%  F1: 0.9022  |  Val Loss: 0.7262  Bal.Acc: 79.1%  F1: 0.7957  |  LR: 1.00e-04  (9.0s)
Epoka [ 20/20]  Train Loss: 0.1906  Bal.Acc: 92.6%  F1: 0.9259  |  Val Loss: 0.7930  Bal.Acc: 76.8%  F1: 0.7730  |  LR: 1.00e-04  (8.7s)

Trening zakończony. Checkpoint: checkpoints/mobilenetv3_large_fold4_best.pt
Log CSV: results/mobilenetv3_large_fold4_training_log.csv
Załadowano najlepsze wagi z epoki 18

Ewaluacja modelu: mobilenetv3_large_fold4
----------------------------------------
  Balanced Accuracy: 78.67%
  F1 (macro):        0.7855
  Quadratic Cohen's Kappa: 0.8945

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.88      0.79      0.84        87
    Doubtful       0.72      0.78      0.75        81
        Mild       0.69      0.72      0.71        40
    Moderate       0.86      0.84      0.85        37
      Severe       0.78      0.80      0.79        35

    accuracy                           0.79       280
   macro avg       0.79      0.79      0.79       280
weighted avg       0.79      0.79      0.79       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: mobilenetv3_large
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large_fold5
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.1570  Bal.Acc: 38.1%  F1: 0.3667  |  Val Loss: 2.4172  Bal.Acc: 29.9%  F1: 0.2439  |  LR: 1.00e-04  (8.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 2.4172)
Epoka [  2/20]  Train Loss: 1.2593  Bal.Acc: 56.6%  F1: 0.5475  |  Val Loss: 1.1507  Bal.Acc: 57.1%  F1: 0.5683  |  LR: 1.00e-04  (9.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.1507)
Epoka [  3/20]  Train Loss: 0.9307  Bal.Acc: 65.0%  F1: 0.6368  |  Val Loss: 0.9830  Bal.Acc: 65.9%  F1: 0.6637  |  LR: 1.00e-04  (8.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9830)
Epoka [  4/20]  Train Loss: 0.7888  Bal.Acc: 70.9%  F1: 0.7000  |  Val Loss: 0.9287  Bal.Acc: 66.1%  F1: 0.6493  |  LR: 1.00e-04  (8.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9287)
Epoka [  5/20]  Train Loss: 0.6804  Bal.Acc: 74.3%  F1: 0.7339  |  Val Loss: 1.1115  Bal.Acc: 72.1%  F1: 0.7152  |  LR: 1.00e-04  (9.3s)
Epoka [  6/20]  Train Loss: 0.6166  Bal.Acc: 77.6%  F1: 0.7651  |  Val Loss: 0.8087  Bal.Acc: 71.2%  F1: 0.7052  |  LR: 1.00e-04  (8.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8087)
Epoka [  7/20]  Train Loss: 0.4845  Bal.Acc: 81.9%  F1: 0.8119  |  Val Loss: 0.8378  Bal.Acc: 70.9%  F1: 0.7084  |  LR: 1.00e-04  (9.1s)
Epoka [  8/20]  Train Loss: 0.4906  Bal.Acc: 81.2%  F1: 0.8019  |  Val Loss: 0.7003  Bal.Acc: 76.7%  F1: 0.7689  |  LR: 1.00e-04  (10.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7003)
Epoka [  9/20]  Train Loss: 0.4297  Bal.Acc: 84.5%  F1: 0.8348  |  Val Loss: 0.7868  Bal.Acc: 73.6%  F1: 0.7434  |  LR: 1.00e-04  (8.2s)
Epoka [ 10/20]  Train Loss: 0.4099  Bal.Acc: 84.9%  F1: 0.8394  |  Val Loss: 0.7582  Bal.Acc: 74.4%  F1: 0.7527  |  LR: 1.00e-04  (10.1s)
Epoka [ 11/20]  Train Loss: 0.3742  Bal.Acc: 86.6%  F1: 0.8607  |  Val Loss: 0.7084  Bal.Acc: 78.7%  F1: 0.7911  |  LR: 1.00e-04  (8.3s)
Epoka [ 12/20]  Train Loss: 0.3513  Bal.Acc: 87.0%  F1: 0.8656  |  Val Loss: 0.6079  Bal.Acc: 79.4%  F1: 0.7928  |  LR: 1.00e-04  (8.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6079)
Epoka [ 13/20]  Train Loss: 0.2577  Bal.Acc: 90.2%  F1: 0.8907  |  Val Loss: 1.4660  Bal.Acc: 74.4%  F1: 0.7503  |  LR: 1.00e-04  (9.3s)
Epoka [ 14/20]  Train Loss: 0.2831  Bal.Acc: 89.7%  F1: 0.8925  |  Val Loss: 0.6585  Bal.Acc: 76.6%  F1: 0.7754  |  LR: 1.00e-04  (8.5s)
Epoka [ 15/20]  Train Loss: 0.2598  Bal.Acc: 90.4%  F1: 0.9010  |  Val Loss: 0.6255  Bal.Acc: 79.8%  F1: 0.7918  |  LR: 1.00e-04  (9.0s)
Epoka [ 16/20]  Train Loss: 0.2459  Bal.Acc: 90.4%  F1: 0.8998  |  Val Loss: 0.6012  Bal.Acc: 80.4%  F1: 0.8081  |  LR: 1.00e-04  (8.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6012)
Epoka [ 17/20]  Train Loss: 0.2779  Bal.Acc: 90.0%  F1: 0.8941  |  Val Loss: 0.6255  Bal.Acc: 81.1%  F1: 0.8200  |  LR: 1.00e-04  (9.4s)
Epoka [ 18/20]  Train Loss: 0.3073  Bal.Acc: 88.2%  F1: 0.8815  |  Val Loss: 0.6711  Bal.Acc: 79.1%  F1: 0.7982  |  LR: 1.00e-04  (9.3s)
Epoka [ 19/20]  Train Loss: 0.2083  Bal.Acc: 92.6%  F1: 0.9195  |  Val Loss: 0.7824  Bal.Acc: 78.2%  F1: 0.7833  |  LR: 1.00e-04  (8.3s)
Epoka [ 20/20]  Train Loss: 0.2058  Bal.Acc: 91.8%  F1: 0.9183  |  Val Loss: 0.6148  Bal.Acc: 80.8%  F1: 0.8063  |  LR: 5.00e-05  (8.9s)

Trening zakończony. Checkpoint: checkpoints/mobilenetv3_large_fold5_best.pt
Log CSV: results/mobilenetv3_large_fold5_training_log.csv
Załadowano najlepsze wagi z epoki 16

Ewaluacja modelu: mobilenetv3_large_fold5
----------------------------------------
  Balanced Accuracy: 80.45%
  F1 (macro):        0.8081
  Quadratic Cohen's Kappa: 0.8876

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.77      0.91      0.83        87
    Doubtful       0.78      0.65      0.71        81
        Mild       0.76      0.74      0.75        39
    Moderate       0.82      0.97      0.89        38
      Severe       1.00      0.74      0.85        35

    accuracy                           0.80       280
   macro avg       0.83      0.80      0.81       280
weighted avg       0.81      0.80      0.80       280

  Metryki zapisane: results/mobilenetv3_large_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/mobilenetv3_large_fold5_test_probs.npz

ZAKOŃCZONO: mobilenetv3_large. Średnia Kappa z 5 foldów: 0.8969 ±0.0165

================================================================================
 ROZPOCZĘCIE TRENINGU MODELU: convnext_tiny
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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
model.safetensors: 100% 114M/114M [00:02<00:00, 56.8MB/s]
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold1
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.6040  Bal.Acc: 31.0%  F1: 0.2930  |  Val Loss: 1.5504  Bal.Acc: 26.8%  F1: 0.1560  |  LR: 1.00e-04  (37.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5504)
Epoka [  2/20]  Train Loss: 1.3543  Bal.Acc: 40.3%  F1: 0.3672  |  Val Loss: 2.2827  Bal.Acc: 25.2%  F1: 0.1946  |  LR: 1.00e-04  (26.8s)
Epoka [  3/20]  Train Loss: 1.4730  Bal.Acc: 38.2%  F1: 0.3513  |  Val Loss: 1.1505  Bal.Acc: 49.2%  F1: 0.4140  |  LR: 1.00e-04  (26.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.1505)
Epoka [  4/20]  Train Loss: 1.0391  Bal.Acc: 56.1%  F1: 0.5407  |  Val Loss: 1.0286  Bal.Acc: 56.4%  F1: 0.5106  |  LR: 1.00e-04  (27.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0286)
Epoka [  5/20]  Train Loss: 0.8455  Bal.Acc: 65.6%  F1: 0.6400  |  Val Loss: 0.9395  Bal.Acc: 58.8%  F1: 0.5611  |  LR: 1.00e-04  (28.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9395)
Epoka [  6/20]  Train Loss: 0.7671  Bal.Acc: 68.0%  F1: 0.6721  |  Val Loss: 0.8371  Bal.Acc: 67.0%  F1: 0.6431  |  LR: 1.00e-04  (27.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8371)
Epoka [  7/20]  Train Loss: 0.6737  Bal.Acc: 72.3%  F1: 0.7133  |  Val Loss: 0.7309  Bal.Acc: 69.2%  F1: 0.7066  |  LR: 1.00e-04  (28.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7309)
Epoka [  8/20]  Train Loss: 0.5197  Bal.Acc: 78.1%  F1: 0.7740  |  Val Loss: 0.7499  Bal.Acc: 69.1%  F1: 0.6472  |  LR: 1.00e-04  (28.7s)
Epoka [  9/20]  Train Loss: 0.5527  Bal.Acc: 78.2%  F1: 0.7718  |  Val Loss: 0.8775  Bal.Acc: 68.2%  F1: 0.6888  |  LR: 1.00e-04  (28.3s)
Epoka [ 10/20]  Train Loss: 0.4733  Bal.Acc: 82.4%  F1: 0.8159  |  Val Loss: 0.6262  Bal.Acc: 77.7%  F1: 0.7586  |  LR: 1.00e-04  (26.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6262)
Epoka [ 11/20]  Train Loss: 0.5293  Bal.Acc: 79.5%  F1: 0.7851  |  Val Loss: 0.5757  Bal.Acc: 80.5%  F1: 0.8016  |  LR: 1.00e-04  (27.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5757)
Epoka [ 12/20]  Train Loss: 0.3753  Bal.Acc: 86.1%  F1: 0.8565  |  Val Loss: 0.7137  Bal.Acc: 76.8%  F1: 0.7730  |  LR: 1.00e-04  (28.0s)
Epoka [ 13/20]  Train Loss: 0.3725  Bal.Acc: 85.3%  F1: 0.8457  |  Val Loss: 0.6165  Bal.Acc: 78.4%  F1: 0.7896  |  LR: 1.00e-04  (28.3s)
Epoka [ 14/20]  Train Loss: 0.3497  Bal.Acc: 87.0%  F1: 0.8648  |  Val Loss: 0.7896  Bal.Acc: 70.2%  F1: 0.7014  |  LR: 1.00e-04  (26.8s)
Epoka [ 15/20]  Train Loss: 0.5657  Bal.Acc: 78.6%  F1: 0.7827  |  Val Loss: 0.6232  Bal.Acc: 78.5%  F1: 0.7968  |  LR: 5.00e-05  (26.3s)
Epoka [ 16/20]  Train Loss: 0.3247  Bal.Acc: 86.8%  F1: 0.8656  |  Val Loss: 0.4817  Bal.Acc: 81.6%  F1: 0.8032  |  LR: 5.00e-05  (26.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.4817)
Epoka [ 17/20]  Train Loss: 0.2413  Bal.Acc: 90.9%  F1: 0.9045  |  Val Loss: 0.4851  Bal.Acc: 85.0%  F1: 0.8395  |  LR: 5.00e-05  (28.5s)
Epoka [ 18/20]  Train Loss: 0.2478  Bal.Acc: 90.6%  F1: 0.9025  |  Val Loss: 0.5150  Bal.Acc: 83.0%  F1: 0.8258  |  LR: 5.00e-05  (27.1s)
Epoka [ 19/20]  Train Loss: 0.1626  Bal.Acc: 94.5%  F1: 0.9421  |  Val Loss: 0.5357  Bal.Acc: 83.9%  F1: 0.8373  |  LR: 5.00e-05  (26.5s)
Epoka [ 20/20]  Train Loss: 0.1396  Bal.Acc: 95.1%  F1: 0.9492  |  Val Loss: 0.5576  Bal.Acc: 83.5%  F1: 0.8307  |  LR: 2.50e-05  (26.6s)

Trening zakończony. Checkpoint: checkpoints/convnext_tiny_fold1_best.pt
Log CSV: results/convnext_tiny_fold1_training_log.csv
Załadowano najlepsze wagi z epoki 16

Ewaluacja modelu: convnext_tiny_fold1
----------------------------------------
  Balanced Accuracy: 81.56%
  F1 (macro):        0.8032
  Quadratic Cohen's Kappa: 0.9095

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.85      0.84      0.85        88
    Doubtful       0.71      0.63      0.67        81
        Mild       0.64      0.74      0.69        39
    Moderate       0.90      0.92      0.91        38
      Severe       0.87      0.94      0.90        35

    accuracy                           0.79       281
   macro avg       0.79      0.82      0.80       281
weighted avg       0.79      0.79      0.79       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold2
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.7657  Bal.Acc: 20.8%  F1: 0.1867  |  Val Loss: 1.6237  Bal.Acc: 20.0%  F1: 0.0895  |  LR: 1.00e-04  (26.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.6237)
Epoka [  2/20]  Train Loss: 1.5220  Bal.Acc: 34.8%  F1: 0.3360  |  Val Loss: 1.2006  Bal.Acc: 57.5%  F1: 0.4874  |  LR: 1.00e-04  (26.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2006)
Epoka [  3/20]  Train Loss: 1.0661  Bal.Acc: 56.2%  F1: 0.5348  |  Val Loss: 0.9978  Bal.Acc: 57.7%  F1: 0.5811  |  LR: 1.00e-04  (27.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9978)
Epoka [  4/20]  Train Loss: 0.9559  Bal.Acc: 58.8%  F1: 0.5792  |  Val Loss: 0.8396  Bal.Acc: 60.6%  F1: 0.5733  |  LR: 1.00e-04  (27.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8396)
Epoka [  5/20]  Train Loss: 0.8125  Bal.Acc: 66.4%  F1: 0.6478  |  Val Loss: 1.1305  Bal.Acc: 47.6%  F1: 0.4212  |  LR: 1.00e-04  (27.8s)
Epoka [  6/20]  Train Loss: 0.7336  Bal.Acc: 68.5%  F1: 0.6741  |  Val Loss: 0.7680  Bal.Acc: 67.8%  F1: 0.6882  |  LR: 1.00e-04  (26.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7680)
Epoka [  7/20]  Train Loss: 0.5996  Bal.Acc: 75.9%  F1: 0.7505  |  Val Loss: 0.8350  Bal.Acc: 69.7%  F1: 0.6900  |  LR: 1.00e-04  (27.5s)
Epoka [  8/20]  Train Loss: 0.6025  Bal.Acc: 77.4%  F1: 0.7724  |  Val Loss: 0.7668  Bal.Acc: 66.4%  F1: 0.6545  |  LR: 1.00e-04  (26.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7668)
Epoka [  9/20]  Train Loss: 0.5494  Bal.Acc: 79.1%  F1: 0.7825  |  Val Loss: 1.5760  Bal.Acc: 59.1%  F1: 0.5449  |  LR: 1.00e-04  (27.3s)
Epoka [ 10/20]  Train Loss: 0.7542  Bal.Acc: 69.4%  F1: 0.6797  |  Val Loss: 0.7223  Bal.Acc: 73.4%  F1: 0.7184  |  LR: 1.00e-04  (26.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7223)
Epoka [ 11/20]  Train Loss: 0.4633  Bal.Acc: 80.9%  F1: 0.8067  |  Val Loss: 0.6526  Bal.Acc: 74.3%  F1: 0.7229  |  LR: 1.00e-04  (28.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6526)
Epoka [ 12/20]  Train Loss: 0.4009  Bal.Acc: 83.5%  F1: 0.8269  |  Val Loss: 0.6194  Bal.Acc: 75.0%  F1: 0.7508  |  LR: 1.00e-04  (28.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6194)
Epoka [ 13/20]  Train Loss: 0.4157  Bal.Acc: 82.6%  F1: 0.8205  |  Val Loss: 0.5946  Bal.Acc: 79.3%  F1: 0.7976  |  LR: 1.00e-04  (28.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5946)
Epoka [ 14/20]  Train Loss: 0.4113  Bal.Acc: 83.0%  F1: 0.8245  |  Val Loss: 0.5547  Bal.Acc: 79.8%  F1: 0.8045  |  LR: 1.00e-04  (28.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5547)
Epoka [ 15/20]  Train Loss: 0.3770  Bal.Acc: 84.4%  F1: 0.8352  |  Val Loss: 0.8012  Bal.Acc: 68.4%  F1: 0.6768  |  LR: 1.00e-04  (28.3s)
Epoka [ 16/20]  Train Loss: 0.2979  Bal.Acc: 88.8%  F1: 0.8855  |  Val Loss: 0.6210  Bal.Acc: 74.6%  F1: 0.7379  |  LR: 1.00e-04  (28.1s)
Epoka [ 17/20]  Train Loss: 0.5803  Bal.Acc: 78.5%  F1: 0.7700  |  Val Loss: 0.6310  Bal.Acc: 73.5%  F1: 0.7369  |  LR: 1.00e-04  (27.1s)
Epoka [ 18/20]  Train Loss: 0.3592  Bal.Acc: 86.6%  F1: 0.8633  |  Val Loss: 0.5930  Bal.Acc: 77.9%  F1: 0.7753  |  LR: 5.00e-05  (26.6s)
Epoka [ 19/20]  Train Loss: 0.2727  Bal.Acc: 89.7%  F1: 0.8955  |  Val Loss: 0.4683  Bal.Acc: 83.8%  F1: 0.8186  |  LR: 5.00e-05  (26.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.4683)
Epoka [ 20/20]  Train Loss: 0.1996  Bal.Acc: 91.6%  F1: 0.9130  |  Val Loss: 0.4871  Bal.Acc: 81.9%  F1: 0.8215  |  LR: 5.00e-05  (28.1s)

Trening zakończony. Checkpoint: checkpoints/convnext_tiny_fold2_best.pt
Log CSV: results/convnext_tiny_fold2_training_log.csv
Załadowano najlepsze wagi z epoki 19

Ewaluacja modelu: convnext_tiny_fold2
----------------------------------------
  Balanced Accuracy: 83.83%
  F1 (macro):        0.8186
  Quadratic Cohen's Kappa: 0.9396

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.91      0.93      0.92        88
    Doubtful       0.91      0.63      0.74        81
        Mild       0.54      0.79      0.65        39
    Moderate       0.80      0.92      0.85        38
      Severe       0.94      0.91      0.93        35

    accuracy                           0.82       281
   macro avg       0.82      0.84      0.82       281
weighted avg       0.85      0.82      0.82       281

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold3
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.7159  Bal.Acc: 18.4%  F1: 0.1544  |  Val Loss: 1.6347  Bal.Acc: 20.0%  F1: 0.0500  |  LR: 1.00e-04  (32.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.6347)
Epoka [  2/20]  Train Loss: 1.6405  Bal.Acc: 19.8%  F1: 0.1680  |  Val Loss: 1.6046  Bal.Acc: 20.0%  F1: 0.0502  |  LR: 1.00e-04  (26.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.6046)
Epoka [  3/20]  Train Loss: 1.5820  Bal.Acc: 26.2%  F1: 0.2266  |  Val Loss: 1.8047  Bal.Acc: 20.5%  F1: 0.1056  |  LR: 1.00e-04  (26.6s)
Epoka [  4/20]  Train Loss: 1.5579  Bal.Acc: 27.3%  F1: 0.2441  |  Val Loss: 1.4569  Bal.Acc: 34.3%  F1: 0.2559  |  LR: 1.00e-04  (27.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4569)
Epoka [  5/20]  Train Loss: 1.4535  Bal.Acc: 36.0%  F1: 0.3416  |  Val Loss: 1.4084  Bal.Acc: 44.4%  F1: 0.4099  |  LR: 1.00e-04  (26.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4084)
Epoka [  6/20]  Train Loss: 1.2316  Bal.Acc: 46.9%  F1: 0.4383  |  Val Loss: 1.5952  Bal.Acc: 43.8%  F1: 0.3170  |  LR: 1.00e-04  (28.1s)
Epoka [  7/20]  Train Loss: 1.0193  Bal.Acc: 58.7%  F1: 0.5661  |  Val Loss: 0.9542  Bal.Acc: 60.3%  F1: 0.5493  |  LR: 1.00e-04  (26.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9542)
Epoka [  8/20]  Train Loss: 0.8218  Bal.Acc: 66.9%  F1: 0.6593  |  Val Loss: 0.7978  Bal.Acc: 67.5%  F1: 0.6870  |  LR: 1.00e-04  (27.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7978)
Epoka [  9/20]  Train Loss: 0.8792  Bal.Acc: 62.3%  F1: 0.6117  |  Val Loss: 0.8070  Bal.Acc: 67.7%  F1: 0.6416  |  LR: 1.00e-04  (28.2s)
Epoka [ 10/20]  Train Loss: 0.6115  Bal.Acc: 74.6%  F1: 0.7358  |  Val Loss: 0.7690  Bal.Acc: 70.0%  F1: 0.6725  |  LR: 1.00e-04  (28.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7690)
Epoka [ 11/20]  Train Loss: 0.5814  Bal.Acc: 76.7%  F1: 0.7575  |  Val Loss: 0.6472  Bal.Acc: 74.3%  F1: 0.7319  |  LR: 1.00e-04  (28.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6472)
Epoka [ 12/20]  Train Loss: 0.5343  Bal.Acc: 79.8%  F1: 0.7864  |  Val Loss: 0.8186  Bal.Acc: 71.2%  F1: 0.7298  |  LR: 1.00e-04  (28.4s)
Epoka [ 13/20]  Train Loss: 0.4992  Bal.Acc: 80.2%  F1: 0.7940  |  Val Loss: 0.9856  Bal.Acc: 67.6%  F1: 0.6253  |  LR: 1.00e-04  (28.2s)
Epoka [ 14/20]  Train Loss: 0.4129  Bal.Acc: 83.2%  F1: 0.8249  |  Val Loss: 0.8763  Bal.Acc: 71.5%  F1: 0.7236  |  LR: 1.00e-04  (26.9s)
Epoka [ 15/20]  Train Loss: 0.4011  Bal.Acc: 83.6%  F1: 0.8283  |  Val Loss: 0.7289  Bal.Acc: 71.9%  F1: 0.7181  |  LR: 5.00e-05  (26.4s)
Epoka [ 16/20]  Train Loss: 0.3095  Bal.Acc: 88.3%  F1: 0.8778  |  Val Loss: 0.7388  Bal.Acc: 78.1%  F1: 0.7655  |  LR: 5.00e-05  (26.6s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.6472

Trening zakończony. Checkpoint: checkpoints/convnext_tiny_fold3_best.pt
Log CSV: results/convnext_tiny_fold3_training_log.csv
Załadowano najlepsze wagi z epoki 11

Ewaluacja modelu: convnext_tiny_fold3
----------------------------------------
  Balanced Accuracy: 74.26%
  F1 (macro):        0.7319
  Quadratic Cohen's Kappa: 0.8910

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.81      0.93      0.87        87
    Doubtful       0.75      0.57      0.65        81
        Mild       0.56      0.60      0.58        40
    Moderate       0.78      0.76      0.77        37
      Severe       0.75      0.86      0.80        35

    accuracy                           0.75       280
   macro avg       0.73      0.74      0.73       280
weighted avg       0.75      0.75      0.74       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold4
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.7818  Bal.Acc: 18.6%  F1: 0.1795  |  Val Loss: 1.5990  Bal.Acc: 30.2%  F1: 0.1223  |  LR: 1.00e-04  (26.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5990)
Epoka [  2/20]  Train Loss: 1.5744  Bal.Acc: 27.9%  F1: 0.2331  |  Val Loss: 1.5512  Bal.Acc: 26.4%  F1: 0.1346  |  LR: 1.00e-04  (27.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5512)
Epoka [  3/20]  Train Loss: 1.5256  Bal.Acc: 29.6%  F1: 0.2759  |  Val Loss: 1.4395  Bal.Acc: 35.8%  F1: 0.2490  |  LR: 1.00e-04  (27.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4395)
Epoka [  4/20]  Train Loss: 1.3611  Bal.Acc: 40.1%  F1: 0.3724  |  Val Loss: 1.3242  Bal.Acc: 40.6%  F1: 0.3239  |  LR: 1.00e-04  (28.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3242)
Epoka [  5/20]  Train Loss: 0.9672  Bal.Acc: 58.9%  F1: 0.5748  |  Val Loss: 0.7818  Bal.Acc: 65.6%  F1: 0.6292  |  LR: 1.00e-04  (26.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7818)
Epoka [  6/20]  Train Loss: 0.8272  Bal.Acc: 65.8%  F1: 0.6476  |  Val Loss: 0.8869  Bal.Acc: 62.5%  F1: 0.6378  |  LR: 1.00e-04  (26.6s)
Epoka [  7/20]  Train Loss: 0.6811  Bal.Acc: 73.4%  F1: 0.7267  |  Val Loss: 0.9721  Bal.Acc: 64.1%  F1: 0.6451  |  LR: 1.00e-04  (27.9s)
Epoka [  8/20]  Train Loss: 0.6208  Bal.Acc: 75.6%  F1: 0.7480  |  Val Loss: 0.7574  Bal.Acc: 72.4%  F1: 0.7103  |  LR: 1.00e-04  (26.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7574)
Epoka [  9/20]  Train Loss: 0.5734  Bal.Acc: 76.9%  F1: 0.7635  |  Val Loss: 0.8551  Bal.Acc: 69.7%  F1: 0.6723  |  LR: 1.00e-04  (27.7s)
Epoka [ 10/20]  Train Loss: 0.7021  Bal.Acc: 70.6%  F1: 0.6983  |  Val Loss: 0.8501  Bal.Acc: 68.6%  F1: 0.7032  |  LR: 1.00e-04  (26.9s)
Epoka [ 11/20]  Train Loss: 0.4669  Bal.Acc: 81.6%  F1: 0.8103  |  Val Loss: 0.8559  Bal.Acc: 72.8%  F1: 0.7403  |  LR: 1.00e-04  (26.4s)
Epoka [ 12/20]  Train Loss: 0.4147  Bal.Acc: 84.8%  F1: 0.8422  |  Val Loss: 0.6849  Bal.Acc: 71.9%  F1: 0.7193  |  LR: 1.00e-04  (26.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6849)
Epoka [ 13/20]  Train Loss: 0.4159  Bal.Acc: 83.9%  F1: 0.8322  |  Val Loss: 0.7420  Bal.Acc: 73.8%  F1: 0.7521  |  LR: 1.00e-04  (28.0s)
Epoka [ 14/20]  Train Loss: 0.3093  Bal.Acc: 87.5%  F1: 0.8679  |  Val Loss: 0.6053  Bal.Acc: 80.1%  F1: 0.7878  |  LR: 1.00e-04  (26.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6053)
Epoka [ 15/20]  Train Loss: 0.2788  Bal.Acc: 89.3%  F1: 0.8879  |  Val Loss: 0.8018  Bal.Acc: 75.7%  F1: 0.7458  |  LR: 1.00e-04  (27.9s)
Epoka [ 16/20]  Train Loss: 0.2991  Bal.Acc: 88.5%  F1: 0.8792  |  Val Loss: 0.6039  Bal.Acc: 81.5%  F1: 0.8091  |  LR: 1.00e-04  (26.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6039)
Epoka [ 17/20]  Train Loss: 0.3163  Bal.Acc: 88.7%  F1: 0.8818  |  Val Loss: 0.6862  Bal.Acc: 78.3%  F1: 0.7686  |  LR: 1.00e-04  (27.6s)
Epoka [ 18/20]  Train Loss: 0.2445  Bal.Acc: 90.8%  F1: 0.9042  |  Val Loss: 0.7045  Bal.Acc: 77.6%  F1: 0.7693  |  LR: 1.00e-04  (26.8s)
Epoka [ 19/20]  Train Loss: 0.3069  Bal.Acc: 88.0%  F1: 0.8728  |  Val Loss: 0.8526  Bal.Acc: 75.5%  F1: 0.7687  |  LR: 1.00e-04  (26.5s)
Epoka [ 20/20]  Train Loss: 0.3103  Bal.Acc: 89.0%  F1: 0.8870  |  Val Loss: 0.6423  Bal.Acc: 77.9%  F1: 0.7686  |  LR: 5.00e-05  (26.5s)

Trening zakończony. Checkpoint: checkpoints/convnext_tiny_fold4_best.pt
Log CSV: results/convnext_tiny_fold4_training_log.csv
Załadowano najlepsze wagi z epoki 16

Ewaluacja modelu: convnext_tiny_fold4
----------------------------------------
  Balanced Accuracy: 81.52%
  F1 (macro):        0.8091
  Quadratic Cohen's Kappa: 0.9190

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.93      0.78      0.85        87
    Doubtful       0.75      0.80      0.77        81
        Mild       0.67      0.80      0.73        40
    Moderate       0.80      0.89      0.85        37
      Severe       0.90      0.80      0.85        35

    accuracy                           0.81       280
   macro avg       0.81      0.82      0.81       280
weighted avg       0.82      0.81      0.81       280

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
/content/drive/MyDrive/Knee_Project/dataset.py:160: UserWarning: Argument(s) 'value' are not valid for transform Rotate
  A.Rotate(limit=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),

Buduję model: convnext_tiny
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny_fold5
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.6630  Bal.Acc: 25.7%  F1: 0.2422  |  Val Loss: 1.4599  Bal.Acc: 32.9%  F1: 0.2543  |  LR: 1.00e-04  (26.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4599)
Epoka [  2/20]  Train Loss: 1.3608  Bal.Acc: 42.5%  F1: 0.3864  |  Val Loss: 1.3287  Bal.Acc: 49.8%  F1: 0.4167  |  LR: 1.00e-04  (26.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3287)
Epoka [  3/20]  Train Loss: 1.0072  Bal.Acc: 58.0%  F1: 0.5602  |  Val Loss: 0.7605  Bal.Acc: 65.8%  F1: 0.6437  |  LR: 1.00e-04  (28.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7605)
Epoka [  4/20]  Train Loss: 0.8975  Bal.Acc: 62.3%  F1: 0.6069  |  Val Loss: 1.2152  Bal.Acc: 44.8%  F1: 0.4500  |  LR: 1.00e-04  (28.1s)
Epoka [  5/20]  Train Loss: 0.6989  Bal.Acc: 73.2%  F1: 0.7239  |  Val Loss: 0.8151  Bal.Acc: 70.0%  F1: 0.7078  |  LR: 1.00e-04  (28.1s)
Epoka [  6/20]  Train Loss: 0.6020  Bal.Acc: 76.0%  F1: 0.7507  |  Val Loss: 0.7693  Bal.Acc: 69.3%  F1: 0.6707  |  LR: 1.00e-04  (26.7s)
Epoka [  7/20]  Train Loss: 0.4703  Bal.Acc: 82.3%  F1: 0.8148  |  Val Loss: 0.5760  Bal.Acc: 76.7%  F1: 0.7579  |  LR: 1.00e-04  (26.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5760)
Epoka [  8/20]  Train Loss: 0.4943  Bal.Acc: 80.5%  F1: 0.7974  |  Val Loss: 0.7920  Bal.Acc: 66.9%  F1: 0.6531  |  LR: 1.00e-04  (28.0s)
Epoka [  9/20]  Train Loss: 0.5427  Bal.Acc: 77.9%  F1: 0.7740  |  Val Loss: 1.0452  Bal.Acc: 62.7%  F1: 0.5493  |  LR: 1.00e-04  (26.7s)
Epoka [ 10/20]  Train Loss: 0.4079  Bal.Acc: 84.5%  F1: 0.8389  |  Val Loss: 0.5427  Bal.Acc: 77.6%  F1: 0.7657  |  LR: 1.00e-04  (26.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5427)
Epoka [ 11/20]  Train Loss: 0.3320  Bal.Acc: 86.8%  F1: 0.8621  |  Val Loss: 0.6143  Bal.Acc: 76.1%  F1: 0.7614  |  LR: 1.00e-04  (27.3s)
Epoka [ 12/20]  Train Loss: 0.4114  Bal.Acc: 83.1%  F1: 0.8269  |  Val Loss: 0.5983  Bal.Acc: 75.3%  F1: 0.7630  |  LR: 1.00e-04  (26.9s)
Epoka [ 13/20]  Train Loss: 0.2976  Bal.Acc: 88.6%  F1: 0.8813  |  Val Loss: 0.7042  Bal.Acc: 74.5%  F1: 0.7415  |  LR: 1.00e-04  (26.6s)
Epoka [ 14/20]  Train Loss: 0.3645  Bal.Acc: 86.1%  F1: 0.8547  |  Val Loss: 0.6638  Bal.Acc: 78.7%  F1: 0.7944  |  LR: 5.00e-05  (26.3s)
Epoka [ 15/20]  Train Loss: 0.2463  Bal.Acc: 91.0%  F1: 0.9039  |  Val Loss: 0.7055  Bal.Acc: 73.4%  F1: 0.7409  |  LR: 5.00e-05  (26.3s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.5427

Trening zakończony. Checkpoint: checkpoints/convnext_tiny_fold5_best.pt
Log CSV: results/convnext_tiny_fold5_training_log.csv
Załadowano najlepsze wagi z epoki 10

Ewaluacja modelu: convnext_tiny_fold5
----------------------------------------
  Balanced Accuracy: 77.61%
  F1 (macro):        0.7657
  Quadratic Cohen's Kappa: 0.8828

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.79      0.89      0.84        87
    Doubtful       0.76      0.62      0.68        81
        Mild       0.79      0.69      0.74        39
    Moderate       0.73      1.00      0.84        38
      Severe       0.77      0.69      0.73        35

    accuracy                           0.77       280
   macro avg       0.77      0.78      0.77       280
weighted avg       0.77      0.77      0.77       280

  Metryki zapisane: results/convnext_tiny_fold5_metrics.json
  Prawdopodobieństwa zapisane: results/convnext_tiny_fold5_test_probs.npz

 ZAKOŃCZONO: convnext_tiny. Średnia Kappa z 5 foldów: 0.9084 ±0.0202

=========================================================================================================
PODSUMOWANIE POJEDYNCZYCH FOLDÓW
=========================================================================================================
Model                        Kappa   F1-Mac |      KL0      KL1      KL2      KL3      KL4
---------------------------------------------------------------------------------------------------------
convnext_tiny_fold2         0.9396   0.8186 |   0.9213   0.7445   0.6458   0.8537   0.9275
densenet121_fold1           0.9313   0.8506 |   0.8814   0.7123   0.7556   0.9744   0.9296
densenet121_fold2           0.9295   0.8196 |   0.8950   0.7550   0.6667   0.8800   0.9014
convnext_tiny_fold4         0.9190   0.8091 |   0.8500   0.7738   0.7273   0.8462   0.8485
mobilenetv3_large_fold1     0.9157   0.8301 |   0.8914   0.7799   0.6923   0.9333   0.8533
densenet121_fold5           0.9150   0.8281 |   0.8362   0.7564   0.7907   0.8861   0.8710
mobilenetv3_large_fold2     0.9146   0.8345 |   0.8667   0.7821   0.7848   0.8861   0.8529
resnet50_fold1              0.9142   0.7717 |   0.8324   0.6667   0.6136   0.8471   0.8986
resnet50_fold2              0.9095   0.7485 |   0.8791   0.6569   0.5510   0.7945   0.8611
convnext_tiny_fold1         0.9095   0.8032 |   0.8457   0.6667   0.6905   0.9091   0.9041
resnet50_fold4              0.9030   0.7679 |   0.8276   0.7342   0.6747   0.7848   0.8182
efficientnet_b3_fold2       0.8992   0.7582 |   0.8063   0.6395   0.6329   0.8267   0.8857
densenet121_fold3           0.8964   0.8046 |   0.8941   0.7089   0.6905   0.8611   0.8684
densenet121_fold4           0.8961   0.7900 |   0.8182   0.7531   0.7027   0.8493   0.8267
mobilenetv3_large_fold4     0.8945   0.7855 |   0.8364   0.7456   0.7073   0.8493   0.7887
convnext_tiny_fold3         0.8910   0.7319 |   0.8663   0.6479   0.5783   0.7671   0.8000
mobilenetv3_large_fold5     0.8876   0.8081 |   0.8316   0.7114   0.7532   0.8916   0.8525
convnext_tiny_fold5         0.8828   0.7657 |   0.8370   0.6803   0.7397   0.8444   0.7273
efficientnet_b3_fold5       0.8820   0.7609 |   0.7979   0.6755   0.6667   0.8462   0.8182
resnet50_fold5              0.8758   0.7495 |   0.8333   0.6301   0.6263   0.8462   0.8116
efficientnet_b3_fold1       0.8730   0.7179 |   0.8317   0.6412   0.5135   0.7654   0.8378
mobilenetv3_large_fold3     0.8722   0.7751 |   0.8663   0.6763   0.6813   0.8406   0.8108
resnet50_fold3              0.8720   0.7433 |   0.8914   0.6842   0.5843   0.8169   0.7397
efficientnet_b3_fold4       0.8606   0.7600 |   0.7200   0.6780   0.7442   0.8462   0.8116
efficientnet_b3_fold3       0.7878   0.6510 |   0.7429   0.5714   0.4615   0.7568   0.7222
=========================================================================================================
Posortowane wg Cohen's Kappa 

===================================================================================================================
PODSUMOWANIE CROSS-VALIDATION — ŚREDNIA Z 5 FOLDÓW
===================================================================================================================
Model                         Kappa         F1-Mac |      KL0      KL1      KL2      KL3      KL4
-------------------------------------------------------------------------------------------------------------------
resnet50              0.8949 ±0.0176  0.7562 ±0.0114 |   0.8528   0.6744   0.6100   0.8179   0.8258
efficientnet_b3       0.8605 ±0.0385  0.7296 ±0.0425 |   0.7798   0.6411   0.6038   0.8083   0.8151
densenet121           0.9137 ±0.0153  0.8186 ±0.0206 |   0.8650   0.7371   0.7212   0.8902   0.8794
mobilenetv3_large     0.8969 ±0.0165  0.8067 ±0.0235 |   0.8585   0.7391   0.7238   0.8802   0.8316
convnext_tiny         0.9084 ±0.0202  0.7857 ±0.0323 |   0.8641   0.7026   0.6763   0.8441   0.8415
================================================================================================================
