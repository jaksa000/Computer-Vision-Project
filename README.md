Dataset is too big to upload here, downolad it from here: https://www.kaggle.com/datasets/tommyngx/digital-knee-xray/data?select=MedicalExpert-I
unzip and rename to data and put in project directory, should work from here

SOLUTION EXAMPLE

D:\Pycharmprojects\NNproject2\.venv1\Scripts\python.exe D:\Pycharmprojects\Computer-Vision-Project\main.py 
Urządzenie: cuda
Modele do uruchomienia: ['resnet50', 'efficientnet_b3', 'densenet121', 'mobilenetv3_large', 'convnext_tiny']
============================================================
Ładowanie danych z: data\MedicalExpert-I
============================================================
  Klasa 0 (0Normal): 514 obrazów
  Klasa 1 (1Doubtful): 477 obrazów
  Klasa 2 (2Mild): 232 obrazów
  Klasa 3 (3Moderate): 221 obrazów
  Klasa 4 (4Severe): 206 obrazów

  Łącznie: 1650 obrazów

Podział danych:
  Train: 1152 (69.8%)
  Val:   245 (14.8%)
  Test:  253 (15.3%)

Wagi klas (wyrównanie imbalance):
  Klasa 0 (Normal): waga = 0.642  (count = 359)
  Klasa 1 (Doubtful): waga = 0.692  (count = 333)
  Klasa 2 (Mild): waga = 1.422  (count = 162)
  Klasa 3 (Moderate): waga = 1.496  (count = 154)
  Klasa 4 (Severe): waga = 1.600  (count = 144)

DataLoadery gotowe.
============================================================

############################################################
# MODEL: resnet50
############################################################

Buduję model: resnet50
  timm ID:     resnet50
  Pretrained:  True
  Opis:        ResNet-50 — klasyk, dobry baseline
  Parametry:   23,518,277 łącznie, 23,518,277 trenowalnych

============================================================
TRENING: resnet50
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.6021  Acc: 25.6%  |  Val Loss: 1.5880  Acc: 39.6%  |  LR: 1.00e-04  (21.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5880)
Epoka [  2/20]  Train Loss: 1.5632  Acc: 41.7%  |  Val Loss: 1.5464  Acc: 47.8%  |  LR: 1.00e-04  (21.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5464)
Epoka [  3/20]  Train Loss: 1.4917  Acc: 50.3%  |  Val Loss: 1.4868  Acc: 45.3%  |  LR: 1.00e-04  (21.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4868)
Epoka [  4/20]  Train Loss: 1.3577  Acc: 58.4%  |  Val Loss: 1.2786  Acc: 58.0%  |  LR: 1.00e-04  (21.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2786)
Epoka [  5/20]  Train Loss: 1.1510  Acc: 59.9%  |  Val Loss: 1.0408  Acc: 64.1%  |  LR: 1.00e-04  (21.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0408)
Epoka [  6/20]  Train Loss: 0.9342  Acc: 68.6%  |  Val Loss: 0.9305  Acc: 65.7%  |  LR: 1.00e-04  (21.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9305)
Epoka [  7/20]  Train Loss: 0.7525  Acc: 74.4%  |  Val Loss: 0.8821  Acc: 66.5%  |  LR: 1.00e-04  (21.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8821)
Epoka [  8/20]  Train Loss: 0.5616  Acc: 82.4%  |  Val Loss: 0.7769  Acc: 72.2%  |  LR: 1.00e-04  (21.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7769)
Epoka [  9/20]  Train Loss: 0.4499  Acc: 83.6%  |  Val Loss: 0.6956  Acc: 72.2%  |  LR: 1.00e-04  (21.8s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6956)
Epoka [ 10/20]  Train Loss: 0.3045  Acc: 91.1%  |  Val Loss: 0.6860  Acc: 73.5%  |  LR: 1.00e-04  (21.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6860)
Epoka [ 11/20]  Train Loss: 0.2563  Acc: 92.0%  |  Val Loss: 0.9297  Acc: 69.4%  |  LR: 1.00e-04  (21.8s)
Epoka [ 12/20]  Train Loss: 0.1949  Acc: 93.8%  |  Val Loss: 0.7209  Acc: 73.9%  |  LR: 1.00e-04  (21.8s)
Epoka [ 13/20]  Train Loss: 0.1682  Acc: 95.1%  |  Val Loss: 0.8177  Acc: 71.8%  |  LR: 1.00e-04  (21.8s)
Epoka [ 14/20]  Train Loss: 0.1601  Acc: 95.0%  |  Val Loss: 0.9862  Acc: 68.2%  |  LR: 5.00e-05  (21.8s)
Epoka [ 15/20]  Train Loss: 0.1333  Acc: 96.0%  |  Val Loss: 0.7071  Acc: 76.3%  |  LR: 5.00e-05  (21.8s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.6860

Trening zakończony. Checkpoint: checkpoints\resnet50_best.pt
Log CSV: results\resnet50_training_log.csv
D:\Pycharmprojects\Computer-Vision-Project\train.py:193: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
Załadowano najlepsze wagi z epoki 10

Ewaluacja modelu: resnet50
----------------------------------------
  Accuracy:      76.68%
  F1 (weighted): 0.7723
  F1 (macro):    0.7678
  Quadratic Cohen's Kappa: 0.8700

  F1 per class:
    Normal: 0.8310
    Doubtful: 0.7368
    Mild: 0.6098
    Moderate: 0.8308
    Severe: 0.8308

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.92      0.76      0.83        78
    Doubtful       0.71      0.77      0.74        73
        Mild       0.54      0.69      0.61        36
    Moderate       0.87      0.79      0.83        34
      Severe       0.82      0.84      0.83        32

    accuracy                           0.77       253
   macro avg       0.77      0.77      0.77       253
weighted avg       0.79      0.77      0.77       253

  Historia treningowa zapisana: results\resnet50_training_history.png
  Metryki zapisane: results\resnet50_metrics.json
  Prawdopodobieństwa zapisane: results\resnet50_test_probs.npz

############################################################
# MODEL: efficientnet_b3
############################################################

Buduję model: efficientnet_b3
  timm ID:     efficientnet_b3
  Pretrained:  True
  Opis:        EfficientNet-B3 — dobry stosunek dokładności do rozmiaru
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
D:\Pycharmprojects\NNproject2\.venv1\Lib\site-packages\huggingface_hub\file_download.py:129: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\mader\.cache\huggingface\hub\models--timm--efficientnet_b3.ra2_in1k. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
  Parametry:   10,703,917 łącznie, 10,703,917 trenowalnych

============================================================
TRENING: efficientnet_b3
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 2.0180  Acc: 39.4%  |  Val Loss: 1.8174  Acc: 35.9%  |  LR: 1.00e-04  (181.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.8174)
Epoka [  2/20]  Train Loss: 0.5273  Acc: 77.5%  |  Val Loss: 1.5763  Acc: 49.0%  |  LR: 1.00e-04  (179.6s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.5763)
Epoka [  3/20]  Train Loss: 0.2212  Acc: 91.0%  |  Val Loss: 1.3195  Acc: 58.4%  |  LR: 1.00e-04  (179.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3195)
Epoka [  4/20]  Train Loss: 0.1300  Acc: 95.4%  |  Val Loss: 1.1721  Acc: 61.6%  |  LR: 1.00e-04  (179.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.1721)
Epoka [  5/20]  Train Loss: 0.1346  Acc: 95.6%  |  Val Loss: 1.0353  Acc: 64.1%  |  LR: 1.00e-04  (186.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0353)
Epoka [  6/20]  Train Loss: 0.0714  Acc: 97.8%  |  Val Loss: 1.0446  Acc: 66.5%  |  LR: 1.00e-04  (185.8s)
Epoka [  7/20]  Train Loss: 0.0482  Acc: 98.4%  |  Val Loss: 1.1614  Acc: 63.7%  |  LR: 1.00e-04  (183.5s)
Epoka [  8/20]  Train Loss: 0.0377  Acc: 99.0%  |  Val Loss: 1.1744  Acc: 62.9%  |  LR: 1.00e-04  (179.1s)
Epoka [  9/20]  Train Loss: 0.0454  Acc: 98.8%  |  Val Loss: 1.1613  Acc: 64.9%  |  LR: 5.00e-05  (177.9s)
Epoka [ 10/20]  Train Loss: 0.0318  Acc: 99.1%  |  Val Loss: 1.1353  Acc: 65.7%  |  LR: 5.00e-05  (177.9s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 1.0353

Trening zakończony. Checkpoint: checkpoints\efficientnet_b3_best.pt

  checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
Załadowano najlepsze wagi z epoki 5

Ewaluacja modelu: efficientnet_b3
----------------------------------------
  Accuracy:      62.45%
  F1 (weighted): 0.6287
  F1 (macro):    0.6241
  Quadratic Cohen's Kappa: 0.7399

  F1 per class:
    Normal: 0.6912
    Doubtful: 0.5939
    Mild: 0.3836
    Moderate: 0.7536
    Severe: 0.6984

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.81      0.60      0.69        78
    Doubtful       0.53      0.67      0.59        73
        Mild       0.38      0.39      0.38        36
    Moderate       0.74      0.76      0.75        34
      Severe       0.71      0.69      0.70        32

    accuracy                           0.62       253
   macro avg       0.63      0.62      0.62       253
weighted avg       0.65      0.62      0.63       253

  Historia treningowa zapisana: results\efficientnet_b3_training_history.png
  Metryki zapisane: results\efficientnet_b3_metrics.json
  Prawdopodobieństwa zapisane: results\efficientnet_b3_test_probs.npz

############################################################
# MODEL: densenet121
############################################################

Buduję model: densenet121
  timm ID:     densenet121
  Pretrained:  True
  Opis:        DenseNet-121 — popularny w medycznym imagingu
  Parametry:   6,958,981 łącznie, 6,958,981 trenowalnych

============================================================
TRENING: densenet121
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.4385  Acc: 38.1%  |  Val Loss: 1.3930  Acc: 39.6%  |  LR: 1.00e-04  (183.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.3930)
Epoka [  2/20]  Train Loss: 0.8929  Acc: 74.0%  |  Val Loss: 0.9697  Acc: 67.3%  |  LR: 1.00e-04  (190.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9697)
Epoka [  3/20]  Train Loss: 0.5660  Acc: 85.2%  |  Val Loss: 0.7595  Acc: 72.2%  |  LR: 1.00e-04  (187.7s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7595)
Epoka [  4/20]  Train Loss: 0.3445  Acc: 92.4%  |  Val Loss: 0.6450  Acc: 78.0%  |  LR: 1.00e-04  (187.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6450)
Epoka [  5/20]  Train Loss: 0.2134  Acc: 95.4%  |  Val Loss: 0.6350  Acc: 78.4%  |  LR: 1.00e-04  (193.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.6350)
Epoka [  6/20]  Train Loss: 0.1454  Acc: 97.2%  |  Val Loss: 0.5976  Acc: 77.1%  |  LR: 1.00e-04  (197.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5976)
Epoka [  7/20]  Train Loss: 0.1039  Acc: 97.6%  |  Val Loss: 0.6592  Acc: 77.6%  |  LR: 1.00e-04  (184.9s)
Epoka [  8/20]  Train Loss: 0.0772  Acc: 98.7%  |  Val Loss: 0.5895  Acc: 78.0%  |  LR: 1.00e-04  (152.9s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.5895)
Epoka [  9/20]  Train Loss: 0.0575  Acc: 99.1%  |  Val Loss: 0.6466  Acc: 78.4%  |  LR: 1.00e-04  (153.3s)
Epoka [ 10/20]  Train Loss: 0.0533  Acc: 99.0%  |  Val Loss: 0.6866  Acc: 77.1%  |  LR: 1.00e-04  (148.4s)
Epoka [ 11/20]  Train Loss: 0.0617  Acc: 98.7%  |  Val Loss: 0.8025  Acc: 72.7%  |  LR: 1.00e-04  (152.2s)
Epoka [ 12/20]  Train Loss: 0.0606  Acc: 98.4%  |  Val Loss: 0.8307  Acc: 75.5%  |  LR: 5.00e-05  (150.6s)
Epoka [ 13/20]  Train Loss: 0.0463  Acc: 99.0%  |  Val Loss: 0.6680  Acc: 79.6%  |  LR: 5.00e-05  (148.9s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.5895

Trening zakończony. Checkpoint: checkpoints\densenet121_best.pt
Log CSV: results\densenet121_training_log.csv
Załadowano najlepsze wagi z epoki 8

Ewaluacja modelu: densenet121
----------------------------------------
  Accuracy:      79.84%
  F1 (weighted): 0.7981
  F1 (macro):    0.8006
  Quadratic Cohen's Kappa: 0.8915

  F1 per class:
    Normal: 0.8375
    Doubtful: 0.7534
    Mild: 0.6761
    Moderate: 0.8571
    Severe: 0.8788

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.82      0.86      0.84        78
    Doubtful       0.75      0.75      0.75        73
        Mild       0.69      0.67      0.68        36
    Moderate       0.93      0.79      0.86        34
      Severe       0.85      0.91      0.88        32

    accuracy                           0.80       253
   macro avg       0.81      0.80      0.80       253
weighted avg       0.80      0.80      0.80       253

  Historia treningowa zapisana: results\densenet121_training_history.png
  Metryki zapisane: results\densenet121_metrics.json
  Prawdopodobieństwa zapisane: results\densenet121_test_probs.npz

############################################################
# MODEL: mobilenetv3_large
############################################################

Buduję model: mobilenetv3_large
  timm ID:     mobilenetv3_large_100
  Pretrained:  True
  Opis:        MobileNetV3 — lekki, szybki
  Parametry:   4,208,437 łącznie, 4,208,437 trenowalnych

============================================================
TRENING: mobilenetv3_large
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.9624  Acc: 40.7%  |  Val Loss: 2.2398  Acc: 44.1%  |  LR: 1.00e-04  (8.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 2.2398)
Epoka [  2/20]  Train Loss: 0.6208  Acc: 74.5%  |  Val Loss: 1.2592  Acc: 55.9%  |  LR: 1.00e-04  (8.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2592)
Epoka [  3/20]  Train Loss: 0.3368  Acc: 86.6%  |  Val Loss: 1.0829  Acc: 66.1%  |  LR: 1.00e-04  (8.1s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0829)
Epoka [  4/20]  Train Loss: 0.1966  Acc: 92.4%  |  Val Loss: 1.0146  Acc: 65.7%  |  LR: 1.00e-04  (8.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.0146)
Epoka [  5/20]  Train Loss: 0.1089  Acc: 97.1%  |  Val Loss: 1.0670  Acc: 68.6%  |  LR: 1.00e-04  (8.1s)
Epoka [  6/20]  Train Loss: 0.0922  Acc: 97.2%  |  Val Loss: 0.9838  Acc: 69.8%  |  LR: 1.00e-04  (8.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9838)
Epoka [  7/20]  Train Loss: 0.0721  Acc: 97.7%  |  Val Loss: 0.9771  Acc: 71.0%  |  LR: 1.00e-04  (8.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.9771)
Epoka [  8/20]  Train Loss: 0.0704  Acc: 97.7%  |  Val Loss: 1.0049  Acc: 69.8%  |  LR: 1.00e-04  (8.1s)
Epoka [  9/20]  Train Loss: 0.0522  Acc: 98.4%  |  Val Loss: 1.0725  Acc: 71.0%  |  LR: 1.00e-04  (8.2s)
Epoka [ 10/20]  Train Loss: 0.0526  Acc: 98.4%  |  Val Loss: 0.9788  Acc: 71.0%  |  LR: 1.00e-04  (8.1s)
Epoka [ 11/20]  Train Loss: 0.0423  Acc: 98.4%  |  Val Loss: 1.0468  Acc: 71.0%  |  LR: 5.00e-05  (8.3s)
Epoka [ 12/20]  Train Loss: 0.0397  Acc: 99.1%  |  Val Loss: 1.0022  Acc: 71.4%  |  LR: 5.00e-05  (8.1s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.9771

Trening zakończony. Checkpoint: checkpoints\mobilenetv3_large_best.pt
Log CSV: results\mobilenetv3_large_training_log.csv
Załadowano najlepsze wagi z epoki 7

Ewaluacja modelu: mobilenetv3_large
----------------------------------------
  Accuracy:      73.91%
  F1 (weighted): 0.7357
  F1 (macro):    0.7328
  Quadratic Cohen's Kappa: 0.8655

  F1 per class:
    Normal: 0.8256
    Doubtful: 0.6614
    Mild: 0.5455
    Moderate: 0.8060
    Severe: 0.8254

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.76      0.91      0.83        78
    Doubtful       0.78      0.58      0.66        73
        Mild       0.51      0.58      0.55        36
    Moderate       0.82      0.79      0.81        34
      Severe       0.84      0.81      0.83        32

    accuracy                           0.74       253
   macro avg       0.74      0.74      0.73       253
weighted avg       0.75      0.74      0.74       253

  Historia treningowa zapisana: results\mobilenetv3_large_training_history.png
  Metryki zapisane: results\mobilenetv3_large_metrics.json
  Prawdopodobieństwa zapisane: results\mobilenetv3_large_test_probs.npz

############################################################
# MODEL: convnext_tiny
############################################################

Buduję model: convnext_tiny
  timm ID:     convnext_tiny
  Pretrained:  True
  Opis:        ConvNeXt-Tiny — nowoczesna architektura
  Parametry:   27,823,973 łącznie, 27,823,973 trenowalnych

============================================================
TRENING: convnext_tiny
Urządzenie: cuda
============================================================
Epoka [  1/20]  Train Loss: 1.6326  Acc: 25.4%  |  Val Loss: 1.4983  Acc: 20.8%  |  LR: 1.00e-04  (541.0s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.4983)
Epoka [  2/20]  Train Loss: 1.2658  Acc: 45.2%  |  Val Loss: 1.2806  Acc: 48.6%  |  LR: 1.00e-04  (569.3s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 1.2806)
Epoka [  3/20]  Train Loss: 0.9576  Acc: 58.2%  |  Val Loss: 0.8666  Acc: 72.7%  |  LR: 1.00e-04  (573.2s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.8666)
Epoka [  4/20]  Train Loss: 0.7166  Acc: 69.0%  |  Val Loss: 0.7857  Acc: 73.5%  |  LR: 1.00e-04  (581.5s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7857)
Epoka [  5/20]  Train Loss: 0.4938  Acc: 77.9%  |  Val Loss: 0.8538  Acc: 66.9%  |  LR: 1.00e-04  (574.6s)
Epoka [  6/20]  Train Loss: 0.4065  Acc: 81.6%  |  Val Loss: 0.8506  Acc: 60.0%  |  LR: 1.00e-04  (571.6s)
Epoka [  7/20]  Train Loss: 0.2759  Acc: 87.3%  |  Val Loss: 1.2033  Acc: 74.7%  |  LR: 1.00e-04  (579.4s)
Epoka [  8/20]  Train Loss: 0.2996  Acc: 90.1%  |  Val Loss: 0.7162  Acc: 79.2%  |  LR: 1.00e-04  (581.4s)
  ✓ Zapisano najlepszy checkpoint (val_loss: 0.7162)
Epoka [  9/20]  Train Loss: 0.2367  Acc: 90.6%  |  Val Loss: 0.8507  Acc: 75.9%  |  LR: 1.00e-04  (579.3s)
Epoka [ 10/20]  Train Loss: 0.1036  Acc: 95.4%  |  Val Loss: 1.0463  Acc: 71.0%  |  LR: 1.00e-04  (580.1s)
Epoka [ 11/20]  Train Loss: 0.0564  Acc: 97.4%  |  Val Loss: 1.8481  Acc: 64.9%  |  LR: 1.00e-04  (582.3s)
Epoka [ 12/20]  Train Loss: 0.0803  Acc: 96.9%  |  Val Loss: 1.0981  Acc: 72.2%  |  LR: 5.00e-05  (582.8s)
Epoka [ 13/20]  Train Loss: 0.0605  Acc: 97.1%  |  Val Loss: 1.3646  Acc: 75.9%  |  LR: 5.00e-05  (579.1s)

  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.7162

Trening zakończony. Checkpoint: checkpoints\convnext_tiny_best.pt
Log CSV: results\convnext_tiny_training_log.csv
D:\Pycharmprojects\Computer-Vision-Project\train.py:193: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
Załadowano najlepsze wagi z epoki 8

Ewaluacja modelu: convnext_tiny
----------------------------------------
  Accuracy:      76.28%
  F1 (weighted): 0.7631
  F1 (macro):    0.7560
  Quadratic Cohen's Kappa: 0.9013

  F1 per class:
    Normal: 0.8387
    Doubtful: 0.7211
    Mild: 0.5833
    Moderate: 0.8116
    Severe: 0.8254

  Classification Report:
              precision    recall  f1-score   support

      Normal       0.84      0.83      0.84        78
    Doubtful       0.72      0.73      0.72        73
        Mild       0.58      0.58      0.58        36
    Moderate       0.80      0.82      0.81        34
      Severe       0.84      0.81      0.83        32

    accuracy                           0.76       253
   macro avg       0.76      0.76      0.76       253
weighted avg       0.76      0.76      0.76       253

  Historia treningowa zapisana: results\convnext_tiny_training_history.png
  Metryki zapisane: results\convnext_tiny_metrics.json
  Prawdopodobieństwa zapisane: results\convnext_tiny_test_probs.npz

================================================================================
PODSUMOWANIE — PORÓWNANIE MODELI
================================================================================
Model                       Accuracy   F1 (wgt)      Kappa
--------------------------------------------------------------------------------
convnext_tiny                 76.28%     0.7631     0.9013
densenet121                   79.84%     0.7981     0.8915
resnet50                      76.68%     0.7723     0.8700
mobilenetv3_large             73.91%     0.7357     0.8655
efficientnet_b3               62.45%     0.6287     0.7399
================================================================================
↑ Posortowane wg Cohen's Kappa 

✓ Gotowe! Wyniki w folderze: results

Process finished with exit code 0
