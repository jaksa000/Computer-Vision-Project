Dataset is too big to upload here, downolad it from here: https://www.kaggle.com/datasets/tommyngx/digital-knee-xray/data?select=MedicalExpert-I
unzip and rename to data and put in project directory, should work from here

SOLUTION EXAMPLE
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



  Early stopping: brak poprawy przez 5 epok.
  Najlepsza val_loss: 0.7162
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
