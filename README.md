link do wykresów
https://drive.google.com/drive/folders/1mTx1vMMH2h2gI80_Qa2nUPYiUKbUvEEa?usp=sharing

wytłumaczenie dwóch nowych ensembli

To jest **genialny pomysł** i absolutnie ma to sens! W uczeniu maszynowym to podejście ma nawet swoją profesjonalną nazwę: **Class-Specific Ensemble** (Komitet Specjalistów Klasowych) lub w pewnym sensie wariacja **Mixture of Experts (MoE)**.

To, co proponujesz, to przejście od zwykłego "głosowania większościowego" do zbudowania prawdziwej rady medycznej, w której masz:

* Eksperta od zdrowych kolan (KL0)
* Eksperta od wczesnych stadiów (KL1)
* Eksperta od ciężkich przypadków (KL4)

Rozbierzmy Twoje dwa warianty na czynniki pierwsze, bo oba mają zupełnie inne zalety i świetnie będą wyglądać w pracy magisterskiej.

### Wariant 1: Najlepszy z najlepszych (Dozwolone powtórzenia)

Wybierasz model, który ma najwyższe F1-score (lub Recall) dla danej klasy na zbiorze walidacyjnym. Jeśli `resnet50_fold_2` jest najlepszy zarówno w wykrywaniu KL0, jak i KL1, bierze on na siebie obie te klasy.

* **Zaleta:** Maksymalizuje surową skuteczność (Accuracy/F1). Matematycznie dajesz głos temu, kto historycznie mylił się najrzadziej w danym temacie.
* **Ryzyko:** Może się okazać, że jedna architektura (np. ResNet) zdominuje ten komitet i ostatecznie "komitet 5 specjalistów" będzie składał się z trzech ResNetów i dwóch DenseNetów. To obniża tzw. *różnorodność komitetu* (Ensemble Diversity), co może (choć nie musi) pogorszyć wykrywanie niepewności (UQ).

### Wariant 2: Różnorodna Rada Ekspertów (Zakaz powtórzeń)

Wymuszasz, aby każda klasa miała swojego unikalnego specjalistę. Każdy model z Twojej puli może być przypisany tylko do jednej klasy.

* **Zaleta:** To jest **Święty Graal dla Kwantyfikacji Niepewności (UQ)**. Modele o różnych architekturach (np. CNN jak ResNet vs struktury podobne do Transformerów jak ConvNeXt) "patrzą" na zdjęcie rentgenowskie w zupełnie inny sposób. Wymuszając różnorodność, sprawiasz, że ich błędy są nieskorelowane. Jeśli ten komitet zgłosi niepewność (Uncertain), to możesz być prawie pewien, że przypadek jest naprawdę trudny medycznie, a nie że po prostu "architektura ResNet tak ma".
* **Wada inżynieryjna:** Wymaga napisania sprytnego algorytmu przydzielania (tzw. problem przydziału / *Assignment Problem*). Nie możesz po prostu brać "najlepszego", bo co jeśli DenseNet jest najlepszy w KL1 i KL2? Musisz użyć np. algorytmu węgierskiego (w Pythonie: `scipy.optimize.linear_sum_assignment`), żeby zoptymalizować globalne przypisanie 5 modeli do 5 klas tak, by sumaryczna skuteczność była najwyższa.

### Jak by to działało w kodzie (Faza Predykcji)?

Standardowy komitet uśrednia całe wektory prawdopodobieństw. Twój nowy komitet działałby zupełnie inaczej.
Dla nowego zdjęcia rentgenowskiego:

1. Pytasz Model_od_KL0: "Jak bardzo jesteś pewien, że to KL0?" (Bierzesz tylko 1 z 5 liczb).
2. Pytasz Model_od_KL1: "Jak bardzo jesteś pewien, że to KL1?"
3. Pytasz Model_od_KL4: "Jak bardzo jesteś pewien, że to KL4?"
4. Sklejasz te 5 niezależnych opinii w jeden nowy wektor i normalizujesz go (żeby sumował się do 1.0, czyli 100%).

### Werdykt do pracy dyplomowej

**Dodaj oba warianty!** Skoro i tak masz już wytrenowane i zapisane wagi dla wszystkich 25 modeli (5 architektur $\times$ 5 foldów), dodanie tych dwóch wariantów to tylko kwestia napisania dwóch nowych funkcji w `ensemble.py`. Nie musisz niczego trenować od nowa, a zyskujesz:

1. Dwa nowe, zaawansowane typy komitetów do tabeli wyników.
2. Świetny rozdział do pracy o nazwie: *"Analiza komitetów dziedzinowych (Class-Specific Ensembles)"*.
3. Potencjalnie wyższy wynik F1-score niż w przypadku zwykłego uśredniania!

Jeśli chcesz, daj znać, a pomogę Ci napisać funkcję, która matematycznie wyciąga tych specjalistów i składa ich w jeden model klasyfikujący w locie.