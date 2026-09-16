# Mushroom Classification — zaharli qo'ziqorinni aniqlash

Qo'ziqorinning tashqi belgilariga qarab uning **zaharli (poisonous)** yoki **zaharsiz (edible)** ekanini aniqlaydigan klassifikatsiya modeli. To'rtta algoritm qurilib solishtirildi, eng yaxshisi tanlanib deploy uchun saqlandi.

**Natija:** Decision Tree test to'plamida 100% aniqlik beradi va buni 117 ta xususiyatdan atigi **12 tasi** va **14 ta qoida** bilan qiladi — ya'ni model qora quti emas, uning butun mantiqini bir rasmda ko'rish mumkin.

---

## Model natijalari

Test to'plami: 1625 qator (842 zaharsiz, 783 zaharli)

| Model | Accuracy | Precision | Recall | F1 | FN* |
|---|---|---|---|---|---|
| **Decision Tree** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0** |
| KNN (k=5) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| Logistic Regression | 0.9994 | 1.0000 | 0.9987 | 0.9994 | 1 |

\* **FN (False Negative)** — zaharli qo'ziqorinni "zaharsiz" deb aytish. Bu masalada eng qimmat xato turi, shuning uchun alohida ustunda.

![Confusion Matrix](images/02-confusion-matrix.png)

---

## Ma'lumotlar to'plami

[UCI Mushroom Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification) — **8124 qator, 23 ustun**, barchasi kategorik (harflar bilan kodlangan).

Target ustuni `class`: `e` (edible) 4208 ta, `p` (poisonous) 3916 ta — **balanslashgan**, shuning uchun accuracy chalg'itmaydi.

---

## Muhim topilmalar

### 1. Yashirin bo'sh qiymatlar

`df.isnull().sum()` **0** qaytaradi — dataset toza ko'rinadi. Aslida emas:

```python
df['stalk-root'].value_counts()
# b    3776
# ?    2480      ← barcha qatorlarning 30.5%
# e    1120
```

`?` belgisi CSV'da oddiy matn sifatida yozilgan, shuning uchun pandas uni `NaN` deb tanimaydi. Yashirin bo'sh qiymatlarni topish uchun har bir ustunning unikal qiymatlarini tekshirish shart.

### 2. `?` ni to'ldirish — xato bo'lardi

```python
pd.crosstab(df['stalk-root'], df['class'], normalize='index')
#             e      p
# ?       0.290  0.710      ← 71% zaharli
# b       0.508  0.492
# r       1.000  0.000
```

`?` qatorlarining **71% zaharli**, `b` da esa 49%. Ya'ni yo'qolgan qiymatning o'zi signal tashiydi (*missing not at random*). `mode()` bilan to'ldirilsa yoki ustun tashlansa, bu ma'lumot yo'qolardi — shuning uchun `?` alohida toifa sifatida saqlab qolindi.

### 3. Kategorik bog'liqlikni o'lchash

Barcha ustunlar kategorik bo'lgani uchun `df.corr()` (Pearson) mos kelmaydi. O'rniga **Cramér's V** (bias tuzatilgan, chi-square asosida) ishlatildi:

![Cramér's V](images/01-cramers-v.png)

`odor` (hid) — 0.97, deyarli target'ning o'zi. `veil-type` — 0.00, chunki barcha qatorlarda bitta qiymat (`p`), ya'ni foydasiz ustun.

### 4. Logistic Regression nega bitta xato qildi

Adashgan qator: `odor = n` (hidsiz). Datasetda hidsiz qo'ziqorinlarning **96.6% zaharsiz**, bu esa qolgan 3.4% istisnodan biri edi.

Model unga **0.4533** ehtimol bergan — ya'ni deyarli ikkilangan, lekin `predict()` 0.5 bilan kesgani uchun javob "zaharsiz" chiqdi.

Sabab strukturaviy: LogReg **chiziqli va qo'shimchali** model, har xususiyatga bitta koeffitsient beradi. Decision Tree esa shartlarni ketma-ket qo'yib ("hidsiz **va** oq sporali **va** clustered") tor istisno hududlarini kesib oladi.

Threshold'ni pasaytirish muammoni butunlay hal qiladi:

| Threshold | FN | FP |
|---|---|---|
| 0.5 | 1 | 0 |
| **0.4** | **0** | **0** |
| 0.3 | 0 | 0 |

---

## Tanlangan model: Decision Tree

Uchta model bir xil mukammal metrika bergani uchun tanlov qo'shimcha mezonlar bo'yicha qilindi:

| Mezon | Decision Tree | KNN | Random Forest |
|---|---|---|---|
| Ishlatilgan xususiyatlar | **12 / 117** | hammasi | hammasi |
| Model hajmi | **9 KB** | ~1 MB (train saqlanadi) | ~1 MB (100 daraxt) |
| Qoidalarni ko'rsatish | **mumkin** | mumkin emas | juda qiyin |
| Bashorat tezligi | **tez** | sekin (har safar masofa) | o'rtacha |

![Feature importance](images/03-feature-importance.png)

Butun modelning mantiqi — 14 ta qoida:

![Decision Tree](images/04-decision-tree.png)

Birinchi qoidani odam tilida o'qish mumkin: *"agar qo'ziqorinning hidi bo'lsa va u anise ham, almond ham bo'lmasa — zaharli."* Bu mikologiyadagi haqiqiy qoidaga mos keladi.

---

## Ishlatish

```bash
pip install -r requirements.txt
```

```python
import joblib, pandas as pd

bundle = joblib.load("mushroom_dt.joblib")
model, FEATURES = bundle["model"], bundle["features"]

namuna = {
    "cap-shape": "x", "cap-surface": "s", "cap-color": "n", "bruises": "t",
    "odor": "p", "gill-attachment": "f", "gill-spacing": "c", "gill-size": "n",
    "gill-color": "k", "stalk-shape": "e", "stalk-root": "e",
    "stalk-surface-above-ring": "s", "stalk-surface-below-ring": "s",
    "stalk-color-above-ring": "w", "stalk-color-below-ring": "w",
    "veil-type": "p", "veil-color": "w", "ring-number": "o", "ring-type": "p",
    "spore-print-color": "k", "population": "s", "habitat": "u",
}

row   = pd.DataFrame([namuna], columns=FEATURES)
proba = model.predict_proba(row)[0, 1]
print("ZAHARLI" if proba >= 0.5 else "zaharsiz", f"({proba:.2%})")
# ZAHARLI (100.00%)
```

Model `Pipeline` sifatida saqlangan — `OneHotEncoder` ichida, shuning uchun **xom qiymatlarni to'g'ridan-to'g'ri** berish mumkin. Qo'lda encoding qilish shart emas.

---

## Loyiha tuzilishi

```
mushroom-classification/
├── README.md
├── requirements.txt
├── mushroom_dt.joblib          # o'qitilgan Pipeline (OneHotEncoder + DecisionTree)
├── predict.py                  # bashorat funksiyasi + validatsiya
├── results.csv                 # 4 model metrikalari
├── notebook.ipynb              # to'liq tahlil
└── images/
    ├── 01-cramers-v.png
    ├── 02-confusion-matrix.png
    ├── 03-feature-importance.png
    └── 04-decision-tree.png
```

---

## Texnologiyalar

Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · joblib

---

## Keyingi qadamlar

- **Cross-validation** — 1.0 natija bitta omadli split'dan kelib chiqmaganiga ishonch hosil qilish
- **`class_weight='balanced'`** — zaharli sinfdagi xatoni og'irroq hisoblash
- **Threshold sozlash** — xavfsizlik uchun 0.5 o'rniga pastroq chegara
- **`max_depth` cheklash** — yanada soddaroq yetarli model topish (*Occam's razor*)

> **Eslatma:** bu o'quv loyihasi. Haqiqiy qo'ziqorin terishda ishlatmang — model faqat shu datasetdagi 23 turdagi belgilar bo'yicha o'qitilgan va tabiatdagi barcha turlarni qamramaydi.
