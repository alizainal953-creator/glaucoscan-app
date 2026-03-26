# GlaucoScan AI — Streamlit Dashboard
**IDSC 2025 · Glaucoma GON Detection**

## Struktur Folder

```
glaucoscan/
├── app.py                  ← aplikasi utama
├── requirements.txt
├── assets/                 ← hasil training (grafik, csv)
│   ├── fold_results.csv
│   ├── history_fold0..4.png
│   ├── confusion_matrix_fold0..4.png
│   └── dev_set.csv, test_set.csv
└── streamlit_models/       ← checkpoint model (taruh di sini)
    ├── fold0_cnn.pth
    ├── fold0_svm.joblib
    ├── fold1_cnn.pth
    ├── fold1_svm.joblib
    ├── ... (sampai fold4)
    └── model_config.json
```

## Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Taruh semua file .pth dan .joblib ke folder streamlit_models/

# 3. Jalankan
streamlit run app.py
```

## Penjelasan Ensemble

Daripada memilih satu fold terbaik, aplikasi ini me-**load semua 5 fold** sekaligus dan
merata-ratakan probabilitas prediksinya. Cara ini menghasilkan prediksi yang lebih
robust karena setiap fold melihat subset data yang berbeda saat training.

| Fold | Val AUC |
|------|---------|
| 0    | 0.9966  |
| 1    | 0.9805  |
| 2    | 0.9986  |
| 3    | 0.9904  |
| 4    | 0.9478  |
| **Mean** | **0.9828 ± 0.0188** |
