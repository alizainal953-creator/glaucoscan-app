# GlaucoScan AI — Streamlit Dashboard
**IDSC 2025 · Glaucoma GON Detection**

This is a GlaucoScan AI, we use collab notebook to run the training by using EfficientNetB3 Backbone to extract the image  + Late Fusion (image quality score) + SVM Kernel RBF. We split it by K-Fold Patient Train Val Split to avoid data leakage. So, each patient has a chance to be the train dataset or validation dataset. We're going to show you how to run the dashboard. 
Prepare : 
## Folder Structure
glaucoscan/
├── app.py                  
├── requirements.txt
├── assets/                 
│   ├── fold_results.csv
│   ├── history_fold0..4.png
│   ├── confusion_matrix_fold0..4.png
│   └── dev_set.csv, test_set.csv
└── streamlit_models/       
    ├── fold0_cnn.pth
    ├── fold0_svm.joblib
    ├── fold1_cnn.pth
    ├── fold1_svm.joblib
    ├── ... (until fold4)
    └── model_config.json
```

## Step by Step

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put all of the files .pt and .joblib into folder streamlit_models/

# 3. Run 
streamlit run app.py
```

## Explanation

This model uses 5 fold training so instead of using one fold as the best result, we ensemble and average it by using 5 folds of model training. Here are the AUC Validation score in all fold :

| Fold | Val AUC |
|------|---------|
| 0    | 0.9966  |
| 1    | 0.9805  |
| 2    | 0.9986  |
| 3    | 0.9904  |
| 4    | 0.9478  |
| **Mean** | **0.9828 ± 0.0188** |

## Dashboard Link
glaucoscan