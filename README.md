# Credit Card Fraud Detection — XGBoost + SHAP

Credit card fraud detection using XGBoost, SMOTE and SHAP explainability.
Full ML pipeline with threshold optimization and FastAPI deployment.

## Results

| Model              | AUROC  | F1 Fraud | Accuracy |
|--------------------|--------|----------|----------|
| Logistic Regression| 0.9817 | 0.12     | 99.9%    |
| Random Forest      | 0.9815 | 0.86     | 99.9%    |
| **XGBoost (tuned)**| **0.9777** | **0.83** | **99.9%** |

- False positives : 3
- False negatives : 19
- Optimal threshold : 0.991

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284 807 transactions — 30 features (V1-V28 PCA + Time + Amount) — 0.17% fraud — CC0 license

## Project Structure

```
fraud-detection-xgboost-shap/
├── src/
│   ├── preprocess.py       # data cleaning and SMOTE
│   ├── train.py            # model training and GridSearchCV
│   ├── evaluate.py         # metrics and error analysis
│   ├── explain.py          # SHAP explainability
│   └── api.py              # FastAPI inference route
├── tests/
│   └── test_pipeline.py    # unit tests
├── notebooks/
│   └── fraud_detection.ipynb
├── outputs/
│   └── metriques.json
├── checkpoints/
│   └── best_model.pkl
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py --data creditcard.csv --seed 42 --cv 3
```

## API

```bash
uvicorn src.api:app --reload
# POST /predict  →  { "fraude": true, "probabilite": 0.99, "top3_features": [...] }
```

## Environment

- Python 3.10
- CPU Google Colab (Intel Xeon ~12GB RAM)
- No GPU required
- See requirements.txt

## License

MIT
