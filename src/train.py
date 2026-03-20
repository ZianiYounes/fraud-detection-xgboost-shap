import argparse
import random
import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.metrics import precision_recall_curv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

from preprocess import load_and_clean, split_data, apply_smote

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def compare_baselines(X_train_sm, y_train_sm, X_val, y_val):
    modeles = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost':             XGBClassifier(
                                   n_estimators=300, max_depth=5, learning_rate=0.05,
                                   use_label_encoder=False, eval_metric='logloss',
                                   random_state=42, verbosity=0
                               )
    }

    resultats = {}
    for nom, modele in modeles.items():
        modele.fit(X_train_sm, y_train_sm)
        proba = modele.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        resultats[nom] = {
            'AUROC': round(roc_auc_score(y_val, proba), 4),
            'F1':    round(f1_score(y_val, pred), 4)
        }
        print(f'{nom:25s} -> AUROC: {resultats[nom]["AUROC"]}  F1: {resultats[nom]["F1"]}')

    return resultats


def tune_xgboost(X_train_sm, y_train_sm, cv=3):
    param_grid = {
        'max_depth':     [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators':  [200, 300]
    }

    xgb_base = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )

    grid_search = GridSearchCV(
        xgb_base, param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_sm, y_train_sm)

    print('Meilleurs hyperparametres :', grid_search.best_params_)
    print('Meilleur AUROC CV         :', round(grid_search.best_score_, 4))

    return grid_search.best_estimator_, grid_search.best_params_


def find_optimal_threshold(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    idx_best  = np.argmax(f1_scores)

    seuil_optimal = float(thresholds[idx_best])
    print(f'Seuil optimal : {seuil_optimal:.3f}  |  F1 : {f1_scores[idx_best]:.4f}')

    return seuil_optimal


def main(args):
    set_seeds(args.seed)

    print('Chargement et nettoyage...')
    df, scaler = load_and_clean(args.data)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=args.seed)

    print('Application SMOTE...')
    X_train_sm, y_train_sm = apply_smote(X_train, y_train, seed=args.seed)

    print('\nComparaison des modeles de base :')
    compare_baselines(X_train_sm, y_train_sm, X_val, y_val)

    print('\nTuning XGBoost...')
    best_model, best_params = tune_xgboost(X_train_sm, y_train_sm, cv=args.cv)

    seuil = find_optimal_threshold(best_model, X_val, y_val)
    proba_val = best_model.predict_proba(X_val)[:, 1]
    pred_val  = (proba_val >= seuil).astype(int)
    print(f'\nVal  -> AUROC: {roc_auc_score(y_val, proba_val):.4f}  F1: {f1_score(y_val, pred_val):.4f}')

    proba_test = best_model.predict_proba(X_test)[:, 1]
    pred_test  = (proba_test >= seuil).astype(int)
    auroc_test = roc_auc_score(y_test, proba_test)
    f1_test    = f1_score(y_test, pred_test)
    print(f'Test -> AUROC: {auroc_test:.4f}  F1: {f1_test:.4f}')
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(best_model, 'checkpoints/best_model.pkl')
    joblib.dump(scaler,     'checkpoints/scaler.pkl')

    metriques = {
        'auroc_test':  round(auroc_test, 4),
        'f1_test':     round(f1_test, 4),
        'seuil':       round(seuil, 3),
        'best_params': best_params
    }
    with open('outputs/metriques.json', 'w') as f:
        json.dump(metriques, f, indent=2)

    print('\nCheckpoint sauvegarde : checkpoints/best_model.pkl')
    print('Metriques :', metriques)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='creditcard.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv',   type=int, default=3)
    args = parser.parse_args()
    main(args)