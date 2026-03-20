import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score,
    f1_score, roc_curve
)
from collections import Counter


def full_evaluation(model, X_test, y_test, seuil, class_names=('Normal', 'Fraude')):
    """
    Evaluation complete sur le test set.
    Retourne les predictions, probabilites et metriques.
    """
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test  = (proba_test >= seuil).astype(int)

    auroc = roc_auc_score(y_test, proba_test)
    f1    = f1_score(y_test, pred_test)

    print('=== Resultats test set ===')
    print(f'AUROC : {auroc:.4f}')
    print(f'F1    : {f1:.4f}')
    print()
    print(classification_report(y_test, pred_test, target_names=class_names))

    return pred_test, proba_test, auroc, f1


def plot_confusion_matrix(y_test, pred_test, class_names=('Normal', 'Fraude')):
    cm   = confusion_matrix(y_test, pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Matrice de confusion - test set')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=120)
    plt.show()
    print('Matrice sauvegardee : outputs/confusion_matrix.png')


def plot_roc_curve(y_test, proba_test, auroc):
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', label=f'XGBoost (AUROC={auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Aleatoire')
    ax.set_xlabel('Taux faux positifs')
    ax.set_ylabel('Taux vrais positifs')
    ax.set_title('Courbe ROC')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/roc_curve.png', dpi=120)
    plt.show()


def error_analysis(X_test, y_test, pred_test, proba_test):
    """
    Identifie les faux positifs et faux negatifs.
    Cherche des patterns dans les erreurs.
    """
    X_copy = X_test.copy()
    X_copy['y_true'] = y_test.values
    X_copy['y_pred'] = pred_test
    X_copy['proba']  = proba_test

    fp = X_copy[(X_copy['y_true'] == 0) & (X_copy['y_pred'] == 1)]
    fn = X_copy[(X_copy['y_true'] == 1) & (X_copy['y_pred'] == 0)]

    print(f'Faux positifs : {len(fp)}')
    print(f'Faux negatifs : {len(fn)}')

    # transaction la plus mal classee (fraude ratee avec la plus faible probabilite)
    if len(fn) > 0:
        worst = fn.sort_values('proba').iloc[0]
        print(f'\nPire faux negatif : index {worst.name}')
        print(f'Probabilite predite : {worst["proba"]:.4f}')
        print('V14 :', round(worst.get('V14', float("nan")), 4),
              ' V17 :', round(worst.get('V17', float("nan")), 4))
        print('=> Profil atypique : valeurs V14/V17 proches des transactions normales')

    return fp, fn


if __name__ == '__main__':
    import sys
    sys.path.insert(0, 'src')
    from preprocess import load_and_clean, split_data, apply_smote

    df, scaler = load_and_clean('creditcard.csv')
    _, _, X_test, _, _, y_test = split_data(df)

    model  = joblib.load('checkpoints/best_model.pkl')
    params = json.load(open('outputs/metriques.json'))
    seuil  = params['seuil']

    pred_test, proba_test, auroc, f1 = full_evaluation(model, X_test, y_test, seuil)
    plot_confusion_matrix(y_test, pred_test)
    plot_roc_curve(y_test, proba_test, auroc)
    error_analysis(X_test, y_test, pred_test, proba_test)