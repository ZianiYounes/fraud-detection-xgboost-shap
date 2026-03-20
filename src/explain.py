import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import json


def shap_global(model, X_test, save=True):
    """
    Calcule les SHAP values et affiche l'importance globale
    et le beeswarm plot.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # importance globale
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.title('Importance globale des features (SHAP)')
    plt.tight_layout()
    if save:
        plt.savefig('outputs/shap_importance.png', dpi=120)
    plt.show()

    # beeswarm
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('Impact des features — SHAP beeswarm')
    plt.tight_layout()
    if save:
        plt.savefig('outputs/shap_beeswarm.png', dpi=120)
    plt.show()

    return explainer, shap_values


def shap_local(explainer, shap_values, X_test, idx, proba):
    """
    Waterfall plot pour une transaction specifique.
    Utile pour expliquer pourquoi une fraude a ete ratee.
    """
    print(f'Transaction index {idx} — probabilite predite : {proba:.4f}')
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[idx],
            base_values   = explainer.expected_value,
            data          = X_test.iloc[idx],
            feature_names = X_test.columns.tolist()
        ),
        show=False
    )
    plt.title(f'Explication SHAP — transaction {idx}')
    plt.tight_layout()
    plt.savefig(f'outputs/shap_waterfall_{idx}.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    import sys, pandas as pd
    sys.path.insert(0, 'src')
    from preprocess import load_and_clean, split_data

    df, _    = load_and_clean('creditcard.csv')
    _, _, X_test, _, _, y_test = split_data(df)

    model  = joblib.load('checkpoints/best_model.pkl')
    params = json.load(open('outputs/metriques.json'))
    seuil  = params['seuil']

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test  = (proba_test >= seuil).astype(int)

    explainer, shap_values = shap_global(model, X_test)

    # waterfall sur le premier faux negatif
    fn_indices = [i for i in range(len(y_test))
                  if y_test.iloc[i] == 1 and pred_test[i] == 0]
    if fn_indices:
        shap_local(explainer, shap_values, X_test, fn_indices[0], proba_test[fn_indices[0]])