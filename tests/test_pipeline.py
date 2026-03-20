import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import apply_smote, split_data


def make_dummy_df(n_normal=500, n_fraud=10):
    """Cree un petit dataset synthetique pour les tests."""
    np.random.seed(42)
    normal = pd.DataFrame(np.random.randn(n_normal, 30),
                          columns=[f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled'])
    normal['Class'] = 0

    fraud = pd.DataFrame(np.random.randn(n_fraud, 30),
                         columns=[f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled'])
    fraud['Class'] = 1

    return pd.concat([normal, fraud], ignore_index=True)


class TestSplit:
    def test_split_sizes(self):
        df = make_dummy_df()
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=42)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(df), "La somme des splits doit egaliser le dataset complet"

    def test_split_stratification(self):
        df = make_dummy_df(n_normal=1000, n_fraud=20)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=42)
        # chaque split doit contenir au moins 1 fraude
        assert y_train.sum() > 0
        assert y_val.sum()   > 0
        assert y_test.sum()  > 0


class TestSMOTE:
    def test_smote_balances_classes(self):
        df = make_dummy_df(n_normal=500, n_fraud=10)
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_sm, y_sm = apply_smote(X, y, seed=42)
        # apres SMOTE les deux classes doivent etre equilibrees
        assert y_sm.sum() == (y_sm == 0).sum(), \
            "SMOTE doit produire autant de fraudes que de normales"

    def test_smote_no_data_leakage(self):
        df = make_dummy_df(n_normal=500, n_fraud=10)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=42)
        original_val_size  = len(X_val)
        original_test_size = len(X_test)
        X_sm, y_sm = apply_smote(X_train, y_train, seed=42)
        # val et test ne doivent pas changer
        assert len(X_val)  == original_val_size,  "SMOTE ne doit pas modifier le val set"
        assert len(X_test) == original_test_size, "SMOTE ne doit pas modifier le test set"


class TestFeatures:
    def test_no_missing_values(self):
        df = make_dummy_df()
        assert df.isnull().sum().sum() == 0

    def test_feature_count(self):
        df = make_dummy_df()
        X = df.drop('Class', axis=1)
        assert X.shape[1] == 30, "Le dataset doit avoir exactement 30 features"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])