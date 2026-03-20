import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_and_clean(csv_path: str):
    df = pd.read_csv(csv_path)
    if df.isnull().sum().sum() > 0:
        df.dropna(inplace=True)

    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df.drop(columns=['Amount', 'Time'], inplace=True)

    return df, scaler


def split_data(df, seed=42):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=seed, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, seed=42):
    smote = SMOTE(random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


if __name__ == '__main__':
    df, scaler = load_and_clean('creditcard.csv')
    print('Shape apres nettoyage :', df.shape)
    print('Fraudes :', df['Class'].sum(), '/', len(df))

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print('Train:', X_train.shape, '| Val:', X_val.shape, '| Test:', X_test.shape)

    X_sm, y_sm = apply_smote(X_train, y_train)
    print('Apres SMOTE - fraudes:', y_sm.sum(), '/ total:', len(y_sm))