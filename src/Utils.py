import pandas as pd
import numpy as np

def pdays_transform(x):
    if x == -1:
        return 0
    elif x < 100:
        return 1
    elif x < 200:
        return 2
    elif x < 400:
        return 3
    else:
        return 4

def process_pdays(X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()
    X_out['wasContacted'] = X_out['pdays'].map(lambda x: 0 if x == -1 else 1)
    X_out['pdaysTransformed'] = X_out['pdays'].map(pdays_transform)
    return X_out

def process_poutcome(X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()
    X_out['DepositPropensity'] = X_out[['poutcome', 'pdays']].apply(calculate_propensity, axis=1)
    return X_out

def process_skewness(X: pd.DataFrame) -> pd.DataFrame:
    X_out = X.copy()
    if 'duration' in X.columns:
        X_out['duration'] = np.log1p(X_out['duration'])
        
    if 'balance' in X.columns:
        X_out['balance'] = np.sign(X_out['balance']) * np.log1p(np.abs(X_out['balance']))

    return X_out

def calculate_propensity(row):
    if row['poutcome'] != 'success':
        return 0
    elif row['pdays'] < 0:
        return 0
    else:
        return 1/(1 + row['pdays'])

def process_job(X):
    X_out = X.copy()
    X_out['job'] = X_out['job'].fillna('unknown')
    return X_out

def drop_columns(X):
    return X.drop(['pdays'], axis=1, errors='ignore') 
