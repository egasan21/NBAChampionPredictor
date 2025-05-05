import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    if 'champion' in data.columns:
        data['champion'] = data['champion'].fillna(0)
    data = data.dropna().copy()

    data['win_percentage'] = data['w'] / (data['w'] + data['l'])

    data['o_rtg_original'] = data['o_rtg']
    data['d_rtg_original'] = data['d_rtg']
    data['win_percentage_original'] = data['win_percentage']
    data['mov_original'] = data['mov']
    data['pace_original'] = data['pace']
    data['ts_percent_original'] = data['ts_percent']
    data['f_tr_original'] = data['f_tr']
    data['x3p_ar_original'] = data['x3p_ar']
    data['age_original'] = data['age']
    data['srs_original'] = data['srs']

    # Normalize the features
    scaler = StandardScaler()
    numerical_features = ['o_rtg', 'd_rtg', 'win_percentage', 'mov', 'pace', 'ts_percent', 'f_tr', 'x3p_ar', 'age', 'srs']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

def split_data(data):
    X = data[['o_rtg', 'd_rtg', 'win_percentage', 'mov', 'pace', 'ts_percent', 'f_tr', 'x3p_ar', 'age', 'srs']]
    y = data['champion']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test