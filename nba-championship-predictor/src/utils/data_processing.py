def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Example preprocessing steps
    data = data.dropna()  # Remove missing values
    # Add more preprocessing steps as needed
    return data

def split_data(data, test_size=0.2):
    from sklearn.model_selection import train_test_split
    return train_test_split(data, test_size=test_size, random_state=42)