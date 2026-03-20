def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path, index_col=False, compression='gzip')

def save_model(model, file_path):
    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    import joblib
    return joblib.load(file_path)

def print_shape(data):
    print(f"Data shape: {data.shape}")

def print_best_params(model):
    print(model.best_estimator_.get_params())