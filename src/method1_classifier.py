import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

def load_data(data_dir):
    X = pd.DataFrame()
    y = pd.DataFrame()
    all_features = []
    all_classes = []
    
    files = os.listdir(data_dir)
    for file in files:
        if os.path.isfile(os.path.join(data_dir, file)):
            print(file)
            temp = pd.read_csv(os.path.join(data_dir, file), index_col=False, compression='gzip')
            all_features.append(temp.drop('class', axis=1))
            all_classes.append(temp['class'])
    
    X = pd.concat(all_features)
    y = pd.concat(all_classes)
    return X, y

def setup_training_data(X, y, test_size=0.20):
    return train_test_split(X, y, test_size=test_size, stratify=y)

def tune_hyperparameters(X_train, y_train):
    param_dist = {
        "max_depth": [1, 5, 10, 15, 20, 25, 30],
        "n_estimators": [5, 10, 50, 100, 110, 120, 200],
        "max_features": ['sqrt', 'log2'],
        "min_samples_split": [2, 4, 8],
        "bootstrap": [True, False],
        "ccp_alpha": [0.0, 0.01, 0.02]
    }
    
    clf = GridSearchCV(RandomForestClassifier(random_state=0), param_dist, cv=5)
    return clf

def train_model(clf, X_train, y_train):
    start = time.perf_counter()
    print('Started training...')
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('Training completed in:', end - start)
    return clf

def save_model(clf, filename):
    joblib.dump(clf, filename)

def load_model(filename):
    return joblib.load(filename)