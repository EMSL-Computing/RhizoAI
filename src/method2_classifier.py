import cv2 as cv
import os
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_image_data(img_path, count_max=204):
    files = os.listdir(img_path)
    max_decay = max_dead = max_live = 0
    X, y = [], []
    
    for filename in files:
        category = ''
        if 'decay' in filename and max_decay < count_max:
            category = 'decay'
            max_decay += 1
        elif 'dead' in filename and max_dead < count_max:
            category = 'dead'
            max_dead += 1
        elif 'live' in filename and max_live < count_max:
            category = 'live'
            max_live += 1
        else:
            continue
        
        image = np.load(os.path.join(img_path, filename))
        image_res = cv.resize(image, (128, 128))
        hog_features = extract_hog_features(image_res)
        X.append(hog_features)
        y.append(category)
    
    return np.array(X), np.array(y)

def extract_hog_features(image):
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features

def train_random_forest(X_train, y_train):
    param_dist = {
        "max_depth": [1, 5, 10, 15, 20, 25, 30],
        "n_estimators": [5, 10, 50, 100, 110, 120, 200],
        "max_features": ['sqrt', 'log2'],
        "min_samples_split": [2, 4, 8],
        "bootstrap": [True, False],
        "ccp_alpha": [0.0, 0.01, 0.02]
    }
    
    clf = GridSearchCV(RandomForestClassifier(random_state=0, class_weight='balanced'), param_dist, cv=5)
    clf.fit(X_train, y_train)
    
    return clf

def save_model(clf, filename):
    joblib.dump(clf, filename)

def load_model(filename):
    return joblib.load(filename)

def main(img_path, model_filename):
    X, y = load_image_data(img_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)
    
    clf = train_random_forest(X_train, y_train)
    print("Best estimator found by grid search:")
    print(clf.best_score_)
    print(clf.best_estimator_.score(X_train, y_train))
    print(clf.best_estimator_.score(X_test, y_test))
    
    save_model(clf, model_filename)

if __name__ == "__main__":
    img_path = 'mar+may_2024_threshold/'
    model_filename = 'OCT_RF_classifier_method_2_all_data_threshold.pkl'
    main(img_path, model_filename)