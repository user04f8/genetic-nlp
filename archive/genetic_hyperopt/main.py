import argparse
import pandas as pd
from preprocess_data import load_data
from data_processing import preprocess_data, extract_features
from models import XGBoostClassifier, LogisticRegressionClassifier
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch
import os

RANDOM_STATE = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(RANDOM_STATE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split', type=float, default=1.0, help='Percentage of data to use')
    parser.add_argument('--balance_classes', action='store_true', help='Balance classes in training data')
    parser.add_argument('--hyperparameter_search', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--model', type=str, default='xgboost', help='Model to use: xgboost, logistic')
    args = parser.parse_args()

    train_df, test_df = load_data()

    if args.data_split < 1.0:
        train_df = train_df.sample(frac=args.data_split, random_state=RANDOM_STATE)

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE)

    X_train_raw, y_train = preprocess_data(train_df)
    X_val_raw, y_val = preprocess_data(val_df)
    X_test_raw = preprocess_data(test_df, training=False)

    X_train_features = extract_features(X_train_raw, fit=True)
    X_val_features = extract_features(X_val_raw, fit=False)
    X_test_features = extract_features(X_test_raw, fit=False)

    y_val = y_val.iloc[X_val_features.shape[0]]

    if args.balance_classes:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_features, y_train = sm.fit_resample(X_train_features, y_train)

    if args.hyperparameter_search:
        from sklearn.model_selection import GridSearchCV
        if args.model == 'xgboost':
            param_grid = {'max_depth': [6, 8, 10], 'n_estimators': [100, 200]}
            base_clf = XGBoostClassifier()
        elif args.model == 'logistic':
            param_grid = {'C': [0.1, 1, 10]}
            base_clf = LogisticRegressionClassifier()
        else:
            raise ValueError('Unknown model: {}'.format(args.model))

        clf = GridSearchCV(base_clf, param_grid, cv=3)
    else:
        if args.model == 'xgboost':
            clf = XGBoostClassifier()
        elif args.model == 'logistic':
            clf = LogisticRegressionClassifier()
        else:
            raise ValueError('Unknown model: {}'.format(args.model))

    clf.fit(X_train_features, y_train)
    y_val_pred = clf.predict(X_val_features)
    evaluate_model(y_val, y_val_pred)

    y_test_pred = clf.predict(X_test_features)
    test_df['Score'] = y_test_pred
    submission = test_df[['Id', 'Score']]
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
