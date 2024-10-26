import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
RANDOM_STATE = 42

class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=6, n_estimators=100, random_state=RANDOM_STATE):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                 max_depth=self.max_depth, n_estimators=self.n_estimators,
                                 random_state=self.random_state)
    
    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=1000, random_state=RANDOM_STATE):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
    
    def fit(self, X, y):
        self.clf.fit(X, y)
        return self
    
    def predict(self, X):
        return self.clf.predict(X)
