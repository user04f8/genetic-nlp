import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

text_vectorizer = None
user_encoder = None
product_encoder = None
user_onehot = None
product_onehot = None
svd_user = None
svd_product = None
RANDOM_STATE = 42

def preprocess_data(df, training=True):
    df['Text'] = df['Text'].fillna('')
    df['Summary'] = df['Summary'].fillna('')
    df['UserId'] = df['UserId'].fillna('unknown_user')
    df['ProductId'] = df['ProductId'].fillna('unknown_product')
    if training:
        X = df[['Text', 'Summary', 'UserId', 'ProductId']]
        y = df['Score']
        return X, y
    else:
        X = df[['Text', 'Summary', 'UserId', 'ProductId']]
        return X

def extract_features(X, fit=True):
    global text_vectorizer, user_encoder, product_encoder
    global user_onehot, product_onehot, svd_user, svd_product

    if fit:
        text_vectorizer = TfidfVectorizer(max_features=5000)
        X_text = text_vectorizer.fit_transform(X['Text'] + ' ' + X['Summary'])

        user_encoder = LabelEncoder()
        X_user = user_encoder.fit_transform(X['UserId'])
        product_encoder = LabelEncoder()
        X_product = product_encoder.fit_transform(X['ProductId'])

        user_onehot = OneHotEncoder(handle_unknown='ignore')
        X_user_onehot = user_onehot.fit_transform(X_user.reshape(-1, 1))
        product_onehot = OneHotEncoder(handle_unknown='ignore')
        X_product_onehot = product_onehot.fit_transform(X_product.reshape(-1, 1))

        svd_user = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
        X_user_embed = svd_user.fit_transform(X_user_onehot)
        svd_product = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
        X_product_embed = svd_product.fit_transform(X_product_onehot)
    else:
        unseen_indices = []
        X_text = text_vectorizer.transform(X['Text'] + ' ' + X['Summary'])

        # Check if UserId exists in the encoder
        try:
            X_user = user_encoder.transform(X['UserId'])
        except ValueError as e:
            unseen_indices.extend(np.where(np.isin(X['UserId'], e.args[0][1:-1].split(',')))[0])
            X_user = X['UserId'].apply(lambda x: -1 if x in e.args[0][1:-1].split(',') else user_encoder.transform([x])[0])

        # Check if ProductId exists in the encoder
        try:
            X_product = product_encoder.transform(X['ProductId'])
        except ValueError as e:
            unseen_indices.extend(np.where(np.isin(X['ProductId'], e.args[0][1:-1].split(',')))[0])
            X_product = X['ProductId'].apply(lambda x: -1 if x in e.args[0][1:-1].split(',') else product_encoder.transform([x])[0])

        # Remove unseen indices
        unseen_indices = list(set(unseen_indices))
        if unseen_indices:
            X_text = np.delete(X_text.toarray(), unseen_indices, axis=0)
            X_user = np.delete(X_user, unseen_indices, axis=0)
            X_product = np.delete(X_product, unseen_indices, axis=0)

        X_user_onehot = user_onehot.transform(X_user.reshape(-1, 1))
        X_user_embed = svd_user.transform(X_user_onehot)
        X_product_onehot = product_onehot.transform(X_product.reshape(-1, 1))
        X_product_embed = svd_product.transform(X_product_onehot)

    X_features = hstack([X_text, X_user_embed, X_product_embed])

    return X_features