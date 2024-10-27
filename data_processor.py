import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from preprocess_data import load_data
from dynamic_als import update_df_and_get_als

class DataProcessor:
    def __init__(self, no_unknowns=False, unknown_threshold=1):
        self.user_encoder = None
        self.product_encoder = None
        self.num_users = None
        self.num_products = None
        self.unknown_user_idx = None
        self.unknown_product_idx = None
        self.no_unknowns = no_unknowns
        self.unknown_threshold = unknown_threshold  # Frequency threshold

    def fit_encoders(self, reviews_df):
        # Compute frequencies
        user_counts = reviews_df['UserId'].value_counts()
        product_counts = reviews_df['ProductId'].value_counts()

        # Identify low-frequency users/products
        self.low_freq_users = set(user_counts[user_counts <= self.unknown_threshold].index)
        self.low_freq_products = set(product_counts[product_counts <= self.unknown_threshold].index)

        # Map low-frequency users/products to 'unknown'
        reviews_df['UserId'] = reviews_df['UserId'].apply(
            lambda x: '<unknown_user>' if x in self.low_freq_users else x
        )
        reviews_df['ProductId'] = reviews_df['ProductId'].apply(
            lambda x: '<unknown_product>' if x in self.low_freq_products else x
        )

        # Fit LabelEncoders
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()

        self.user_encoder.fit(reviews_df['UserId'])
        self.product_encoder.fit(reviews_df['ProductId'])

        self.num_users = len(self.user_encoder.classes_)
        self.num_products = len(self.product_encoder.classes_)

        # Set unknown user/product indices
        if not self.no_unknowns:
            self.unknown_user_idx = self.user_encoder.transform(['<unknown_user>'])[0] \
                if '<unknown_user>' in self.user_encoder.classes_ else None
            self.unknown_product_idx = self.product_encoder.transform(['<unknown_product>'])[0] \
                if '<unknown_product>' in self.product_encoder.classes_ else None
        else:
            self.unknown_user_idx = None
            self.unknown_product_idx = None

        # Save the encoders
        joblib.dump(self.user_encoder, 'user_encoder.joblib')
        joblib.dump(self.product_encoder, 'product_encoder.joblib')

    def set_encoders(self, user_encoder, product_encoder):
        self.user_encoder = user_encoder
        self.product_encoder = product_encoder
        self.num_users = len(self.user_encoder.classes_) + 1  # Include unknown
        self.num_products = len(self.product_encoder.classes_) + 1  # Include unknown
        self.unknown_user_idx = self.num_users - 1
        self.unknown_product_idx = self.num_products - 1

    def load_encoders(self):
        self.user_encoder = joblib.load('user_encoder.joblib')
        self.product_encoder = joblib.load('product_encoder.joblib')
        self.num_users = len(self.user_encoder.classes_) + 1  # Include unknown
        self.num_products = len(self.product_encoder.classes_) + 1  # Include unknown
        self.unknown_user_idx = self.num_users - 1
        self.unknown_product_idx = self.num_products - 1

    def transform_with_unknown(self, encoder, labels, unknown_value):
        label_to_index = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        return [label_to_index.get(label, unknown_value) for label in labels]

    def process_reviews(self, df, is_training=True):
        if is_training:
            # Map low-frequency users/products in training data
            df['UserId'] = df['UserId'].apply(
                lambda x: '<unknown_user>' if x in self.low_freq_users else x
            )
            df['ProductId'] = df['ProductId'].apply(
                lambda x: '<unknown_product>' if x in self.low_freq_products else x
            )

            df['user_idx'] = self.user_encoder.transform(df['UserId'])
            df['product_idx'] = self.product_encoder.transform(df['ProductId'])
        else:
            # Map unknown users/products
            if self.no_unknowns:
                # Raise error if unknown users/products are encountered
                unknown_users = set(df['UserId']) - set(self.user_encoder.classes_)
                unknown_products = set(df['ProductId']) - set(self.product_encoder.classes_)

                if unknown_users or unknown_products:
                    raise ValueError(f"Unknown users: {unknown_users}, Unknown products: {unknown_products}")

                df['user_idx'] = self.user_encoder.transform(df['UserId'])
                df['product_idx'] = self.product_encoder.transform(df['ProductId'])
            else:
                # Map unknown users/products to '<unknown_user>'/'<unknown_product>'
                df['UserId'] = df['UserId'].apply(
                    lambda x: x if x in self.user_encoder.classes_ else '<unknown_user>'
                )
                df['ProductId'] = df['ProductId'].apply(
                    lambda x: x if x in self.product_encoder.classes_ else '<unknown_product>'
                )

                df['user_idx'] = self.user_encoder.transform(df['UserId'])
                df['product_idx'] = self.product_encoder.transform(df['ProductId'])
        return df

    def get_num_users_products(self):
        return self.num_users, self.num_products

    def get_encoders(self):
        return self.user_encoder, self.product_encoder
    # datasets.py

import torch
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.text_tokens = df['Text_glove_tokens_np'].values
        self.summary_tokens = df['Summary_glove_tokens_np'].values
        self.user_idx = df['user_idx'].values
        self.product_idx = df['product_idx'].values
        self.labels = df['Score'].values.astype(np.int64) - 1  # Ratings from 0 to 4
        self.helpfulness_ratio = df['HelpfulnessRatio'].values
        self.log_helpfulness_denominator = df['LogHelpfulnessDenominator'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'text': torch.tensor(self.text_tokens[idx], dtype=torch.long),
            'summary': torch.tensor(self.summary_tokens[idx], dtype=torch.long),
            'user': torch.tensor(self.user_idx[idx], dtype=torch.long),
            'product': torch.tensor(self.product_idx[idx], dtype=torch.long),
            'helpfulness_ratio': torch.tensor(self.helpfulness_ratio[idx], dtype=torch.float),
            'log_helpfulness_denominator': torch.tensor(self.log_helpfulness_denominator[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

class TestReviewDataset(Dataset):
    def __init__(self, df):
        self.text_tokens = df['Text_glove_tokens_np'].values
        self.summary_tokens = df['Summary_glove_tokens_np'].values
        self.user_idx = df['user_idx'].values
        self.product_idx = df['product_idx'].values
        self.ids = df['Id'].values  # Add `Id` to return for submission
        self.helpfulness_ratio = df['HelpfulnessRatio'].values
        self.log_helpfulness_denominator = df['LogHelpfulnessDenominator'].values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            'text': torch.tensor(self.text_tokens[idx], dtype=torch.long),
            'summary': torch.tensor(self.summary_tokens[idx], dtype=torch.long),
            'user': torch.tensor(self.user_idx[idx], dtype=torch.long),
            'product': torch.tensor(self.product_idx[idx], dtype=torch.long),
            'Id': self.ids[idx],  # Return the Id for the final submission
            'helpfulness_ratio': torch.tensor(self.helpfulness_ratio[idx], dtype=torch.float),
            'log_helpfulness_denominator': torch.tensor(self.log_helpfulness_denominator[idx], dtype=torch.float)
        }
        return item

from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    summaries = [item['summary'] for item in batch]
    users = torch.stack([item['user'] for item in batch])
    products = torch.stack([item['product'] for item in batch])
    helpfulness_ratios = torch.tensor([item['helpfulness_ratio'] for item in batch], dtype=torch.float)
    log_helpfulness_denominators = torch.tensor([item['log_helpfulness_denominator'] for item in batch], dtype=torch.float)

    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    summaries_padded = pad_sequence(summaries, batch_first=True, padding_value=0)

    batch_dict = {
        'text': texts_padded,
        'summary': summaries_padded,
        'user': users,
        'product': products,
        'helpfulness_ratio': helpfulness_ratios,
        'log_helpfulness_denominator': log_helpfulness_denominators
    }

    # Include 'label' if present
    if 'label' in batch[0]:
        labels = torch.stack([item['label'] for item in batch])
        batch_dict['label'] = labels

    # Include 'Id' if present
    if 'Id' in batch[0]:
        ids = [item['Id'] for item in batch]
        batch_dict['Id'] = ids

    return batch_dict