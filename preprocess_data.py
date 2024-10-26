import os
import zipfile
import numpy as np
import pandas as pd
import tiktoken
from multiprocessing import Pool, cpu_count

data_dir = './data/'
train_parquet_path = os.path.join(data_dir, 'train_data.parquet')
test_parquet_path = os.path.join(data_dir, 'test_data.parquet')

def tokenize_column(text):
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer.encode(text) if pd.notnull(text) else []

def parallel_tokenize(column_data):
    with Pool(cpu_count()) as pool:
        tokens = pool.map(tokenize_column, column_data)
    return tokens

def preprocess():
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    # Check if train_data.parquet and test_data.parquet already exist
    if os.path.exists(train_parquet_path) and os.path.exists(test_parquet_path):
        print('Parquet files already exist. Skipping preprocessing.')
        return

    if os.path.exists(train_path) and os.path.exists(test_path):
        print('train.csv and test.csv already exist. Skipping unzipping.')
    else:
        zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
        assert len(zip_files) == 1, "There should be exactly one .zip file in ./data/ directory."
        zip_path = os.path.join(data_dir, zip_files[0])

        print('Unzipping...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    print('Reading CSVs...')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print('Handling missing values in Score...')
    # Split the rows in train_df where 'Score' is null
    null_score_train_df = train_df[train_df['Score'].isnull()]
    
    # Filter the test_df to only include rows that match the 'Id' in the null_score_train_df
    test_df = null_score_train_df[null_score_train_df['Id'].isin(test_df['Id'])].copy()

    # Remove the 'Score' column from test_df (since it's null and not needed in the test set)
    test_df = test_df.drop(columns=['Score'])

    # Remove rows with null 'Score' from train_df
    train_df = train_df[train_df['Score'].notnull()]

    print('Tokenizing in parallel...')
    train_df['Summary_tokens'] = parallel_tokenize(train_df['Summary'])
    train_df['Text_tokens'] = parallel_tokenize(train_df['Text'])

    test_df['Summary_tokens'] = parallel_tokenize(test_df['Summary'])
    test_df['Text_tokens'] = parallel_tokenize(test_df['Text'])

    print('Saving dataset as Parquet...')
    train_df.to_parquet(train_parquet_path, compression='snappy')
    test_df.to_parquet(test_parquet_path, compression='snappy')

    print("Data successfully compressed and saved as Parquet.")

def load_preprocessed_data():
    if os.path.exists(train_parquet_path) and os.path.exists(test_parquet_path):
        print('Loading preprocessed Parquet files...', end='')
        train_df = pd.read_parquet(train_parquet_path)
        print('...', end='')
        test_df = pd.read_parquet(test_parquet_path)
        print(' Done!')
        return train_df, test_df
    else:
        raise FileNotFoundError("Preprocessed Parquet files not found. Run the preprocess() function first.")

if __name__ == '__main__':
    preprocess()
