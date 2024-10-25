import os
import zipfile
import numpy as np
import pandas as pd
import tiktoken
from multiprocessing import Pool, cpu_count

data_dir = './data/'
train_parquet_path = os.path.join(data_dir, 'train_data.parquet')
test_parquet_path = os.path.join(data_dir, 'test_data.parquet')

# Move the tokenize_column function outside preprocess
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

    # Check if train.csv and test.csv already exist
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

    print('Tokenizing in parallel...')
    train_df['Summary_tokens'] = parallel_tokenize(train_df['Summary'])
    train_df['Text_tokens'] = parallel_tokenize(train_df['Text'])

    print('Handling missing values in Score...')
    train_df['Score'] = train_df['Score'].fillna(0).astype(int)
    test_df['Score'] = np.nan

    print('Saving dataset as Parquet...')
    train_df.to_parquet(train_parquet_path, compression='snappy')
    test_df.to_parquet(test_parquet_path, compression='snappy')

    print("Data successfully compressed and saved as Parquet.")

def load_preprocessed_data():
    if os.path.exists(train_parquet_path) and os.path.exists(test_parquet_path):
        print('Loading preprocessed Parquet files...')
        train_df = pd.read_parquet(train_parquet_path)
        test_df = pd.read_parquet(test_parquet_path)
        return train_df, test_df
    else:
        raise FileNotFoundError("Preprocessed Parquet files not found. Run the preprocess() function first.")

if __name__ == '__main__':
    preprocess()
