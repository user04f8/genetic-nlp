import os
import zipfile
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from multiprocessing import Pool, cpu_count

data_dir = './data/'
train_parquet_path = os.path.join(data_dir, 'train_data.parquet')
test_parquet_path = os.path.join(data_dir, 'test_data.parquet')

# Load GloVe embeddings
glove_path = 'glove.840B.300d.word2vec.bin'
glove_model = None
glove_vocab = None
glove_embedding_matrix = None

def load_glove():
    global glove_model, glove_vocab
    if glove_model is None:
        print('Loading GloVe... ', end="")
        glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=True, limit=1500000)
        glove_vocab = {word: idx for idx, word in enumerate(glove_model.index_to_key)}
        print('Done!')

def get_glove_vocab():
    global glove_vocab
    load_glove()
    return glove_vocab

def get_glove(freeze=True):
    global glove_embedding_matrix
    import torch
    load_glove()

    if glove_embedding_matrix is None:
        vocab_size = len(glove_model.key_to_index)  # Number of words in GloVe
        embedding_dim = glove_model.vector_size  # 300 in the case of GloVe 300d

        # Create embedding matrix (vocab_size x embedding_dim)
        embedding_matrix = torch.zeros((vocab_size, embedding_dim))

        for i, word in enumerate(glove_model.key_to_index):
            embedding_matrix[i] = torch.tensor(glove_model[word])

    # Use the embedding matrix with PyTorch nn.Embedding
    return torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze)

# Tokenization function for GloVe
def glove_tokenize(text):
    if pd.notnull(text):
        tokens = text.lower().split()  # Simple whitespace-based tokenization
        token_indices = [glove_vocab.get(token, glove_vocab.get('<unk>', 0)) for token in tokens]
        return token_indices
    return []  # Return empty list for None or NaN values

def parallel_glove_tokenize(column_data):
    with Pool(cpu_count()) as pool:
        tokens = pool.map(glove_tokenize, column_data)
    return tokens

def preprocess():
    load_glove()

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
    train_df['Summary_glove_tokens'] = parallel_glove_tokenize(train_df['Summary'])
    train_df['Text_glove_tokens'] = parallel_glove_tokenize(train_df['Text'])

    test_df['Summary_glove_tokens'] = parallel_glove_tokenize(test_df['Summary'])
    test_df['Text_glove_tokens'] = parallel_glove_tokenize(test_df['Text'])

    # Convert token lists to numpy arrays for efficient storage
    train_df['Summary_glove_tokens_np'] = train_df['Summary_glove_tokens'].apply(lambda x: np.array(x, dtype=np.int32))
    train_df['Text_glove_tokens_np'] = train_df['Text_glove_tokens'].apply(lambda x: np.array(x, dtype=np.int32))

    test_df['Summary_glove_tokens_np'] = test_df['Summary_glove_tokens'].apply(lambda x: np.array(x, dtype=np.int32))
    test_df['Text_glove_tokens_np'] = test_df['Text_glove_tokens'].apply(lambda x: np.array(x, dtype=np.int32))

    print('Determining helpfulness ratios...')
    train_df['HelpfulnessRatio'] = np.minimum(np.where(
        train_df['HelpfulnessDenominator'] > 0,
        train_df['HelpfulnessNumerator'] / train_df['HelpfulnessDenominator'],
        0
    ), 1)
    test_df['HelpfulnessRatio'] = np.minimum(np.where(
        test_df['HelpfulnessDenominator'] > 0,
        test_df['HelpfulnessNumerator'] / test_df['HelpfulnessDenominator'],
        0
    ), 1)

    train_df['LogHelpfulnessDenominator'] = np.log(train_df['HelpfulnessDenominator'] + 1)
    test_df['LogHelpfulnessDenominator'] = np.log(test_df['HelpfulnessDenominator'] + 1)

    print('Saving dataset as Parquet...')
    train_df.to_parquet(train_parquet_path, compression='snappy')
    test_df.to_parquet(test_parquet_path, compression='snappy')

    print("Data successfully compressed and saved as Parquet.")

def load_data():
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
