import os
import pickle
import numpy as np
import pandas as pd
import implicit
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder

from preprocess_data import load_data

def update_df_and_get_als(reviews_df, n_factors=100, cache_dir='./cache/'):
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    user_emb_path = os.path.join(cache_dir, f'user_embeddings_{n_factors}.npy')
    item_emb_path = os.path.join(cache_dir, f'item_embeddings_{n_factors}.npy')
    num_users_path = os.path.join(cache_dir, 'num_users.pkl')
    num_products_path = os.path.join(cache_dir, 'num_products.pkl')
    user_encoder_path = os.path.join(cache_dir, 'user_encoder.pkl')
    product_encoder_path = os.path.join(cache_dir, 'product_encoder.pkl')

    # Check if cached files exist
    if (os.path.exists(user_emb_path) and os.path.exists(item_emb_path) and
        os.path.exists(num_users_path) and os.path.exists(num_products_path) and
        os.path.exists(user_encoder_path) and os.path.exists(product_encoder_path)):
        print('Loading cached ALS outputs...')
        user_embeddings = np.load(user_emb_path)
        item_embeddings = np.load(item_emb_path)
        with open(num_users_path, 'rb') as f:
            num_users = pickle.load(f)
        with open(num_products_path, 'rb') as f:
            num_products = pickle.load(f)
        with open(user_encoder_path, 'rb') as f:
            user_encoder = pickle.load(f)
        with open(product_encoder_path, 'rb') as f:
            product_encoder = pickle.load(f)

        # Use the encoders to transform UserId and ProductId
        reviews_df['user_idx'] = user_encoder.transform(reviews_df['UserId'])
        reviews_df['product_idx'] = product_encoder.transform(reviews_df['ProductId'])
    else:
        print('Computing ALS outputs...')
        user_embeddings, item_embeddings, num_users, num_products, user_encoder, product_encoder = compute_als(reviews_df, n_factors)

        # Save outputs to cache
        np.save(user_emb_path, user_embeddings)
        np.save(item_emb_path, item_embeddings)
        with open(num_users_path, 'wb') as f:
            pickle.dump(num_users, f)
        with open(num_products_path, 'wb') as f:
            pickle.dump(num_products, f)
        with open(user_encoder_path, 'wb') as f:
            pickle.dump(user_encoder, f)
        with open(product_encoder_path, 'wb') as f:
            pickle.dump(product_encoder, f)

    return user_embeddings, item_embeddings, num_users, num_products, user_encoder, product_encoder

def compute_als(reviews_df, n_factors):
    # Map UserId and ProductId to integer indices
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    reviews_df['user_idx'] = user_encoder.fit_transform(reviews_df['UserId'])
    reviews_df['product_idx'] = product_encoder.fit_transform(reviews_df['ProductId'])

    num_users = len(user_encoder.classes_)
    num_products = len(product_encoder.classes_)

    print(f"Number of users: {num_users}")
    print(f"Number of products: {num_products}")

    # Create the interaction matrix
    user_indices = reviews_df['user_idx'].values
    product_indices = reviews_df['product_idx'].values
    ratings = reviews_df['Score'].values.astype(np.float32)

    # Build the sparse interaction matrix in COO format
    interaction_matrix = coo_matrix(
        (ratings, (user_indices, product_indices)),
        shape=(num_users, num_products),
        dtype=np.float32
    )

    # Convert to CSR format for efficient operations
    interaction_matrix_csr = interaction_matrix.tocsr()

    # Initialize the ALS model
    als_model = implicit.als.AlternatingLeastSquares(
        factors=n_factors,
        regularization=0.1,
        iterations=500,
        use_gpu=False  # Disable GPU to avoid CuPy
    )

    # Fit the model (implicit expects item-user matrix)
    als_model.fit(interaction_matrix_csr.T)

    # Extract embeddings
    user_embeddings = als_model.user_factors  # Shape: (num_users, factors)
    item_embeddings = als_model.item_factors  # Shape: (num_products, factors)

    user_embeddings, item_embeddings = item_embeddings, user_embeddings  # swap due to library implementation issue
    # NOTE: this was a fun 2-hour debugging adventure. Something with implicit's handling of the csr matrix above is a bit interesting...

    return user_embeddings, item_embeddings, num_users, num_products, user_encoder, product_encoder

