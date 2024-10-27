import os
import pickle
import joblib
import numpy as np
import pandas as pd
import implicit
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder

from preprocess_data import load_data

def update_df_and_get_als(reviews_df, data_processor, n_factors=100, n_iterations=10, cache_dir='./cache/'):
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
        data_processor.user_encoder = joblib.load(user_encoder_path)
        data_processor.product_encoder = joblib.load(product_encoder_path)
    else:
        print('Computing ALS outputs...')

        # Process reviews to get encoded indices
        reviews_df = data_processor.process_reviews(reviews_df, is_training=True)
        user_indices = reviews_df['user_idx'].values
        product_indices = reviews_df['product_idx'].values
        ratings = reviews_df['Score'].values.astype(np.float32)

        num_users = data_processor.num_users
        num_products = data_processor.num_products

        user_embeddings, item_embeddings = compute_als(
            user_indices, product_indices, ratings, num_users, num_products,
            n_factors, n_iterations
        )

        # Save outputs to cache
        np.save(user_emb_path, user_embeddings)
        np.save(item_emb_path, item_embeddings)
        with open(num_users_path, 'wb') as f:
            pickle.dump(num_users, f)
        with open(num_products_path, 'wb') as f:
            pickle.dump(num_products, f)
        joblib.dump(data_processor.user_encoder, user_encoder_path)
        joblib.dump(data_processor.product_encoder, product_encoder_path)

    return user_embeddings, item_embeddings, num_users, num_products

def compute_als(user_indices, product_indices, ratings, num_users, num_products, n_factors, n_iterations=10, regularization=0.4):
    print(f"Number of users: {num_users}")
    print(f"Number of products: {num_products}")

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
        regularization=regularization,
        iterations=n_iterations,
        use_gpu=False  # Disable GPU to avoid CuPy
    )

    # Fit the model (implicit expects item-user matrix)
    als_model.fit(interaction_matrix_csr.T)

    # Extract embeddings
    user_embeddings = als_model.user_factors  # Shape: (num_users, factors)
    item_embeddings = als_model.item_factors  # Shape: (num_products, factors)

    # Swap embeddings due to library implementation issue
    user_embeddings, item_embeddings = item_embeddings, user_embeddings

    return user_embeddings, item_embeddings