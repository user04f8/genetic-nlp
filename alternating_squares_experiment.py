import numpy as np
import pandas as pd
import implicit
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from preprocess_data import load_data
from utils import seed_everything

# Set random seed for reproducibility
seed_everything(42)

# Load data
reviews_df, _ = load_data()

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


for n_factors in (2, 5, 10, 20, 50, 100):
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

    # Save embeddings to disk
    np.save(f'als_embeddings/user_embeddings_{n_factors}.npy', user_embeddings)
    np.save(f'als_embeddings/product_embeddings_{n_factors}.npy', item_embeddings)

    print("Embeddings saved successfully.")