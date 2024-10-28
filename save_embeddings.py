import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

from model_utils import (
    initialize_environment, load_trained_model, process_data, create_dataloader, generate_submission
)
from data_processor import DataProcessor, ReviewDataset, collate_fn
from dynamic_als import update_df_and_get_als
from preprocess_data import load_data
from fuzzy_ngram_matrix_factors_model import SentimentModel

initialize_environment(3)  # reproducability
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_ids = []
    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to device
            text = batch['text'].to(device)
            summary = batch['summary'].to(device)
            user = batch['user'].to(device)
            product = batch['product'].to(device)
            helpfulness_ratio = batch['helpfulness_ratio'].to(device)
            log_helpfulness_denominator = batch['log_helpfulness_denominator'].to(device)

            # Extract embeddings
            embeddings = model.get_embeddings(
                text, summary, user, product, helpfulness_ratio, log_helpfulness_denominator
            )
            all_embeddings.append(embeddings.cpu())

            # Collect labels and IDs
            if 'label' in batch:
                labels = batch['label'].cpu()
                all_labels.append(labels)
            if 'Id' in batch:
                ids = batch['Id']
                all_ids.extend(ids)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else None
    ids = all_ids if all_ids else None

    return embeddings.numpy(), labels.numpy() if labels is not None else None, ids

# from hparams.yaml
# extern_params:
#   als_factors: 10
#   als_iterations: 10
#   als_regularization: 0.1
#   cull_unknown_threshold: 2
cull_unknown_threshold = 2
user_product_embed_size = 10
als_iterations = 10
als_regularization = 0.1
batch_size = 512  # Adjust based on your GPU memory

# Load data
train_df, test_df = load_data()

# Initialize DataProcessor and set encoders
data_processor = DataProcessor(no_unknowns=False, unknown_threshold=cull_unknown_threshold)
data_processor.fit_encoders(train_df)

# Generate ALS embeddings and update DataFrame
user_embeddings, item_embeddings, num_users, num_products = update_df_and_get_als(
    train_df,
    data_processor,
    n_factors=user_product_embed_size,
    n_iterations=als_iterations,
    regularization=als_regularization,
    cache_dir='./cache/'
)

# Process train_df to include 'user_idx' and 'product_idx'
train_df = data_processor.process_reviews(train_df, is_training=True)

# Prepare the dataset and dataloaders for training data
dataset = ReviewDataset(train_df)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=8,
    pin_memory=True
)

# Process test_df
data_processor = DataProcessor(unknown_threshold=None)
# NOTE original unknown_threshold is from hparams.yaml
data_processor.load_encoders()

print('Running process_data()')
processed_test_df = process_data(test_df, data_processor, is_training=False)

print('Creating dataloader')
batch_size = 512
test_dataloader = create_dataloader(processed_test_df, batch_size, is_training=False)
# Load the trained model
BEST_CHECKPOINT_FILENAME = 'lightning_logs/heavy_regularization/version_1/checkpoints/epoch=40-step=11931.ckpt'
model = SentimentModel.load_from_checkpoint(BEST_CHECKPOINT_FILENAME)
model.to(device)

# Extract embeddings for the training set
print("Extracting embeddings for the training set...")
train_embeddings, train_labels, _ = extract_embeddings(model, train_loader, device)

# Save train embeddings and labels
np.savez_compressed('train_embeddings.npz', embeddings=train_embeddings, labels=train_labels)
print("Training embeddings saved to 'train_embeddings.npz'.")

# Extract embeddings for the test set
print("Extracting embeddings for the test set...")
test_embeddings, _, test_ids = extract_embeddings(model, test_dataloader, device)

# Save test embeddings and IDs
np.savez_compressed('test_embeddings.npz', embeddings=test_embeddings, ids=np.array(test_ids))
print("Test embeddings saved to 'test_embeddings.npz'.")
