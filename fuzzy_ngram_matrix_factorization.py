# train_model.py

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from data_processor import DataProcessor, ReviewDataset, collate_fn
from fuzzy_ngram_matrix_factors_model import SentimentModel
from preprocess_data import load_data
from dynamic_als import update_df_and_get_als
from utils import seed_everything

FINAL_CHECKPOINT_FILENAME = "checkpoints/fuzzy_ngram_model.ckpt"

# Reproducibility
seed_everything(3)
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with an optional checkpoint path")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to the checkpoint file")
    args = parser.parse_args()

    # Load the data
    reviews_df, test_df = load_data()

    # Initialize DataProcessor and fit encoders
    data_processor = DataProcessor()
    data_processor.fit_encoders(reviews_df)
    reviews_df = data_processor.process_reviews(reviews_df, is_training=True)
    num_users, num_products = data_processor.get_num_users_products()

    # Prepare the dataset and dataloaders
    dataset = ReviewDataset(reviews_df)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 512  # Adjust based on GPU memory
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    # Initialize the model
    model = SentimentModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=300,  # GloVe embedding size
        n_filters=100,
        filter_sizes=[3, 4, 5, 7],  # fuzzy n-gram sizes
        user_emb_dim=50,
        product_emb_dim=50,
        output_dim=5,  # Ratings from 0 to 4
        dropout=0.4,
        # user_embedding_weights=torch.tensor(user_embeddings, dtype=torch.float32),
        # product_embedding_weights=torch.tensor(item_embeddings, dtype=torch.float32)
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False,
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
    )

    # Trainer
    trainer = Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices='auto',
        strategy='ddp',
        callbacks=[
            # early_stopping,
            lr_monitor,
            checkpoint_callback
        ],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
    trainer.save_checkpoint(FINAL_CHECKPOINT_FILENAME)
    print(f'Training finished! Final checkpoint saved at {FINAL_CHECKPOINT_FILENAME}')
