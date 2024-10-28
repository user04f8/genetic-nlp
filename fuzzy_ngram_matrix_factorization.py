import argparse
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data_processor import DataProcessor, ReviewDataset, collate_fn
from fuzzy_ngram_matrix_factors_model import SentimentModel
from preprocess_data import load_data
from dynamic_als import update_df_and_get_als
from utils import seed_everything

FINAL_CHECKPOINT_FILENAME = "checkpoints/fuzzy_ngram_model.ckpt"

# Reproducibility
RANDOM_STATE = 4  # NOTE: before this was added RANDOM_STATE = 3 generally
seed_everything(RANDOM_STATE)
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model with an optional checkpoint path")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to the checkpoint file")
    args = parser.parse_args()

    cull_unknown_threshold = 0
    user_product_embed_size = 50
    als_iterations = 1
    als_regularization = 0.1

    # Load the data
    reviews_df, _ = load_data()

    # Initialize DataProcessor and set encoders
    data_processor = DataProcessor(no_unknowns=False, unknown_threshold=cull_unknown_threshold)
    data_processor.fit_encoders(reviews_df)

    # Generate ALS embeddings and update DataFrame
    user_embeddings, item_embeddings, num_users, num_products = update_df_and_get_als(
        reviews_df, data_processor, n_factors=user_product_embed_size, n_iterations=als_iterations, regularization=als_regularization, 
        cache_dir='./cache/'
    )

    # Process reviews_df to include 'user_idx' and 'product_idx'
    reviews_df = data_processor.process_reviews(reviews_df, is_training=True)

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

    model = SentimentModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=300,  # GloVe embedding size
        n_filters=100,
        filter_sizes=[3, 4, 5, 7],  # fuzzy n-gram sizes
        user_emb_dim=user_product_embed_size,
        product_emb_dim=user_product_embed_size,
        output_dim=5,  # Ratings from 0 to 4
        dropout=0.4,
        user_embedding_weights=torch.tensor(user_embeddings, dtype=torch.float32),
        product_embedding_weights=torch.tensor(item_embeddings, dtype=torch.float32),
        blend_factor=0.0,  # Adjust to taste :P
        unfreeze_epoch=0,
        weight_decay=None,
        extern_params={
            'als_factors': user_product_embed_size,
            'als_iterations': als_iterations,
            'als_regularization': als_regularization,
            'cull_unknown_threshold': cull_unknown_threshold,
            'random_state': RANDOM_STATE
        }
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

    logger = TensorBoardLogger("lightning_logs", name="recreate_v65")

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
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
    trainer.save_checkpoint(FINAL_CHECKPOINT_FILENAME)
    print(f'Training finished! Final checkpoint saved at {FINAL_CHECKPOINT_FILENAME}')