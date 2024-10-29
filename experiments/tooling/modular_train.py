import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import json

from data_processor import DataProcessor, ReviewDataset, collate_fn
from fuzzy_ngram_matrix_factors_model import SentimentModel
from preprocess_data import load_data
from dynamic_als import update_df_and_get_als

FINAL_CHECKPOINT_FILENAME = "checkpoints/fuzzy_ngram_model.ckpt"

# Reproducibility
RANDOM_STATE = 4
seed_everything(RANDOM_STATE)
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    reviews_df, _ = load_data()
    
    with open('manual_tuning.json', 'r') as f:
        best_trial = json.load(f)
    best_params = best_trial['params']
    
    # Map the filter_sizes_key back to actual filter_sizes
    FILTER_SIZES_OPTIONS = {
        'small': [2, 3, 4],
        'medium': [3, 4, 5],
        'large': [3, 5, 7],
        'all': [2, 3, 4, 5, 7]
    }
    filter_sizes = FILTER_SIZES_OPTIONS[best_params['filter_sizes']]
    
    # Initialize DataProcessor and set encoders
    data_processor = DataProcessor(no_unknowns=False, unknown_threshold=best_params.get('cull_unknown_threshold', 0))
    data_processor.fit_encoders(reviews_df)
    
    # Generate ALS embeddings and update DataFrame
    user_embeddings, item_embeddings, num_users, num_products = update_df_and_get_als(
        reviews_df, data_processor, n_factors=best_params['user_product_embed_size'], 
        n_iterations=best_params['als_iterations'], 
        regularization=best_params['als_regularization'], cache_dir='./cache/'
    )
    
    # Process reviews_df to include 'user_idx' and 'product_idx'
    processed_reviews_df = data_processor.process_reviews(reviews_df, is_training=True)
    
    # Prepare the dataset and dataloaders
    dataset = ReviewDataset(processed_reviews_df)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = 512  # Adjust based on GPU memory
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    
    # Hard-coded embedding_dim
    embedding_dim = 300
    
    # Initialize the model with best hyperparameters
    model = SentimentModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=embedding_dim,
        n_filters=best_params['n_filters'],
        filter_sizes=filter_sizes,
        user_emb_dim=best_params['user_product_embed_size'],
        product_emb_dim=best_params['user_product_embed_size'],
        output_dim=5,
        dropout=best_params['dropout'],
        learning_rate=best_params['learning_rate'],
        user_embedding_weights=torch.tensor(user_embeddings, dtype=torch.float32),
        product_embedding_weights=torch.tensor(item_embeddings, dtype=torch.float32),
        blend_factor=best_params['blend_factor'],
        unfreeze_epoch=best_params['unfreeze_epoch'],
        weight_decay=best_params['weight_decay'],
        extern_params={
            'als_factors': best_params['user_product_embed_size'],
            'als_iterations': best_params['als_iterations'],
            'als_regularization': best_params['als_regularization'],
            'cull_unknown_threshold': best_params.get('cull_unknown_threshold', 0),
            'random_state': RANDOM_STATE
        }
    )
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=False,
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        # dirpath='checkpoints/',
        filename='best_model_manual_tune'
    )
    
    logger = TensorBoardLogger("lightning_logs", name="final_training")
    
    # Trainer
    trainer = Trainer(
        max_epochs=150,
        accelerator='gpu',
        devices='auto',
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[early_stopping, lr_monitor, checkpoint_callback],
        logger=logger,
        deterministic=True,
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(FINAL_CHECKPOINT_FILENAME)
    print(f'Training finished! Final checkpoint saved at {FINAL_CHECKPOINT_FILENAME}')
