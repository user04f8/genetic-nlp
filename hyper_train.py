import argparse
import optuna
import os
import json
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from data_processor import DataProcessor, ReviewDataset, collate_fn
from fuzzy_ngram_matrix_factors_model import SentimentModel
from dynamic_als import update_df_and_get_als
import numpy as np

# Reproducibility
RANDOM_STATE = 4

# Constants
BATCH_SIZE = 1024
MAX_EPOCHS = 15

# Number of GPUs available
NUM_GPUS = 8
print(f"Number of GPUs available: {NUM_GPUS}")

# Load the data outside the objective function to avoid redundant computations
from preprocess_data import load_data
reviews_df, _ = load_data()

# Mapping for filter_sizes
FILTER_SIZES_OPTIONS = {
    'small': [2, 3, 4],
    'medium': [3, 4, 5],
    'large': [3, 5, 7],
    'all': [2, 3, 4, 5, 7]
}

# Lock and counter for assigning GPUs
gpu_lock = multiprocessing.Lock()
gpu_counter = multiprocessing.Value('i', 0)

def objective(trial):
    # Assign a GPU to this trial
    with gpu_lock:
        gpu_id = gpu_counter.value % NUM_GPUS
        gpu_counter.value += 1
    print(f"Trial {trial.number} is using GPU {gpu_id}")

    # Set the GPU device index in the Trainer
    devices = [gpu_id]

    # Set seed for reproducibility
    seed = RANDOM_STATE + trial.number
    seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('high')

    # Hyperparameter space
    cull_unknown_threshold = 0  # Must be static
    user_product_embed_size = trial.suggest_int('user_product_embed_size', 5, 50)
    als_iterations = trial.suggest_int('als_iterations', 1, 50)
    als_regularization = trial.suggest_float('als_regularization', 1e-4, 5e-1, log=True)
    n_filters = trial.suggest_int('n_filters', 50, 200)
    filter_sizes_key = trial.suggest_categorical('filter_sizes', list(FILTER_SIZES_OPTIONS.keys()))
    filter_sizes = FILTER_SIZES_OPTIONS[filter_sizes_key]
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    blend_factor = trial.suggest_float('blend_factor', 0.0, 1.0)
    unfreeze_epoch = trial.suggest_int('unfreeze_epoch', 0, 10)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    # Hard-coded embedding_dim
    embedding_dim = 300

    # Initialize DataProcessor and set encoders
    data_processor = DataProcessor(no_unknowns=False, unknown_threshold=cull_unknown_threshold)
    data_processor.fit_encoders(reviews_df)

    # Generate ALS embeddings and update DataFrame
    user_embeddings, item_embeddings, num_users, num_products = update_df_and_get_als(
        reviews_df, data_processor, n_factors=user_product_embed_size, n_iterations=als_iterations,
        regularization=als_regularization, cache_dir='./cache/'
    )

    # Process reviews_df to include 'user_idx' and 'product_idx'
    processed_reviews_df = data_processor.process_reviews(reviews_df, is_training=True)

    print('Generating dataset / split')
    # Prepare the dataset and dataloaders
    dataset = ReviewDataset(processed_reviews_df)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Increase num_workers to utilize more CPU cores
    num_workers = max(1, (os.cpu_count() // NUM_GPUS) - 1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )

    # Initialize the model
    model = SentimentModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=embedding_dim,
        n_filters=n_filters,
        filter_sizes=filter_sizes,
        user_emb_dim=user_product_embed_size,
        product_emb_dim=user_product_embed_size,
        output_dim=5,
        dropout=dropout,
        learning_rate=learning_rate,
        user_embedding_weights=torch.tensor(user_embeddings, dtype=torch.float32),
        product_embedding_weights=torch.tensor(item_embeddings, dtype=torch.float32),
        blend_factor=blend_factor,
        unfreeze_epoch=unfreeze_epoch,
        weight_decay=weight_decay,
        extern_params={
            'als_factors': user_product_embed_size,
            'als_iterations': als_iterations,
            'als_regularization': als_regularization,
            'cull_unknown_threshold': cull_unknown_threshold,
            'random_state': seed
        }
    )

    # Custom PyTorch Lightning pruning callback
    class OptunaPruningCallback(Callback):
        def __init__(self, trial, monitor):
            super().__init__()
            self.trial = trial
            self.monitor = monitor

        def on_validation_end(self, trainer, pl_module):
            current_score = trainer.callback_metrics.get(self.monitor)
            if current_score is None:
                return
            self.trial.report(current_score, step=trainer.current_epoch)
            if self.trial.should_prune():
                message = f"Trial was pruned at epoch {trainer.current_epoch}."
                raise optuna.exceptions.TrialPruned(message)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    pruning_callback = OptunaPruningCallback(trial, monitor='val_acc')

    # Logger
    logger = TensorBoardLogger("lightning_logs", name=f"trial_{trial.number}")

    # Trainer
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='gpu',
        devices=devices,
        callbacks=[early_stopping, lr_monitor, pruning_callback],
        logger=logger,
        enable_checkpointing=False,
        deterministic=True,
    )

    # Train the model
    try:
        trainer.fit(model, train_loader, val_loader)
    except optuna.exceptions.TrialPruned as e:
        # Handle pruning
        print(f"Trial {trial.number} pruned: {e}")
        raise optuna.exceptions.TrialPruned()

    # Get validation accuracy
    val_acc = trainer.callback_metrics.get('val_acc')
    if isinstance(val_acc, torch.Tensor):
        val_acc = val_acc.item()

    # Handle the case where validation accuracy is not available
    if val_acc is None:
        val_acc = 0.0

    # Save trial details to a logfile
    trial_results = {
        'trial_number': trial.number,
        'value': val_acc,
        'params': trial.params
    }
    with open(f'optuna_trial_{trial.number}.json', 'w') as f:
        json.dump(trial_results, f, indent=4)

    return val_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument('--n_trials', type=int, default=100, help="Number of trials for hyperparameter search")
    args = parser.parse_args()

    # Create Optuna study
    study = optuna.create_study(direction='maximize')

    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=NUM_GPUS)
    except KeyboardInterrupt:
        print("Catching interrupt. Saving study and exiting gracefully.")
    finally:
        print('Number of finished trials:', len(study.trials))
        print('Best trial:')
        trial = study.best_trial

        print('  Value: {:.4f}'.format(trial.value))
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

        best_trial_results = {
            'trial_number': trial.number,
            'value': trial.value,
            'params': trial.params
        }
        with open('optuna_best_trial.json', 'w') as f:
            json.dump(best_trial_results, f, indent=4)

    # Automatically run final training script with best hyperparameters
    # print("Starting final training with best hyperparameters...")
    # os.system('python final_training.py')
