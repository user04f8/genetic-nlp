import optuna
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from fuzzy_ngram_matrix_factorization import (
    seed_everything,
    num_users,
    num_products,
    train_dataset,
    val_dataset,
    SentimentModel,
    batch_size,
    collate_fn,
)
import torch
import os

# Define filter size options
filter_sizes_options = {
    'small': [2, 3, 4],
    'medium': [3, 4, 5],
    'large': [2, 3, 4, 5]
}

def objective(trial):
    # Set seed for reproducibility
    seed_everything(2)
    
    # Suggest hyperparameters
    n_filters = trial.suggest_categorical('n_filters', [50, 100, 200])
    filter_size_key = trial.suggest_categorical('filter_sizes', list(filter_sizes_options.keys()))
    filter_sizes = filter_sizes_options[filter_size_key]
    user_emb_dim = trial.suggest_int('user_emb_dim', 20, 100, step=10)
    product_emb_dim = trial.suggest_int('product_emb_dim', 20, 100, step=10)
    dropout = trial.suggest_float('dropout', 0.2, 0.7, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    # Initialize the model
    model = SentimentModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=300,
        n_filters=n_filters,
        filter_sizes=filter_sizes,
        user_emb_dim=user_emb_dim,
        product_emb_dim=product_emb_dim,
        output_dim=5,
        dropout=dropout,
        learning_rate=learning_rate
    )

    # Modify DataLoader to have a small num_workers during HPO
    from torch.utils.data import DataLoader

    def get_dataloaders():
        # Use a subset of the dataset to speed up HPO
        train_subset_size = int(0.1 * len(train_dataset))
        val_subset_size = int(0.1 * len(val_dataset))
        train_subset, _ = torch.utils.data.random_split(train_dataset, [train_subset_size, len(train_dataset) - train_subset_size])
        val_subset, _ = torch.utils.data.random_split(val_dataset, [val_subset_size, len(val_dataset) - val_subset_size])

        train_loader_hpo = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True
        )
        val_loader_hpo = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True
        )
        return train_loader_hpo, val_loader_hpo

    train_loader_hpo, val_loader_hpo = get_dataloaders()

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')

    # Determine the GPU to use for this trial
    available_gpus = torch.cuda.device_count()
    gpu_id = trial.number % available_gpus

    # Trainer with specified GPU
    trainer = Trainer(
        max_epochs=2,  # Reduce epochs to speed up HPO for initial test
        accelerator='gpu',
        devices=[gpu_id],
        callbacks=[early_stopping],
        enable_progress_bar=False,  # Disable progress bar for cleaner output
        logger=False,  # Disable logging to save resources
    )

    # Train the model
    trainer.fit(model, train_loader_hpo, val_loader_hpo)

    # Get the validation loss
    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss

if __name__ == '__main__':
    # Create an Optuna study
    study = optuna.create_study(direction='minimize')

    # Start the optimization with parallel trials
    study.optimize(objective, n_trials=20, n_jobs=8)  # Run 8 trials in parallel

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial

    print(f"  Val Loss: {trial.value}")
    print("  Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Re-train the model with the best hyperparameters on the full dataset
    best_model = SentimentModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=300,
        n_filters=trial.params['n_filters'],
        filter_sizes=filter_sizes_options[trial.params['filter_sizes']],
        user_emb_dim=trial.params['user_emb_dim'],
        product_emb_dim=trial.params['product_emb_dim'],
        output_dim=5,
        dropout=trial.params['dropout'],
        learning_rate=trial.params['learning_rate']
    )

    # Use the full DataLoader with original num_workers
    from torch.utils.data import DataLoader

    train_loader_full = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    val_loader_full = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    # Trainer with multiple GPUs for final training
    trainer = Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices='auto',
        strategy='ddp',  # Use DDP for multi-GPU training
        callbacks=[early_stopping],
    )

    # Fit the model
    trainer.fit(best_model, train_loader_full, val_loader_full)

    # Save the final model checkpoint
    FINAL_CHECKPOINT_FILENAME = "checkpoints/best_sentiment_model.ckpt"
    trainer.save_checkpoint(FINAL_CHECKPOINT_FILENAME)
