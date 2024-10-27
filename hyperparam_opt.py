import optuna
from optuna.integration import PyTorchLightningPruningCallback

from fuzzy_ngram_matrix_factorization import SentimentModel, num_products, num_users, Trainer, early_stopping, lr_monitor, train_loader, val_loader

def objective(trial):
    # Suggest hyperparameters
    n_filters = trial.suggest_int("n_filters", 50, 200)
    dropout = trial.suggest_float("dropout", 0.3, 0.7)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    
    # Create model with suggested hyperparameters
    model = SentimentModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=300,
        n_filters=n_filters,
        filter_sizes=[3, 4, 5],
        user_emb_dim=50,
        product_emb_dim=50,
        output_dim=5,
        dropout=dropout,
        learning_rate=learning_rate
    )

    # Define trainer with Optuna's pruning callback
    trainer = Trainer(
        max_epochs=10,
        accelerator='gpu',
        devices='auto',
        strategy='ddp',
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            early_stopping,
            lr_monitor
        ],
        enable_progress_bar=False
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Return validation loss for Optuna
    return trainer.callback_metrics["val_loss"].item()

# Create an Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print(f"Best trial: {study.best_trial.value}")
print(f"Best hyperparameters: {study.best_trial.params}")