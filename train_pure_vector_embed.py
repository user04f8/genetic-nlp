import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import tiktoken
import numpy as np
import random
import pandas as pd
import os
import sys
import signal
import argparse
import hashlib
import json

from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from utils import preprocess_data, get_train_val_split
from preprocess_data import load_data

# Set float32 matmul precision to 'medium' to leverage tensor cores
torch.set_float32_matmul_precision('medium')

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--latent_dim', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--max_length', type=int, default=256)
parser.add_argument('--scheduler_type', type=str, default='cosine')
parser.add_argument('--max_steps', type=int, default=1000)
parser.add_argument('--model_id', type=str, default='model_0')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--optimizer_type', type=str, default='adam')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--activation_function', type=str, default='relu')
parser.add_argument('--dropout_rate', type=float, default=0.0)
parser.add_argument('--num_layers', type=int, default=1)
args = parser.parse_args()

# Set random seed based on model_id
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)

def get_seed_from_model_id(model_id):
    seed = int(hashlib.sha256(model_id.encode('utf-8')).hexdigest(), 16) % (2**32)
    return seed

seed = get_seed_from_model_id(args.model_id)
seed_everything(seed)

# Dataset class
class ReviewDataset(Dataset):
    def __init__(self, tokens_list, labels, max_length):
        self.tokens_list = tokens_list
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list.iloc[idx]
        label = self.labels.iloc[idx] if self.labels is not None else None

        # Ensure tokens are in list format
        if isinstance(tokens, str):
            # Convert string representation of list to actual list
            import ast
            tokens = ast.literal_eval(tokens)
        else:
            tokens = list(tokens)

        # Truncate or pad tokens
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)

        if label is not None:
            label = torch.tensor(int(label) - 1, dtype=torch.long)  # Adjust labels to start from 0
            return tokens, label
        else:
            return tokens

# DataModule for PyTorch Lightning
class ReviewDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, batch_size=64, max_length=256):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.batch_size = batch_size
        self.max_length = max_length
        self.encoder = tiktoken.get_encoding('gpt2')  # Using GPT-2 tokenizer

    def setup(self, stage=None):
        self.train_dataset = ReviewDataset(
            tokens_list=self.train_df['Text_tokens'],
            labels=self.train_df['Score'],
            max_length=self.max_length
        )
        self.val_dataset = ReviewDataset(
            tokens_list=self.val_df['Text_tokens'],
            labels=self.val_df['Score'],
            max_length=self.max_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Set to False when using DDP
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

class ReviewClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        num_classes=5,
        latent_dim=3,
        lr=1e-3,
        scheduler_type='cosine',
        warmup_steps=0,
        max_steps=1000,
        model_id='model_0',
        optimizer_type='adam',
        weight_decay=0.0,
        activation_function='relu',
        dropout_rate=0.0,
        num_layers=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Activation function
        if activation_function == 'relu':
            self.activation = nn.ReLU()
        elif activation_function == 'tanh':
            self.activation = nn.Tanh()
        elif activation_function == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Build layers
        layers = []
        input_dim = embedding_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, latent_dim))
            layers.append(self.activation)
            layers.append(self.dropout)
            input_dim = latent_dim
        self.latent_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(input_dim, num_classes)

        self.lr = lr
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay

        # For storing validation outputs
        self.validation_accs = []

    def forward(self, x):
        # x: batch_size x seq_length
        embedded = self.embedding(x)  # batch_size x seq_length x embedding_dim

        # Global average pooling
        pooled = embedded.mean(dim=1)  # batch_size x embedding_dim

        # Pass through latent layers
        latent = self.latent_layers(pooled)

        # Output logits
        logits = self.output_layer(latent)  # batch_size x num_classes

        return logits

    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', acc, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels = batch
        logits = self(tokens)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        # Accumulate validation accuracy
        self.validation_accs.append(acc.detach())

    def on_validation_epoch_end(self):
        # Calculate average validation accuracy
        avg_val_acc = torch.stack(self.validation_accs).mean().item()

        # Save status to a file
        status = {
            'model_id': self.hparams.model_id,
            'epoch': self.current_epoch,
            'val_acc': avg_val_acc,
            'hyperparameters': vars(self.hparams)
            # Include any other relevant information
        }

        os.makedirs('status', exist_ok=True)
        status_file = f'./status/{self.hparams.model_id}_status.json'
        with open(status_file, 'w') as f:
            json.dump(status, f)

        # Clear the accumulated accuracies
        self.validation_accs.clear()

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type {self.optimizer_type}")

        if self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.max_steps, eta_min=1e-6
            )
        elif self.scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )
        elif self.scheduler_type == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1
            )
        else:
            scheduler = None

        if scheduler is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                }
            }
        else:
            return optimizer

# Function to handle graceful shutdown
def save_model_checkpoint(trainer, model):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'./checkpoints/{args.model_id}.ckpt'
    trainer.save_checkpoint(checkpoint_path)

def handle_sigterm(*args):
    save_model_checkpoint(trainer, model)
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == '__main__':
    # Load data
    train_df, test_df = load_data()

    # Preprocess data
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Split train data into train and validation
    train_df, val_df = get_train_val_split(train_df, random_state=42)

    # Create data module
    data_module = ReviewDataModule(
        train_df,
        val_df,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    data_module.setup()

    # Get vocab_size
    vocab_size = data_module.encoder.n_vocab

    # Create model with hyperparameters from args
    model = ReviewClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        latent_dim=args.latent_dim,
        lr=args.lr,
        scheduler_type=args.scheduler_type,
        max_steps=args.max_steps,
        model_id=args.model_id,
        optimizer_type=args.optimizer_type,
        weight_decay=args.weight_decay,
        activation_function=args.activation_function,
        dropout_rate=args.dropout_rate,
        num_layers=args.num_layers,
    )

    # Define callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')

    # Use custom logger to log only per-epoch metrics
    class EpochCSVLogger(CSVLogger):
        def log_metrics(self, metrics, step=None):
            if 'epoch' in metrics:
                super().log_metrics(metrics, step)

    logger = EpochCSVLogger(save_dir='logs/', name='review_classifier', version=args.model_id)

    # Define trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,  # Each process uses one GPU
        logger=logger,
        callbacks=[lr_monitor, early_stopping],
        deterministic=True,
        log_every_n_steps=0,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save the final model checkpoint
    save_model_checkpoint(trainer, model)
