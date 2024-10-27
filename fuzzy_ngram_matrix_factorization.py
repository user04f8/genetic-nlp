import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from sklearn.preprocessing import LabelEncoder

from preprocess_data import load_data, get_glove
from utils import seed_everything

FINAL_CHECKPOINT_FILENAME = "checkpoints/fuzzy_ngram_model.ckpt"

seed_everything(1)
torch.set_float32_matmul_precision('high')

# Load the data
reviews_df, test_df = load_data()


# Encode User IDs and Product IDs
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

reviews_df['user_idx'] = user_encoder.fit_transform(reviews_df['UserId'])
reviews_df['product_idx'] = product_encoder.fit_transform(reviews_df['ProductId'])

# Get the number of unique users and products
num_users = len(user_encoder.classes_)
num_products = len(product_encoder.classes_)

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.text_tokens = df['Text_glove_tokens_np'].values
        self.summary_tokens = df['Summary_glove_tokens_np'].values
        self.user_idx = df['user_idx'].values
        self.product_idx = df['product_idx'].values
        self.labels = df['Score'].values.astype(np.int64) - 1  # Ratings from 0 to 4

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.text_tokens[idx]
        summary = self.summary_tokens[idx]
        user = self.user_idx[idx]
        product = self.product_idx[idx]
        label = self.labels[idx]
        return {
            'text': torch.tensor(text, dtype=torch.long),
            'summary': torch.tensor(summary, dtype=torch.long),
            'user': torch.tensor(user, dtype=torch.long),
            'product': torch.tensor(product, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TestReviewDataset(Dataset):
    def __init__(self, df):
        self.text_tokens = df['Text_glove_tokens_np'].values
        self.summary_tokens = df['Summary_glove_tokens_np'].values
        self.user_idx = df['user_idx'].values
        self.product_idx = df['product_idx'].values
        self.ids = df['Id'].values  # Add `Id` to return for submission

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        text = self.text_tokens[idx]
        summary = self.summary_tokens[idx]
        user = self.user_idx[idx]
        product = self.product_idx[idx]
        review_id = self.ids[idx]
        return {
            'text': torch.tensor(text, dtype=torch.long),
            'summary': torch.tensor(summary, dtype=torch.long),
            'user': torch.tensor(user, dtype=torch.long),
            'product': torch.tensor(product, dtype=torch.long),
            'Id': review_id  # Return the Id for the final submission
        }

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    summaries = [item['summary'] for item in batch]
    users = torch.stack([item['user'] for item in batch])
    products = torch.stack([item['product'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    summaries_padded = pad_sequence(summaries, batch_first=True, padding_value=0)

    return {
        'text': texts_padded,
        'summary': summaries_padded,
        'user': users,
        'product': products,
        'label': labels
    }


# Split the dataset into training and validation sets
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


class SentimentModel(pl.LightningModule):
    def __init__(self, num_users, num_products, embedding_dim=300, n_filters=100, filter_sizes=[3,4,5], 
                 user_emb_dim=50, product_emb_dim=50, output_dim=5, dropout=0.5):
        super(SentimentModel, self).__init__()

        # Load pre-trained GloVe embeddings
        self.embedding = get_glove()
        self.embedding.weight.requires_grad = False  # Freeze embeddings

        # Text CNN
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        # User and Product Embeddings
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.product_embedding = nn.Embedding(num_products, product_emb_dim)

        # Fully Connected Layer
        self.fc = nn.Linear(len(filter_sizes)*n_filters*2 + user_emb_dim + product_emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, summary, user_idx, product_idx):
        # Text Embedding
        embedded_text = self.embedding(text)  # [batch_size, text_len, emb_dim]
        embedded_text = embedded_text.unsqueeze(1)  # [batch_size, 1, text_len, emb_dim]

        # Summary Embedding
        embedded_summary = self.embedding(summary)  # [batch_size, summary_len, emb_dim]
        embedded_summary = embedded_summary.unsqueeze(1)  # [batch_size, 1, summary_len, emb_dim]

        # Convolutions on Text
        text_conved = [F.relu(conv(embedded_text)).squeeze(3) for conv in self.convs]
        text_pooled = [F.max_pool1d(t, t.size(2)).squeeze(2) for t in text_conved]

        # Convolutions on Summary
        summary_conved = [F.relu(conv(embedded_summary)).squeeze(3) for conv in self.convs]
        summary_pooled = [F.max_pool1d(s, s.size(2)).squeeze(2) for s in summary_conved]

        # Concatenate pooled features
        text_features = torch.cat(text_pooled, dim=1)
        summary_features = torch.cat(summary_pooled, dim=1)
        text_cat = torch.cat([text_features, summary_features], dim=1)

        text_cat = self.dropout(text_cat)

        # User and Product Embeddings
        user_embedded = self.user_embedding(user_idx)
        product_embedded = self.product_embedding(product_idx)

        # Concatenate all features
        combined = torch.cat([text_cat, user_embedded, product_embedded], dim=1)

        combined = self.dropout(combined)

        # Final output
        output = self.fc(combined)

        return output

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['text'], batch['summary'], batch['user'], batch['product'])
        loss = F.cross_entropy(outputs, batch['label'])
        acc = (outputs.argmax(1) == batch['label']).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch['text'], batch['summary'], batch['user'], batch['product'])
        loss = F.cross_entropy(outputs, batch['label'])
        acc = (outputs.argmax(1) == batch['label']).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }

# Initialize the model
model = SentimentModel(
    num_users=num_users,
    num_products=num_products,
    embedding_dim=300,  # GloVe embedding size
    n_filters=100,
    filter_sizes=[3, 4, 5],
    user_emb_dim=50,
    product_emb_dim=50,
    output_dim=5,  # Ratings from 0 to 4
    dropout=0.5
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(
    max_epochs=50,
    accelerator='gpu',
    devices='auto',  # Automatically use available GPUs
    strategy='ddp',
    callbacks=[early_stopping, lr_monitor]  # Add callbacks
)


def evaluate_model(model, dataloader):
    model.eval()
    total_correct = 0
    total_count = 0
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(batch['text'], batch['summary'], batch['user'], batch['product'])
            predictions = outputs.argmax(1)
            total_correct += (predictions == batch['label']).sum().item()
            total_count += batch['label'].size(0)
    accuracy = total_correct / total_count
    print(f'Final validation Accuracy: {accuracy:.4f}')

if __name__ == '__main__':

    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint(FINAL_CHECKPOINT_FILENAME)


    evaluate_model(model, val_loader)
