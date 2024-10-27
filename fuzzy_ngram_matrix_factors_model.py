import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from preprocess_data import get_glove

class SentimentModel(pl.LightningModule):
    def __init__(
        self,
        num_users,
        num_products,
        embedding_dim=300,
        n_filters=100,
        filter_sizes=[3, 4, 5],
        user_emb_dim=50,
        product_emb_dim=50,
        output_dim=5,
        dropout=0.5,
        learning_rate=1e-3,
        weight_decay: float = None,
        user_embedding_weights=None,
        product_embedding_weights=None,
        als_freeze=False,
        latent_user_product_dim=25,
        enable_user_product_dim_reduce=False,
        no_load_glove=False,
        blend_factor=0.5,
        unfreeze_epoch=5,  # Unfreeze after 5 epochs
        extern_params=None
    ):
        super(SentimentModel, self).__init__()

        # Save hyperparameters, excluding large embedding weights
        self.save_hyperparameters(
            ignore=['user_embedding_weights', 'product_embedding_weights']
        )

        if no_load_glove:
            self.embedding = nn.Identity()
            print('WARN: not loading GloVe (this should only be done during model analysis)')
        else:
            # Load pre-trained GloVe embeddings
            self.embedding = get_glove()
            self.embedding.weight.requires_grad = False  # Freeze embeddings

        # Fuzzy n-grams
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        # Initialize User and Product Embeddings
        if user_embedding_weights is not None:
            # Ensure embedding dimensions match
            if user_embedding_weights.shape[1] != user_emb_dim or product_embedding_weights.shape[1] != product_emb_dim:
                print("WARNING, emb_dim appears mismatched: ", user_emb_dim, product_emb_dim)
                user_emb_dim = user_embedding_weights.shape[1]
                product_emb_dim = product_embedding_weights.shape[1]

            # Random embeddings
            self.user_embedding = self._blend_embeddings(nn.Embedding(num_users, user_emb_dim), user_embedding_weights, blend_factor)
            self.product_embedding = self._blend_embeddings(nn.Embedding(num_products, product_emb_dim), product_embedding_weights, blend_factor)
        else:
            # If no pre-trained embeddings, use only random embeddings
            self.user_embedding = nn.Embedding(num_users, user_emb_dim)
            self.product_embedding = nn.Embedding(num_products, product_emb_dim)

        self.unfreeze_epoch = unfreeze_epoch  # Store unfreeze epoch as a hyperparameter

        self.enable_user_product_dim_reduce = enable_user_product_dim_reduce
        if self.enable_user_product_dim_reduce:
            self.user_product_dim_reduction = nn.Linear(
                user_emb_dim + product_emb_dim, latent_user_product_dim
            )
            self.user_product_pool = F.relu
            combined_user_product_dim = latent_user_product_dim
        else:
            self.user_product_dim_reduction = nn.Identity()
            self.user_product_pool = nn.Identity()
            combined_user_product_dim = user_emb_dim + product_emb_dim

        # Feature weighting
        self.feature_weights = nn.Linear(
            len(filter_sizes) * n_filters * 2 + combined_user_product_dim + 2, output_dim
        )

        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Initially freeze embeddings
        self._freeze_embeddings()

    def _blend_embeddings(self, random_embedding_weights, pretrained_embedding_weights, blend_factor):
        """
        Blend pre-trained and random embeddings during initialization.
        """
        blended_user_weights = (
            blend_factor * pretrained_embedding_weights +
            (1 - blend_factor) * random_embedding_weights.weight.data
        )
        return nn.Embedding.from_pretrained(
            blended_user_weights
        )

    def _freeze_embeddings(self):
        self.user_embedding.weight.requires_grad = False
        self.product_embedding.weight.requires_grad = False

    def _unfreeze_embeddings(self):
        self.user_embedding.weight.requires_grad = True
        self.product_embedding.weight.requires_grad = True

    def forward(self, text, summary, user_idx, product_idx, helpfulness_ratio, log_helpfulness_denominator):
        # Text Embedding
        embedded_text = self.embedding(text)  # [batch_size, text_len, emb_dim]
        embedded_text = embedded_text.unsqueeze(1)  # [batch_size, 1, text_len, emb_dim]

        # Summary Embedding
        embedded_summary = self.embedding(summary)  # [batch_size, summary_len, emb_dim]
        embedded_summary = embedded_summary.unsqueeze(1)  # [batch_size, 1, summary_len, emb_dim]

        # Fuzzy n-grams on Text
        text_n_grams = [F.relu(conv(embedded_text)).squeeze(3) for conv in self.convs]
        text_pooled = [F.max_pool1d(t, t.size(2)).squeeze(2) for t in text_n_grams]

        # Fuzzy n-grams on Summary
        summary_n_grams = [F.relu(conv(embedded_summary)).squeeze(3) for conv in self.convs]
        summary_pooled = [F.max_pool1d(s, s.size(2)).squeeze(2) for s in summary_n_grams]

        # Concatenate pooled features
        text_features = torch.cat(text_pooled, dim=1)
        summary_features = torch.cat(summary_pooled, dim=1)
        text_cat = torch.cat([text_features, summary_features], dim=1)

        text_cat = self.dropout(text_cat)

        user_embedded = self.user_embedding(user_idx)
        product_embedded = self.product_embedding(product_idx)

        # Ensure helpfulness_ratio and log_helpfulness_denominator are 2D tensors
        helpfulness_ratio = helpfulness_ratio.unsqueeze(1)                      # [batch_size, 1]
        log_helpfulness_denominator = log_helpfulness_denominator.unsqueeze(1)  # [batch_size, 1]

        # Combine user and product embeddings
        user_product_features = self.user_product_pool(
            self.user_product_dim_reduction(torch.cat([user_embedded, product_embedded], dim=1))
        )

        # Concatenate all features
        combined = torch.cat(
            [text_cat, user_product_features, helpfulness_ratio, log_helpfulness_denominator], dim=1
        )

        combined = self.dropout(combined)

        return self.feature_weights(combined)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch['text'], batch['summary'], batch['user'], batch['product'],
            batch['helpfulness_ratio'], batch['log_helpfulness_denominator']
        )
        loss = F.cross_entropy(outputs, batch['label'])
        acc = (outputs.argmax(1) == batch['label']).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            batch['text'], batch['summary'], batch['user'], batch['product'],
            batch['helpfulness_ratio'], batch['log_helpfulness_denominator']
        )
        loss = F.cross_entropy(outputs, batch['label'])
        acc = (outputs.argmax(1) == batch['label']).float().mean()
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.weight_decay is None:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
            )
        else:
            no_decay = ['bias', 'LayerNorm.weight', 'embedding']  # Embedding layers shouldn't have weight decay
            decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
            no_decay_params = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]

            optimizer = torch.optim.AdamW(
                [{'params': decay, 'weight_decay': self.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}],  # No weight decay for embeddings
                lr=self.learning_rate
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }

    def on_train_epoch_start(self):
        # Unfreeze embeddings after unfreeze_epoch
        if self.current_epoch == self.unfreeze_epoch:
            self._unfreeze_embeddings()
            # Reconfigure optimizer to include new parameters
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
            )
            self.trainer.optimizers = [optimizer]
            print(f"Unfreezing embeddings at epoch {self.current_epoch}")

    def on_save_checkpoint(self, checkpoint):
        """Exclude the GloVe embeddings from the checkpoint."""
        if 'state_dict' in checkpoint and 'embedding.weight' in checkpoint['state_dict']:
            del checkpoint['state_dict']['embedding.weight']

    def load_state_dict(self, state_dict, strict=True):
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
        # Remove 'embedding.weight' from missing_keys
        if 'embedding.weight' in missing_keys:
            missing_keys.remove('embedding.weight')
        if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
            raise RuntimeError(f"Missing or unexpected keys: {missing_keys} {unexpected_keys}")
