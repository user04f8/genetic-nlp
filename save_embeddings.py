import torch
import numpy as np

from fuzzy_ngram_matrix_factors_model import SentimentModel
from data_processor import train_loader, test_loader

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
            embeddings = model.get_embeddings(text, summary, user, product, helpfulness_ratio, log_helpfulness_denominator)
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

model = SentimentModel.load_from_checkpoint('path_to_your_checkpoint.ckpt')
model.to(device)

# Extract embeddings for the training set
train_embeddings, train_labels, _ = extract_embeddings(model, train_loader, device)

# Save train embeddings and labels
np.savez_compressed('train_embeddings.npz', embeddings=train_embeddings, labels=train_labels)

# Extract embeddings for the test set
test_embeddings, _, test_ids = extract_embeddings(model, test_loader, device)

# Save test embeddings and IDs
np.savez_compressed('test_embeddings.npz', embeddings=test_embeddings, ids=np.array(test_ids))
