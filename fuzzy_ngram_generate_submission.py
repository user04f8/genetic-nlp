import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from preprocess_data import load_data, get_glove
from fuzzy_ngram_matrix_factorization import SentimentModel, TestReviewDataset, collate_fn, test_df, user_encoder, product_encoder, FINAL_CHECKPOINT_FILENAME

test_df['user_idx'] = user_encoder.fit_transform(test_df['UserId'])
test_df['product_idx'] = product_encoder.fit_transform(test_df['ProductId'])

# Create the test dataset and dataloader
test_dataset = TestReviewDataset(test_df)
batch_size = 512
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

# Load the trained model
model = SentimentModel(
    num_users=len(user_encoder.classes_),
    num_products=len(product_encoder.classes_),
    embedding_dim=300,  # GloVe embedding size
    n_filters=100,
    filter_sizes=[3, 4, 5],
    user_emb_dim=50,
    product_emb_dim=50,
    output_dim=5,  # Ratings from 0 to 4
    dropout=0.5
)

# Load model weights
model = model.load_from_checkpoint(FINAL_CHECKPOINT_FILENAME)

# Set the model to evaluation mode
model.eval()

# Run inference on the test set and generate predictions
submission = []

with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch['text'], batch['summary'], batch['user'], batch['product'])
        predictions = outputs.argmax(1) + 1  # Convert back to 1-5 ratings
        for i, review_id in enumerate(batch['Id']):
            submission.append((review_id, predictions[i].item()))

# Convert the submission to a DataFrame
submission_df = pd.DataFrame(submission, columns=['Id', 'Score'])

# Save to CSV
submission_file = f'submission_{1}.csv'
submission_df.to_csv(submission_file, index=False)

print(f'Submission saved to {submission_file}')