# generate_submission.py

import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from data_processor import DataProcessor, TestReviewDataset, collate_fn
from fuzzy_ngram_matrix_factors_model import SentimentModel
from preprocess_data import load_data
from utils import seed_everything

seed_everything(3)

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the test data
reviews_df, test_df = load_data()

# Load encoders
data_processor = DataProcessor()
data_processor.load_encoders()

# Process test data
test_df = data_processor.process_reviews(test_df, is_training=False)

# Create the test dataset and dataloader
test_dataset = TestReviewDataset(test_df)
batch_size = 512
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

# Load model weights
BEST_CHECKPOINT_FILENAME = 'lightning_logs/version_64/checkpoints/epoch=45-step=13386.ckpt'
model = SentimentModel.load_from_checkpoint(BEST_CHECKPOINT_FILENAME)
model.to(device)
model.eval()

# Run inference on the test set and generate predictions
submission = []
num_examples_to_print = 5
example_counter = 0

with torch.no_grad():
    for batch in test_loader:
        # Move data to device
        batch_on_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(
            batch_on_device['text'], batch_on_device['summary'], batch_on_device['user'],
            batch_on_device['product'], batch_on_device['helpfulness_ratio'], batch_on_device['log_helpfulness_denominator']
        )
        predictions = outputs.argmax(1) + 1  # Convert back to 1-5 ratings
        for i, review_id in enumerate(batch['Id']):
            submission.append((review_id, predictions[i].item()))
            if example_counter < num_examples_to_print:
                print(f"Review ID: {review_id}")
                print(f"User ID: {test_df.loc[test_df['Id'] == review_id, 'UserId'].values[0]}")
                print(f"Product ID: {test_df.loc[test_df['Id'] == review_id, 'ProductId'].values[0]}")
                # For text and summary, we'll print the raw text if available
                text_tokens = batch['text'][i]
                summary_tokens = batch['summary'][i]
                # Assuming you have a function 'decode_tokens' to convert tokens back to text
                print(f"Text tokens: {text_tokens.cpu().numpy()}")
                print(f"Summary tokens: {summary_tokens.cpu().numpy()}")
                print(f"Prediction: {predictions[i].item()}")
                print('-' * 50)
                example_counter += 1

# Create the submission DataFrame
submission_df = pd.DataFrame(submission, columns=['Id', 'Score'])
submission_file = 'submission.csv'
submission_df.to_csv(submission_file, index=False)
print(f'Submission saved to {submission_file}')
