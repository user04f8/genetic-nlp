import torch
import pandas as pd
from torch.utils.data import DataLoader

from data_processor import DataProcessor, ReviewDataset, TestReviewDataset, collate_fn
from fuzzy_ngram_matrix_factors_model import SentimentModel
from preprocess_data import load_data
from utils import seed_everything

# Function to seed everything
def initialize_environment(seed=3):
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

# Function to load the trained model
def load_trained_model(checkpoint_path, device, no_load_glove=False):
    model = SentimentModel.load_from_checkpoint(checkpoint_path, no_load_glove=no_load_glove)
    model.to(device)
    model.eval()
    return model

# Function to process the data
def process_data(data_df, data_processor, is_training=False):
    processed_df = data_processor.process_reviews(data_df, is_training=is_training)
    return processed_df

# Function to create DataLoader
def create_dataloader(data_df, batch_size, is_training=False):
    if is_training:
        dataset = ReviewDataset(data_df)
    else:
        dataset = TestReviewDataset(data_df)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=is_training, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    return dataloader

# Function to make predictions on a DataLoader
def make_predictions(model, dataloader, device):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            batch_on_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(
                batch_on_device['text'], batch_on_device['summary'], batch_on_device['user'],
                batch_on_device['product'], batch_on_device['helpfulness_ratio'], batch_on_device['log_helpfulness_denominator']
            )
            predictions = outputs.argmax(1) + 1  # Convert back to 1-5 ratings
            all_predictions.extend(predictions.cpu().numpy())
    return all_predictions

# Function to test the model on arbitrary data samples
def test_model_on_samples(model, data_df, data_processor, device, num_samples=10):
    model.to(device)
    model.eval()
    # Sample random rows from data_df
    samples = data_df.sample(n=num_samples)
    # Process samples
    processed_samples = process_data(samples, data_processor, is_training=False)
    # Create DataLoader
    dataloader = create_dataloader(processed_samples, batch_size=num_samples, is_training=False)
    # Make predictions
    predictions = make_predictions(model, dataloader, device)
    # Compare predictions with ground truth if available
    for idx, (index, row) in enumerate(samples.iterrows()):
        print(f"Sample {idx+1}:")
        print(f"Review ID: {row.get('Id', 'N/A')}")
        print(f"User ID: {row.get('UserId', 'N/A')}")
        print(f"Product ID: {row.get('ProductId', 'N/A')}")
        print(f"Text: {row.get('Text', '')}")
        print(f"Summary: {row.get('Summary', '')}")
        print(f"Prediction: {predictions[idx]}")
        ground_truth = row.get('Score')
        if ground_truth is not None:
            print(f"Ground Truth: {ground_truth}")
        print("-" * 50)

# Function to generate submission file
def generate_submission(model, test_dataloader, test_df, submission_file, device):
    predictions = make_predictions(model, test_dataloader, device)
    submission = []
    for idx, review_id in enumerate(test_df['Id']):
        submission.append((review_id, predictions[idx]))
    submission_df = pd.DataFrame(submission, columns=['Id', 'Score'])
    submission_df.to_csv(submission_file, index=False)
    print(f'Submission saved to {submission_file}')
