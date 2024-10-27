import torch
from model_utils import (
    initialize_environment, load_trained_model, process_data, create_dataloader, generate_submission
)
from data_processor import DataProcessor
from preprocess_data import load_data
from utils import seed_everything

# Initialize environment and device
device = initialize_environment(seed=3)

# Load the test data
_, test_df = load_data()

# Load encoders
data_processor = DataProcessor()
data_processor.load_encoders()

# Process test data
processed_test_df = process_data(test_df, data_processor, is_training=False)

# Create the test dataset and dataloader
batch_size = 512
test_dataloader = create_dataloader(processed_test_df, batch_size, is_training=False)

# Load model weights
BEST_CHECKPOINT_FILENAME = 'lightning_logs/version_64/checkpoints/epoch=45-step=13386.ckpt'
model = load_trained_model(BEST_CHECKPOINT_FILENAME, device, no_load_glove=False)

# Generate submission
submission_file = 'submission.csv'
generate_submission(model, test_dataloader, processed_test_df, submission_file, device)
