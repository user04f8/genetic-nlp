import torch
from model_utils import (
    initialize_environment, load_trained_model, process_data, create_dataloader, generate_submission
)
from data_processor import DataProcessor
from preprocess_data import load_data

# Initialize environment and device
device = initialize_environment(seed=3)
print(device)

# Load the test data
_, test_df = load_data()

# Load encoders
data_processor = DataProcessor(unknown_threshold=None)
# NOTE original unknown_threshold is from hparams.yaml
data_processor.load_encoders()

print('Running process_data()')
processed_test_df = process_data(test_df, data_processor, is_training=False)

print('Creating dataloader')
batch_size = 512
test_dataloader = create_dataloader(processed_test_df, batch_size, is_training=False)

print('Loading model')
# BEST_CHECKPOINT_FILENAME = 'lightning_logs/version_64/checkpoints/epoch=45-step=13386.ckpt'
BEST_CHECKPOINT_FILENAME = 'lightning_logs/heavy_regularization/version_1/checkpoints/epoch=40-step=11931.ckpt'
model = load_trained_model(BEST_CHECKPOINT_FILENAME, device, no_load_glove=False)

print('Generating. . .')
submission_file = 'submission.csv'
generate_submission(model, test_dataloader, processed_test_df, submission_file, device)

print(f'Complete! Submission file exists at {submission_file}')
