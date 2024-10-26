# train_ddp.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets.user_product import get_data_loaders
from models.matrix_factorization import MatrixFactorizationLogitModel
from utils.train import train_epoch, validate_epoch, get_optimizer_scheduler
from utils.evaluation import evaluate_accuracy, print_confusion_matrix, save_confusion_matrix_plot
from preprocess_data import load_preprocessed_data
import os

# Define the training function for each GPU process
def train_ddp(rank, world_size):
    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    # Set the device for each process (each process gets its own GPU)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Load the preprocessed data
    train_df, test_df = load_preprocessed_data()

    # Compute class weights (optional)
    class_weights = compute_class_weights(train_df).to(device)

    # Create data loaders with DistributedSampler
    train_loader, val_loader, n_users, n_products = get_data_loaders(train_df, sampler_rank=rank, world_size=world_size)

    # Initialize the model and wrap it with DDP
    model = MatrixFactorizationLogitModel(n_users, n_products).to(device)
    model = DDP(model, device_ids=[rank])

    # Set up optimizer, scheduler, and loss function
    optimizer, scheduler = get_optimizer_scheduler(model)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Early stopping variables
    early_stopping_patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    n_epochs = 50
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate_epoch(model, val_loader, loss_fn, device)

        if rank == 0:  # Only rank 0 logs the results
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if rank == 0:  # Save the best model from rank 0
                torch.save(model.module.state_dict(), 'best_model.pt')
                print(f"Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            if rank == 0:
                print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        if epochs_no_improve >= early_stopping_patience:
            if rank == 0:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Cleanup the process group
    dist.destroy_process_group()

def compute_class_weights(train_df):
    score_counts = train_df['Score'].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = total_samples / (len(score_counts) * score_counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32)
    return class_weights

def main():
    world_size = torch.cuda.device_count()  # Number of GPUs available
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Spawn processes for each GPU
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
