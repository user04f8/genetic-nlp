import torch
from datasets.user_product import get_data_loaders
from models.matrix_factorization import MatrixFactorizationLogitModel
from utils.train import train_epoch, validate_epoch, get_optimizer_scheduler
from utils.evaluation import evaluate_accuracy, print_confusion_matrix, save_confusion_matrix_plot

# Load data and compute class weights
from preprocess_data import load_data
train_df, test_df = load_data()

# Compute class weights
def compute_class_weights(train_df):
    score_counts = train_df['Score'].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = total_samples / (len(score_counts) * score_counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32)
    return class_weights

class_weights = compute_class_weights(train_df)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders
train_loader, val_loader, n_users, n_products = get_data_loaders(train_df)

# Initialize model
model = MatrixFactorizationLogitModel(n_users, n_products).to(device)

# Get optimizer and scheduler
optimizer, scheduler = get_optimizer_scheduler(model)

# Loss function with class weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

# Early stopping variables
early_stopping_patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop
n_epochs = 50

for epoch in range(n_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss = validate_epoch(model, val_loader, loss_fn, device)
    
    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"Validation loss improved. Model saved.")
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

    if epochs_no_improve >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))

# Evaluate accuracy
from utils.evaluation import evaluate_accuracy, print_confusion_matrix, save_confusion_matrix_plot
accuracy, predictions, true_labels = evaluate_accuracy(model, val_loader, device)
print(f"Validation Accuracy: {accuracy:.4f}")

# Print or save the confusion matrix
print_confusion_matrix(true_labels, predictions)
save_confusion_matrix_plot(true_labels, predictions, filename="diagrams/confusion_matrix.png")
