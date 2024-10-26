import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from preprocess_data import load_preprocessed_data

train_df, test_df = load_preprocessed_data()


class UserProductDataset(Dataset):
    def __init__(self, df):
        # Encode UserId and ProductId as categorical codes
        self.users = torch.tensor(df['UserId'].astype('category').cat.codes.values, dtype=torch.long)  # Long for embedding layers
        self.products = torch.tensor(df['ProductId'].astype('category').cat.codes.values, dtype=torch.long)
        self.scores = torch.tensor(df['Score'].values - 1, dtype=torch.long)  # Subtract 1 to shift scores from [1, 5] to [0, 4] for classification

        # Store the number of users and products for embedding layer dimensions
        self.n_users = len(df['UserId'].unique())
        self.n_products = len(df['ProductId'].unique())

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.users[idx], self.products[idx], self.scores[idx]

# Split data into train and validation sets
train_size = int(0.8 * len(train_df))
val_size = len(train_df) - train_size
train_data, val_data = torch.utils.data.random_split(train_df, [train_size, val_size])

# Create PyTorch datasets
train_dataset = UserProductDataset(train_data.dataset.iloc[train_data.indices])
val_dataset = UserProductDataset(val_data.dataset.iloc[val_data.indices])

# Create DataLoaders with multi-threading for faster data loading, larger batch size, and pinning memory for GPU
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)


class MatrixFactorizationLogitModel(nn.Module):
    def __init__(self, n_users, n_products, n_factors=50):
        super(MatrixFactorizationLogitModel, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)  # User embedding
        self.product_factors = nn.Embedding(n_products, n_factors)  # Product embedding
        self.fc = nn.Linear(n_factors, 5)  # Fully connected layer to output logits for 5 possible scores

    def forward(self, user, product):
        user_embedding = self.user_factors(user)  # Shape: (batch_size, n_factors)
        product_embedding = self.product_factors(product)  # Shape: (batch_size, n_factors)
        
        interaction = user_embedding * product_embedding  # Element-wise product
        logits = self.fc(interaction)  # Pass the interaction through the fully connected layer
        return logits  # Output logits for the 5 score classes

print(f"User tensor sample: {train_dataset[0][0]}")  # Should print a user ID as an integer
print(f"Product tensor sample: {train_dataset[0][1]}")  # Should print a product ID as an integer

# Initialize the model
n_factors = 50  # Number of latent factors
model = MatrixFactorizationLogitModel(train_dataset.n_users, train_dataset.n_products, n_factors)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function: CrossEntropyLoss for classification
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Early stopping to mitigate overfitting
early_stopping_patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0

# Training and validation loops
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for user, product, score in train_loader:
        # Move data to GPU
        user, product, score = user.to(device, non_blocking=True), product.to(device, non_blocking=True), score.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(user, product)
        loss = loss_fn(logits, score)  # Use CrossEntropyLoss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user, product, score in val_loader:
            # Move data to GPU
            user, product, score = user.to(device, non_blocking=True), product.to(device, non_blocking=True), score.to(device, non_blocking=True)
            logits = model(user, product)
            loss = loss_fn(logits, score)  # Use CrossEntropyLoss
            total_loss += loss.item()

    return total_loss / len(val_loader)

# Training loop with early stopping and LR scheduler
n_epochs = 50  # Allow for more epochs but early stopping will stop when there's no improvement

for epoch in range(n_epochs):
    train_loss = train(model, train_loader, optimizer, loss_fn, device)
    val_loss = validate(model, val_loader, loss_fn, device)
    
    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss (CrossEntropy): {train_loss:.4f}, Val Loss (CrossEntropy): {val_loss:.4f}")
    
    # Step the LR scheduler based on validation loss
    scheduler.step(val_loss)
    
    # Check if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0  # Reset early stopping counter
        torch.save(model.state_dict(), 'best_model.pt')  # Save the best model
        print(f"Validation loss improved. Saving model with val_loss: {val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
    
    # Early stopping
    if epochs_no_improve >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# Load the best model for evaluation
model.load_state_dict(torch.load('best_model.pt'))

# Evaluate accuracy on the validation set
def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for user, product, score in data_loader:
            user, product, score = user.to(device), product.to(device), score.to(device)
            logits = model(user, product)
            predictions = torch.argmax(logits, dim=1)  # Get the predicted class (0-4, corresponding to scores 1-5)
            correct += (predictions == score).sum().item()
            total += score.size(0)

    return correct / total

accuracy = evaluate_accuracy(model, val_loader)
print(f"Validation Accuracy: {accuracy:.4f}")