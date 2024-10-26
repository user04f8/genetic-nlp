import torch
from torch.utils.data import Dataset, DataLoader

class UserProductDataset(Dataset):
    def __init__(self, df):
        # Encode UserId and ProductId as categorical codes
        self.users = torch.tensor(df['UserId'].astype('category').cat.codes.values, dtype=torch.long)
        self.products = torch.tensor(df['ProductId'].astype('category').cat.codes.values, dtype=torch.long)
        self.scores = torch.tensor(df['Score'].values - 1, dtype=torch.long)  # Shift scores [1, 5] to [0, 4]

        # Store the number of users and products for embedding layer dimensions
        self.n_users = len(df['UserId'].unique())
        self.n_products = len(df['ProductId'].unique())

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.users[idx], self.products[idx], self.scores[idx]

def get_data_loaders(train_df, batch_size=4096, num_workers=4):
    # Split data into train and validation sets
    train_size = int(0.8 * len(train_df))
    val_size = len(train_df) - train_size
    train_data, val_data = torch.utils.data.random_split(train_df, [train_size, val_size])

    # Create PyTorch datasets
    train_dataset = UserProductDataset(train_data.dataset.iloc[train_data.indices])
    val_dataset = UserProductDataset(val_data.dataset.iloc[val_data.indices])

    # Create DataLoaders with multi-threading
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset.n_users, train_dataset.n_products
