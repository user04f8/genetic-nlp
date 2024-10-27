import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os
import zipfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from preprocess_data import load_data
from utils import preprocess_data, get_train_val_split


def load_glove_embeddings(glove_zip_file='glove.840B.300d.zip', glove_txt_file='glove.840B.300d.txt', binary_output_file='glove.840B.300d.word2vec.bin', embedding_dim=300):
    """
    Loads GloVe embeddings from a zip file, directly in Word2Vec format, and saves as binary format if necessary.

    Parameters:
    - glove_zip_file: The path to the GloVe zip file (e.g., 'glove.840B.300d.zip').
    - glove_txt_file: The GloVe .txt file inside the zip (e.g., 'glove.840B.300d.txt').
    - binary_output_file: The binary format file to save (e.g., 'glove.840B.300d.word2vec.bin').
    - embedding_dim: The dimensionality of the GloVe embeddings (default: 300).

    Returns:
    - embedding_matrix: A NumPy array containing the GloVe embeddings.
    - word_to_idx: A dictionary mapping words to indices in the embedding matrix.
    """

    # Check if the binary format file already exists
    if not os.path.exists(binary_output_file):
        # Extract the GloVe .txt file from the zip archive if necessary
        if not os.path.exists(glove_txt_file):
            print(f"Extracting {glove_txt_file} from {glove_zip_file}...")
            with zipfile.ZipFile(glove_zip_file, 'r') as z:
                z.extract(glove_txt_file)

        # Load GloVe .txt file directly using Gensim with no_header=True
        print(f"Loading GloVe embeddings from {glove_txt_file}...")
        model = KeyedVectors.load_word2vec_format(glove_txt_file, binary=False, no_header=True)

        # Save the model in binary format to reduce size
        print(f"Saving the embeddings in binary format as {binary_output_file}...")
        model.save_word2vec_format(binary_output_file, binary=True)

    else:
        print(f"Binary file {binary_output_file} already exists, loading embeddings...")

    # Load the binary format file using Gensim
    model = KeyedVectors.load_word2vec_format(binary_output_file, binary=True)

    # Create word to index mapping and embedding matrix
    word_to_idx = {word: idx + 1 for idx, word in enumerate(model.key_to_index)}  # +1 to reserve index 0 for padding
    embedding_matrix = np.zeros((len(word_to_idx) + 1, embedding_dim))  # idx 0 is reserved for padding
    for word, idx in word_to_idx.items():
        embedding_matrix[idx] = model[word]

    return embedding_matrix, word_to_idx

class ReviewsDataset(Dataset):
    def __init__(self, df, word_to_idx, max_length=100):
        self.df = df
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.encoder = LabelEncoder().fit([1, 2, 3, 4, 5])  # Encode ratings into categories

    def _tokenize_and_pad(self, text):
        tokens = text.split()
        token_ids = [self.word_to_idx.get(token, 0) for token in tokens[:self.max_length]]  # 0 is for padding
        return token_ids + [0] * (self.max_length - len(token_ids))  # Pad to max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        summary = row['Summary']
        text = row['Text']
        combined_text = summary + " " + text
        tokens = self._tokenize_and_pad(combined_text)

        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        label_tensor = torch.tensor(self.encoder.transform([int(row['Score'])])[0], dtype=torch.long)

        return tokens_tensor, label_tensor

# Step 3: Model Definition
class GloveFineTuneModel(nn.Module):
    def __init__(self, embedding_matrix, num_classes=5, embedding_dim=300, hidden_dim=128):
        super(GloveFineTuneModel, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)  # Get GloVe embeddings
        x = x.mean(dim=1)  # Average pooling
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 4: Training Loop
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Validation step
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total}%")

# Step 5: Putting Everything Together
if __name__ == "__main__":
    print('Loading GloVe. . .')
    embedding_matrix, word_to_idx = load_glove_embeddings()
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Number of words in vocabulary: {len(word_to_idx)}")

    from preprocess_data import load_data
    train_df, _ = load_data()
    train_df = preprocess_data(train_df)
    train_df, val_df = get_train_val_split(train_df)

    print('Creating datasets. . .')
    train_dataset = ReviewsDataset(train_df, word_to_idx)
    val_dataset = ReviewsDataset(val_df, word_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print('Training. . .')
    model = GloveFineTuneModel(embedding_matrix)
    train_model(model, train_loader, val_loader)
