import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from data_processor import DataProcessor, ReviewDataset, collate_fn
from dynamic_als import update_df_and_get_als
from preprocess_data import load_data
from fuzzy_ngram_matrix_factors_model import SentimentModel
from model_utils import initialize_environment

device = initialize_environment(4)
xgb_random_seed = 4

# For reproducibility
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load embeddings and labels
print("Loading embeddings and labels...")
train_data = np.load('train_embeddings.npz')
X_train = train_data['embeddings']
y_train = train_data['labels']

val_data = np.load('val_embeddings.npz')
X_val = val_data['embeddings']
y_val = val_data['labels']

test_data = np.load('test_embeddings.npz')
X_test = test_data['embeddings']
test_ids = test_data['ids']

# Train the XGBoost classifier
print("Training the XGBoost classifier...")
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=5,  # Assuming labels are from 0 to 4
    random_state=xgb_random_seed,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=1,
    seed=xgb_random_seed
)
xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

# Evaluate on validation set
print("Evaluating on the validation set...")
y_val_pred = xgb_clf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc}")
print(classification_report(y_val, y_val_pred))

# Save confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
labels = ['1', '2', '3', '4', '5']

def save_confusion_matrix(cm, labels, filename='xgb_confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on Validation Set')
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved as '{filename}'.")

save_confusion_matrix(cm, labels)

# Predict on test set
print("Generating predictions on the test set...")
y_test_pred = xgb_clf.predict(X_test)
y_test_pred_adjusted = y_test_pred + 1  # Adjust labels from 0-4 to 1-5

# Generate submission file
print("Generating submission file...")
submission_df = pd.DataFrame({
    'Id': test_ids,
    'Score': y_test_pred_adjusted
})
submission_df.to_csv('submission_xgb.csv', index=False)
print("Submission file 'submission_xgb.csv' has been generated.")

# For reproducibility, print versions of libraries used
print("\nLibrary Versions:")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"Random Seed: {xgb_random_seed}")
