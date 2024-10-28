import torch
import numpy as np
import pandas as pd
from model_utils import initialize_environment

# Set device and seeds for reproducibility
device = initialize_environment(4)
xgb_random_seed = 4

# For reproducibility
import random
import os
import xgboost as xgb

random.seed(xgb_random_seed)
np.random.seed(xgb_random_seed)
os.environ['PYTHONHASHSEED'] = str(xgb_random_seed)
torch.manual_seed(xgb_random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(xgb_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
xgb.set_config(verbosity=1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load embeddings and labels
print("Loading embeddings and labels...")
train_data = np.load('train_embeddings.npz')
X_full = train_data['embeddings']
y_full = train_data['labels']

test_data = np.load('test_embeddings.npz')
X_test = test_data['embeddings']
test_ids = test_data['ids']

# Split the data
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=xgb_random_seed, stratify=y_full
)

# Convert labels to 0-based if necessary
if y_train.min() == 1:
    y_train -= 1
    y_val -= 1

# Configure and train the XGBoost classifier
print("Training the XGBoost classifier...")
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=5,  # Labels from 0 to 4
    tree_method='hist',
    device='cuda',
    random_state=xgb_random_seed,
    n_estimators=500,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    seed=xgb_random_seed,
    early_stopping_rounds=50,
    eval_metric='mlogloss',  # Monitor log loss for multi-class
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Evaluate on validation set
print("Evaluating on the validation set...")
y_val_pred = xgb_clf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")
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
y_test_pred_adjusted = y_test_pred + 1  # Adjust labels back to 1-5

# Generate submission file
print("Generating submission file...")
submission_df = pd.DataFrame({
    'Id': test_ids,
    'Score': y_test_pred_adjusted
})
submission_df.to_csv('submission_xgb.csv', index=False)
print("Submission file 'submission_xgb.csv' has been generated.")

# Print library versions
print("\nLibrary Versions:")
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"XGBoost: {xgb.__version__}")
print(f"Random Seed: {xgb_random_seed}")
