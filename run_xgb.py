import torch
import numpy as np
import pandas as pd

# Set device and seeds for reproducibility
from model_utils import initialize_environment

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
xgb.set_config(verbosity=2)

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

# Load embeddings and labels
print("Loading embeddings and labels...")
train_data = np.load('train_embeddings.npz')
X_full = train_data['embeddings']
y_full = train_data['labels']

test_data = np.load('test_embeddings.npz')
X_test = test_data['embeddings']
test_ids = test_data['ids']

# Convert labels to 0-based if necessary
if y_full.min() == 1:
    y_full -= 1

# Analyze class distribution
def plot_class_distribution(labels, title):
    classes, counts = np.unique(labels, return_counts=True)
    plt.bar(classes + 1, counts)  # Adjust if classes start from 0
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

print("Original class distribution:")
plot_class_distribution(y_full, 'Original Training Set Class Distribution')

# Oversample the minority classes
ros = RandomOverSampler(random_state=xgb_random_seed)
X_resampled, y_resampled = ros.fit_resample(X_full, y_full)

print("After oversampling:")
plot_class_distribution(y_resampled, 'Resampled Training Set Class Distribution')

# Split the data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=xgb_random_seed, stratify=y_resampled
)

# Prepare the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1.0, 2.0, 5.0],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

# Initialize the XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=5,  # Labels from 0 to 4
    tree_method='hist',
    device='cuda',
    random_state=xgb_random_seed,
    seed=xgb_random_seed,
    eval_metric='mlogloss',  # Monitor log loss for multi-class
    use_label_encoder=False
)

# Use RandomizedSearchCV for hyperparameter tuning
print("Starting hyperparameter tuning with RandomizedSearchCV...")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=xgb_random_seed)

random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=5,
    scoring='accuracy',
    cv=skf,
    verbose=2,
    random_state=xgb_random_seed,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"Best Hyperparameters: {random_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {random_search.best_score_:.4f}")

# Get the best estimator
best_xgb_clf = random_search.best_estimator_

# Train the best model on the training set
print("Training the best XGBoost classifier on the training set...")
best_xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=True
)

# Evaluate on validation set
print("Evaluating on the validation set...")
y_val_pred = best_xgb_clf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")
print(classification_report(y_val, y_val_pred, digits=4))

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
y_test_pred = best_xgb_clf.predict(X_test)
y_test_pred_adjusted = y_test_pred + 1  # Adjust labels back to 1-5

# Generate submission file
print("Generating submission file...")
submission_df = pd.DataFrame({
    'Id': test_ids,
    'Score': y_test_pred_adjusted
})
submission_df.to_csv('submission_xgb.csv', index=False)
print("Submission file 'submission_xgb.csv' has been generated.")
