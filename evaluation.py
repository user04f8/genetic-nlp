import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

def evaluate_model(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])
    print(classification_report(y_true, y_pred))
    if not os.path.exists('diagrams'):
        os.makedirs('diagrams')
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('diagrams/confusion_matrix.png')
    plt.close()
