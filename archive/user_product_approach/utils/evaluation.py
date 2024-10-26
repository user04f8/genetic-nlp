import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for *model_inputs, score in data_loader:
            model_inputs = [input_.to(device) for input_ in model_inputs]
            score = score.to(device)

            logits = model(*model_inputs)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == score).sum().item()
            total += score.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(score.cpu().numpy())

    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)

def print_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

def save_confusion_matrix_plot(true_labels, predictions, filename="confusion_matrix.png"):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(5), labels=[1, 2, 3, 4, 5], rotation=45)
    plt.yticks(np.arange(5), labels=[1, 2, 3, 4, 5])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()
