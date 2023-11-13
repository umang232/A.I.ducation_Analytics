import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='micro')

    # Print Confusion Matrix
    print("Confusion Matrix:")
    print(cm)

    # Print Metrics
    print("\nMetrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F1-Score: {micro_f1:.4f}")
