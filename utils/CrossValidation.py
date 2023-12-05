import torch
from models.cnn_model import Variant1CNN
from utils.CustomImageDataset import get_dataloader
from utils.train_eval import train_model
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Function to perform k-fold cross-validation
def crossValidation(model, dataset, k_folds=10, num_epochs=50, batch_size=32, learning_rate=0.001):
    # Initialize lists to store metrics for each fold
    all_accuracies, all_precisions, all_recalls, all_f1_scores = [], [], [], []

    # Create k-fold cross-validation splitter
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_index, val_index) in enumerate(kf.split(dataset.augmented_dataset, dataset.labels)):
        print(f'Fold {fold + 1}/{k_folds}')

        # Create data loaders for the current fold
        train_dataset_fold = torch.utils.data.Subset(dataset, train_index)
        val_dataset_fold = torch.utils.data.Subset(dataset, val_index)

        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False, num_workers=2)

        # Train the model on the current fold
        train_model(model, train_loader_fold, val_loader_fold, num_epochs, learning_rate)

        # Evaluate the model on the validation set for this fold
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in val_loader_fold:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.numpy())

        # Calculate metrics for this fold
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

        # Print or log metrics for this fold
        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}')

        # Store metrics for this fold
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)

    # Calculate and print the average metrics across all folds
    average_accuracy = sum(all_accuracies) / len(all_accuracies)
    average_precision = sum(all_precisions) / len(all_precisions)
    average_recall = sum(all_recalls) / len(all_recalls)
    average_f1 = sum(all_f1_scores) / len(all_f1_scores)

    print(f'Average Accuracy: {average_accuracy}, Average Precision: {average_precision}, '
          f'Average Recall: {average_recall}, Average F1-Score: {average_f1}')
