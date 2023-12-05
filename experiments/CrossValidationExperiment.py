import torch
from models.cnn_model import Variant1CNN
from utils.CustomImageDataset import get_dataloader
from utils.train_eval import train_model
from utils.evaluation_matrix import evaluate_model
from utils.CrossValidation import crossValidation

if __name__ == '__main__':
    # Define hyperparameters
    num_epochs = 50
    num_classes = 4
    learning_rate = 0.001

    # Load data
    train_loader = get_dataloader(root='../Dataset/Training', batch_size=32, train=True)
    val_loader = get_dataloader(root='../Dataset/Training', batch_size=32, train=False)
    test_loader = get_dataloader(root='../Dataset/Testing', batch_size=32, train=False)

    # Experiment 2: Variant 1 - Vary Number of Convolutional Layers
    print("\nExperiment 2: Variant 1 - Vary Number of Convolutional Layers")
    model2 = Variant1CNN()
    train_model(model2, train_loader, val_loader, num_epochs, learning_rate)
    crossValidation(model2, train_loader, num_epochs=num_epochs, learning_rate=learning_rate)





