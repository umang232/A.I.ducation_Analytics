import torch
from models.cnn_model import CNN
from models.cnn_model import Variant1CNN
from models.cnn_model import Variant2CNN
from utils.dataloader import get_dataloader
from utils.train_eval import train_model, evaluate_model

if __name__ == '__main__':
    # Define hyperparameters
    num_epochs = 10
    num_classes = 4
    learning_rate = 0.001

    # Load data
    train_loader = get_dataloader(root='../Dataset/Training', batch_size=32, train=True)
    val_loader = get_dataloader(root='../Dataset/Training', batch_size=32, train=False)
    test_loader = get_dataloader(root='../Dataset/Testing', batch_size=32, train=False)

    # Experiment 1: Base Architecture
    print("Experiment 1: Base Architecture")
    model1 = CNN()
    train_model(model1, train_loader, val_loader, num_epochs, learning_rate)
    evaluate_model(model1, test_loader)
    torch.save(model1.state_dict(), '../saved_models/experiment1_model.pth')

    # Experiment 2: Variant 1 - Vary Number of Convolutional Layers
    print("\nExperiment 2: Variant 1 - Vary Number of Convolutional Layers")
    model2 = Variant1CNN()
    train_model(model2, train_loader, val_loader, num_epochs, learning_rate)
    evaluate_model(model2, test_loader)
    torch.save(model2.state_dict(), '../saved_models/experiment2_model.pth')

    # Experiment 3: Variant 2 - Experiment with Different Kernel Sizes
    print("\nExperiment 3: Variant 2 - Experiment with Different Kernel Sizes")
    model3 = Variant2CNN()
    train_model(model3, train_loader, val_loader, num_epochs, learning_rate)
    evaluate_model(model3, test_loader)
    torch.save(model3.state_dict(), '../saved_models/experiment3_model.pth')
