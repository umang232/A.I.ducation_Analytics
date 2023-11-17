import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch

def git git git (root, batch_size=32, train=True):
    # Imagenet standards
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Train uses data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # Validation does not use augmentation
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # Apply transforms to the datasets directly
    dataset = ImageFolder(root=root, transform=train_transforms if train else test_transforms)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if train:
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return loader
