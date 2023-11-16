from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split


def get_dataloader(root, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root=root, transform=transform)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if train:
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return loader
