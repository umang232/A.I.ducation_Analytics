import os
from utils.dataloader import get_dataloader
from PIL.Image import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import random


class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None, train=True, attribute_filter=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.attribute_filter = attribute_filter
        self.dataset = ImageFolder(root=self.root, transform=None)  # Use a temporary transform for dynamic augmentation

        # Imagenet standards
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Define age and gender distribution targets
        target_distribution = {'young_male': 500, 'young_female': 500, 'middle-aged_male': 500,
                               'middle-aged_female': 500,
                               'senior_male': 500, 'senior_female': 500}

        # Group images by age and gender
        self.image_groups = {}
        for img_path, label in self.dataset.samples:
            age, gender = self.extract_attributes(os.path.basename(img_path))
            group_key = f'{age}_{gender}'

            if group_key not in self.image_groups:
                self.image_groups[group_key] = []

            self.image_groups[group_key].append((img_path, label))

        # Augment the dataset to meet the target distribution
        self.augmented_dataset = self.balance_dataset(target_distribution)

        # Apply attribute filter if specified
        if self.attribute_filter:
            self.augmented_dataset = [entry for entry in self.augmented_dataset if self.attribute_filter in entry[0]]

    def extract_attributes(self, filename):
        parts = filename.split('_')
        age = parts[0]  # Assuming age is an integer
        gender = parts[1]
        return age, gender

    def balance_dataset(self, target_distribution):
        augmented_dataset = []

        for target_class, target_count in target_distribution.items():
            group_key = target_class.split('_')
            age, gender = group_key[0], group_key[1]

            if len(self.image_groups[group_key]) < target_count:
                # Calculate the required number of additional images for augmentation
                num_additional_images = target_count - len(self.image_groups[group_key])

                # Randomly sample existing images to augment the dataset
                sampled_entries = random.sample(self.image_groups[group_key], num_additional_images)

                # Apply the specified augmentations to the sampled entries
                for img_path, label in sampled_entries:
                    augmented_dataset.append((img_path, label))

        return augmented_dataset

    def __len__(self):
        return len(self.augmented_dataset)

    def __getitem__(self, index):
        img_path, label = self.augmented_dataset[index]

        # Apply the actual transformations to the image
        img = self.load_and_apply_transform(img_path)

        return img, label

    def load_and_apply_transform(self, img_path):
        # Load the image using torchvision transforms
        img = transforms.ToTensor()(
            transforms.Grayscale(num_output_channels=1)(transforms.Resize((224, 224))(Image.open(img_path))))

        if self.transform is not None:
            img = self.transform(img)

        return img

    def get_dataloader(root, batch_size=32, train=True, attribute_to_filter=''):
        # Use the same data loader code, but replace ImageFolder with CustomImageDataset
        train_loader = get_dataloader(root, batch_size=batch_size, train=True, custom_dataset=CustomImageDataset,
                                      attribute_filter=attribute_to_filter)
        val_loader = get_dataloader(root, batch_size=batch_size, train=False, custom_dataset=CustomImageDataset,
                                    attribute_filter=attribute_to_filter)

