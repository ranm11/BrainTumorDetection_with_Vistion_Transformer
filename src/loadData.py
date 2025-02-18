import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
from torch.utils.data import  random_split
from torch.utils.data import ConcatDataset
# Define image transformations

class DatasetLoader:

    def __init__(self,base_dir):
        self.dataset_path = base_dir
    
    def loadDataset(self):     
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images to 128x128
            transforms.ToTensor(),          # Convert  to PyTorch tensors
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        #add data augmentation
        augmented_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert 3-channel grayscale to 1-channel
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Works for grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.15], std=[0.12])  # Adjusted for single-channel
        ])
        # Load the dataset
        original_dataset = datasets.ImageFolder(self.dataset_path, transform=transform)
        augmented_dataset = datasets.ImageFolder(self.dataset_path, transform=augmented_transform)
        # Create DataLoader for batching and shuffling
        batch_size = 32
        # Define the split sizes
        train_size = int(0.9 * len(original_dataset))  # 90% for training
        val_size = int((len(original_dataset) - train_size + 1 )//2)
        test_size = len(original_dataset) - train_size - val_size

        train_loader,val_loader, test_loader = random_split(original_dataset, [train_size, val_size,test_size])

        train_data_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=11,persistent_workers=True)
        val_data_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=True, num_workers=11,persistent_workers=True)
        test_data_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True, num_workers=11,persistent_workers=True)
        return train_data_loader , val_data_loader, test_data_loader , original_dataset ,augmented_dataset
    