import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np
from torch.utils.data import  random_split
# Define image transformations

class DatasetLoader:

    def __init__(self,base_dir):
        self.dataset_path = base_dir
    
    def loadDataset(self):     
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images to 128x128
            transforms.ToTensor(),          # Convert images to PyTorch tensors
           # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        # Load the dataset
        #dataset_path = 'C:\\Users\\ranmi\\dev\\GraphNeuralNet\\VisionTransformers\\brain_tumor_detection_transformers\\brain_tumor_dataset'
        dataset = datasets.ImageFolder(self.dataset_path, transform=transform)

        # Create DataLoader for batching and shuffling
        batch_size = 32
        # Define the split sizes
        train_size = int(0.9 * len(dataset))  # 90% for training
        test_size = len(dataset) - train_size  # Remaining 10% for testing
        train_loader,val_loader, test_loader = random_split(dataset, [train_size, test_size//2,test_size//2])

        train_data_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=11)
        val_data_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=True, num_workers=11)
        test_data_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True, num_workers=11)
        return train_data_loader , val_data_loader, test_data_loader
    # Display dataset info
    # print(f"Number of samples: {len(dataset)}")
    # print(f"Classes: {dataset.classes}")  # Class names
# 
# 
    # plt.ion()
    #image = np.transpose(data_loader.dataset[52][0], (1, 2, 0))  # New shape: (128, 128, 3)
    # image = (data_loader.dataset[0][0]).permute(1, 2, 0).numpy()
    # image_pil = ToPILImage()(image)
    # image_pil.show()

    # image_yes = (data_loader.dataset[100][0]).permute(1, 2, 0).numpy()
    # image_pil_yes = ToPILImage()(image_yes)
    # image_pil_yes.show()

    # # Iterate through the DataLoader
    # for images, labels in data_loader:
    #     print(f"Batch size: {images.size()}")  # Example: torch.Size([32, 3, 128, 128])
    #     print(f"Labels: {labels}")  # Class indices
    #     break
