# src/data/splitters.py
import torch
from torch.utils.data import DataLoader, random_split
from .loader import HAM10000Dataset

def create_data_loaders(csv_file, img_dir, batch_size=32, train_transform=None, val_transform=None, split_ratio=0.8):
    """
    Create train/val splits and data loaders
    
    Args:
        csv_file: Path to metadata CSV
        img_dir: Directory containing images  
        batch_size: Batch size for data loaders
        train_transform: Transformations for training
        val_transform: Transformations for validation
        split_ratio: Ratio of data to use for training
    
    Returns:
        tuple: (train_loader, val_loader, full_dataset)
    """
    # Create dataset
    dataset = HAM10000Dataset(csv_file, img_dir, transform=None)
    
    # Split dataset
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset