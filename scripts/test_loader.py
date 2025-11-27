# scripts/test_loader.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import HAM10000Dataset
from torchvision import transforms

def quick_test():
    """Quick test for development"""
    print("🧪 Quick Data Loader Test...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = HAM10000Dataset(
        csv_file='data/raw/HAM10000_metadata.csv',
        img_dir='data/',
        transform=transform
    )
    
    print(f"✅ Loaded {len(dataset)} samples")
    print(f"📊 Classes: {dataset.classes}")
    
    # Test first sample
    image, label = dataset[0]
    print(f"🎯 Sample 0: shape {image.shape}, label {label} ({dataset.classes[label]})")
    
    # Class distribution
    dist = dataset.get_class_distribution()
    print(f"📈 Class dist: nv={dist['nv']}, mel={dist['mel']}")
    
    print("✅ Quick test passed!")

if __name__ == '__main__':
    quick_test()