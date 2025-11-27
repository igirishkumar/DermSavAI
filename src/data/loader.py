import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_id = self.metadata.iloc[idx]['image_id']
        dx = self.metadata.iloc[idx]['dx']
        
        img_path = self._find_image_path(img_id)
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[dx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def _find_image_path(self, img_id):
        """
        Search for image in raw data directory
        """
        possible_paths = [
            os.path.join(self.img_dir, 'raw', 'HAM10000_images_part_1', img_id + '.jpg'),
            os.path.join(self.img_dir, 'raw', 'HAM10000_images_part_2', img_id + '.jpg'),
            os.path.join(self.img_dir, 'raw', img_id + '.jpg'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError(f"Image {img_id} not found in: {possible_paths}")
    
    def get_class_distribution(self):
        """Return class distribution for analysis"""
        return self.metadata['dx'].value_counts()
    
    def get_sample_info(self, idx):
        """Get additional sample information"""
        return self.metadata.iloc[idx].to_dict()