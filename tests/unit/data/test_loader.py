# tests/unit/data/test_loader.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pytest
from src.data.loader import HAM10000Dataset
from torchvision import transforms

class TestHAM10000Dataset:
    @pytest.fixture
    def dataset(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor()
        ])
        return HAM10000Dataset(
            csv_file='data/raw/HAM10000_metadata.csv',
            img_dir='data/',
            transform=transform
        )
    
    def test_dataset_length(self, dataset):
        assert len(dataset) == 10015
    
    def test_classes(self, dataset):
        expected_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        assert dataset.classes == expected_classes
    
    def test_class_distribution(self, dataset):
        class_dist = dataset.get_class_distribution()
        assert class_dist.sum() == 10015
        assert class_dist['nv'] == 6705
    
    def test_sample_loading(self, dataset):
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)
        assert 0 <= label < 7