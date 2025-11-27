import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import numpy as np
import os
import time
from datetime import datetime

from src.data.loader import HAM10000Dataset

class DermSavantTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Create output directory
        os.makedirs('models', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
    
    def _build_model(self):
        """Build and initialize the model"""
        print("🛠️ Building EfficientNet-B3 model...")
        
        # Load pre-trained EfficientNet
        model = models.efficientnet_b3(pretrained=True)
        
        # Replace the classifier for 7 classes
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 7)
        
        # Move model to appropriate device
        model = model.to(self.device)
        
        print(f"✅ Model built with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def _get_data_transforms(self):
        """Define data transformations for training and validation"""
        # Enhanced training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((300, 300)),  # Slightly larger for better details
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Simple validation transforms
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def prepare_data(self):
        """Prepare data loaders with proper splitting"""
        print("📊 Preparing data loaders...")
        
        train_transform, val_transform = self._get_data_transforms()
        
        # Create dataset
        dataset = HAM10000Dataset(
            csv_file=self.config.csv_file,
            img_dir=self.config.img_dir,
            transform=None  # We'll apply transforms after split
        )
        
        # Handle class imbalance - stratified split would be better, but random for now
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"📈 Training samples: {len(train_dataset):,}")
        print(f"📈 Validation samples: {len(val_dataset):,}")
        
        return self.train_loader, self.val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0:
                batch_acc = 100. * correct / total
                print(f'Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        return epoch_loss, epoch_acc, epoch_time
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate F1-score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return epoch_loss, epoch_acc, f1, all_preds, all_labels
    
    def train(self):
        """Main training loop"""
        print("🎯 Starting training...")
        self.prepare_data()
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, train_acc, train_time = self.train_epoch(epoch + 1)
            
            # Validation phase
            val_loss, val_acc, val_f1, val_preds, val_labels = self.validate_epoch(epoch + 1)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{self.config.epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"   LR: {current_lr:.6f} | Time: {train_time:.1f}s")
            print('-' * 60)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('models/best_model.pth')
                print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        self.save_model('models/final_model.pth')
        print(f"✅ Training completed! Best Val Acc: {best_val_acc:.2f}%")
    
    def save_model(self, path):
        """Save model weights and training info"""
        torch.save({
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'config': self.config.__dict__
        }, path)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', linewidth=2)
        ax1.plot(self.val_losses, label='Val Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', linewidth=2)
        ax2.plot(self.val_accuracies, label='Val Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3.plot(self.learning_rates, color='red', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        
        # Validation metrics
        epochs = range(1, len(self.val_accuracies) + 1)
        ax4.bar(epochs, self.val_accuracies, alpha=0.7, color='green')
        ax4.set_title('Validation Accuracy per Epoch', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Training plots saved to outputs/training_history.png")

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n🧪 Evaluating model...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Classification report
        print("\n📈 Classification Report:")
        print(classification_report(all_labels, all_preds, 
                                  target_names=self.train_loader.dataset.dataset.classes))
        
        # Confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.train_loader.dataset.dataset.classes,
                   yticklabels=self.train_loader.dataset.dataset.classes)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Confusion matrix saved to outputs/confusion_matrix.png")