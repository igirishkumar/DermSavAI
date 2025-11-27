import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.trainer import DermSavantTrainer
from configs.training_config import TrainingConfig

def main():
    print("=" * 70)
    print("🚀 DermSavant AI - Model Training")
    print("=" * 70)
    
    # Initialize config and trainer
    config = TrainingConfig()
    trainer = DermSavantTrainer(config)
    
    # Start training
    trainer.train()
    
    # Generate plots and evaluation
    trainer.plot_training_history()
    trainer.evaluate_model()
    
    print("🎉 Training completed successfully!")
    print("📁 Models saved in: models/")
    print("📊 Plots saved in: outputs/")

if __name__ == '__main__':
    main()