class TrainingConfig:
    def __init__(self):
        # Data paths
        self.csv_file = 'data/raw/HAM10000_metadata.csv'
        self.img_dir = 'data/'
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 30
        self.patience = 5  # Early stopping patience
        
        # Model settings
        self.model_name = 'efficientnet_b3'
        
        # Output paths
        self.model_save_path = 'models/dermsavant_model.pth'