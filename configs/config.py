class Config:
    PROJECT_NAME = "DermSavant AI"
    # Data
    CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # Model
    BACKBONE = "efficientnet_b3"
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001