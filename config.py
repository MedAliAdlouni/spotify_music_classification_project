# Define the hyperparameter grid for the architecture and training parameters
param_grid = {
    'hidden_layers': [[64, 32], [16, 16, 16], [128, 16]],
    'batch_normalization': [True, False],
    'dropout_rate': [0.3, 0.7],
    'learning_rate': [1e-3, 1e-4],
    'batch_size': [16, 32],
    'epochs': [20],
    'optimizer': ['Adam', 'SGD'],
    'clip_value': [0.5, 1.5]
}

# param_grid = {
#     'hidden_layers': [[64, 32],],
#     'batch_normalization': [True,],
#     'dropout_rate': [0.3,],
#     'learning_rate': [1e-3,],
#     'batch_size': [16, 32],
#     'epochs': [10],
#     'optimizer': ['Adam',],
#     'clip_value': [0.5,]
# }

# Model architecture parameters
MODEL_CONFIG_numerical = {
    "input_size": 14,
    "hidden_layers": [128, 16],
    "output_size": 5,
    "batch_normalization": False,
    "dropout_rate": 0.3,
}   
MODEL_CONFIG_textual = {
    "input_size": 384,
    "hidden_layers": [64, 32, 16],
    "output_size": 5,
    "batch_normalization": True,
    "dropout_rate": 0.5,
}
MODEL_CONFIG_numerical_and_textual = {
    "input_size": 398,
    "hidden_layers": [64, 32, 16],
    "output_size": 5,
    "batch_normalization": True,
    "dropout_rate": 0.5,
}

MODEL_CONFIG_numerical_and_textual_kaggle = {
    "input_size": 398,
    "hidden_layers": [64, 32, 16],
    "output_size": 6,
    "batch_normalization": True,
    "dropout_rate": 0.5,
}

# Training parameters
TRAINING_CONFIG = {
    "criterion": "CrossEntropyLoss",  # Loss function
    "optimizer": "Adam",             # Optimizer
    "learning_rate": 0.001,          # Learning rate
    "weight_decay": 1e-4,            # L2 regularization
    "batch_size": 32,                # Batch size
    "epochs": 50,                    # Number of training epochs
    "gradient_clipping": True,       # Apply gradient clipping
    "clip_value": 1.0,               # Maximum gradient norm (if gradient clipping is True)
}

# Data parameters
DATA_CONFIG = {
    "data_path": r"data/raw/train.csv",
    "test_dataset_path": r"data/raw/test.csv",
    "data_path_kaggle": r"data/raw/spotify_songs.csv",
    "test_size": 0.2,       # Fraction of data for validation
    "random_state": 42,     # Random seed for reproducibility
}
