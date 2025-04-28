from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_function=nn.ReLU, batch_normalization=True, dropout_rate=0.5):

        super(NeuralNetwork, self).__init__()

        # Store parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate

        layers = []
        in_size = input_size

        # Hidden layers
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_units))
            if self.batch_normalization==True:
                layers.append(nn.BatchNorm1d(hidden_units))  # Add Batch Normalization
            layers.append(nn.ReLU())  # Activation function
            layers.append(nn.Dropout(p=self.dropout_rate))  # Dropout layer          
            in_size = hidden_units

        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        
        self.network = nn.Sequential(*layers)

        
    def forward(self, X):
        return self.network(X)


class XGBoostModel:
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        
    def save_model(self, filename):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a saved model from a file."""
        try:
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")


class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)

    def save_model(self, filename):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load a saved model from a file."""
        try:
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")
































# class NeuralNetworkModel:
#     def __init__(self, input_size, hidden_sizes=(100,), output_size=1, activation='relu', lr=0.001, **kwargs):
#         """
#         Initialize the Neural Network model with custom hidden layers.
#         Parameters are only set at initialization and don't change after.
#         """
#         self.input_size = input_size
#         self.hidden_sizes = hidden_sizes
#         self.output_size = output_size
#         self.activation = activation
#         self.lr = lr
        
#         # Define the network layers based on user input
#         layers = []
#         prev_size = self.input_size
#         for h_size in self.hidden_sizes:
#             layers.append(nn.Linear(prev_size, h_size))
#             if self.activation == 'relu':
#                 layers.append(nn.ReLU())
#             prev_size = h_size
        
#         layers.append(nn.Linear(prev_size, self.output_size))
        
#         # Combine layers into a module
#         self.model = nn.Sequential(*layers)
        
#         # Initialize the optimizer and loss function
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
#         self.criterion = nn.BCEWithLogitsLoss() if self.output_size == 1 else nn.CrossEntropyLoss()

#     def fit(self, X_train, y_train, batch_size=64, epochs=10):
#         # Convert data to PyTorch tensors
#         X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#         y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        
#         # Create DataLoader for batching
#         dataset = TensorDataset(X_train_tensor, y_train_tensor)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
#         # Training loop
#         for epoch in range(epochs):
#             self.model.train()  # Set model to training mode
#             running_loss = 0.0
            
#             for X_batch, y_batch in dataloader:
#                 # Zero gradients
#                 self.optimizer.zero_grad()
                
#                 # Forward pass
#                 outputs = self.model(X_batch)
                
#                 # Compute the loss
#                 loss = self.criterion(outputs.squeeze(), y_batch)
                
#                 # Backward pass
#                 loss.backward()
                
#                 # Update weights
#                 self.optimizer.step()
                
#                 # Accumulate loss
#                 running_loss += loss.item()

#             print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")

#     def predict(self, X_test):
#         self.model.eval()  # Set model to evaluation mode
#         X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#         with torch.no_grad():  # Disable gradient computation
#             outputs = self.model(X_test_tensor)
#         return torch.sigmoid(outputs).round().numpy()  # For binary classification

#     def get_params(self, deep=True):
#         """
#         Return model's hyperparameters as a dictionary.
#         This is required for sklearn cloning.
#         """
#         return {
#             "input_size": self.input_size,
#             "hidden_sizes": self.hidden_sizes,
#             "output_size": self.output_size,
#             "activation": self.activation,
#             "lr": self.lr  # Make sure learning rate is included here
#         }

#     def set_params(self, **params):
#         """
#         Set model's hyperparameters.
#         This is required for sklearn cloning.
#         """
#         self.input_size = params['input_size']
#         self.hidden_sizes = params['hidden_sizes']
#         self.output_size = params['output_size']
#         self.activation = params['activation']
#         self.lr = params.get('lr', 0.001)  # Ensure that the learning rate is correctly set
        
#         # Instead of reinitializing, modify the model architecture directly.
#         self._update_model()

#     def __repr__(self):
#         """
#         Return a string representation of the model, required for cloning.
#         """
#         return f"NeuralNetworkModel(input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, output_size={self.output_size}, activation={self.activation}, lr={self.lr})"
