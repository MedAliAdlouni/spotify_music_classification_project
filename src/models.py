from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import joblib

class NeuralNetwork(nn.Module):
    """
    A customizable feedforward neural network with optional batch normalization and dropout.
    """
    def __init__(self, input_size, hidden_layers, output_size, activation_function=nn.ReLU, batch_normalization=True, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()

        # Store network parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_function = activation_function
        self.batch_normalization = batch_normalization
        self.dropout_rate = dropout_rate

        layers = []
        in_size = input_size

        # Create hidden layers dynamically
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_units))
            if self.batch_normalization:
                layers.append(nn.BatchNorm1d(hidden_units))  # Apply Batch Normalization
            layers.append(self.activation_function())  # Apply activation function
            layers.append(nn.Dropout(p=self.dropout_rate))  # Apply Dropout
            in_size = hidden_units

        # Add the final output layer
        layers.append(nn.Linear(in_size, output_size))

        # Assemble layers into a Sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        """
        Forward pass through the network.
        """
        return self.network(X)

class XGBoostModel:
    """
    Wrapper class for XGBoost Classifier with added model persistence methods.
    """
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X_train, y_train):
        """
        Train the XGBoost model.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the trained XGBoost model.
        """
        return self.model.predict(X_test)
    
    def get_params(self, deep=True):
        """
        Get hyperparameters of the XGBoost model.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """
        Set hyperparameters for the XGBoost model.
        """
        self.model.set_params(**params)
        
    def save_model(self, filename):
        """
        Save the trained model to a file using joblib.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load a trained model from a file.
        """
        try:
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")

class RandomForestModel:
    """
    Wrapper class for RandomForest Classifier with added model persistence methods.
    """
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X_train, y_train):
        """
        Train the Random Forest model.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the trained Random Forest model.
        """
        return self.model.predict(X_test)

    def get_params(self, deep=True):
        """
        Get hyperparameters of the Random Forest model.
        """
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """
        Set hyperparameters for the Random Forest model.
        """
        self.model.set_params(**params)

    def save_model(self, filename):
        """
        Save the trained model to a file using joblib.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load a trained model from a file.
        """
        try:
            self.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")
