import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score


class PyTorchTrainer:
    def __init__(self, model, criterion, optimizer, X, y, batch_size=32, gradient_clipping=True, clip_value=1.0, load_weights=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # Targets must be integers for classification

        self.batch_size = batch_size
        self.gradient_clipping = gradient_clipping
        self.clip_value = clip_value
        self.best_val_loss = float('inf')  # Best validation loss seen so far
        self.patience = 10  # Early stopping patience
        self.epochs_no_improve = 0  # Number of epochs with no improvement

        # List to store historic loss values
        self.loss_history = []

        # Load weights if provided
        if load_weights:
            # Load the saved state_dict
            saved_state_dict = torch.load(load_weights)
            # Get the current model's state_dict
            model_state_dict = self.model.state_dict()
            
            # Filter out incompatible layers (e.g., last layer weights and biases)
            excluded_layers = [k for k in saved_state_dict.keys() if k not in model_state_dict or saved_state_dict[k].size() != model_state_dict[k].size()]
            compatible_state_dict = {k: v for k, v in saved_state_dict.items() if k not in excluded_layers}

            # Load compatible layers
            model_state_dict.update(compatible_state_dict)
            self.model.load_state_dict(model_state_dict)
            
            print(f"Loaded weights from {load_weights}, excluding incompatible layers: {excluded_layers}")

    def train(self, epochs=10, validation_data=None):
        self.model.train()
        dataset = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if validation_data:
            val_X, val_y = validation_data
            val_X = torch.tensor(val_X, dtype=torch.float32)
            val_y = torch.tensor(val_y, dtype=torch.long)

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                # Apply gradient clipping if enabled
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)

                self.optimizer.step()

                epoch_loss += loss.item()
                # Compute accuracy for the batch
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            accuracy = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Store the epoch loss in the history
            self.loss_history.append(epoch_loss)

            # Validate and check for early stopping
            if validation_data:
                val_loss, val_accuracy = self._validate(val_X, val_y)
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    print("Validation loss improved.")
                else:
                    self.epochs_no_improve += 1
                    print(f"No improvement in validation loss for {self.epochs_no_improve} epoch(s).")

                if self.epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    break
        
        # Return the loss history for plotting
        return self.loss_history

    def plot_losses(self):
        # Plot the loss history
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    def evaluate(self, X_test, y_test):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = self.model(X_test)
            loss = self.criterion(outputs, y_test).item()

            _, predicted = torch.max(outputs, dim=1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)

            # Calculate precision, recall, and F1 score
            precision = precision_score(y_test.cpu(), predicted.cpu(), average='weighted')
            recall = recall_score(y_test.cpu(), predicted.cpu(), average='weighted')
            f1 = f1_score(y_test.cpu(), predicted.cpu(), average='weighted')

        # Return all the metrics as a dictionary
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": predicted.cpu().numpy()  # Return predicted values as a numpy array
        }

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pth")
        print("Model saved successfully!")




class ModelTrainer:

    def __init__(self, model, X, y, param_grid=None):
        self.model = model
        self.X = X
        self.y = y
        self.param_grid = param_grid

    def cross_validate(self, cv=5, hyperparam_tuning=False):
        model_name = self.model.__class__.__name__.lower()

        if hyperparam_tuning:
            grid_search = GridSearchCV(estimator=self.model.model,param_grid=self.param_grid,cv=cv,scoring='accuracy',n_jobs=-1, return_train_score=True)
            grid_search.fit(self.X, self.y)
            # Save results and model
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            cv_results = grid_search.cv_results_
            self.save_results(model_name, best_params=best_params, best_score=best_score)
            self.plot_learning_curve(cv_results, model_name)
            
            return best_score, best_params, cv_results
        else:
            scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='accuracy', n_jobs=-1)
            return np.mean(scores)


    def plot_learning_curve(self, cv_results, model_name):
        # Plot the learning curve (training vs. validation loss for each fold)
        print(cv_results.keys())  # Print all the keys in cv_results
        mean_train_score = cv_results['mean_train_score']
        mean_test_score = cv_results['mean_test_score']
        
        # Create a 'results' folder if it doesn't exist
        results_dir = 'results'
        model_dir = os.path.join(results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Plot the learning curve
        plt.figure(figsize=(8, 6))
        plt.plot(mean_train_score, label='Training accuracy', color='blue')
        plt.plot(mean_test_score, label='Validation accuracy', color='orange')
        plt.xlabel('Fold number')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True)
    
        # Save the plot to the appropriate folder
        plot_filename = os.path.join(model_dir, f'learning_curve_{model_name}.png')
        plt.savefig(plot_filename)
        print(f"Learning curve saved as {plot_filename}")

    def save_results(self, model_name, best_params=None, best_score=None):
        """Save model and results in dedicated folders."""
        # Create directories if they do not exist
        model_folder = os.path.join('models', model_name)
        results_folder = os.path.join('results', model_name)
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        # Save best parameters
        if best_params is not None:
            param_filename = os.path.join(results_folder, 'best_params.json')
            with open(param_filename, 'w') as f:
                json.dump(best_params, f)

        # Save best accuracy
        if best_score is not None:
            accuracy_filename = os.path.join(results_folder, 'best_accuracy.json')
            with open(accuracy_filename, 'w') as f:
                json.dump({'best_accuracy': best_score}, f)

        # Save the best model
        model_path = os.path.join(model_folder, f'{model_name}_model.joblib')
        self.model.save_model(model_path)
