import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_model(y_test, y_pred):
    """
    Evaluate the performance of a classification model using standard metrics.
    
    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
    
    Prints:
        Accuracy, Precision, Recall, and F1-Score of the model.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy of model: {accuracy:.4f}")
    print(f"Precision of model: {precision:.4f}")
    print(f"Recall of model: {recall:.4f}")
    print(f"F1-score of model: {f1:.4f}")

def set_seed(seed=42):
    """
    Set the random seed across NumPy, random, and PyTorch for reproducibility.
    
    Args:
        seed (int): Seed value to use (default is 42).
    """
    random.seed(seed)                # Python's built-in random module
    np.random.seed(seed)             # NumPy random generator
    torch.manual_seed(seed)          # PyTorch (CPU operations)

def plot_all_columns(data, columns):
    """
    Plot boxplots and distribution plots for all specified columns in the DataFrame.    
    
    """
    # Determine the number of rows and columns for the grid layout
    n_cols = 2  # Each feature will have two plots (boxplot and distplot)
    n_rows = len(columns)  # One row per column
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 5 * n_rows), 
                             gridspec_kw={'wspace': 0.4, 'hspace': 0.6})
    
    # Loop through each column and plot
    for i, col in enumerate(columns):
        sns.boxplot(data=data, x=col, ax=axes[i, 0])
        axes[i, 0].set_title(f'Boxplot of {col}')
        
        sns.histplot(data[col], kde=True, ax=axes[i, 1], color='#ff4125')  # Updated to `sns.histplot`
        axes[i, 1].set_title(f'Distribution of {col}')
    
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    plt.show()
