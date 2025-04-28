import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
import torch
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy of XGBoost model: {accuracy}%")
    print(f"Precision of XGBoost model: {precision}%")
    print(f"Recall of XGBoost model: {recall}%")
    print(f"F1-score of XGBoost model: {f1}%")

def set_seed(seed=42):
    random.seed(seed)                # Python's random
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch CPU

