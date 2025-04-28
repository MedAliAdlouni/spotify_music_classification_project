import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from data_loader import *
from models import *
from utils import *
from training import *
from config import *
import warnings
import logging
import pandas as pd
import os 

from sklearn.model_selection import ParameterGrid
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


def neural_network_tuning_main():    
    df_path = DATA_CONFIG["data_path"]
    X, y, y_label = load_and_preprocess_numerical_data(df_path)

    # Initialize the grid search using the param_grid imported from config.py
    grid = ParameterGrid(param_grid)
    # Get the total number of configurations
    total_configs = len(grid)
    # List to store results
    results = []

    # Iterate over all hyperparameter combinations
    for idx, params in enumerate(grid, start=1):
        print(f"Training configuration {idx}/{total_configs}: {params}")
        # Update the config dictionaries with the current hyperparameters
        MODEL_CONFIG_numerical["hidden_layers"] = params["hidden_layers"]
        MODEL_CONFIG_numerical["batch_normalization"] = params["batch_normalization"]
        MODEL_CONFIG_numerical["dropout_rate"] = params["dropout_rate"]
        
        TRAINING_CONFIG["learning_rate"] = params["learning_rate"]
        TRAINING_CONFIG["batch_size"] = params["batch_size"]
        TRAINING_CONFIG["epochs"] = params["epochs"]
        TRAINING_CONFIG["optimizer"] = params["optimizer"]
        TRAINING_CONFIG["clip_value"] = params["clip_value"]

        # Run the training
        print(f"Training with configuration: {params}")
        scores = neural_network_tuning(X, y)  # Train and evaluate the model
        
        # Store the results with the selected metrics (loss and accuracy)
        results.append({
            "hidden_layers": params["hidden_layers"],
            "batch_normalization": params["batch_normalization"],
            "dropout_rate": params["dropout_rate"],
            "learning_rate": params["learning_rate"],
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "optimizer": params["optimizer"],
            "clip_value": params["clip_value"],
            "loss": scores["loss"],
            "accuracy": scores["accuracy"]
        })

    # Convert results into a DataFrame for easier comparison
    results_df = pd.DataFrame(results)
    # Save the results to a CSV file
    results_df.to_csv('results/tuning_results.csv', index=False)
    print(results_df)
    # Sort the results based on the desired metric (e.g., accuracy or loss)
    best_results = results_df.sort_values(by="accuracy", ascending=False).iloc[0]  # For highest accuracy

    print("Best Hyperparameter Configuration:")
    print(best_results)


def neural_network_tuning(X, y):
    set_seed(42)
    MODEL_CONFIG = MODEL_CONFIG_numerical

    # Convert X and y into PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)  # Ensure X is in float32
    y = torch.tensor(y, dtype=torch.long)     # Ensure y is in long (for classification)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=DATA_CONFIG["test_size"], 
        random_state=DATA_CONFIG["random_state"]
    )

    # Initialize the model
    model = NeuralNetwork(
        input_size=MODEL_CONFIG["input_size"],
        hidden_layers=MODEL_CONFIG["hidden_layers"],
        output_size=MODEL_CONFIG["output_size"],
        batch_normalization=MODEL_CONFIG["batch_normalization"],
        dropout_rate=MODEL_CONFIG["dropout_rate"],
    )

    # Configure the loss function and optimizer
    criterion = getattr(nn, TRAINING_CONFIG["criterion"])()  # Dynamically get the criterion
    optimizer = getattr(optim, TRAINING_CONFIG["optimizer"])(
        model.parameters(), 
        lr=TRAINING_CONFIG["learning_rate"], 
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )

    # Training parameters
    batch_size = TRAINING_CONFIG["batch_size"]
    epochs = TRAINING_CONFIG["epochs"]
    gradient_clipping = TRAINING_CONFIG["gradient_clipping"]
    clip_value = TRAINING_CONFIG["clip_value"]

    # Initialize the trainer and start training
    trainer = PyTorchTrainer(
        model, criterion, optimizer, 
        X_train, y_train, 
        batch_size=batch_size, 
        gradient_clipping=gradient_clipping, 
        clip_value=clip_value,
    )
    trainer.train(epochs=epochs)
    scores = trainer.evaluate(X_val, y_val)
    #trainer.plot_losses()
    print(f"Evaluation Scores: {scores}")

    return scores


def neural_network_main(data="local", type =None):
    df_path, X, y, y_label, test_dataset_path, X_test, y_test, y_label_test, MODEL_CONFIG, model_path, y_pred = (None, None, None, None, None, None, None, None, None, None, None)
    set_seed(42)
    if (data == "local"):
    # Load and preprocess the data
        df_path = DATA_CONFIG["data_path"]
        test_dataset_path = DATA_CONFIG["test_dataset_path"]
    elif (data == "kaggle"):
        df_path = DATA_CONFIG["data_path_kaggle"]

    if (type == "numerical"):
        X, y, y_label = load_and_preprocess_numerical_data(df_path)
        X_test, y_test, y_label_test = load_and_preprocess_numerical_data(test_dataset_path)
        MODEL_CONFIG = MODEL_CONFIG_numerical
    elif (type== "textual"):
        X, y, y_label = load_and_preprocess_textual_data(df_path)
        X_test, y_test, y_label_test = load_and_preprocess_textual_data(test_dataset_path)
        MODEL_CONFIG = MODEL_CONFIG_textual
    elif (data == "local" and type =="numerical_and_textual"):
        X, y, y_label = load_and_preprocess_numerical_and_textual_data(df_path, separator='tab')
        X_test, y_test, y_label_test = load_and_preprocess_numerical_and_textual_data(test_dataset_path)
        MODEL_CONFIG = MODEL_CONFIG_numerical_and_textual
    elif (data == "kaggle"):
        X, y, y_label = load_and_preprocess_numerical_and_textual_data(df_path, separator='comma')
        X, X_test, y, y_test = train_test_split(X, y, test_size=DATA_CONFIG["test_size"], random_state=DATA_CONFIG["random_state"])
        MODEL_CONFIG = MODEL_CONFIG_numerical_and_textual_kaggle

    # Convert X and y into PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)  # Ensure X is in float32
    y = torch.tensor(y, dtype=torch.long)     # Ensure y is in long (for classification)
    X_test = torch.tensor(X_test, dtype=torch.float32)  # Ensure X is in float32
    y_test= torch.tensor(y_test, dtype=torch.long)     # Ensure y is in long (for classification)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=DATA_CONFIG["test_size"], 
        random_state=DATA_CONFIG["random_state"]
    )

    # Initialize the model
    model = NeuralNetwork(
        input_size=MODEL_CONFIG["input_size"],
        hidden_layers=MODEL_CONFIG["hidden_layers"],
        output_size=MODEL_CONFIG["output_size"],
        batch_normalization=MODEL_CONFIG["batch_normalization"],
        dropout_rate=MODEL_CONFIG["dropout_rate"],
    )

    # Load the model's state dict with strict=False to allow ignoring mismatched output layer
    if data == "kaggle":
        model_path = r"models/model_local_and_numerical_and_textual__data.pth"
        state_dict = torch.load(model_path)

        # Remove the last layer's weights (it should be named 'network.12.weight' and 'network.12.bias' based on your model)
        del state_dict['network.12.weight']
        del state_dict['network.12.bias']

        # Load the state_dict into the model, ignoring the last layer
        model.load_state_dict(state_dict, strict=False)

        # Replace the output layer with the new one (6 output classes)
        model.network[-1] = torch.nn.Linear(in_features=16, out_features=6)  # Adjust for new number of output classes

    # Configure the loss function and optimizer
    criterion = getattr(nn, TRAINING_CONFIG["criterion"])()  # Dynamically get the criterion
    optimizer = getattr(optim, TRAINING_CONFIG["optimizer"])(
        model.parameters(), 
        lr=TRAINING_CONFIG["learning_rate"], 
        weight_decay=TRAINING_CONFIG["weight_decay"]
    )

    # Training parameters
    batch_size = TRAINING_CONFIG["batch_size"]
    epochs = TRAINING_CONFIG["epochs"]
    gradient_clipping = TRAINING_CONFIG["gradient_clipping"]
    clip_value = TRAINING_CONFIG["clip_value"]

    # Initialize the trainer and start training
    trainer = PyTorchTrainer(
        model, criterion, optimizer, 
        X_train, y_train, 
        batch_size=batch_size, 
        gradient_clipping=gradient_clipping, 
        clip_value=clip_value,
        load_weights=model_path
    )
    trainer.train(epochs=epochs)
    scores = trainer.evaluate(X_val, y_val)
    trainer.plot_losses()
    print(f"Evaluation Scores: {scores}")

    if (data == "local" and type =="numerical_and_textual"):
        # Save the model
        model_path = r"models/model_local_and_numerical_and_textual__data.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved successfully at {model_path}!")

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        predicted_indices = y_pred.argmax(dim=1)
        label_mapping = {0: 'latin', 1: 'pop', 2: 'r&b', 3: 'rap', 4: 'rock'}
        # Map the predicted indices to the correct labels using the mapping
        y_pred_labels = [label_mapping[i] for i in predicted_indices.cpu().numpy()]
        test_scores = accuracy_score(y_test.numpy(), predicted_indices.numpy())
    
    print(f"Test Accuracy: {test_scores:.4f}")

    print(predicted_indices[:20])
    print(y_test[:20])
    print(y_pred_labels[:20])
    print(y_label_test[:20])
    # Save test results
    test_data = pd.read_csv(test_dataset_path, sep="\t")
    test_data['y_test'] = y_label_test
    test_data['y_pred'] = y_pred_labels
    result_file = "test_results.csv"
    test_data.to_csv(result_file, index=False)
    print(f"Test results saved to {result_file}")
    
    return scores, test_scores, y_pred_labels


def baseline_model_main():    
    df_path = os.path.join('data', 'raw', 'train.csv')
    X, y, y_label = load_and_preprocess_numerical_data(df_path)

    # Convert X and y into PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)  # Ensure X is in float32
    y = torch.tensor(y, dtype=torch.long)     # Ensure y is in long (for classification)
    # Print the types and shapes of the tensors
    print(f"X_tensor type: {X.type()} and shape: {X.shape}")
    print(f"y_tensor type: {y.type()} and shape: {y.shape}")

    models = {
        'XGBoost': XGBoostModel(colsample_bytree=0.9,learning_rate=0.1,max_depth=10,n_estimators=300,subsample=0.9,n_jobs=-1),
        'RandomForest': RandomForestModel(bootstrap=True,max_depth=20,min_samples_leaf=4,min_samples_split=5,n_estimators=300,random_state=42,n_jobs=-1)
    }
    models = {'XGBoost': XGBoostModel(colsample_bytree=0.9,learning_rate=0.1,max_depth=10,n_estimators=300,subsample=0.9,n_jobs=-1)}

    # Define hyperparameter grids for each model
    xgb_param_grid = {
        'learning_rate': [0.1,],
        'n_estimators': [100,],
        'max_depth': [5,],
        'subsample': [0.9,],
        'colsample_bytree': [0.9,]
    }
    rf_param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [5,20],
        'min_samples_split': [5,],
        'min_samples_leaf': [4,],
        'bootstrap': [True,]
    }
    param_grids = {
        'XGBoost': xgb_param_grid,
        'RandomForest': rf_param_grid
    }
    for model_name, model in models.items():
        param_grid = param_grids[model_name]
        trainer = ModelTrainer(model, X, y, param_grid=param_grid)
        accuracy, best_params, cv_results = trainer.cross_validate(cv=5, hyperparam_tuning=True)
        print(f'{model_name} cross-validation accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    baseline_model_main()
    #neural_network_main(data="local", type ="numerical_and_textual")  # data="local", type ="numerical_and_textual"  data="kaggle"  data="local", type ="numerical"  data="local", type ="textual"
    #neural_network_tuning_main()
