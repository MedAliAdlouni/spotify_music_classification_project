# import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sentence_transformers import SentenceTransformer

def load_and_preprocess_numerical_and_textual_data(data_path, separator='tab'):
    if separator == 'comma':
        separator = ','
    if separator == 'tab':
        separator = '\t'

    # Load the dataset
    df = pd.read_csv(data_path, sep=separator, encoding='utf-8')

    # --- Preprocess textual data ---
    # Extract and preprocess the textual data
    y_label = df['playlist_genre']
    y = y_label.astype('category').cat.codes

    # Combine 'track_album_name' and 'track_name' into a single sentence column
    df['combined_text'] = df['track_album_name'] + ' ' + df['track_name']
    
    # Initialize the Sentence-BERT model
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model

    # Apply Sentence-BERT to each sentence (row) in the combined_text column
    textual_embeddings = sentence_model.encode(df['combined_text'].fillna(''))  # Handle NaNs if any

    # Convert the embeddings into a numpy array
    X_textual_embeddings = np.array(textual_embeddings)

    # --- Preprocess numerical data ---
    # Convert 'track_album_release_date' to datetime and extract year
    df['date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce', format='mixed')
    df['year'] = df['date'].dt.year

    # Select numerical columns for processing
    X_numerical = df.select_dtypes(exclude='object')

    # Normalize the numerical data (scale features between 0 and 1)
    scaler = StandardScaler()
    X_to_normalize = X_numerical.drop(columns=['mode', 'date'])
    X_normalized = scaler.fit_transform(X_to_normalize)

    # Concatenate the 'mode' column back to X_normalized
    mode_column = X_numerical[['mode']].values  # Convert 'mode' column to numpy array
    X_numerical_final = np.hstack((X_normalized, mode_column))  # Concatenate along columns

    # Combine both textual and numerical data into a final feature set
    X_combined = np.hstack((X_textual_embeddings, X_numerical_final))

    # Print the shapes of the processed data
    print(f"Shape of X (combined data): {X_combined.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Shape of y_label: {y_label.shape}")

    return X_combined, y, y_label

def load_and_preprocess_textual_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
    y_label = df['playlist_genre']
    y = y_label.astype('category').cat.codes

    # Combine 'track_album_name' and 'track_name' into a single sentence column
    df['combined_text'] = df['track_album_name'] + ' ' + df['track_name']

    # Select the 'combined_text' column for transformation
    X_textual = df[['combined_text']]  # Now we have a single textual column

    # Initialize the Sentence-BERT model
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model

    # Apply Sentence-BERT to each sentence (row) in the combined_text column
    textual_embeddings = sentence_model.encode(X_textual['combined_text'].fillna(''))  # Handle NaNs if any

    # Convert the embeddings into a numpy array
    X_textual_embeddings = np.array(textual_embeddings)
    scaler = StandardScaler()
    X_textual_embeddings = scaler.fit_transform(X_textual_embeddings)

    # Print the shape of the processed features
    print(f"Shape of X (textual data): {X_textual_embeddings.shape}")

    return X_textual_embeddings, y, y_label

def load_and_preprocess_numerical_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
    df['date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce', format='mixed')
    df['year'] = df['date'].dt.year

    X = df.select_dtypes(exclude='object')

    y_label = df['playlist_genre']
    y = y_label.astype('category').cat.codes

    # Normalize the input data (scale features between 0 and 1)
    scaler = StandardScaler()
    X_to_normalize = X.drop(columns=['mode', 'date'])
    X_normalized = scaler.fit_transform(X_to_normalize)

    # Concatenate the 'mode' column back to X_normalized
    mode_column = X[['mode']].values  # Convert 'mode' column to numpy array
    X_final = np.hstack((X_normalized, mode_column))  # Concatenate along columns

    # Print the shapes
    print(f"Shape of X: {X_final.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Shape of y_label: {y_label.shape}")

    return X_final, y, y_label

    # # Convert the DataFrame to numpy arrays, then to torch tensors
    # X = torch.from_numpy(X_df.to_numpy()).type(torch.float)
    # y = torch.from_numpy(y_df.to_numpy()).type(torch.float).squeeze()

    # # Split into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # return X_train, X_test, y_train, y_test, genres

