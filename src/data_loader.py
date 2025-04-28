import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

def load_dataset(data_path, separator='tab'):
    """
    Load the dataset from a CSV file with specified separator.
    
    Args:
        data_path (str): Path to the dataset.
        separator (str): Separator type ('tab' or 'comma').
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    sep = {'tab': '\t', 'comma': ','}.get(separator, separator)
    return pd.read_csv(data_path, sep=sep, encoding='utf-8')

def preprocess_labels(df, label_column='playlist_genre'):
    """
    Preprocess target labels by converting them to numerical codes.
    
    Args:
        df (pd.DataFrame): Dataset containing the label column.
        label_column (str): Name of the label column.
        
    Returns:
        tuple: Encoded labels (np.ndarray), original labels (pd.Series)
    """
    y_label = df[label_column]
    y = y_label.astype('category').cat.codes
    return y, y_label

def create_text_embeddings(text_data, model_name='all-MiniLM-L6-v2'):
    """
    Generate sentence embeddings from textual data using a SentenceTransformer.
    
    Args:
        text_data (pd.Series): Text data to embed.
        model_name (str): Name of the pretrained SentenceTransformer model.
        
    Returns:
        np.ndarray: Textual embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_data.fillna(''))
    return np.array(embeddings)

def preprocess_numerical_data(df, drop_columns=None):
    """
    Preprocess numerical features: extract year, normalize values, and concatenate columns if needed.
    
    Args:
        df (pd.DataFrame): Dataset containing numerical data.
        drop_columns (list, optional): Columns to exclude from normalization.
        
    Returns:
        np.ndarray: Processed numerical features.
    """
    # Parse release date and extract release year
    df['date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce', format='mixed')
    df['year'] = df['date'].dt.year

    numerical_df = df.select_dtypes(exclude='object')
    
    if drop_columns:
        X_to_normalize = numerical_df.drop(columns=drop_columns)
    else:
        X_to_normalize = numerical_df

    # Standardize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_to_normalize)

    # Optionally, concatenate the 'mode' column back
    if drop_columns and 'mode' in drop_columns:
        mode_column = numerical_df[['mode']].values
        X_final = np.hstack((X_normalized, mode_column))
    else:
        X_final = X_normalized

    return X_final

def load_and_preprocess_numerical_and_textual_data(data_path, separator='tab'):
    """
    Load and preprocess both numerical and textual features.
    
    Args:
        data_path (str): Path to the dataset.
        separator (str): Separator type for the CSV ('tab' or 'comma').
        
    Returns:
        tuple: Combined features (np.ndarray), encoded labels (np.ndarray), original labels (pd.Series)
    """
    df = load_dataset(data_path, separator)
    y, y_label = preprocess_labels(df)

    # Combine album and track name for text embedding
    df['combined_text'] = df['track_album_name'] + ' ' + df['track_name']
    X_textual = create_text_embeddings(df['combined_text'])

    X_numerical = preprocess_numerical_data(df, drop_columns=['mode', 'date'])

    # Merge textual and numerical features
    X_combined = np.hstack((X_textual, X_numerical))

    print(f"Shape of X (combined): {X_combined.shape}")
    print(f"Shape of y: {y.shape}")

    return X_combined, y, y_label

def load_and_preprocess_textual_data(data_path, separator='tab'):
    """
    Load and preprocess only textual features using Sentence-BERT embeddings.
    
    Args:
        data_path (str): Path to the dataset.
        separator (str): Separator type for the CSV ('tab' or 'comma').
        
    Returns:
        tuple: Textual embeddings (np.ndarray), encoded labels (np.ndarray), original labels (pd.Series)
    """
    df = load_dataset(data_path, separator)
    y, y_label = preprocess_labels(df)

    df['combined_text'] = df['track_album_name'] + ' ' + df['track_name']
    X_textual = create_text_embeddings(df['combined_text'])

    # Standardize textual embeddings
    scaler = StandardScaler()
    X_textual_scaled = scaler.fit_transform(X_textual)

    print(f"Shape of X (textual): {X_textual_scaled.shape}")

    return X_textual_scaled, y, y_label

def load_and_preprocess_numerical_data(data_path, separator='tab'):
    """
    Load and preprocess only numerical features.
    
    Args:
        data_path (str): Path to the dataset.
        separator (str): Separator type for the CSV ('tab' or 'comma').
        
    Returns:
        tuple: Numerical features (np.ndarray), encoded labels (np.ndarray), original labels (pd.Series)
    """
    df = load_dataset(data_path, separator)
    y, y_label = preprocess_labels(df)

    X_numerical = preprocess_numerical_data(df, drop_columns=['mode', 'date'])

    print(f"Shape of X (numerical): {X_numerical.shape}")
    print(f"Shape of y: {y.shape}")

    return X_numerical, y, y_label
