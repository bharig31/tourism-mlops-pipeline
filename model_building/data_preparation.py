"""
Data Preparation Script
Loads dataset from local file, performs cleaning, splits into train/test,
and uploads to Hugging Face.
"""

from datasets import Dataset, DatasetDict
import pandas as pd
from huggingface_hub import login
from sklearn.model_selection import train_test_split
import os
import sys

# Get your Hugging Face username from environment variable or command line argument
HF_USERNAME = os.getenv('HF_USERNAME') or (sys.argv[1] if len(sys.argv) > 1 else 'your-username-here')

# Login to Hugging Face if needed
# login(token="your_huggingface_token")

def prepare_data():
    """Prepare and split the dataset for model training."""
    
    if HF_USERNAME == 'your-username-here':
        print("ERROR: Please set HF_USERNAME environment variable or pass username as argument")
        print("Usage: python data_preparation.py your_huggingface_username")
        return
    
    # Load raw dataset from local file
    print("Loading dataset from local file...")
    df = pd.read_csv('../data/tourism.csv')
    
    # Data cleaning
    print("Cleaning data...")
    # Remove CustomerID as it's not useful for prediction
    df = df.drop('CustomerID', axis=1)
    
    # Convert Gender to numeric (Female=0, Male=1)
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    # Check for missing values
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    # Encode categorical columns
    print("Encoding categorical columns...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = pd.factorize(df[col])[0]
    
    # Separate features and target
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']
    
    # Split into train and test sets (80-20 split)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Combine features and target for train and test
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Create dataset dictionary
    prepared_dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    # Upload to Hugging Face
    print("Uploading prepared dataset to Hugging Face...")
    prepared_dataset.push_to_hub(f"{HF_USERNAME}/wellness-tourism-customers-prepared")
    
    print("Data preparation completed successfully!")
    print(f"Dataset ID: {HF_USERNAME}/wellness-tourism-customers-prepared")
    
    return prepared_dataset

if __name__ == "__main__":
    prepare_data()
