"""
Data Registration Script
Registers the customer dataset to Hugging Face dataset hub.
"""

from datasets import Dataset, DatasetDict
import pandas as pd
from huggingface_hub import login
import os

# Login to Hugging Face (requires HF_TOKEN environment variable or user input)
# Uncomment and set your token if needed
# login(token="your_huggingface_token")

# Get your Hugging Face username from environment variable or command line argument
# Set this: set HF_USERNAME=your_huggingface_username
# Or pass as argument: python register_data.py your_username
import sys
HF_USERNAME = os.getenv('HF_USERNAME') or (sys.argv[1] if len(sys.argv) > 1 else 'your-username-here')

def register_dataset():
    """Register the customer dataset to Hugging Face."""
    
    if HF_USERNAME == 'your-username-here':
        print("ERROR: Please set HF_USERNAME environment variable or pass username as argument")
        print("Usage: python register_data.py your_huggingface_username")
        return
    
    # Load the raw dataset
    df = pd.read_csv('../data/tourism.csv')
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'raw': dataset
    })
    
    # Push to Hugging Face using your username
    dataset_id = f"{HF_USERNAME}/wellness-tourism-customers"
    dataset_dict.push_to_hub(dataset_id)
    
    print("Dataset successfully registered to Hugging Face!")
    print(f"Dataset ID: {dataset_id}")

if __name__ == "__main__":
    register_dataset()
