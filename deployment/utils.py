"""
Utility functions for the Wellness Tourism Package Prediction application.
This module provides helper functions for data preprocessing and model operations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_input(input_dict):
    """
    Preprocess user input data before making predictions.
    
    Args:
        input_dict (dict): Dictionary containing raw user input features
        
    Returns:
        pd.DataFrame: Preprocessed feature dataframe
    """
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Ensure all features are present and in correct order
    feature_order = [
        'age', 'gender', 'income', 'travel_frequency', 'previous_trips',
        'travel_spend', 'interest_adventure', 'interest_culture',
        'interest_relaxation', 'interest_wellness', 'loyalty_member',
        'marketing_emails_clicked', 'website_visits', 'last_booking_months'
    ]
    
    # Reorder columns
    df = df[feature_order]
    
    return df


def validate_input(input_dict):
    """
    Validate user input to ensure all required fields are present
    and values are within acceptable ranges.
    
    Args:
        input_dict (dict): Dictionary containing user input features
        
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    required_fields = [
        'age', 'gender', 'income', 'travel_frequency', 'previous_trips',
        'travel_spend', 'interest_adventure', 'interest_culture',
        'interest_relaxation', 'interest_wellness', 'loyalty_member',
        'marketing_emails_clicked', 'website_visits', 'last_booking_months'
    ]
    
    # Check all required fields are present
    for field in required_fields:
        if field not in input_dict:
            return False, f"Missing required field: {field}"
    
    # Validate ranges
    if not (18 <= input_dict['age'] <= 80):
        return False, "Age must be between 18 and 80"
    
    if not (20000 <= input_dict['income'] <= 200000):
        return False, "Income must be between $20,000 and $200,000"
    
    if not (0 <= input_dict['travel_frequency'] <= 10):
        return False, "Travel frequency must be between 0 and 10"
    
    if input_dict['gender'] not in [0, 1]:
        return False, "Gender must be 0 (Female) or 1 (Male)"
    
    return True, ""


def format_probability(probability):
    """
    Format probability value for display.
    
    Args:
        probability (float): Probability value between 0 and 1
        
    Returns:
        str: Formatted percentage string
    """
    return f"{probability * 100:.2f}%"


def get_feature_importance(model, feature_names):
    """
    Extract and format feature importance from the model.
    
    Args:
        model: Trained sklearn model
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        return feature_df
    else:
        return None
