import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import yaml
import os
import re

def load_config():
    """Load configuration from config.yaml"""
    with open('src/config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_data(config):
    """Load the raw data from CSV file"""
    return pd.read_csv(config['data']['raw_data_path'])

def parse_remaining_lease(lease_str):
    """Convert remaining lease string to numeric years"""
    if pd.isna(lease_str):
        return np.nan
    
    # Extract years and months using regex
    match = re.search(r'(\d+)\s*years?\s*(?:(\d+)\s*months?)?', str(lease_str))
    if not match:
        return np.nan
    
    years = float(match.group(1))
    months = float(match.group(2)) if match.group(2) else 0
    
    return years + months/12

def calculate_remaining_lease(df):
    """Calculate remaining lease in years"""
    # If remaining_lease is already numeric, return as is
    if pd.api.types.is_numeric_dtype(df['remaining_lease']):
        return df
    
    # Convert string format to numeric years
    df['remaining_lease'] = df['remaining_lease'].apply(parse_remaining_lease)
    return df

def preprocess_data(df, config):
    """Preprocess the data including cleaning and feature engineering"""
    # Calculate remaining lease if needed
    df = calculate_remaining_lease(df)
    
    # Handle categorical variables
    le = LabelEncoder()
    categorical_columns = [
        'town_name',
        'flat_type',
        'storey_range',
        'flatm_name',
        'block'
    ]
    
    # Impute missing categorical values with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    for col in categorical_columns:
        if col in df.columns:
            # Fill missing values
            df[col] = cat_imputer.fit_transform(df[[col]]).ravel()
            # Encode categorical variables
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Select numerical features
    numerical_columns = [
        'floor_area_sqm',
        'lease_commence_date',
        'remaining_lease'
    ]
    
    # Impute missing numerical values with median
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_columns] = pd.DataFrame(
        num_imputer.fit_transform(df[numerical_columns]),
        columns=numerical_columns,
        index=df.index
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_columns] = pd.DataFrame(
        scaler.fit_transform(df[numerical_columns]),
        columns=numerical_columns,
        index=df.index
    )
    
    # Convert price_category to numeric
    df['price_category'] = (df['price_category'] == 'Above Median').astype(int)
    
    return df

def split_data(df, config):
    """Split data into train and test sets"""
    # Select features
    feature_columns = [
        'town_name',
        'flat_type',
        'storey_range',
        'flatm_name',
        'block',
        'floor_area_sqm',
        'lease_commence_date',
        'remaining_lease'
    ]
    
    # Filter feature columns that exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[feature_columns]
    y = df['price_category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['preprocessing']['test_size'],
        random_state=config['preprocessing']['random_state'],
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, config):
    """Save processed data to CSV files"""
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(config['data']['processed_data_path']), exist_ok=True)
    
    # Save processed data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(config['data']['train_data_path'], index=False)
    test_data.to_csv(config['data']['test_data_path'], index=False)

def main():
    """Main function to run the data preparation pipeline"""
    # Load configuration
    config = load_config()
    
    # Load and preprocess data
    df = load_data(config)
    df = preprocess_data(df, config)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, config)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, config)
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main() 