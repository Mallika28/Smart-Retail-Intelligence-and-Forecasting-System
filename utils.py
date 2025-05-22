import pandas as pd
import numpy as np

def load_and_preprocess(dataset_path):
    """
    Load and preprocess the retail dataset
    
    Parameters:
    dataset_path (str): Path to the dataset file
    
    Returns:
    pd.DataFrame: Preprocessed dataframe
    """
    df = pd.read_csv(dataset_path)
    
    # Handle missing values
    df['Item_Weight'] = df['Item_Weight'].interpolate(method='linear')
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, np.nan).interpolate(method='linear')
    
    # Feature engineering
    df['Item_Identifier'] = df['Item_Identifier'].apply(lambda x: x[:2])
    df['Outlet_age'] = 2025 - df['Outlet_Establishment_Year']
    
    # Standardize categorical values
    df.replace({
        'Item_Fat_Content': {
            'Low Fat': 'LF', 
            'low fat': 'LF', 
            'reg': 'Regular'
        }
    }, inplace=True)
    
    # Remove redundant features
    df.drop(['Outlet_Establishment_Year'], axis=1, inplace=True)
    
    return df

def format_currency(value):
    """
    Format a numeric value as Indian currency
    
    Parameters:
    value (float): Numeric value to format
    
    Returns:
    str: Formatted currency string
    """
    return f"${value:,.2f}"

def get_sales_status(value):
    """
    Determine the status of sales value
    
    Parameters:
    value (float): Sales value
    
    Returns:
    str: Status indicator ('low', 'medium', 'high')
    """
    if value < 1000:
        return "low"
    elif value > 3000:
        return "high"
    else:
        return "medium"
