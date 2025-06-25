"""
Preprocessing utilities for time series forecasting of respiratory-morbidity rates.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional, List

def load_city_data(filepath: str) -> pd.DataFrame:
    """
    Load a city's data from CSV. Assumes columns: 'CD_MUN', 'week', 'target', plus features.
    Sorts by 'CD_MUN' and 'week' if present. No date parsing.
    """
    df = pd.read_csv(filepath)
    sort_cols = []
    if 'CD_MUN' in df.columns:
        sort_cols.append('CD_MUN')
    if 'week' in df.columns:
        sort_cols.append('week')
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df

def load_all_cities(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all city CSVs in a directory. Returns dict: city_name -> DataFrame.
    """
    city_dfs = {}
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv'):
            city = os.path.splitext(fname)[0]
            city_dfs[city] = load_city_data(os.path.join(data_dir, fname))
    return city_dfs

def filter_city(df: pd.DataFrame, cd_mun: Optional[int] = None) -> pd.DataFrame:
    """
    Filter DataFrame for a specific city by CD_MUN. If cd_mun is None, returns all data.
    """
    if cd_mun is not None and 'CD_MUN' in df.columns:
        return df[df['CD_MUN'] == cd_mun].copy()
    return df

def clean_timeseries(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean time series data:
    - Forward fill NaNs in target and features
    - Optionally, drop leading/trailing NaNs
    - Optionally, treat zeros as missing for small cities (if desired)
    """
    # Forward fill NaNs
    df = df.sort_values(['CD_MUN', 'week']) if 'CD_MUN' in df.columns and 'week' in df.columns else df
    df = df.ffill().bfill()
    # Optionally, treat zeros as missing for target (if city is small)
    # Uncomment if desired:
    # df.loc[df[target_column] == 0, target_column] = np.nan
    # df[target_column] = df[target_column].interpolate().ffill().bfill()
    return df

def create_sequences(
    data: np.ndarray, 
    seq_length: int, 
    forecast_horizon: int = 1,
    target_mode: str = 'sum'  # 'sum' for sum of next horizon, 'last' for last value
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input/output sequences for time series models.
    If target_mode == 'sum', y is the sum of the next forecast_horizon steps.
    If target_mode == 'last', y is the value at t+seq_length+forecast_horizon-1 (legacy behavior).
    """
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        if target_mode == 'sum':
            y.append(np.sum(data[i+seq_length:i+seq_length+forecast_horizon, 0]))  # assumes target is first column
        else:
            y.append(data[i+seq_length+forecast_horizon-1, 0])
    return np.array(X), np.array(y)

def prepare_data_for_model(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    forecast_horizon: int = 1,
    test_size: int = 52,
    val_size: int = 52,
    feature_columns: Optional[List[str]] = None,
    target_mode: str = 'sum'
) -> Dict:
    """
    Full data preparation pipeline: split, scale, and create sequences.
    
    This function implements a leak-proof pipeline:
    1. Splits data chronologically into train, validation, and test sets.
    2. Fits a StandardScaler ONLY on the training data.
    3. Applies the fitted scaler to transform train, validation, and test sets.
    4. Creates input (X) and output (y) sequences for the model.

    Args:
        df (pd.DataFrame): The input dataframe with time series data.
        target_column (str): The name of the column to be predicted.
        sequence_length (int): The number of time steps in each input sequence.
        forecast_horizon (int): The number of time steps to forecast ahead.
        test_size (int): The number of samples for the test set.
        val_size (int): The number of samples for the validation set.
        feature_columns (Optional[List[str]]): List of columns to use as features. 
                                               If None, all columns except identifiers are used.
        target_mode (str): 'sum' or 'last'. Defines how the target `y` is created.

    Returns:
        Dict: A dictionary containing all data splits (X_train, y_train, etc.), 
              the fitted scaler, and the original data splits (for plotting/analysis).
    """
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in ['CD_MUN', 'week', 'city', 'state']]

    if target_column in feature_columns:
        feature_columns.remove(target_column)
    ordered_features = [target_column] + feature_columns
    data_df = df[ordered_features].copy()

    if len(data_df) < test_size + val_size + sequence_length:
        raise ValueError(f"Not enough data for given test/val/sequence sizes.")

    train_df = data_df.iloc[:-(test_size + val_size)]
    val_df = data_df.iloc[-(test_size + val_size):-test_size]
    test_df = data_df.iloc[-test_size:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    X_train, y_train = create_sequences(train_scaled, sequence_length, forecast_horizon, target_mode)
    X_val, y_val = create_sequences(val_scaled, sequence_length, forecast_horizon, target_mode)
    X_test, y_test = create_sequences(test_scaled, sequence_length, forecast_horizon, target_mode)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'feature_columns': ordered_features,
        'original_train_df': train_df,
        'original_val_df': val_df,
        'original_test_df': test_df,
    }
