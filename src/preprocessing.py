"""
Preprocessing utilities for time series forecasting of respiratory-morbidity rates.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
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

def normalize_data(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Z-score normalization for selected columns. Returns normalized df and scaler.
    """
    if columns is None:
        columns = df.columns.tolist()
    if scaler is None:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    return df, scaler

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

def time_series_train_test_split(
    df: pd.DataFrame, 
    test_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/test sets, preserving time order.
    """
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test

def prepare_data_for_model(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int,
    forecast_horizon: int = 1,
    normalization: Optional[str] = 'zscore',
    feature_columns: Optional[List[str]] = None,
    test_size: int = 52,
    val_size: Optional[int] = None
) -> Dict:
    """
    Full pipeline: normalization, split, sequence creation.
    Returns dict with X_train, y_train, X_val, y_val, X_test, y_test, scaler, test_df.
    If feature_columns is None, uses all columns except 'CD_MUN' and 'week'.
    """
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in ['CD_MUN', 'week']]
    feature_columns = [c for c in feature_columns if c != target_column]
    feature_columns = [target_column] + feature_columns
    scaler = None
    if normalization == 'zscore':
        df, scaler = normalize_data(df.copy(), columns=feature_columns)
    # Split train/test
    train_df, test_df = time_series_train_test_split(df, test_size)
    # Validation split from train
    if val_size is None:
        val_size = max(1, int(0.1 * len(train_df)))  # 10% of train, at least 1
    if val_size > 0 and len(train_df) > val_size + sequence_length + forecast_horizon:
        val_df = train_df.iloc[-val_size:]
        train_df = train_df.iloc[:-val_size]
        X_val, y_val = create_sequences(
            val_df[feature_columns].values, sequence_length, forecast_horizon, target_mode='sum')
    else:
        X_val, y_val = None, None
    # Sequences
    X_train, y_train = create_sequences(
        train_df[feature_columns].values, sequence_length, forecast_horizon, target_mode='sum')
    X_test, y_test = create_sequences(
        test_df[feature_columns].values, sequence_length, forecast_horizon, target_mode='sum')
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'test_df': test_df,
        'feature_columns': feature_columns
    }
