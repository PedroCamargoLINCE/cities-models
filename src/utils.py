"""
Utility functions for respiratory-morbidity forecasting models.

This module contains helper functions for:
- Plotting time series data and forecasts
- Custom evaluation metrics
- Helper functions for data processing
- File and results management
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import glob
from datetime import datetime
from matplotlib.figure import Figure


def plot_time_series(
    data: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    date_format: str = '%Y-%m-%d',
    color: str = 'royalblue'
) -> Figure:
    """
    Plot a time series.
    
    Args:
        data: DataFrame with time series data
        column: Column to plot
        title: Plot title
        figsize: Figure size
        date_format: Date format for x-axis ticks
        color: Line color
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the data
    ax.plot(data.index, data[column], color=color)
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(column, fontsize=12)
    
    # Format x-axis ticks
    ax.tick_params(axis='both', labelsize=10)
    plt.xticks(rotation=45)
    
    # Format dates
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_forecast(
    true_values: Union[pd.Series, np.ndarray],
    predictions: Union[pd.Series, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    true_label: str = 'Actual',
    pred_label: str = 'Forecast',
    true_color: str = 'royalblue',
    pred_color: str = 'crimson',
    include_metrics: bool = True,
    quantiles: Optional[Dict[str, np.ndarray]] = None
) -> Figure:
    """
    Plot actual values vs. forecasts.
    
    Args:
        true_values: Actual values
        predictions: Predicted values
        dates: Dates for x-axis (if None, uses indices)
        title: Plot title
        figsize: Figure size
        true_label: Label for actual values
        pred_label: Label for predictions
        true_color: Color for actual values
        pred_color: Color for predictions
        include_metrics: Whether to include metrics in the title
        quantiles: Optional dictionary of quantile predictions
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to arrays
    y_true = np.asarray(true_values).flatten()
    y_pred = np.asarray(predictions).flatten()
    
    # Use dates if provided, otherwise use indices
    x = dates if dates is not None else np.arange(len(y_true))
    
    # Plot quantiles first (if provided)
    if quantiles:
        for q_name, q_values in quantiles.items():
            q_values = np.asarray(q_values).flatten()
            ax.plot(x, q_values, linestyle='--', alpha=0.5, label=q_name)
        
        # Plot prediction intervals if p10 and p90 are provided
        if 'p10' in quantiles and 'p90' in quantiles:
            p10 = np.asarray(quantiles['p10']).flatten()
            p90 = np.asarray(quantiles['p90']).flatten()
            ax.fill_between(x, p10, p90, alpha=0.2, color='gray', label='80% Prediction Interval')
    
    # Plot actual and predicted values
    ax.plot(x, y_true, color=true_color, marker='o', markersize=4, label=true_label)
    ax.plot(x, y_pred, color=pred_color, marker='x', markersize=4, label=pred_label)
    
    # Calculate metrics
    if include_metrics:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics_text = f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}'
        if title:
            title = f'{title}\n{metrics_text}'
        else:
            title = metrics_text
    
    # Set title and labels
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date' if dates is not None else 'Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    # Format x-axis ticks if using dates
    if dates is not None:
        plt.xticks(rotation=45)
        fig.autofmt_xdate()
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_forecast_error(
    true_values: Union[pd.Series, np.ndarray],
    predictions: Union[pd.Series, np.ndarray],
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Forecast Error",
    figsize: Tuple[int, int] = (12, 6),
    color: str = 'crimson'
) -> Figure:
    """
    Plot forecast errors over time.
    
    Args:
        true_values: Actual values
        predictions: Predicted values
        dates: Dates for x-axis (if None, uses indices)
        title: Plot title
        figsize: Figure size
        color: Color for error line
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to arrays and calculate errors
    y_true = np.asarray(true_values).flatten()
    y_pred = np.asarray(predictions).flatten()
    errors = y_pred - y_true
    
    # Use dates if provided, otherwise use indices
    x = dates if dates is not None else np.arange(len(errors))
    
    # Plot errors
    ax.bar(x, errors, color=color, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(errors)
    
    metrics_text = f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, Bias: {bias:.2f}'
    title = f'{title}\n{metrics_text}'
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date' if dates is not None else 'Time Step', fontsize=12)
    ax.set_ylabel('Error (Predicted - Actual)', fontsize=12)
    
    # Format x-axis ticks if using dates
    if dates is not None:
        plt.xticks(rotation=45)
        fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_training_history(
    history: object,  # expects keras.callbacks.History or similar with .history attribute
    metrics: List[str] = ['loss'],
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Training History"
) -> Figure:
    """
    Plot training history for Keras models.
    
    Args:
        history: Keras History object from model.fit() (must have .history dict)
        metrics: List of metrics to plot
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    if not hasattr(history, 'history'):
        raise ValueError("history must be a Keras History object with a .history attribute")
    
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    
    # Convert to list if only one metric
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get training values
        values = history.history.get(metric, [])
        
        # Plot training values
        ax.plot(values, label=f'Training {metric}')
        
        # Plot validation values if available
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            val_values = history.history[val_metric]
            ax.plot(val_values, label=f'Validation {metric}')
        
        # Add title and labels
        if i == 0:
            ax.set_title(title, fontsize=14)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-label for bottom plot
    axes[-1].set_xlabel('Epoch', fontsize=12)
    
    plt.tight_layout()
    
    return fig


def create_results_summary(
    results_dir: str,
    pattern: str = "*_metrics.csv"
) -> pd.DataFrame:
    """
    Create a summary DataFrame of all results.
    
    Args:
        results_dir: Directory with result files
        pattern: Pattern to match result files
        
    Returns:
        DataFrame with results summary
    """
    # Find all result files
    files = glob.glob(os.path.join(results_dir, pattern))
    
    if not files:
        return pd.DataFrame()
    
    # Read and combine all files
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Concatenate all dataframes
    summary = pd.concat(dfs, ignore_index=True)
    
    return summary


def generate_report_markdown(results_dir: str) -> str:
    """
    Generate a Markdown report of all results.
    
    Args:
        results_dir: Directory with result files
        
    Returns:
        Markdown string with report
    """
    # Create summary DataFrame
    summary = create_results_summary(results_dir)
    
    if summary.empty:
        return "# No results found"
    
    # Start the report
    report = [
        "# Forecast Models Results Report",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
    ]
    
    # Add model comparison table
    report.append("### Model Performance Comparison")
    report.append("")
    
    # Group by city and model
    grouped = summary.groupby(['city', 'model']).mean().reset_index()
    
    # Format as markdown table
    table = ["| City | Model | MAE | RMSE | R² |", "| ---- | ----- | --- | ---- | -- |"]
    
    for _, row in grouped.iterrows():
        city = row['city']
        model = row['model']
        mae = row.get('mae', float('nan'))
        rmse = row.get('rmse', float('nan'))
        r2 = row.get('r2', float('nan'))
        
        table.append(f"| {city} | {model} | {mae:.4f} | {rmse:.4f} | {r2:.4f} |")
    
    report.extend(table)
    report.append("")
    
    # Best model per city
    report.append("### Best Model Per City")
    report.append("")
    
    best_models = []
    cities = sorted(summary['city'].unique())
    
    for city in cities:
        city_data = summary[summary['city'] == city]
        best_idx = city_data['mae'].idxmin()
        best_model = city_data.loc[best_idx, 'model']
        best_mae = city_data.loc[best_idx, 'mae']
        best_rmse = city_data.loc[best_idx, 'rmse']
        
        best_models.append(f"- **{city}**: {best_model} (MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f})")
    
    report.extend(best_models)
    
    # Join all lines
    return "\n".join(report)


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: List[float]
) -> Dict[str, float]:
    """
    Calculate pinball loss for quantile forecasts.
    
    Args:
        y_true: True values
        y_pred: Predicted quantiles (shape: n_samples x n_quantiles)
        quantiles: List of quantiles
        
    Returns:
        Dictionary of pinball losses for each quantile
    """
    # Ensure arrays are flat
    y_true = np.asarray(y_true).flatten()
    
    losses = {}
    
    # Calculate pinball loss for each quantile
    for i, q in enumerate(quantiles):
        y_pred_q = y_pred[:, i]
        error = y_true - y_pred_q
        
        # Apply pinball loss formula
        loss = np.mean(np.maximum(q * error, (q - 1) * error))
        
        losses[f'pinball_q{int(q*100)}'] = loss
    
    # Add average
    losses['pinball_avg'] = np.mean(list(losses.values()))
    
    return losses


def calculate_prediction_interval_coverage(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> float:
    """
    Calculate the prediction interval coverage.
    
    Args:
        y_true: True values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        
    Returns:
        Coverage ratio (0 to 1)
    """
    # Ensure arrays are flat
    y_true = np.asarray(y_true).flatten()
    lower_bound = np.asarray(lower_bound).flatten()
    upper_bound = np.asarray(upper_bound).flatten()
    
    # Count how many true values fall within the interval
    count = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
    
    # Calculate coverage
    coverage = float(count) / float(len(y_true))
    
    return coverage


def evaluate_model_time_series_split(
    model_builder,
    X,
    y,
    n_splits: int = 5,
    scaler=None,
    fit_params: dict = None,
    predict_params: dict = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluate a model using time series cross-validation (TimeSeriesSplit).
    
    Args:
        model_builder: Function that returns a new, untrained model instance (e.g., lambda: build_lstm(...))
        X: Input features (numpy array)
        y: Target values (numpy array)
        n_splits: Number of splits for TimeSeriesSplit
        scaler: Optional scaler for inverse transform
        fit_params: Dict of parameters for model.fit()
        predict_params: Dict of parameters for model.predict()
        random_state: Random seed
        
    Returns:
        DataFrame with metrics for each split
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    split_num = 1
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = model_builder()
        
        if fit_params is None:
            fit_params = {}
        if predict_params is None:
            predict_params = {}
        
        model.fit(X_train, y_train, **fit_params)
        y_pred = model.predict(X_test, **predict_params)
        
        # Inverse transform if scaler is provided
        if scaler is not None:
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_test_inv = y_test
            y_pred_inv = y_pred
        
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        results.append({
            'split': split_num,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
        
        split_num += 1
    
    return pd.DataFrame(results)
