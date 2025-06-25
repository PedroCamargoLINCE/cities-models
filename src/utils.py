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
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 7),
    metrics: Optional[Dict[str, float]] = None
) -> Figure:
    """
    Plot actual values vs. forecasts and optionally display metrics.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(dates, y_true, color='royalblue', marker='.', linestyle='-', label='Actual')
    ax.plot(dates, y_pred, color='red', marker='.', linestyle='-', label='Forecast')
    
    plot_title = title if title else 'Forecast vs Actual'
    if metrics:
        metrics_text = f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.2f}"
        plot_title = f"{plot_title}\n{metrics_text}"
    
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.autofmt_xdate()
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
    
    # Ensure metrics are numeric, coercing errors to NaN
    metric_cols = ['mae', 'rmse', 'r2'] # Add other metrics if they exist
    for col in metric_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors='coerce')
            
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
    # Before grouping, ensure metrics are numeric and handle potential NaNs from coercion
    metric_cols_for_grouping = ['mae', 'rmse', 'r2'] # Ensure these are present and numeric
    
    # Drop rows where essential metrics for grouping might be NaN after coercion, if desired
    # For example, if 'mae' is critical for ranking:
    # grouped_summary = summary.dropna(subset=['mae'])
    # For now, we'll let .mean() handle NaNs by skipping them.
    
    grouped = summary.groupby(['city', 'model'])[metric_cols_for_grouping].mean().reset_index()
    
    # Format as markdown table
    table = ["| City | Model | MAE | RMSE | R² |", "| ---- | ----- | --- | ---- | -- |"]
    
    for _, row in grouped.iterrows():
        city = row['city']
        model = row['model']
        mae = row.get('mae', float('nan')) # .get() is good for safety if a column is missing
        rmse = row.get('rmse', float('nan'))
        r2 = row.get('r2', float('nan'))
        
        # Check if metrics are NaN before formatting, to avoid printing 'nan'
        mae_str = f"{mae:.4f}" if pd.notna(mae) else "N/A"
        rmse_str = f"{rmse:.4f}" if pd.notna(rmse) else "N/A"
        r2_str = f"{r2:.4f}" if pd.notna(r2) else "N/A"
        
        table.append(f"| {city} | {model} | {mae_str} | {rmse_str} | {r2_str} |")
    
    report.extend(table)
    report.append("")
    
    # Best model per city
    report.append("### Best Model Per City")
    report.append("")
    
    best_models = []
    cities = sorted(summary['city'].unique())
    
    for city in cities:
        city_data = summary[summary['city'] == city]
        if not city_data.empty and 'mae' in city_data.columns and city_data['mae'].notna().any():
            best_idx = city_data['mae'].idxmin() # This will fail if all 'mae' are NaN for a city
            best_model = city_data.loc[best_idx, 'model']
            best_mae = city_data.loc[best_idx, 'mae']
            best_rmse = city_data.loc[best_idx, 'rmse'] # Could also be NaN
            
            mae_str = f"{best_mae:.4f}" if pd.notna(best_mae) else "N/A"
            rmse_str = f"{best_rmse:.4f}" if pd.notna(best_rmse) else "N/A"
            
            best_models.append(f"- **{city}**: {best_model} (MAE: {mae_str}, RMSE: {rmse_str})")
        else:
            best_models.append(f"- **{city}**: No valid MAE data to determine best model")
            
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
    scaler=None, # Scaler for X and y if they were scaled *before* creating sequences
    target_idx: int = 0, # Index of target variable for denormalization if y_test/y_pred are multi-dim post-scaling
    fit_params: dict = None,
    predict_params: dict = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluate a model using time series cross-validation (TimeSeriesSplit).
    Uses the refactored evaluate_model for metric calculation.

    Args:
        model_builder: Function that returns a new, untrained model instance.
        X: Input features (numpy array, pre-sequencing).
        y: Target values (numpy array, pre-sequencing).
        n_splits: Number of splits for TimeSeriesSplit.
        scaler: Optional scaler object (e.g., StandardScaler) that was fit on the target variable *before* it was windowed/sequenced.
                This scaler is used to denormalize y_test and y_pred for metric calculation.
        target_idx: Index of the target variable if the scaler was fit on a multi-feature array.
        fit_params: Dict of parameters for model.fit().
        predict_params: Dict of parameters for model.predict().
        random_state: Random seed (currently not used by TimeSeriesSplit directly).
        
    Returns:
        DataFrame with metrics for each split.
    """
    # Ensure predict_params is a dictionary
    if predict_params is None:
        predict_params = {}
    # Ensure forecast_horizon is set in predict_params, defaulting to 1
    if 'forecast_horizon' not in predict_params:
        predict_params['forecast_horizon'] = 1

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    split_num = 1
    
    # Assuming X and y are already prepared (e.g., windowed sequences)
    # If X and y are raw time series, they need to be processed into sequences inside the loop
    # For now, this function assumes X and y are ready for model.fit/predict

    for train_idx, test_idx in tscv.split(X): # X here should be the windowed X_full
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx] # y_full corresponding to X_full
        
        model = model_builder()
        
        if fit_params is None:
            fit_params = {}
        
        # Fit the model
        # Keras models might need verbose=0 in fit_params for cleaner output during CV
        if hasattr(model, 'fit') and 'epochs' in model.fit.__code__.co_varnames: # Basic check for Keras-like model
            if 'verbose' not in fit_params:
                 fit_params_cv = {**fit_params, 'verbose': 0}
            else:
                 fit_params_cv = fit_params
            model.fit(X_train, y_train, **fit_params_cv)
        else:
            model.fit(X_train, y_train, **fit_params) # For sklearn-like models
        
        # Make predictions
        y_pred = model.predict(X_test, **predict_params)
        
        # y_test and y_pred are from the model (potentially scaled if model outputs scaled values
        # or if y_train/y_test were scaled versions of the original target sums).
        # The 'scaler' argument should be the one fit on the original target sums if normalization was applied
        # *before* y_train/y_test were formed.
        
        # Use the main evaluate_model function
        # The scaler here is tricky. If y_test/y_pred are already on the original scale of the target (e.g. sum of next N weeks),
        # then scaler should be None. If they are on a normalized scale, then scaler should be the one
        # used to normalize that target representation.
        metrics = evaluate_model(y_test, y_pred, scaler=scaler, target_idx=target_idx)
        
        results.append({
            'split': split_num,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2']
        })
        split_num += 1
        
    return pd.DataFrame(results)


def plot_target_distribution(
    df: pd.DataFrame, 
    target_column: str, 
    title: str = "Target Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot the distribution of the target variable.
    
    Args:
        df: DataFrame containing the target column
        target_column: Name of the target column
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram with KDE
    sns.histplot(df[target_column], kde=True, bins=30, color='royalblue', ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(target_column)
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_autocorrelation(
    df: pd.DataFrame, 
    target_column: str, 
    title: str = "Target Autocorrelation",
    figsize: Tuple[int, int] = (12, 8),
    lags: int = 40
) -> Figure:
    """
    Plot autocorrelation and partial autocorrelation functions.
    
    Args:
        df: DataFrame containing the target column
        target_column: Name of the target column
        title: Plot title
        figsize: Figure size
        lags: Number of lags to plot    
    
    Returns:
        Matplotlib Figure object
    """
    try:
        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        print("Warning: statsmodels not available. Cannot create autocorrelation plots.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'statsmodels not available\nfor autocorrelation plots', 
                ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Autocorrelation function
    plot_acf(df[target_column].dropna(), ax=ax1, lags=lags, title=f'{title} - ACF')
    
    # Partial autocorrelation function
    plot_pacf(df[target_column].dropna(), ax=ax2, lags=lags, title=f'{title} - PACF')
    
    plt.tight_layout()
    return fig


def plot_time_series_decomposition(
    df: pd.DataFrame, 
    target_column: str, 
    model: str = 'additive', 
    period: int = 52,
    title: str = "Time Series Decomposition",
    figsize: Tuple[int, int] = (12, 10)
) -> Figure:
    """
    Plot time series decomposition (trend, seasonal, residual).
    
    Args:
        df: DataFrame containing the target column
        target_column: Name of the target column
        model: Type of decomposition ('additive' or 'multiplicative')
        period: Period for seasonal decomposition
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        print("Warning: statsmodels not available. Cannot create decomposition plots.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'statsmodels not available\nfor time series decomposition', 
                ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig
    
    # Ensure we have enough data for decomposition
    if len(df) < 2 * period:
        print(f"Warning: Not enough data for decomposition (need at least {2*period} points, have {len(df)})")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Not enough data for decomposition\n(need at least {2*period} points)', 
                ha='center', va='center', fontsize=14)
        ax.set_title(title)
        return fig
    
    # Perform decomposition
    decomposition = seasonal_decompose(df[target_column].dropna(), model=model, period=period)
    
    # Create plots
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    decomposition.observed.plot(ax=axes[0], title=f'{title} - Original')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    metrics: Optional[Dict[str, float]] = None
) -> Figure:
    """
    Create a scatter plot of actual vs. predicted values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='black', lw=2)
    
    plot_title = title if title else 'Actual vs Predicted'
    if metrics:
        metrics_text = f"MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.2f}"
        plot_title = f"{plot_title}\n{metrics_text}"

    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig
