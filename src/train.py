"""
Training and evaluation routines for time series forecasting models.
"""
import numpy as np
import os
from typing import Dict, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def reduce_to_1d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel()
    # For sums, they should already be 1D from prepare_data_for_model or model output
    return arr.ravel() # Ensure 1D

# For deep learning models
try:
    import keras
except ImportError:
    keras = None

def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, patience=10, verbose=1, callbacks=None):
    """
    Train a Keras model.
    """
    if not (keras and isinstance(model, keras.Model)):
        raise TypeError("This train_model function is designed for Keras models.")

    # Define standard callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=max(1, patience // 2), 
        min_lr=1e-6
    )
    
    all_callbacks = [early_stopping, reduce_lr]
    if callbacks:
        all_callbacks.extend(callbacks)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=all_callbacks,
        verbose=verbose
    )
    return history

def evaluate_model(y_true, y_pred, scaler, target_idx=0):
    """
    Evaluate the model, handling denormalization before calculating metrics.

    Args:
        y_true (np.ndarray): True target values (normalized).
        y_pred (np.ndarray): Predicted target values (normalized).
        scaler (StandardScaler): The scaler fitted on the training data.
        target_idx (int): The index of the target column in the scaled data.

    Returns:
        A dictionary with metrics and denormalized true and predicted values.
    """
    y_true_1d = reduce_to_1d(y_true)
    y_pred_1d = reduce_to_1d(y_pred)

    # Denormalize predictions and true values
    num_features = scaler.n_features_in_
    
    true_dummy = np.zeros((len(y_true_1d), num_features))
    true_dummy[:, target_idx] = y_true_1d
    y_true_denorm = scaler.inverse_transform(true_dummy)[:, target_idx]

    pred_dummy = np.zeros((len(y_pred_1d), num_features))
    pred_dummy[:, target_idx] = y_pred_1d
    y_pred_denorm = scaler.inverse_transform(pred_dummy)[:, target_idx]

    # Calculate metrics on the denormalized data
    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))
    r2 = r2_score(y_true_denorm, y_pred_denorm)

    return {
        'mae': mae, 
        'rmse': rmse, 
        'r2': r2, 
        'y_true_denorm': y_true_denorm, 
        'y_pred_denorm': y_pred_denorm
    }

def generate_forecasts(model, X_test):
    """
    Generate forecasts for test set.
    """
    return model.predict(X_test)

def save_predictions(y_true, y_pred, dates, city_name, model_name, output_dir):
    """
    Save predictions to CSV.
    """
    import pandas as pd
    df = pd.DataFrame({'date': dates, 'y_true': y_true, 'y_pred': y_pred})
    out_path = os.path.join(output_dir, f'{city_name}_{model_name}_preds.csv')
    df.to_csv(out_path, index=False)
    return out_path

def save_metrics(metrics: Dict[str, Any], city_name, model_name, output_dir, params: Optional[Dict[str, Any]] = None):
    """
    Save metrics to CSV.
    """
    import pandas as pd
    row = {**metrics, 'city': city_name, 'model': model_name}
    if params:
        row.update(params)
    df = pd.DataFrame([row])
    out_path = os.path.join(output_dir, f'{city_name}_{model_name}_metrics.csv')
    df.to_csv(out_path, index=False)
    return out_path
