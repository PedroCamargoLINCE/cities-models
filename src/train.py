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
    Train a model (supports sklearn-like and keras models).
    """
    if hasattr(model, 'fit') and keras and isinstance(model, keras.Model):
        cb = []
        if patience:
            cb.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
            cb.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1, patience//2), min_lr=1e-6, verbose=1))
        if callbacks:
            cb.extend(callbacks)
        # Always pass verbose as integer for progress bar
        # Keras 3+ expects verbose as str, but for progress bar use 'auto' or 1
        # We'll use 'auto' for best compatibility
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose='auto'  # Use 'auto' for progress bar in most environments
        )
        return history
    else:
        # For sklearn-like or baseline models
        model.fit(X_train, y_train)
        return None

def evaluate_model(model, X_test, y_test, scaler=None, target_idx=0, forecast_horizon=1):
    y_pred_scaled = model.predict(X_test)

    y_true_scaled = reduce_to_1d(y_test)
    y_pred_scaled = reduce_to_1d(y_pred_scaled)

    if scaler:
        # Denormalize using the correct formula for sums of scaled values
        y_true_denorm = y_true_scaled * scaler.scale_[target_idx] + forecast_horizon * scaler.mean_[target_idx]
        y_pred_denorm = y_pred_scaled * scaler.scale_[target_idx] + forecast_horizon * scaler.mean_[target_idx]
    else:
        y_true_denorm = y_true_scaled
        y_pred_denorm = y_pred_scaled

    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))
    r2 = r2_score(y_true_denorm, y_pred_denorm)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

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
