"""
Training and evaluation routines for time series forecasting models.
"""
import numpy as np
import os
from typing import Dict, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

def evaluate_model(model, X_test, y_test, scaler=None, feature_columns=None) -> Dict[str, float]:
    """
    Evaluate model and return MAE, RMSE, R2. Always denormalizes predictions and targets if scaler is provided.
    Robustly handles shape issues for y_test/y_pred (always 1D, sum if 2D with >1 col).
    """
    if hasattr(model, 'predict'):
        try:
            y_pred = model.predict(X_test, feature_columns=feature_columns)
        except TypeError:
            y_pred = model.predict(X_test)
    else:
        raise ValueError('Model does not have predict method')
    # Ensure y_test and y_pred are numpy arrays
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    # Robustly reduce to 1D: if 2D and shape[1]>1, sum across axis=1 (for 4-week sum target)
    def reduce_to_1d(arr):
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                return arr.ravel()
            else:
                return arr.sum(axis=1)
        raise ValueError(f"Unexpected array shape: {arr.shape}")
    y_test = reduce_to_1d(y_test)
    y_pred = reduce_to_1d(y_pred)
    # Inverse transform if scaler is provided (handle multi-feature scaler)
    if scaler is not None and hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
        if scaler.scale_.shape[0] > 1:
            # Only use the first column's params (target)
            y_test = y_test * scaler.scale_[0] + scaler.mean_[0]
            y_pred = y_pred * scaler.scale_[0] + scaler.mean_[0]
        else:
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    # Ensure both are 1D
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
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
