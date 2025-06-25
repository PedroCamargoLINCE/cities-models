"""
Model definitions for time series forecasting of respiratory-morbidity rates.
"""
import numpy as np
from typing import Optional, Tuple

# Baseline 1: Last Value
class LastValueBaseline:
    def fit(self, X, y):
        pass  # No training needed
    def predict(self, X, target_index=None, feature_columns=None, forecast_horizon=1): # Changed default from 4 to 1
        # Predict the sum of the last forecast_horizon values of the target column in the input window
        if feature_columns is not None and 'target' in feature_columns:
            target_index = feature_columns.index('target')
        elif target_index is None:
            target_index = 0  # fallback
        if X.ndim == 3:
            # Sum the last forecast_horizon values in the sequence for the target column
            return X[:, -forecast_horizon:, target_index].sum(axis=1)
        else:
            # Fallback: sum last forecast_horizon values
            return X[:, -forecast_horizon:].sum(axis=1)

# Baseline 2: Moving Average
class MovingAverageBaseline:
    def __init__(self, window: int = 4, mode: str = 'sum'):
        self.window = window
        self.mode = mode  # 'sum' or 'mean'
    def fit(self, X, y):
        pass
    def predict(self, X):
        if self.mode == 'sum':
            # Predict the sum of the last window values in the input window (for 4-week sum forecasting)
            return np.sum(X[:, -self.window:, 0], axis=1)
        else:
            # Default: mean of last window values (legacy behavior)
            return np.mean(X[:, -self.window:, 0], axis=1)

# Deep learning models are defined as factory functions to avoid import errors if keras/tensorflow is not installed

def build_mlp(input_shape: Tuple[int], units: int = 32):
    import keras
    from keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(units, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

def build_lstm(input_shape: Tuple[int], units: int = 32, loss: str = 'mae'):
    import keras
    from keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=loss)
    return model

def build_gru(input_shape: Tuple[int], units: int = 32, loss: str = 'mae'):
    import keras
    from keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(units, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=loss)
    return model

def build_stacked_lstm(input_shape: Tuple[int], loss: str = 'mae'):
    import keras
    from keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=loss)
    return model

def build_stacked_gru(input_shape: Tuple[int], loss: str = 'mae'):
    """Stacked GRU: GRU(64, return_sequences=True) -> Dropout(0.2) -> GRU(32) -> Dense(1)"""
    import keras
    from keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(32, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=loss)
    return model

def build_lstm_attention(input_shape: Tuple[int], loss: str = 'mae'):
    import keras
    from keras import layers, models
    import tensorflow as tf
    class AttentionLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def build(self, input_shape):
            self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='normal', trainable=True)
            self.b = self.add_weight(shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        def call(self, x):
            # x: (batch, timesteps, features)
            # Compute attention scores
            e = tf.math.tanh(tf.matmul(x, self.W) + self.b)  # (batch, timesteps, 1)
            a = tf.nn.softmax(e, axis=1)  # (batch, timesteps, 1)
            output = tf.reduce_sum(x * a, axis=1)  # (batch, features)
            return output
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = AttentionLayer()(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss)
    return model

def build_gru_attention(input_shape: Tuple[int], loss: str = 'mae'):
    """GRU with Attention: GRU(64, return_sequences=True) -> Attention -> Dense(1)"""
    import keras
    from keras import layers, models
    import tensorflow as tf
    class AttentionLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def build(self, input_shape):
            self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='normal', trainable=True)
            self.b = self.add_weight(shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        def call(self, x):
            # x: (batch, timesteps, features)
            # Compute attention scores
            e = tf.math.tanh(tf.matmul(x, self.W) + self.b)  # (batch, timesteps, 1)
            a = tf.nn.softmax(e, axis=1)  # (batch, timesteps, 1)
            output = tf.reduce_sum(x * a, axis=1)  # (batch, features)
            return output
    inputs = layers.Input(shape=input_shape)
    x = layers.GRU(64, return_sequences=True)(inputs)
    x = AttentionLayer()(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss)
    return model

def build_cnn_lstm(input_shape: Tuple[int], loss: str = 'mae'):
    import keras
    from keras import layers, models
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling1D()(x)
    x = layers.LSTM(32)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss)
    return model

def build_cnn_gru(input_shape: Tuple[int], loss: str = 'mae'):
    """CNN + GRU: Conv1D(32) -> MaxPooling1D -> GRU(32) -> Dense(1)"""
    import keras
    from keras import layers, models
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling1D()(x)
    x = layers.GRU(32)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=loss)
    return model

def build_lstm_multivariate(input_shape: Tuple[int], loss: str = 'mae'):
    import keras
    from keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=loss)
    return model

def build_gru_multivariate(input_shape: Tuple[int], loss: str = 'mae'):
    """Multivariate GRU: GRU(64) -> Dense(1)"""
    import keras
    from keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=loss)
    return model

def build_quantile_lstm(input_shape: Tuple[int], quantiles=[0.1, 0.5, 0.9]):
    import keras
    from keras import layers, models
    import numpy as np
    class QuantileLoss(keras.losses.Loss):
        def __init__(self, quantiles):
            super().__init__()
            self.quantiles = quantiles
        def call(self, y_true, y_pred):
            losses = []
            for i, q in enumerate(self.quantiles):
                e = y_true - y_pred[:, i]
                losses.append(np.mean(np.maximum(q * e, (q - 1) * e)))
            return np.mean(losses)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dense(len(quantiles))
    ])
    model.compile(optimizer='adam', loss=QuantileLoss(quantiles))
    return model

def build_quantile_gru(input_shape: Tuple[int], quantiles=[0.1, 0.5, 0.9]):
    """Quantile GRU: GRU(64) -> Dense(len(quantiles))"""
    import keras
    from keras import layers, models
    import tensorflow as tf
    class QuantileLoss(keras.losses.Loss):
        def __init__(self, quantiles):
            super().__init__()
            self.quantiles = quantiles
        def call(self, y_true, y_pred):
            losses = []
            for i, q in enumerate(self.quantiles):
                e = y_true - y_pred[:, i]
                losses.append(tf.reduce_mean(tf.maximum(q * e, (q - 1) * e)))
            return tf.reduce_mean(losses)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64),
        layers.Dense(len(quantiles))
    ])
    model.compile(optimizer='adam', loss=QuantileLoss(quantiles))
    return model
