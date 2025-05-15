# Cities Models - Instructions for AI Assistants

This document provides guidelines for AI assistants (like GitHub Copilot) on how to interact with this repository for forecasting respiratory-morbidity rates in Brazilian municipalities.

## Repository Structure

```
cities-models/
├── data/                  # CSV files for each city
│   ├── df_base_morb_resp.csv  # Respiratory morbidity dataset (main data)
│   └── df_base_mort_dengue.csv # Dengue mortality dataset (not used currently)
├── notebooks/             # Jupyter notebooks for experiments (one per model)
├── results/               # Saved forecasts and evaluation metrics
├── src/                   # Core Python modules with reusable code
│   ├── __init__.py        # Package initialization
│   ├── preprocessing.py   # Data loading, normalization, sequence creation
│   ├── models.py          # All model architectures
│   ├── train.py           # Training and evaluation routines
│   └── utils.py           # Helper functions, plotting, metrics
└── instructions.md        # This file
```

## Key Design Principles

### 1. Modularity is Critical

- All core logic MUST be in the `src` modules, not in notebooks
- Notebooks are for execution and documentation ONLY
- Don't reimplement functions in multiple places

### 2. Data Processing Flow

The standard data flow for models is:

1. Load data from CSV files (`preprocessing.load_city_data`)
2. Normalize data (`preprocessing.normalize_data`) 
3. Create sequences for time-series models (`preprocessing.create_sequences`)
4. Train model (`train.train_model`)
5. Evaluate model (`train.evaluate_model`) 
6. Save predictions and metrics (`train.save_predictions`, `train.save_metrics`)
7. Visualize results (`utils.plot_forecast`)

### 3. Notebook Structure

Each notebook should follow this structure:

```python
# Model: [Model Name]
# City: [City Name]
# Description: [Brief description of the experiment]
# Date: [Date]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Import from src modules
from src.preprocessing import load_city_data, prepare_data_for_model
from src.models import build_lstm_model  # or other model function
from src.train import train_evaluate_model
from src.utils import plot_forecast, plot_training_history

# Load data
city_name = 'city_name'  # Replace with actual city
data_path = f'data/{city_name}.csv'  # Or other path pattern
df = load_city_data(data_path)

# Setup parameters
params = {
    'sequence_length': 24,
    'forecast_horizon': 1,
    'normalization': 'zscore',
    # model-specific parameters
}

# Prepare data
data_dict = prepare_data_for_model(
    df=df,
    target_column='target_column_name',  # Replace with actual column
    sequence_length=params['sequence_length'],
    forecast_horizon=params['forecast_horizon'],
    normalization=params['normalization']
)

# Create and train model
model = build_lstm_model(...)  # Use appropriate model function
results = train_evaluate_model(
    model=model,
    data_dict=data_dict,
    city_name=city_name,
    model_name='model_name',  # E.g., 'lstm', 'gru', etc.
    params=params
)

# Visualize results
plot_forecast(results['true_values'], results['predictions'])
if hasattr(results['training_history'], 'history'):
    plot_training_history(results['training_history'])

# Print metrics
print(f"Evaluation metrics: {results['metrics']}")
```

### 4. Model Types

This repository includes implementations of several model types:

1. Baseline models:
   - Last Value (`models.LastValueBaseline`)
   - Moving Average (`models.MovingAverageBaseline`)

2. Neural Network models:
   - MLP (`models.build_mlp_model`)
   - LSTM (`models.build_lstm_model`)
   - GRU (`models.build_gru_model`)
   - Stacked LSTM (`models.build_stacked_lstm_model`)
   - LSTM with Attention (`models.build_lstm_attention_model`)
   - CNN+LSTM (`models.build_cnn_lstm_model`)
   - LSTM with External Features (`models.build_lstm_with_features_model`)
   - Quantile LSTM (`models.build_quantile_lstm_model`)

### 5. Evaluation Metrics

Always use the following metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

Optional metrics:
- R² (R-squared)
- Pinball Loss (for quantile models)
- Prediction Interval Coverage (for quantile models)

### 6. Best Practices for AI Code Generation

When generating code:

1. ALWAYS import from `src` modules rather than rewriting functions
2. ALWAYS follow the notebook structure outlined above
3. Use proper documentation and comments
4. Use the standardized parameter names across functions
5. Keep visualizations consistent by using the utility functions

## Example Workflow

Here's a typical workflow when exploring a new model:

```python
# Import necessary functions
from src.preprocessing import load_city_data, prepare_data_for_model
from src.models import build_lstm_model  # or other model
from src.train import train_evaluate_model
from src.utils import plot_forecast

# Load and prepare data
df = load_city_data('data/city_name.csv')
data_dict = prepare_data_for_model(df, target_column='column_name')

# Create and train model
model = build_lstm_model(input_shape=(data_dict['X_train'].shape[1], data_dict['X_train'].shape[2]))
results = train_evaluate_model(model, data_dict, 'city_name', 'lstm')

# Visualize results
plot_forecast(results['true_values'], results['predictions'])
print(f"MAE: {results['metrics']['mae']}")
print(f"RMSE: {results['metrics']['rmse']}")
```
