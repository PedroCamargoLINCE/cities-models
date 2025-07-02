import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import timedelta

# Use a direct import which works when running as a module
from preprocessing import load_city_data, filter_city, clean_timeseries


def _exponential_smooth(pred, prev_pred, alpha=0.3):
    """Applies exponential smoothing to the prediction."""
    return alpha * pred + (1 - alpha) * prev_pred


def forecast_city(model, city_df, scaler, params):
    """
    Generates a long-term autoregressive forecast for a single city.

    Args:
        model: The pre-trained XGBoost model.
        city_df (pd.DataFrame): The historical data for the city.
        scaler (StandardScaler): The scaler fitted on the training data.
        params (dict): A dictionary of model parameters, including 'sequence_length'.

    Returns:
        pd.DataFrame: A dataframe containing the forecast with 'data' and 'previsao' columns.
    """
    sequence_length = params["sequence_length"]
    # Ensure 'target' is in the feature columns list for indexing
    feature_cols = list(city_df.columns)
    if "target" not in feature_cols:
        # This case should not happen if data is prepared correctly
        raise ValueError("Column 'target' not found in the dataframe")

    target_idx = feature_cols.index("target")
    
    # Initialize the window with the most recent data (unscaled)
    window = city_df.values[-sequence_length:]
    
    # Determine the forecast horizon
    last_date = city_df.index[-1]
    end_date = pd.to_datetime("2035-12-31")
    # Calculate the number of months to forecast
    months_to_forecast = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)

    preds = []
    dates = []
    
    # Use the last known unscaled value for initial smoothing
    prev_pred = city_df["target"].iloc[-1]
    
    current_date = last_date
    
    for _ in range(months_to_forecast):
        # The window contains unscaled data. We scale it just for prediction.
        X_scaled = scaler.transform(window)
        
        # Predict the next step (the prediction is scaled)
        pred_scaled = float(model.predict(X_scaled.reshape(1, -1))[0])
        
        # To inverse_transform, we need a dummy array of the correct shape
        dummy_row = np.zeros((1, len(feature_cols)))
        dummy_row[0, target_idx] = pred_scaled
        
        # Inverse transform to get the prediction in its original, unscaled value
        pred_unscaled = scaler.inverse_transform(dummy_row)[0, target_idx]
        
        # Apply smoothing to the unscaled prediction
        pred = _exponential_smooth(pred_unscaled, prev_pred)
        prev_pred = pred
        
        preds.append(pred)
        
        # Update date for the next prediction step
        current_date += timedelta(weeks=4)  # Approximate a month
        dates.append(current_date)
        
        # Create the next row for the input window using unscaled data
        new_row = window[-1].copy()  # Copy the last known row
        new_row[target_idx] = pred  # Update the target with the new unscaled prediction
        
        # Append the new row and slide the window
        window = np.vstack([window[1:], new_row])

    return pd.DataFrame({"data": dates, "previsao": preds})


def generate_all_forecasts():
    """
    Finds all trained XGBoost models, generates long-term forecasts for each city,
    and saves them to a structured output directory.
    """
    base_path = Path("./")
    results_path = base_path / "results"
    data_path = base_path / "data"
    output_path = base_path / "long_term_forecasts"

    print("Starting long-term forecast generation...")

    
    # Find all XGBoost model files
    model_paths = list(results_path.glob("**/*xgboost_model.json"))
    
    if not model_paths:
        print("No XGBoost models found in the 'results' directory.")
        return

    for model_path in model_paths:
        model_dir = model_path.parent
        
        # Extract the dataset identifier from the directory name
        # e.g., "xgboost_batch_all_municipalities(morb_circ)" -> "morb_circ"
        name = model_dir.name
        try:
            dataset_id = name.split('(')[1].split(')')[0]
        except IndexError:
            dataset_id = name  # assume name é o ID
        print(f"Processing dataset: {dataset_id}")

        # Construct paths
        data_file = data_path / f"df_base_{dataset_id}.csv"
        scaler_file = model_dir / "scaler.pkl"
        params_file = model_dir / "params.json"
        
        # Create a dedicated output directory for this dataset's forecasts
        forecast_dir = output_path / model_dir.name
        forecast_dir.mkdir(parents=True, exist_ok=True)

        # Load assets
        if not all([data_file.exists(), scaler_file.exists(), params_file.exists()]):
            print(f"Missing required files for {dataset_id} (data, scaler, or params). Skipping.")
            continue
            
        print("Loading data and model assets...")
        df = load_city_data(str(data_file))
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        scaler = joblib.load(scaler_file)
        with open(params_file, 'r') as f:
            params = json.load(f)
            
        all_cities = df["cod_mun"].unique()
        print(f"Found {len(all_cities)} cities in {dataset_id}.csv")

        for i, city_code in enumerate(all_cities):
            print(f"  ({i+1}/{len(all_cities)}) Forecasting for city: {city_code}...")
            
            city_df = filter_city(df, city_code)
            city_df_cleaned, _ = clean_timeseries(city_df)
            
            if city_df_cleaned.shape[0] < params["sequence_length"]:
                print(f"    Skipping city {city_code} due to insufficient data for the model's sequence length.")
                continue

            # Generate forecast
            forecast_df = forecast_city(model, city_df_cleaned, scaler, params)
            
            # Save forecast to the correct sub-directory
            output_file = forecast_dir / f"{city_code}_forecast.csv"
            forecast_df.to_csv(output_file, index=False)
            
    print("\nForecast generation complete.")


if __name__ == "__main__":
    generate_all_forecasts()
