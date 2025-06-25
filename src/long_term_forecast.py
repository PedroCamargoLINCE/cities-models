import os
from typing import List

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from .preprocessing import load_city_data, filter_city, clean_timeseries


def _load_model_components(city_dir: str, city_code: str):
    model_path = os.path.join(city_dir, f"{city_code}_xgboost_model.json")
    scaler_path = os.path.join(city_dir, f"{city_code}_scaler.pkl")
    metrics_path = os.path.join(city_dir, f"{city_code}_xgboost_metrics.csv")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    metrics = pd.read_csv(metrics_path)
    seq_len = int(metrics["sequence_length"].iloc[0])
    horizon = int(metrics["forecast_horizon"].iloc[0])
    return model, scaler, seq_len, horizon


def _prepare_city_dataframe(csv_path: str, city_code: int) -> pd.DataFrame:
    df = load_city_data(csv_path)
    df_city = filter_city(df, cd_mun=city_code)
    df_city = clean_timeseries(df_city, target_column="target")
    df_city = df_city.sort_values("week")
    return df_city


def _exponential_smooth(value: float, prev: float, alpha: float = 0.5) -> float:
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev


def forecast_city(
    city_dir: str,
    dataset_csv: str,
    city_code: str,
    end_year: int = 2035,
) -> pd.DataFrame:
    model, scaler, seq_len, horizon = _load_model_components(city_dir, city_code)
    df_city = _prepare_city_dataframe(dataset_csv, int(city_code))

    feature_cols = [c for c in df_city.columns if c not in ["CD_MUN", "week", "city", "state"]]
    data = df_city[feature_cols].values.astype(float)
    last_week = df_city["week"].iloc[-1] if "week" in df_city.columns else len(df_city) - 1

    window = data[-seq_len:]
    preds: List[float] = []
    dates: List[int] = []
    prev_pred = None

    months = (end_year - 2023) * 12
    for _ in range(months):
        X = window.reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = float(model.predict(X_scaled)[0])
        pred = _exponential_smooth(pred, prev_pred)
        prev_pred = pred

        last_week += horizon
        dates.append(int(last_week))
        preds.append(pred)

        new_row = window[-1].copy()
        if "target" in feature_cols:
            target_idx = feature_cols.index("target")
            new_row[target_idx] = pred
        window = np.vstack([window[1:], new_row])

    return pd.DataFrame({"data": dates, "previsao": preds})


def generate_all_forecasts(results_root: str = "results", output_dir: str = "long_term_forecasts"):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(results_root):
        for file in files:
            if file.endswith("_xgboost_model.json"):
                city_code = file.split("_")[0]
                dataset_key = None
                if "(" in root and ")" in root:
                    dataset_key = root.split("(")[-1].split(")")[0]
                if dataset_key:
                    csv_path = os.path.join("data", f"df_base_{dataset_key}.csv")
                    if not os.path.exists(csv_path):
                        continue
                    df_out = forecast_city(root, csv_path, city_code)
                    out_file = os.path.join(output_dir, f"{city_code}_forecast.csv")
                    df_out.to_csv(out_file, index=False)


if __name__ == "__main__":
    generate_all_forecasts()
