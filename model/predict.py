import pandas as pd
import numpy as np
from model import GRUForecaster
from helper import mase, mda
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys
import os

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load datasets
from data.data_processing import df_elec_train, df_elec_test

def load_forecaster(model_path, seq_len, horizon, units):
    """
    Load GRU model and fit scaler on TRAINING dataset only.
    """
    forecaster = GRUForecaster(seq_len=seq_len, horizon=horizon, units=units)

    # Load model weights
    forecaster.model = load_model(model_path, compile=False)

    # Fit scaler on TRAINING data
    training_data = df_elec_train['Price'].values.reshape(-1, 1)
    forecaster.scaler.fit(training_data)

    return forecaster

forecaster = load_forecaster(
    model_path="saved_models/gru_forecaster_full.h5",
    seq_len=30,
    horizon=15,
    units=64
)

def predict_from_specific_start_mean(
    forecaster, df_test, df_train, start_date, seq_len=30, horizon=15
):
    """
    Predict multi-horizon values starting from a specific date.
    - Averages overlapping predictions
    - Computes MASE and MDA on averaged predictions
    """

    start_date = pd.to_datetime(start_date)

    # --- initial window = seq_len days before start_date ---
    past_data = df_test['Price'].loc[:start_date - pd.Timedelta(days=1)].iloc[-seq_len:].values.reshape(-1, 1)
    current_window = forecaster.scaler.transform(past_data)

    test_dates_to_predict = df_test.index[df_test.index >= start_date]

    forecast_dict = {}

    for current_date in test_dates_to_predict:

        # ---- predict next horizon ----
        X = current_window.reshape(1, seq_len, 1)
        pred_scaled = forecaster.model.predict(X, verbose=0)[0]
        preds = forecaster.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        future_dates = pd.date_range(current_date, periods=horizon, freq="D")

        # Store overlapping predictions inside test set
        for d, p in zip(future_dates, preds):
            if d in df_test.index:
                forecast_dict.setdefault(d, []).append(p)

        # ---- update window using the TRUE value ----
        true_val = df_test.loc[[current_date], ['Price']].values
        new_scaled = forecaster.scaler.transform(true_val)
        current_window = np.vstack([current_window, new_scaled])[-seq_len:]

    # ---- Average overlapping predictions ----
    averaged_predictions = {d: np.mean(vals) for d, vals in forecast_dict.items()}

    df_pred = pd.DataFrame.from_dict(
        averaged_predictions, orient="index", columns=["Predicted"]
    ).sort_index()

    # ---- Compute metrics ----
    actual = df_test.loc[df_pred.index, "Price"].values
    predicted = df_pred["Predicted"].values

    training_series = df_train["Price"].values

    mase_value = mase(actual, predicted, training_series)
    mda_value = mda(actual, predicted)

    return df_pred, mase_value, mda_value



df_predictions_test, mase_val, mda_val = predict_from_specific_start_mean(
    forecaster,
    df_test=df_elec_test,
    df_train=df_elec_train,   # training series needed for MASE
    start_date="2021-02-01",
    seq_len=30,
    horizon=15
)

print("MASE:", mase_val)
print("MDA :", mda_val)


# --- Plot ---
plt.figure(figsize=(14,6))
df_elec_test['Price'].plot(label="True Test Price", linewidth=2)
df_predictions_test['Predicted'].plot(label="15-Day Averaged Prediction", linewidth=2)
plt.title("Multi-Horizon Averaged Forecast (2021-02-01 → 2021-03-01)")
plt.xlabel("Date")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True)
plt.show()



def predict_all_horizons_with_metrics(
    forecaster, df_test, df_train, start_date, seq_len=30, horizon=15
):
    """
    Generate rolling horizon predictions and compute MASE + MDA
    for each forecast, then average errors.

    Returns:
        all_predictions: list of (dates, preds)
        avg_mase: average MASE across all forecast horizons
        avg_mda: average MDA across all forecast horizons
    """

    start_date = pd.to_datetime(start_date)

    # --- initial window = seq_len days before start_date ---
    past_data = df_test['Price'].loc[:start_date - pd.Timedelta(days=1)].iloc[-seq_len:].values.reshape(-1, 1)
    current_window = forecaster.scaler.transform(past_data)

    test_dates_to_predict = df_test.index[df_test.index >= start_date]

    all_predictions = []
    mase_list = []
    mda_list = []

    # Training series for MASE denominator
    training_series = df_train['Price'].values

    for current_date in test_dates_to_predict:

        # ---- predict horizon ----
        X = current_window.reshape(1, seq_len, 1)
        pred_scaled = forecaster.model.predict(X, verbose=0)[0]
        preds = forecaster.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        future_dates = pd.date_range(current_date, periods=horizon, freq="D")

        # Keep only predictions that fully fit inside df_test
        valid_mask = [d in df_test.index for d in future_dates]

        if sum(valid_mask) > 1:   # need >= 2 points for MDA
            valid_dates = future_dates[valid_mask]
            preds_valid = preds[valid_mask]
            actual_valid = df_test.loc[valid_dates, "Price"].values

            # ---- MASE ----
            mase_val = mase(actual_valid, preds_valid, training_series)
            mase_list.append(mase_val)

            # ---- MDA ----
            mda_val = mda(actual_valid, preds_valid)
            mda_list.append(mda_val)

            # Store predictions
            all_predictions.append((valid_dates, preds_valid))

        # ---- update window using TRUE value ----
        true_val = df_test.loc[[current_date], ['Price']].values
        new_scaled = forecaster.scaler.transform(true_val)
        current_window = np.vstack([current_window, new_scaled])[-seq_len:]

    # ---- Compute global averages ----
    avg_mase = float(np.mean(mase_list))
    avg_mda  = float(np.mean(mda_list))

    return all_predictions, avg_mase, avg_mda


all_preds, avg_mase, avg_mda = predict_all_horizons_with_metrics(
    forecaster=forecaster,
    df_test=df_elec_test,
    df_train=df_elec_train,
    start_date="2021-02-01"
)

print("Average MASE:", avg_mase)
print("Average MDA :", avg_mda)

plt.figure(figsize=(14, 6))

# Plot true price
df_elec_test['Price'].plot(label="True Test Price", linewidth=2, color="black")

# Plot each horizon prediction
for dates, preds in all_preds:
    plt.plot(dates, preds, alpha=0.25, linewidth=1, color="blue")

plt.title("All 15-Day Rolling Horizon Predictions (No Averaging)")
plt.xlabel("Date")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(14, 6))

# Plot true price
df_elec_test['Price'].plot(label="True Test Price", linewidth=2, color="black")

# Plot only predictions with full 15-day horizon
for dates, preds in all_preds:
    if len(preds) == 15:  # <-- only plot full 15-day predictions
        plt.plot(dates, preds, alpha=0.25, linewidth=1, color="blue")

plt.title("15-Day Rolling Horizon Predictions")
plt.xlabel("Date")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True)
plt.show()
