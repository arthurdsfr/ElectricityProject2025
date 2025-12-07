import pandas as pd
import numpy as np
from model import GRUForecaster
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import sys
import os

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add it to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

# Load full dataset
from data.data_processing import df_elec_full

def load_forecaster(model_path, seq_len, horizon, units):
    forecaster = GRUForecaster(seq_len=seq_len, horizon=horizon, units=units)

    # Load model weights
    forecaster.model = load_model(model_path,  compile=False)

    # Recreate & fit scaler on the TRAINING dataset (until 2021)
    training_data = df_elec_full[df_elec_full.index <= "2021-03-01"]['Price'].values.reshape(-1, 1)
    forecaster.scaler.fit(training_data)

    return forecaster

forecaster = load_forecaster(
    model_path="saved_models/gru_forecaster_full.h5",
    seq_len=30,
    horizon=15,
    units=64
)

def predict_future_range(forecaster, df_full, start_date, seq_len=30):
    """
    Predict from 'start_date' until the last available date in df_full
    using rolling 15-day forecasts.
    """

    start_date = pd.to_datetime(start_date)
    last_date  = df_full.index.max()

    # Get the last 30 days BEFORE the start date
    past_data = df_full[df_full.index < start_date].tail(seq_len)['Price'].values.reshape(-1, 1)

    # Scale initial window
    current_window = forecaster.scaler.transform(past_data)

    predictions = []
    prediction_dates = []

    current_date = start_date

    while current_date <= last_date:

        X = current_window.reshape(1, seq_len, 1)
        pred_scaled = forecaster.model.predict(X, verbose=0)[0]     # 15 days
        pred = forecaster.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        # Save results
        future_dates = pd.date_range(current_date, periods=len(pred), freq='D')
        predictions.extend(pred)
        prediction_dates.extend(future_dates)

        # Update window
        new_scaled = forecaster.scaler.transform(pred.reshape(-1, 1))
        current_window = np.vstack([current_window, new_scaled])[-seq_len:]

        current_date += pd.Timedelta(days=len(pred))

    # Return DataFrame
    return pd.DataFrame({"Predicted": predictions}, index=prediction_dates)


df_predictions = predict_future_range(
    forecaster,
    df_elec_full,
    start_date="2024-01-01",
    seq_len=30
)

plt.figure(figsize=(14,6))

df_elec_full.loc["2024-01-01":]['Price'].plot(label="True Price")
df_predictions['Predicted'].plot(label="Predicted Price")

plt.title("Electricity Price Forecast (2024 → last date)")
plt.xlabel("Date")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True)
plt.show()
