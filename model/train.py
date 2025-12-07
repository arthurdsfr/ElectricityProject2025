import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import save_model

import sys
import os

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add it to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from model import GRUForecaster
from data.data_processing import df_elec

# ------------------- K-FOLD CROSS-VALIDATION -------------------
def k_fold_cross_validation(folds, price_series, seq_len=30, horizon=15, units=64, epochs=50, batch_size=32):
    """
    Perform K-fold cross-validation on the GRUForecaster model.
    """
    fold_mae_results = []

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        print(f"\n===== FOLD {i+1} =====")
        print(f"Train: {tr_start} → {tr_end}")
        print(f"Test : {te_start} → {te_end}")

        # Extract train/test data by date
        train_series = price_series[tr_start:tr_end].values
        test_series  = price_series[te_start:te_end].values

        # Initialize model
        forecaster = GRUForecaster(seq_len=seq_len, horizon=horizon, units=units)

        # Train on this fold
        forecaster.fit_on_split(train_series, test_series, epochs=epochs, batch_size=batch_size)

        # Predict and evaluate
        preds, true = forecaster.predict_test()
        fold_mae = np.mean([mean_absolute_error(true[j], preds[j]) for j in range(len(preds))])
        print(f"Fold {i+1} MAE = {fold_mae:.4f}")
        fold_mae_results.append(fold_mae)

    # Cross-validation summary
    print("\n===== CROSS-VALIDATION RESULTS =====")
    print("MAE per fold:", fold_mae_results)
    print("Mean MAE:", np.mean(fold_mae_results))
    print("Std MAE :", np.std(fold_mae_results))

    return fold_mae_results

# ------------------- TRAIN FINAL MODEL & SAVE -------------------
def train_and_save_model(forecaster_class, price_series, save_path, seq_len=30, horizon=15, units=64, epochs=60, batch_size=32):
    """
    Train the GRUForecaster on the full dataset and save the trained model.
    """
    # Initialize the forecaster
    forecaster = forecaster_class(seq_len=seq_len, horizon=horizon, units=units)
    
    # Train on the full dataset
    forecaster.fit(price_series.values if hasattr(price_series, "values") else price_series,
                   epochs=epochs, batch_size=batch_size)
    
    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_model(forecaster.model, save_path)
    print(f"Model saved at: {save_path}")
    
    return forecaster

# ------------------- CONFIG -------------------
folds = [
    ("2012-01-02", "2014-12-31", "2015-01-01", "2016-06-30"),
    ("2012-01-02", "2016-06-30", "2016-07-01", "2017-12-31"),
    ("2012-01-02", "2017-12-31", "2018-01-01", "2019-06-30"),
    ("2012-01-02", "2019-06-30", "2019-07-01", "2020-12-31"),
]

prices = df_elec['Price']

# # ------------------- RUN CROSS-VALIDATION -------------------
# fold_mae_results = k_fold_cross_validation(
#     folds, prices,
#     seq_len=30, horizon=15, units=64, epochs=50, batch_size=32
# )

# #------------------- TRAIN FINAL MODEL -------------------
# save_path = "saved_models/gru_forecaster_full.h5"
# forecaster = train_and_save_model(
#    GRUForecaster, prices, save_path,
#    seq_len=30, horizon=15, units=64, epochs=60, batch_size=32
# )
