import numpy as np
from sklearn.metrics import mean_absolute_error
import sys
import os

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add it to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)

from model import GRUForecaster
from data.data_processing import df_elec


folds = [
    # (train_start, train_end, test_start, test_end)
    ("2012-01-02", "2014-12-31", "2015-01-01", "2016-06-30"),
    ("2012-01-02", "2016-06-30", "2016-07-01", "2017-12-31"),
    ("2012-01-02", "2017-12-31", "2018-01-01", "2019-06-30"),
    ("2012-01-02", "2019-06-30", "2019-07-01", "2020-12-31"),
]

prices = df_elec['Price']

seq_len = 30
horizon = 15

fold_mae_results = []

for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
    print(f"\n===== FOLD {i+1} =====")
    print(f"Train: {tr_start} → {tr_end}")
    print(f"Test : {te_start} → {te_end}")

    # Extract train/test data by date
    train_series = prices[tr_start : tr_end].values
    test_series  = prices[te_start : te_end].values

    # Initialize model
    forecaster = GRUForecaster(seq_len=seq_len, horizon=horizon, units=64)

    # Train on this fold
    forecaster.fit_on_split(train_series, test_series, epochs=50, batch_size=32)

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
