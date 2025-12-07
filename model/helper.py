import numpy as np

def mase(actual, predicted, training_series, m=1):
    """
    Mean Absolute Scaled Error.
    - actual: true values
    - predicted: forecast values
    - training_series: full training dataset (1D np.array)
    - m: seasonality (1 for daily)
    """
    n = len(training_series)
    denom = np.mean(np.abs(training_series[m:] - training_series[:-m]))
    return np.mean(np.abs(actual - predicted)) / denom


def mda(actual, predicted):
    """
    Mean Directional Accuracy.
    Compares direction of change.
    """
    actual_diff = np.sign(np.diff(actual))
    pred_diff   = np.sign(np.diff(predicted))
    return np.mean(actual_diff == pred_diff)
