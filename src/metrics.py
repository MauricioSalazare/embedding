import numpy as np
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE) in a robust way.

    Parameters:
    - y_true (np.ndarray): Array of actual values.
    - y_pred (np.ndarray): Array of predicted values.

    Returns:
    - float: MAPE value in percentage.
    """
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)

    # Detect near-zero values to avoid unstable division
    constant_mask = np.abs(y_true) < 10 * np.finfo(y_true.dtype).eps

    # Create a copy to avoid modifying the original array
    y_true_safe = np.copy(y_true)
    y_true_safe[constant_mask] = 1.0  # Avoid division by near-zero values

    mape_value = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return mape_value