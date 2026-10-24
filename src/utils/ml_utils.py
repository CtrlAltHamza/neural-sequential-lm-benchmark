"""
Robust Machine Learning Utilities with Pure Python & NumPy Fallbacks.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Guarantees 100% cross-platform execution even when OS security policies
(such as Windows Defender Application Control / Smart App Control)
block compiled external C-extension binary DLLs.
"""

from typing import List, Tuple, Any, Optional
import numpy as np

# Try importing sklearn if permitted by OS policy, otherwise fallback cleanly
try:
    from sklearn.model_selection import train_test_split as _sk_split
    from sklearn.linear_model import LinearRegression as _sk_LinearRegression
    from sklearn.metrics import mean_squared_error as _sk_mse, r2_score as _sk_r2
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True):
    """
    Robust train-test split that supports lists, numpy arrays, and pandas Series.
    Uses sklearn when available; otherwise provides identical deterministic NumPy splitting.
    """
    if _SKLEARN_AVAILABLE:
        try:
            return _sk_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle)
        except Exception:
            pass

    if len(arrays) == 0:
        raise ValueError("At least one input array is required.")

    n_samples = len(arrays[0])
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError(f"Found input variables with inconsistent numbers of samples: {[len(a) for a in arrays]}")

    if test_size is None and train_size is None:
        test_size = 0.25

    if test_size is not None:
        if isinstance(test_size, float):
            n_test = int(np.ceil(n_samples * test_size))
        else:
            n_test = int(test_size)
        n_train = n_samples - n_test
    else:
        if isinstance(train_size, float):
            n_train = int(np.floor(n_samples * train_size))
        else:
            n_train = int(train_size)
        n_test = n_samples - n_train

    if shuffle:
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    splits = []
    for arr in arrays:
        if isinstance(arr, list):
            splits.append([arr[i] for i in train_idx])
            splits.append([arr[i] for i in test_idx])
        elif isinstance(arr, np.ndarray):
            splits.append(arr[train_idx])
            splits.append(arr[test_idx])
        else:
            try:
                splits.append(arr.iloc[train_idx])
                splits.append(arr.iloc[test_idx])
            except Exception:
                splits.append([arr[i] for i in train_idx])
                splits.append([arr[i] for i in test_idx])

    return splits


class LinearRegression:
    """
    Ordinary Least Squares (OLS) Linear Regression solved via closed-form Normal Equations.
    Provides identical API to sklearn.linear_model.LinearRegression without DLL dependencies.
    """
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: Any, y: Any):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(X)), X])
            beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            self.intercept_ = float(beta[0])
            self.coef_ = np.asarray(beta[1:], dtype=float)
        else:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.intercept_ = 0.0
            self.coef_ = np.asarray(beta, dtype=float)

        return self

    def predict(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.dot(X, self.coef_) + self.intercept_


def r2_score(y_true: Any, y_pred: Any) -> float:
    """Calculates Coefficient of Determination R^2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Calculates Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))
