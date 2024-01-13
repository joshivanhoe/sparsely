import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dataset for testing the regressor."""
    X, y, coef = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=3,
        noise=0.0,
        random_state=0,
        coef=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test, coef
