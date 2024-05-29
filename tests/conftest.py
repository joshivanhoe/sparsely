import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

Dataset = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]


@pytest.fixture
def regression_dataset() -> Dataset:
    """Generate a regression dataset."""
    X, y, coef = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        random_state=0,
        coef=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test, coef


@pytest.fixture
def classification_dataset() -> Dataset:
    """Generate a classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test, None
