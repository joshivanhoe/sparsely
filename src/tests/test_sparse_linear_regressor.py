import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

from sparsely.sparse_linear_regressor import SparseLinearRegressor

Dataset = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@pytest.fixture
def dataset() -> Dataset:
    X, y, coef = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=3,
        noise=0.,
        random_state=0,
        coef=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test, coef


def test_sklearn_compatibility():
    check_estimator(SparseLinearRegressor())


@pytest.mark.parametrize(
    "estimator",
    [
        SparseLinearRegressor(),
        SparseLinearRegressor(normalize=False),
        SparseLinearRegressor(max_selected_features=3),
        SparseLinearRegressor(gamma=1e-2),
    ]
)
def test_sparse_linear_regressor(dataset: Dataset, estimator: SparseLinearRegressor):
    X_train, X_test, y_train, y_test, coef = dataset
    predicted = estimator.fit(X_train, y_train).predict(X_test)
    assert estimator.coef_.shape == (X_train.shape[1],)
    assert predicted.shape == (X_test.shape[0],)
    assert estimator.score(X_train, y_train) > 0.8
    assert estimator.score(X_test, y_test) > 0.8
    assert (np.isclose(estimator.coef_, 0) == np.isclose(coef, 0)).all(), (
            np.argwhere(~np.isclose(estimator.coef_, 0)).flatten(), np.argwhere(~np.isclose(coef, 0)).flatten(),
            estimator.coef_
    )


@pytest.mark.parametrize(
    "estimator",
    [
        SparseLinearRegressor(max_selected_features=0),
        SparseLinearRegressor(max_selected_features=11),
        SparseLinearRegressor(gamma=-1e-2),
    ]
)
def test_sparse_linear_regressor_invalid_params(dataset: Dataset, estimator: SparseLinearRegressor):
    X_train, X_test, y_train, y_test, coef = dataset
    with pytest.raises(ValueError):
        estimator.fit(X_train, y_train)
