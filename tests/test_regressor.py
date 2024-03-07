import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from sparsely import SparseLinearRegressor

Dataset = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def test_sklearn_compatibility():
    check_estimator(SparseLinearRegressor())


@pytest.mark.parametrize(
    "estimator",
    [
        SparseLinearRegressor(),
        SparseLinearRegressor(normalize=False),
        SparseLinearRegressor(k=3),
        SparseLinearRegressor(gamma=1e-2),
        SparseLinearRegressor(feature_groups=[{0, 1}, {2, 3}]),
        SparseLinearRegressor(start={0, 1, 2}),
    ],
)
def test_sparse_linear_regressor(
    regression_dataset: Dataset, estimator: SparseLinearRegressor
):
    X_train, X_test, y_train, y_test, coef = regression_dataset
    predicted = estimator.fit(X_train, y_train).predict(X_test)
    assert estimator._coef.shape == (X_train.shape[1],)
    assert predicted.shape == (X_test.shape[0],)
    if estimator.feature_groups is None:
        assert estimator.score(X_train, y_train) > 0.95
        assert estimator.score(X_test, y_test) > 0.95
        assert estimator._coef.shape == (X_train.shape[1],)
        assert (~np.isclose(coef, 0)).sum() <= estimator._k
        assert (np.isclose(estimator._coef, 0) == np.isclose(coef, 0)).all()
    else:
        assert estimator._coef.shape == (X_train.shape[1],)


@pytest.mark.parametrize(
    "estimator",
    [
        SparseLinearRegressor(k=0),
        SparseLinearRegressor(k=11),
        SparseLinearRegressor(gamma=-1e-2),
        SparseLinearRegressor(start={-1, 0, 1}),
        SparseLinearRegressor(start={0, 1, 1000}),
        SparseLinearRegressor(feature_groups=[{-1, 0, 1}]),
        SparseLinearRegressor(feature_groups=[{0, 1, 1000}]),
        SparseLinearRegressor(feature_groups=[[0, 0, 1]]),
    ],
)
def test_sparse_linear_regressor_invalid_params(
    regression_dataset: Dataset, estimator: SparseLinearRegressor
):
    X_train, X_test, y_train, y_test, coef = regression_dataset
    with pytest.raises((ValueError, TypeError)):
        estimator.fit(X_train, y_train)
