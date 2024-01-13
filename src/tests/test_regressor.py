import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from sparsely.regressor import SparseLinearRegressor

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
    ],
)
def test_sparse_linear_regressor(dataset: Dataset, estimator: SparseLinearRegressor):
    X_train, X_test, y_train, y_test, coef = dataset
    predicted = estimator.fit(X_train, y_train).predict(X_test)
    assert estimator.coef_.shape == (X_train.shape[1],)
    assert predicted.shape == (X_test.shape[0],)
    assert estimator.score(X_train, y_train) > 0.8
    assert estimator.score(X_test, y_test) > 0.8
    assert estimator.coef_.shape == (X_train.shape[1],)
    assert (~np.isclose(coef, 0)).sum() <= estimator.k_
    assert (np.isclose(estimator.coef_, 0) == np.isclose(coef, 0)).all()


@pytest.mark.parametrize(
    "estimator",
    [
        SparseLinearRegressor(k=0),
        SparseLinearRegressor(k=11),
        SparseLinearRegressor(gamma=-1e-2),
    ],
)
def test_sparse_linear_regressor_invalid_params(
    dataset: Dataset, estimator: SparseLinearRegressor
):
    X_train, X_test, y_train, y_test, coef = dataset
    with pytest.raises(ValueError):
        estimator.fit(X_train, y_train)
