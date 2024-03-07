import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from sparsely import SparseLinearClassifier

Dataset = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# TODO: add tests for sklearn compatibility - requires support for multi-class problems


@pytest.mark.parametrize(
    "estimator",
    [
        SparseLinearClassifier(),
        SparseLinearClassifier(normalize=False),
        SparseLinearClassifier(k=3),
        SparseLinearClassifier(gamma=1e-1),
        SparseLinearClassifier(feature_groups=[{0, 1}, {2, 3}]),
        SparseLinearClassifier(start={0, 1, 2}),
    ],
)
def test_sparse_linear_regressor(
    classification_dataset: Dataset, estimator: SparseLinearClassifier
):
    X_train, X_test, y_train, y_test = classification_dataset
    estimator.fit(X_train, y_train)
    predicted = estimator.predict(X_test)
    predicted_proba = estimator.predict_proba(X_test)
    assert estimator._coef.shape == (X_train.shape[1],)
    assert predicted.shape == (X_test.shape[0],)
    assert predicted_proba.shape == (X_test.shape[0], 2)
    assert estimator._coef.shape == (X_train.shape[1],)
    if estimator.feature_groups is None:
        assert balanced_accuracy_score(y_test, predicted) > 0.9
        assert roc_auc_score(y_test, predicted_proba[:, 1]) > 0.9


@pytest.mark.parametrize(
    "estimator",
    [
        SparseLinearClassifier(k=0),
        SparseLinearClassifier(k=11),
        SparseLinearClassifier(gamma=-1e-2),
        SparseLinearClassifier(start={0, 1, 1000}),
        SparseLinearClassifier(feature_groups=[{-1, 0, 1}]),
        SparseLinearClassifier(feature_groups=[{0, 1, 1000}]),
        SparseLinearClassifier(feature_groups=[[0, 0, 1]]),
    ],
)
def test_sparse_linear_regressor_invalid_params(
    classification_dataset: Dataset, estimator: SparseLinearClassifier
):
    X_train, X_test, y_train, y_test = classification_dataset
    with pytest.raises((ValueError, TypeError)):
        estimator.fit(X_train, y_train)
