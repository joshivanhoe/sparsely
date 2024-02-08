"""This module implements a sparse linear model for classification problems."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from .base import BaseSparseEstimator


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))


class SparseLinearClassifier(BaseSparseEstimator, ClassifierMixin):
    """Sparse linear model for classification.

    Currently, only binary classification is supported. The model is trained using the logistic loss function and the
    L2 regularization penalty. The optimal features are selected using a scalable cutting plane algorithm.
    """

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted regressor.

        Args:
            X: The training data. The array should be of shape (n_samples, n_features).

        Returns:
            The predicted values. Array of shape `(n_samples,)`.
        """
        check_is_fitted(estimator=self)
        self._validate_data(X=X)
        if self.normalize:
            X = self.scaler_X_.transform(X)
        return self._predict(X=X, proba=True)

    def _pre_process_y(self, y: np.ndarray) -> np.ndarray:
        self.binarizer_ = LabelBinarizer(neg_label=np.min(y), pos_label=np.max(y))
        return 2 * self.binarizer_.fit_transform(y).flatten() - 1

    def _predict(self, X: np.ndarray, proba: bool = False) -> np.ndarray:
        """Perform inference using the fitted model.

        Args:
            X: The training data. The array should be of shape (n_samples, n_features).
            proba: Whether to return the predicted probabilities. If `False`, then the predicted class labels are
                returned instead.

        Returns:
            The predicted values. The array will be of shape (n_samples,).
        """
        predicted = _sigmoid(np.dot(X, self.coef_) + self.intercept_)
        if proba:
            return np.column_stack([1 - predicted, predicted])
        return self.binarizer_.inverse_transform(predicted, threshold=0.5)

    def _get_coef(self) -> np.ndarray:
        if self.normalize:
            return self.coef_ / self.scaler_X_.scale_
        return self.coef_

    def _get_intercept(self) -> float:
        if self.normalize:
            return (
                self.intercept_ - (self.scaler_X_.mean_ / self.scaler_X_.scale_).sum()
            )
        return self.intercept_

    def _make_callback(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
        def func(selected: np.ndarray) -> tuple[float, np.ndarray]:
            X_subset = X[:, np.round(selected).astype(bool)]
            coef_subset = self._fit_coef_for_subset(X_subset=X_subset, y=y)
            log_odds = np.matmul(X_subset, coef_subset) + self.intercept_
            dual_vars = -y / (1 + np.exp(y * log_odds))
            loss = (
                dual_vars * y * np.log(-dual_vars * y)
                - (1 + dual_vars * y) * np.log(1 + dual_vars * y)
            ).sum() - 0.5 * self.gamma_ * (np.matmul(X_subset.T, dual_vars) ** 2).sum()
            grad = -0.5 * self.gamma_ * np.matmul(X.T, dual_vars) ** 2
            return loss, grad

        return func

    def _fit_coef_for_subset(self, X_subset: np.ndarray, y) -> np.ndarray:
        estimator = LogisticRegression(C=self.gamma_, penalty="l2").fit(X=X_subset, y=y)
        self.intercept_ = estimator.intercept_[0]
        return estimator.coef_[0, :]
