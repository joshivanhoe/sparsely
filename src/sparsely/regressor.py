"""This module implements a sparse linear model for regression problems."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler

from .base import BaseSparseEstimator


class SparseLinearRegressor(BaseSparseEstimator, RegressorMixin):
    """Sparse linear model for regression.

    The model is trained using the L2 loss function and the L2 regularization penalty. The optimal features are
    selected using a scalable cutting plane algorithm.
    """

    def _pre_process_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize the target variable."""
        self._scaler_y = StandardScaler()
        return self._scaler_y.fit_transform(y[:, None])[:, 0]

    def _predict(self, X: np.ndarray, proba: bool = False) -> np.ndarray:
        """Perform inference using the fitted model.

        Args:
            X: The training data. The array should be of shape (n_samples, n_features).
            proba: Not used. Exists for interoperability with the sparse linear classifier.

        Returns:
            The predicted values. The array will be of shape (n_samples,).
        """
        predicted = np.dot(X, self._coef)
        return self._scaler_y.inverse_transform(predicted[:, None])[:, 0]

    def _get_coef(self) -> np.ndarray:
        if self.normalize:
            return self._coef / self._scaler_X.scale_ * self._scaler_y.scale_
        return self._coef

    def _get_intercept(self) -> float:
        if self.normalize:
            return (
                self._scaler_y.mean_
                - (self._scaler_X.mean_ / self._scaler_X.scale_).sum()
                * self._scaler_y.scale_
            )
        return 0

    def _make_callback(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
        def func(selected: np.ndarray) -> tuple[float, np.ndarray]:
            X_subset = X[:, np.round(selected).astype(bool)]
            coef_subset = self._fit_coef_for_subset(X_subset=X_subset, y=y)
            dual_vars = y - np.matmul(X_subset, coef_subset)
            loss = 0.5 * np.dot(y, dual_vars)
            grad = -0.5 * self._gamma * np.matmul(X.T, dual_vars) ** 2
            return loss, grad

        return func

    def _fit_coef_for_subset(self, X_subset: np.ndarray, y) -> np.ndarray:
        return np.matmul(
            np.linalg.inv(
                1 / self._gamma * np.eye(X_subset.shape[1])
                + np.matmul(X_subset.T, X_subset)
            ),
            np.matmul(X_subset.T, y),
        )
