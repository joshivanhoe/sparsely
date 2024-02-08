from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler

from .base import BaseSparseEstimator


class SparseLinearRegressor(BaseSparseEstimator, RegressorMixin):
    """Sparse linear regressor."""

    def _pre_process_y(self, y: np.ndarray) -> np.ndarray:
        self.scaler_y_ = StandardScaler()
        return self.scaler_y_.fit_transform(y[:, None])[:, 0]

    def _predict(self, X: np.ndarray, proba: bool = False) -> np.ndarray:
        predicted = np.dot(X, self.coef_)
        return self.scaler_y_.inverse_transform(predicted[:, None])[:, 0]

    def _get_coef(self) -> np.ndarray:
        if self.normalize:
            return self.coef_ / self.scaler_X_.scale_ * self.scaler_y_.scale_
        return self.coef_

    def _get_intercept(self) -> float:
        if self.normalize:
            return (
                self.scaler_y_.mean_
                - (self.scaler_X_.mean_ / self.scaler_X_.scale_).sum()
                * self.scaler_y_.scale_
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
            grad = -0.5 * self.gamma_ * np.matmul(X.T, dual_vars) ** 2
            return loss, grad

        return func

    def _fit_coef_for_subset(self, X_subset: np.ndarray, y) -> np.ndarray:
        return np.matmul(
            np.linalg.inv(
                1 / self.gamma_ * np.eye(X_subset.shape[1])
                + np.matmul(X_subset.T, X_subset)
            ),
            np.matmul(X_subset.T, y),
        )
