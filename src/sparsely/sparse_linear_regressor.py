from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_scalar


class SparseLinearRegressor(BaseEstimator, RegressorMixin):
    """Sparse linear regressor."""

    def __init__(self, max_features: Optional[int], gamma: Optional[float] = None, normalize: bool = True):
        """Model constructor.

        Args:
            max_features: int or `None`, default=`None`
                The maximum number of features with non-zero coefficients. If `None`, then `max_features` is set to
                the square root of the number of features, rounded up to the nearest integer.
            gamma: float or `None`, default=`None`
                The regularization parameter. If `None`, then `gamma` is set to `1 / sqrt(n_samples)`.
            normalize: bool, default=`True`
                Whether to normalize the data before fitting the model.
        """
        self.max_features = max_features
        self.gamma = gamma
        self.normalize = normalize

    def fit(self, X: np.ndarray, y: np.ndarray) -> SparseLinearRegressor:
        """Fit the regressor to the training data.

        Args:
            X: array-like of shape (n_samples, n_features)
                The training data.
            y: array-like of shape (n_samples,)
                The training labels.
        Returns: SparseLinearRegressor
            The fitted regressor.
        """
        self._validate_data(X=X, y=y)
        self._validate_params()
        if self.normalize:
            self.scaler_X_ = StandardScaler()
            self.scaler_y_ = StandardScaler()
            X = self.scaler_X_.fit_transform(X)
            y = self.scaler_y_.fit_transform(y[:, None])[:, 0]
        # Fitting algorithm here
        self.coef_ = np.zeros(self.n_features_in_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted regressor.

        Args:
            X: array-like of shape (n_samples, n_features)
                The data to predict.

        Returns: array-like of shape (n_samples,)
            The predicted values.
        """
        self._validate_data(X=X)
        if self.normalize:
            X = self.scaler_X_.transform(X)
        predicted = np.dot(X, self.coef_)
        if self.normalize:
            predicted = self.scaler_y_.inverse_transform(predicted[:, None])[:, 0]
        return predicted

    def _validate_params(self):
        check_scalar(
            x=self.max_features,
            name="max_features",
            target_type=int,
            min_val=1,
            max_val=self.n_features_in_,
            include_boundaries=True,
        )
        check_scalar(
            x=self.gamma,
            name="gamma",
            target_type=float,
            min_val=0,
            include_boundaries=False,
        )
        check_scalar(
            x=self.normalize,
            name="normalize",
            target_type=bool,
        )

    @property
    def coef(self) -> np.ndarray:
        """Get the coefficients of the linear model."""
        if self.normalize:
            return self.coef_ / self.scaler_X_.scale_ * self.scaler_y_.scale_
        return self.coef_

    @property
    def intercept(self) -> float:
        """Get the intercept of the linear model."""
        if self.normalize:
            return -self.scaler_X_.mean_ / self.scaler_X_.scale_ * self.scaler_y_.scale_ + self.scaler_y_.mean_
        return 0

