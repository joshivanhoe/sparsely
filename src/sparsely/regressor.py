from __future__ import annotations

from typing import Optional, Callable

import numpy as np
from halfspace import Model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_scalar, check_is_fitted


class SparseLinearRegressor(BaseEstimator, RegressorMixin):
    """Sparse linear regressor."""

    def __init__(
        self,
        k: Optional[int] = None,
        gamma: Optional[float] = None,
        normalize: bool = True,
        max_iters: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        """Model constructor.

        Args:
            k: int or `None`, default=`None`
                The sparsity parameter (i.e. number of non-zero coefficients). If `None`, then `k` is set to the
                square root of the number of features, rounded to the nearest integer.
            gamma: float or `None`, default=`None`
                The regularization parameter. If `None`, then `gamma` is set to `1 / sqrt(n_samples)`.
            normalize: bool, default=`True`
                Whether to normalize the data before fitting the model.
            max_iters: int, default=`500`
                The maximum number of iterations.
            tol: float, default=`1e-4`
                The tolerance for the stopping criterion.
            verbose: bool, default=`False`
                Whether to enable logging of the search progress.
        """
        self.k = k
        self.gamma = gamma
        self.normalize = normalize
        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose

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
        # Perform validation checks
        X, y = self._validate_data(X=X, y=y)
        self._validate_params()

        # Set hyperparameters to default values if not specified
        self.k_ = self.k or int(np.sqrt(X.shape[1]))
        self.gamma_ = self.gamma or 1 / np.sqrt(X.shape[0])

        # Pre-process training data
        if self.normalize:
            self.scaler_X_ = StandardScaler()
            self.scaler_y_ = StandardScaler()
            X = self.scaler_X_.fit_transform(X)
            y = self.scaler_y_.fit_transform(y[:, None])[:, 0]

        # Optimize feature selection
        model = Model(
            max_gap=self.tol, max_gap_abs=self.tol, log_freq=1 if self.verbose else None
        )
        selected = model.add_var_tensor(
            shape=(X.shape[1],), var_type="B", name="selected"
        )
        func, grad = self._make_callbacks(X=X, y=y)
        model.add_objective_term(var=selected, func=func, grad=grad)
        model.add_linear_constr(sum(selected) <= self.k_)
        model.optimize()
        selected = np.round([model.var_value(var) for var in selected]).astype(bool)

        # Compute coefficients
        self.coef_ = np.zeros(self.n_features_in_)
        self.coef_[selected] = self._compute_coef_for_subset(
            X_subset=X[:, selected], y=y
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted regressor.

        Args:
            X: array-like of shape (n_samples, n_features)
                The data to predict.

        Returns: array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(estimator=self)
        self._validate_data(X=X)
        if self.normalize:
            X = self.scaler_X_.transform(X)
        predicted = np.dot(X, self.coef_)
        if self.normalize:
            predicted = self.scaler_y_.inverse_transform(predicted[:, None])[:, 0]
        return predicted

    @property
    def coef(self) -> np.ndarray:
        """Get the coefficients of the linear model."""
        check_is_fitted(estimator=self)
        if self.normalize:
            return self.coef_ / self.scaler_X_.scale_ * self.scaler_y_.scale_
        return self.coef_

    @property
    def intercept(self) -> float:
        """Get the intercept of the linear model."""
        check_is_fitted(estimator=self)
        if self.normalize:
            return (
                -self.scaler_X_.mean_ / self.scaler_X_.scale_ * self.scaler_y_.scale_
                + self.scaler_y_.mean_
            )
        return 0

    def _validate_params(self):
        if self.k is not None:
            check_scalar(
                x=self.k,
                name="max_features",
                target_type=int,
                min_val=1,
                max_val=self.n_features_in_,
                include_boundaries="both",
            )
        if self.gamma is not None:
            check_scalar(
                x=self.gamma,
                name="gamma",
                target_type=float,
                min_val=0,
                include_boundaries="neither",
            )
        check_scalar(
            x=self.normalize,
            name="normalize",
            target_type=bool,
        )

    def _make_callbacks(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
        def func(selected: np.ndarray) -> float:
            X_subset = X[:, np.round(selected).astype(bool)]
            coef = self._compute_coef_for_subset(X_subset=X_subset, y=y)
            return 0.5 * np.dot(y, y - np.matmul(X_subset, coef))

        def grad(selected: np.ndarray) -> np.ndarray:
            X_subset = X[:, np.round(selected).astype(bool)]
            # TODO: remove redundant computation of subset coef for gradient
            coef = self._compute_coef_for_subset(X_subset=X_subset, y=y)
            return (
                -0.5 * self.gamma_ * np.matmul(X.T, y - np.matmul(X_subset, coef)) ** 2
            )

        return func, grad

    def _compute_coef_for_subset(self, X_subset: np.ndarray, y) -> np.ndarray:
        return np.matmul(
            np.linalg.inv(
                1 / self.gamma_ * np.eye(X_subset.shape[1])
                + np.matmul(X_subset.T, X_subset)
            ),
            np.matmul(X_subset.T, y),
        )
