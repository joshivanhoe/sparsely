from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
from halfspace import Model
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_scalar, check_is_fitted


class BaseSparseEstimator(BaseEstimator, ABC):
    """Base class for sparse estimators.

    Attributes:
        k: The sparsity parameter (i.e. number of non-zero coefficients). If `None`, then `k` is set to the square root
            of the number of features, rounded to the nearest integer.
        gamma: The regularization parameter. If `None`, then `gamma` is set to `1 / sqrt(n_samples)`.
        normalize: Whether to normalize the data before fitting the model.
        max_iters: The maximum number of iterations.
        tol: The tolerance for the stopping criterion.
        verbose: Whether to enable logging of the search progress.
    """

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
            k: x
            gamma: x
            normalize: x
            max_iters: x
            tol: x
            verbose: x
        """
        self.k = k
        self.gamma = gamma
        self.normalize = normalize
        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseSparseEstimator:
        """Fit the model to the training data.

        Args:
            X: array-like of shape (n_samples, n_features)
                The training data.
            y: array-like of shape (n_samples,)
                The training labels.
        Returns: self
            The fitted model.
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
            X = self.scaler_X_.fit_transform(X)
        y = self._pre_process_y(y=y)

        # Optimize feature selection
        model = Model(
            max_gap=self.tol,
            max_gap_abs=self.tol,
            log_freq=1 if self.verbose else None,
        )
        selected = model.add_var_tensor(
            shape=(X.shape[1],), var_type="B", name="selected"
        )
        func = self._make_callback(X=X, y=y)
        model.add_objective_term(var=selected, func=func, grad=True)
        model.add_linear_constr(sum(selected) <= self.k_)
        model.add_linear_constr(sum(selected) >= 1)
        model.start = [(selected[i], 1) for i in range(self.k_)]
        model.optimize()
        selected = np.round([model.var_value(var) for var in selected]).astype(bool)

        # Compute coefficients
        self.coef_ = np.zeros(self.n_features_in_)
        self.coef_[selected] = self._fit_coef_for_subset(X_subset=X[:, selected], y=y)

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
        return self._predict(X=X)

    @property
    def coef(self) -> np.ndarray:
        """Get the coefficients of the linear model."""
        check_is_fitted(estimator=self)
        return self._get_coef()

    @property
    def intercept(self) -> float:
        """Get the intercept of the linear model."""
        check_is_fitted(estimator=self)
        return self._get_intercept()

    def _validate_params(self):
        # super()._validate_params()
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

    @abstractmethod
    def _pre_process_y(self, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _predict(self, X: np.ndarray, proba: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def _get_coef(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_intercept(self) -> float:
        pass

    @abstractmethod
    def _make_callback(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
        pass

    @abstractmethod
    def _fit_coef_for_subset(self, X_subset: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass
