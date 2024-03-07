"""This module implements an abstract base class for sparse linear models.

It contains the shared functionality used for both classification and regression models, particularly in terms
of the fitting procedure. Feature selection is optimized using a scalable cutting plane algorithm.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from numbers import Real, Integral
from typing import Optional, Callable, ClassVar, Sequence

import numpy as np
from halfspace import Model
from mip import OptimizationStatus
from sklearn.base import BaseEstimator
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_scalar


class BaseSparseEstimator(BaseEstimator, ABC):
    """Base class for sparse linear models.

    The features are selected using an efficient cutting plane that scales to thousands of features and samples. As
    the parameters and fitting procedure are the same for both regression and classification models, this class
    implements the shared functionality.

    Attributes:
        k: The sparsity parameter (i.e. number of non-zero coefficients). If `None`, then `k` is set to the square root
            of the number of features, rounded to the nearest integer.
        gamma: The regularization parameter. If `None`, then `gamma` is set to `1 / sqrt(n_samples)`.
        normalize: Whether to normalize the data before fitting the model.
        max_iters: The maximum number of iterations.
        tol: The tolerance for the stopping criterion.
        start: The initial guess for the selected features. For example if `start={0, 1, 2}`, then the first three
            features will be selected. If `None`, then the initial guess is randomly selected. Providing a good initial
            guess based on problem-specific knowledge can significantly speed up the search.
        feature_groups: Set of features that are mutually exclusive. For example, if `feature_groups=[{0, 1}, {2, 3}]`,
            then at most one features 0 and 1 will be selected, and at most one features 2 and 3 will be selected. This
            can be used to encode prior knowledge about the problem.
        solver: The solver to use for the optimization problem. The available options are "CBC" and "GUROBI". Support
            for the "HiGHS" solver is also planned for a future release.
        random_state: Controls the random seed for the initial guess if a user-defined initial guess is not provided.
        verbose: Whether to enable logging of the search progress.
    """

    _parameter_constraints: ClassVar[dict[str, list]] = {
        "k": [Interval(type=Integral, left=1, right=None, closed="left"), None],
        "gamma": [Interval(type=Real, left=0, right=None, closed="left"), None],
        "normalize": ["boolean"],
        "max_iters": [Interval(type=Integral, left=1, right=None, closed="left")],
        "tol": [Interval(type=Real, left=0, right=None, closed="left")],
        "start": ["array-like", None],
        "feature_groups": ["array-like", None],
        "solver": [StrOptions({"HiGHS", "CBC", "GUROBI"})],
        "random_state": ["random_state"],
        "verbose": ["boolean"],
    }

    def __init__(
        self,
        k: Optional[int] = None,
        gamma: Optional[float] = None,
        normalize: bool = True,
        max_iters: int = 500,
        tol: float = 1e-4,
        start: Optional[set[int]] = None,
        feature_groups: Optional[Sequence[set[int]]] = None,
        solver: str = "CBC",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """Model constructor.

        Args:
            k: The value for the `k` attribute.
            gamma: The value for the `gamma` attribute.
            normalize: The value for the `normalize` attribute.
            max_iters: The value for the `max_iters` attribute.
            tol: The value for the `tol` attribute.
            start: The value for the `start` attribute.
            feature_groups: The value for the `feature_groups` attribute.
            solver: The value for the `solver` attribute.
            random_state: The value for the `random_state` attribute.
            verbose: The value for the `verbose` attribute.
        """
        self.k = k
        self.gamma = gamma
        self.normalize = normalize
        self.max_iters = max_iters
        self.tol = tol
        self.start = start
        self.feature_groups = feature_groups
        self.solver = solver
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseSparseEstimator:
        """Fit the model to the training data.

        Args:
            X: The training data. The array should be of shape (n_samples, n_features).
            y: The training labels. The array-like should be of shape (n_samples,).

        Returns:
            The fitted model.
        """
        # Perform validation checks
        X, y = self._validate_data(X=X, y=y)
        self._validate_params()

        # Set hyperparameters to default values if not specified
        self._k = self.k or int(np.sqrt(X.shape[1]))
        self._gamma = self.gamma or 1 / np.sqrt(X.shape[0])

        # Pre-process training data
        if self.normalize:
            self._scaler_X = StandardScaler()
            X = self._scaler_X.fit_transform(X)
        y = self._pre_process_y(y=y)

        # Initialize feature selection
        if self.start is None:
            rng = check_random_state(self.random_state)
            start = rng.choice(X.shape[1], size=self._k, replace=False)
        else:
            start = self.start

        # Implement feature selection optimization model
        model = Model(
            solver_name=self.solver,
            max_gap=self.tol,
            max_gap_abs=self.tol,
            log_freq=1 if self.verbose else None,
        )
        selected = model.add_var_tensor(
            shape=(X.shape[1],), var_type="B", name="selected"
        )
        func = self._make_callback(X=X, y=y)
        model.add_objective_term(var=selected, func=func, grad=True)
        model.add_linear_constr(sum(selected) <= self._k)
        model.add_linear_constr(sum(selected) >= 1)
        if self.feature_groups:
            for group in self.feature_groups:
                model.add_linear_constr(sum(selected[i] for i in group) <= 1)
        model.start = [(selected[i], 1) for i in start]

        # Run solve and extract selected features
        status = model.optimize()
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            warnings.warn(
                f"Optimization failed with status: {status}.",
                category=FitFailedWarning,
            )
            return self
        selected = np.round([model.var_value(var) for var in selected]).astype(bool)

        # Compute coefficients
        self._coef = np.zeros(self.n_features_in_)
        self._coef[selected] = self._fit_coef_for_subset(X_subset=X[:, selected], y=y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model.

        Args:
            X: The training data. The array should be of shape (n_samples, n_features).

        Returns:
            The predicted values. The array will be of shape (n_samples,).
        """
        check_is_fitted(estimator=self)
        self._validate_data(X=X)
        if self.normalize:
            X = self._scaler_X.transform(X)
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
        super()._validate_params()
        if self.start is not None:
            for i in self.start:
                check_scalar(
                    x=i,
                    name="start",
                    target_type=int,
                    min_val=0,
                    max_val=self.n_features_in_,
                    include_boundaries="both",
                )
        if self.feature_groups is not None:
            for group in self.feature_groups:
                if not isinstance(group, set):
                    raise TypeError(
                        f"Each feature group must provided as a set, not type '{type(group)}'."
                    )
                for i in group:
                    check_scalar(
                        x=i,
                        name="start",
                        target_type=int,
                        min_val=0,
                        max_val=self.n_features_in_,
                        include_boundaries="both",
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
