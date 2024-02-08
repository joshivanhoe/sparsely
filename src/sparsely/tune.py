"""This module implements a function to tune the sparsity parameter of a linear model using cross-validation."""

from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_scalar
from tqdm.auto import tqdm

from .classifier import SparseLinearClassifier
from .regressor import SparseLinearRegressor

Estimator = Union[SparseLinearRegressor, SparseLinearClassifier]


def tune_estimator(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Estimator,
    k_min: int = 1,
    k_max: int = None,
    step_size: int = 1,
    max_iters_no_improvement: Optional[int] = None,
    cv: int = 3,
    return_search_log: bool = False,
    show_progress_bar: bool = False,
) -> Union[Estimator, tuple[Estimator, pd.DataFrame]]:
    """Tune the sparsity parameter (i.e. number of non-zero coefficients) of a linear model.

    The sparsity parameter is tuned by performing a grid search over the range [k_min, k_max] with step size
    `step_size`. If the test score does not improve for `max_iters_no_improvement` iterations, then the search is
    terminated early.

    Args:
        X: The training data. The array should be of shape (n_samples, n_features)
        y: The training labels. The array should be of shape (n_samples,).
        estimator: The estimator to tune. This must be a `SparseLinearRegressor` instance (for regression problems) or
            a `SparseLinearClassifier` instance (for classification problems).
        k_min: The minimum value for the sparsity parameter (i.e. number of non-zero coefficients).
        k_max: The maximum sparsity for the sparsity parameter (i.e. number of non-zero coefficients). If `None`, then
            this is set to `n_features`.
        step_size: The step size for the search. The sparsity parameter is incremented by this value at each iteration.
            Must be less than or equal to `k_max - k_min`.
        max_iters_no_improvement: The maximum number of iterations without improvement in the CV test score before the
            search is terminated. If `None`, then no early stopping is performed.
        cv: The number of cross-validation folds.
        return_search_log:Whether to return the search log.
        show_progress_bar: Whether to show a progress bar.

    Returns:
        The tuned estimator. If `return_search_log` is `True`, then a tuple of the tuned estimator and the search log.
    """
    # Perform validation checks
    k_max = k_max or X.shape[1]
    check_scalar(
        x=k_min,
        name="k_min",
        target_type=int,
        min_val=1,
        max_val=k_max,
        include_boundaries="left",
    )
    check_scalar(
        x=k_max,
        name="k_max",
        target_type=int,
        min_val=k_min,
        max_val=X.shape[1],
        include_boundaries="right",
    )
    check_scalar(
        x=step_size,
        name="step_size",
        target_type=int,
        min_val=1,
        max_val=k_max - k_min,
        include_boundaries="both",
    )
    if max_iters_no_improvement is not None:
        check_scalar(
            x=max_iters_no_improvement,
            name="max_iters_no_improvement",
            target_type=int,
            min_val=1,
            include_boundaries="left",
        )
    check_scalar(x=cv, name="cv", target_type=int, min_val=2, include_boundaries="left")

    # Initialize the search
    best_score = -np.inf
    best_k = None
    n_iters_no_improvement = 0
    search_log = list()

    for k in tqdm(
        range(k_min, (k_max or X.shape[1]) + 1, step_size),
        disable=not show_progress_bar,
    ):
        # Perform cross-validation
        output = cross_validate(
            estimator=deepcopy(estimator).set_params(k=k),
            X=X,
            y=y,
            cv=cv,
            scoring="auc" if isinstance(estimator, SparseLinearClassifier) else "r2",
            n_jobs=1,  # parallelization will interfere with the MIP solver
        )

        # Update search and check early termination condition
        score = output["test_score"].mean()
        if score > best_score:
            best_score = score
            best_k = k
        else:
            n_iters_no_improvement += 1
            if max_iters_no_improvement is not None:
                if n_iters_no_improvement > max_iters_no_improvement:
                    break
        search_log.append(dict(k=k, score=score, std=output["test_score"].std()))

    # Refit estimator with the best hyperparameter on full dataset and return
    estimator.set_params(k=best_k).fit(X=X, y=y),
    if return_search_log:
        return estimator, pd.DataFrame(search_log)
    return estimator
