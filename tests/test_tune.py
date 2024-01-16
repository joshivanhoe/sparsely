from sparsely import tune_estimator
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("max_iters_no_improvement", [None, 1])
@pytest.mark.parametrize("return_search_log", [True, False])
def test_tune_estimator(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    max_iters_no_improvement: int,
    return_search_log: bool,
):
    X_train, X_test, y_train, y_test, coef = dataset
    output = tune_estimator(
        X_train,
        y_train,
        k_min=1,
        k_max=5,
        max_iters_no_improvement=max_iters_no_improvement,
        return_search_log=return_search_log,
    )
    if return_search_log:
        estimator, search_log = output
        assert isinstance(search_log, pd.DataFrame)
        assert (search_log.columns == ["k", "score", "std"]).all()
        if max_iters_no_improvement is None:
            assert len(search_log) == 5
        else:
            assert 1 < len(search_log) < 5
    else:
        estimator = output
    assert estimator.score(X_train, y_train) > 0.8
    assert estimator.score(X_test, y_test) > 0.8
    assert estimator.coef_.shape == (X_train.shape[1],)
    assert (~np.isclose(coef, 0)).sum() <= estimator.k_
    assert (np.isclose(estimator.coef_, 0) == np.isclose(coef, 0)).all()
