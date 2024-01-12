import numpy as np


def generate_dataset(
    n_samples: int = 1000,
    n_features: int = 100,
    n_true_features: int = 10,
    noise: float = 0.1,
    random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic dataset.

    Args:
        n_samples: int, default=1000
            Number of samples.
        n_features: int, default=100
            Number of features.
        n_true_features: int, default=10
            Number of features with non-zero coefficients.
        noise: float, default=0.1
            The standard deviation of the gaussian noise added to the data.
        random_state: int, default=0
            Random seed.

    Returns: tuple of np.ndarray
        The generated dataset. The first element is the feature matrix, the second element is the target vector, and the
        third element is the true coefficient vector.
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    coef = np.random.randn(n_features)
    coef[np.random.permutation(n_features)[:n_features - n_true_features]] = 0
    y = np.matmul(X, coef) + noise * np.random.randn(n_samples)
    return X, y, coef
