[![CI](https://github.com/joshivanhoe/sparsely/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/joshivanhoe/sparsely/actions/workflows/ci.yml)
[![Version](https://img.shields.io/pypi/v/sparsely?color=blue)](https://pypi.org/project/sparsely/)


# ⚡ sparsely ⚡
`sparsely` is a `sklearn`-compatible Python module for sparse linear regression and classification. It uses an efficient cutting-plane algorithm to optimize feature selection, which scales to thousands of samples and features.
This implementation follows [Bertsimas & Van Parys (2017)](https://arxiv.org/pdf/1709.10029.pdf) for regression, and [Bertsimas, Pauphilet & Van Parys (2021)](https://link.springer.com/article/10.1007/s10994-021-06085-5) for classification.

Full API documentation can be found [here](https://joshivanhoe.github.io/sparsely/).

## Quick start

You can install `sparsely` using `pip` as follows:

```bash
pip install sparsely
```

Here is a simple example of how use a `sparsely` estimator:

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sparsely import SparseLinearRegressor

X,y = make_regression(n_samples=1000, n_features=100, n_informative=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = SparseLinearRegressor(k=10)  # k is the max number of non-zero coefficients
estimator.fit(X_train, y_train)
print(estimator.score(X_test, y_test))
```

## Development

Clone the repository using `git`:

```bash
git clone https://github.com/joshivanhoe/sparsely
````

Create a fresh virtual environment using `venv` or `conda`.
Activate the environment and navigate to the cloned `halfspace` directory.
Install a locally editable version of the package using `pip`:

```bash
pip install -e .
```

To check the installation has worked, you can run the tests (with coverage metrics) using `pytest` as follows:

```bash
pytest --cov=sparsely tests/
```

Contributions are welcome! To see our development priorities, refer to the [open issues](https://github.com/joshivanhoe/sparsely/issues).
Please submit a pull request with a clear description of the changes you've made.
