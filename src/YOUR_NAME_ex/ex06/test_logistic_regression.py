from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest

from . import logistic_regression as lr


# The fixtures are called whenever their name is used as input to a test
# or another fixture. The output from the fixture call is used as the
# corresponding input.
@pytest.fixture
def coef():
    return np.array([1, -1])


@pytest.fixture
def X():
    return np.array([[1, 1], [100, -100], [-100, 100]])


# This function will call the `coef()` and `X()` fixture functions and use what
# those functions return as input.
@pytest.fixture
def y(coef, X):
    return lr.predict_proba(coef, X) > 0.5


def test_sigmoid():
    assert abs(lr.sigmoid(0) - 0.5) < 1e-8
    assert abs(lr.sigmoid(-1e5)) < 1e-8
    assert abs(lr.sigmoid(1e5) - 1) < 1e-8


# This function will call the `coef()` and `X()` fixture functions and use what
# those functions return as input.
def test_predict_proba(coef, X):
    probabilities = lr.predict_proba(coef, X)
    assert abs(probabilities[0] - 0.5) < 1e-8
    assert abs(probabilities[1] - 1) < 1e-8
    assert abs(probabilities[2]) < 1e-8


# This function will call the `coef()`, `X()` and `y(coef(), X())` fixture
# functions and use what those functions return as input.
def test_logistic_gradient(coef, X, y):
    p = lr.predict_proba(coef, X)
    assert np.linalg.norm(lr.logistic_gradient(coef, X, p)) < 1e-8

    assert np.linalg.norm(lr.logistic_gradient(np.array([1, 100]), X, y)) > 1

    gradient_norm = np.linalg.norm(lr.logistic_gradient(coef, X, y))
    assert abs(gradient_norm - 0.7071067811865) < 1e-8


@contextmanager
def patch_with_mock(container, name):
    """Mock and patch an object and reset it upon exit.

    The specified object is replaced with a Mock class that wraps it.
    When the context manager is exited, then it is reset to how it was
    previously.

    Arguments
    ---------
    container : module or class
        Any object that the getattr and setattr functions can be called on.
    name : str
        The name of the object to patch

    Examples
    --------
    >>> import numpy as np
    ... with patch_with_mock(np, 'array'):
    ...     a = np.array([1])
    ...     np.array.call_count  # -> 1
    ...     b = np.array([1, 2])
    ...     np.array.call_count  # -> 2
    ... hasattr(np.array, 'call_count')  # -> False
    """
    old_func = getattr(container, name)
    mocked_function = mock.Mock(wraps=old_func)
    setattr(container, name, mocked_function)
    yield
    setattr(container, name, old_func)


class TestLogisticRegression:
    """Test class for the logistic regression class.
    """

    def test_init(self):
        lr_model = lr.LogisticRegression(
            max_iter=10, tol=1e-6, learning_rate=1e-3, random_state=2
        )
        assert lr_model.max_iter == 10
        assert lr_model.tol == 1e-6
        assert lr_model.learning_rate == 1e-3
        assert lr_model.random_state == 2

    def test_gradient_descent_computes_gradient(self, X, y):
        with patch_with_mock(lr, "logistic_gradient"):
            lr_model = lr.LogisticRegression(max_iter=5)
            lr_model.fit(X, y)
            assert lr.logistic_gradient.call_count >= 5
