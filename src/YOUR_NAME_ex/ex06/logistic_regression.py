# -*- coding: utf-8 -*-

r"""
A logistic regression estimator that follows the sklearn API.
-------------------------------------------------------------

In this exersice, you are supposed to implement a simple statistical and
machine learning model. Namely logistic regression. The way such a model
works is that you have some function :math:`f` that you wish to learn from
pairs of measurements, :math:`\mathbf{x}_i` and target values :math:`y_i`.

Disclaimer:
~~~~~~~~~~~
If you do not understand the following mathematics, that is ok. You are
can solve the solution by looking at the lecture notes for lecture 10
and asking your TA or peers for help.

The functions you should implement
----------------------------------

For this coursework, you must implement the functions required to compute the
gradient and the gradient descent algorithm. Specifically, you should
implement the following five functions.

  * ``sigmoid(z)``
  * ``predict_proba(w, X)``
  * ``logistic_gradient(w, X, y)``
  * ``LogisticRegression.__init__(self, max_iter=1000, tol=1e-5, learning_rate=0.01, random_state=None)``
  * ``LogisticRegression._fit_gradient_descent(self, coef, X, y)``
  * ``LogisticRegression._has_converged(self, coef, X, y)``

Read the docstring of these functions to understand how you should create
them.

Also, in the ``__main__`` block, you should add two lines to create and
fit a logistic regression model to some simulated data.

Note that for the source code, we use the name ``coef`` and ``coef_`` for 
the regression coefficients, whereas we use the name :math:`w` for
mathematical expression. This is to follow the standard set by scikit-learn.

Tips for your code
------------------

 * No function should be larger than 5 lines.
 * You don't need any other imported functions than ``numpy.exp`` and
   ``numpy.linalg.norm``.
 * You will need matrix multiplication (the ``@`` operator).
 * You shoul to use broadcasting (``x[:, np.newaxis]``), although the problem
   can be solved without it.
 * Look at the attributes field of the ``LogisticRegression`` docstring
   to see which attributes you should assign in the ``__init__`` method.

Some theoretical background
---------------------------

Motivation:
~~~~~~~~~~~~

In our case, the dataset may represent urine samples that we screen for
an infection. The :math:`\mathbf{x}_i` measurements may represent metabolomic
measurements (quantities of different key molecules) and :math:`y_i` is
equal to 1 if the patient has the disease we are screening for and is equal
to 0 otherwise.

Those of you familiar with statistics might realise that this problem is
well suited for logistic regression, where we assume that we can find a
function

.. math::

    p(\mathbf{x}; \mathbf{w}) = \frac{1}{1 + exp(-\mathbf{x}_i^T \mathbf{w})},

that gives us the probability of a patient being diseased or not. Our goal
is therefore to find a vector :math:`\mathbf{w}` so that 
:math:`p(\mathbf{x}_i)` is close to 1 if :math:`y_i=1` and close to 0 if 
:math:`y_i=0`.

Notation:
~~~~~~~~~~~~~

In the following paragraphs, it is useful to have the following notation

.. math::

    \hat{y}_i = p(\mathbf{x}; \mathbf{w})

Cost function:
~~~~~~~~~~~~~~
To find the function :math:`p`, we create a *cost function* 
:math:`C(\mathbf{w}; \mathbf{X}, \mathbf{y})` that takes the regression
coefficients, the data matrix and the true labels in as input, and tells us
how poorly our model performs with the given regression coefficients,
:math:`\mathbf{w}`. Intuitively, we say that it tells us the 
*missprediction-cost* of the regression coefficients. Our goal is then to
find a set of regression coefficients that minimises this cost.

If you are good at statistics, you might realise that a good cost function
for this problem is one on the following form:

.. math::

    C(\mathbf{w}; \mathbf{X}, \mathbf{y}) = -\sum_i y_i log(p(\mathbf{x}_i; \mathbf{w})) + (1-y_i) log(1 - p(\mathbf{x}_i; \mathbf{w})),

which, with the notation above, becomes

.. math::

    C(\mathbf{w}; \mathbf{X}, \mathbf{y}) = -\sum_i y_i log(\hat{y}_i) + (1 - y_i) log(1 - \hat{y}_i).

Finding the best model:
~~~~~~~~~~~~~~~~~~~~~~~

Now, we wish to find the :math:`\mathbf{w}` that minimise the cost function
above. To do that, we use *gradient descent*, which we learned about in
lecture 10. This optimisation algorithm works by iteratively modifying
:math:`\mathbf{w}` until we have a good guess for the best set of coefficients,
:math:`\mathbf{w}`.

The way gradient descent works is by realising that the gradient of a function,

.. math::

    \nabla_w C(\mathbf{w}; \mathbf{X}, \mathbf{y}),

is the "direction" in which the cost function changes the most rapidly. Thus,
if we want to make the small change in :math:`\mathbf{w}` that has the maximum
effect on the value of :math:`C`, then we update it the following way

.. math::

    \mathbf{w}^{\text{new}} = w - \eta \nabla_w C(\mathbf{w}; \mathbf{X}, \mathbf{y}),

where :math:`\mathbf{w}^{\text{new}}` is the new guess for a good set of 
regression coefficients and :math:`\eta` is a parameter that specifies how 
large the change in :math:`w` can be.

The difficult part of implementing the gradient descent algorithm is, in
other words, to compute the gradient of the cost function. Luckily, this is
not too complicated in the case of logistic regression. Here, the gradient
is given by

.. math::

    \nabla_w C(\mathbf{w}; \mathbf{X}, \mathbf{y}) = \sum_i \mathbf{x}_i (y_i - \hat{y}_i).

Final note
----------

You may wonder why some of the methods start with a single leading underscore.
In Python, all attributes of a class can be accessed from code outside the
class. However, sometimes, we wish to hide the implementation details. The
leading underscore is a way to tell other developers that this method is not
relevant unless you are actively modifying or inheriting from the class. If
you simply use instances of a class, then you shouldn't worry about (or use)
them.

Code
----
"""


__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, check_X_y


def sigmoid(z):
    r"""Perform a logistic transform on the input.

    This function applies the sigmoidal function element-wise to all
    elements of `z`. The sigmoidal function is on the following form:

    .. math::

        \frac{1}{1 + exp(-\mathbf{z})}.

    Parameters
    ----------
    z : np.ndarray
        Logit to transform.

    Returns
    -------
    sigmoidal_transformed_z : np.ndarray
        Transformed input.
    """
    # Your code here
    pass


def predict_proba(coef, X):
    r"""Predict the class probabilities for each data point in :math:`X`.

    Estimate which class each data point in X corresponds to. This is done
    according to the following formula.

    .. math::

        \hat{y}_i = \sigma(\mathbf{x}_i^T \mathbf{w}),

    where :math:`x_i` is the i-th row in :math:`X` and :math:`\sigma` is
    the sigmoidal function. Alternatively, in matrix-vector form:

    .. math::

        \hat{\mathbf{y}} = \sigma(X \mathbf{w}).

    Parameters
    ----------
    coef : np.ndarray(shape=(r,))
        The weight vector, :math:`w`
    X : np.ndarray(shape=(n, r))
        The data matrix (aka design or measurement matrix)

    Returns
    -------
    p : np.ndarray(shape(n,))
        The predicted class probabilities.
    """
    # Your code here
    pass


def logistic_gradient(coef, X, y):
    r"""Returns the gradient of a logistic regression model.

    The gradient is given by

    .. math::

        \nabla_w L(\mathbf{w}; X, \mathbf{y}) = \sum_i \mathbf{x}_i (y_i - \hat{y}_i),

    or, elementwise,

    .. math::

        \left[\nabla_w L(\mathbf{w}; X, \mathbf{y})\right]_j = \frac{\partial L}{\partial w_j}
                                                             = \sum_i X_{ij} (y_i - \hat{y}_i),

    where :math:`\hat{y}_i` is the predicted value for data point
    :math:`i` and is given by :math:`\sigma(x_i^Tw)`, where
    :math:`\sigma(z)` is the sigmoidal function.

    Parameters
    ----------
    coef : np.ndarray(shape=(r,))
        The weight vector, :math:`w`
    X : np.ndarray(shape=(n, r))
        The data matrix (aka design or measurement matrix)
    y : np.ndarray(shape=(n,))
        The true class labels for each data point.

    Returns
    -------
    gradient : np.ndarray(shape=(r,))
        The gradient of the cross entropy loss related to the linear
        logistic regression model.
    """
    # Your code here
    pass


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """A logistic regression classifier that follows the scikit-learn API.

    Note that the ``__init__`` method of scikit-learn estimators should not do
    any logic or input validation. This is all taken care of in the ``fit``
    method. 

    Parameters
    ----------
    max_iter : int (default=1000)
        Maximum number of gradient descent iterations to run.
    tol : float (default=1e-5)
        The gradient descent iterations will converge when the gradient
        norm is less than this.
    learning_rate : float (default=0.01)
        The step-size for the gradient descent updates.
    random_state : np.random.random_state or int or None (default=None)
        A numpy random state object or a seed for a numpy random state object.

    Attributes
    ----------
    coef_ : np.ndarray(shape=(r,))
        The logistic regression weights (initialised in ``self.fit``)
    max_iter : int (default=1000)
        Maximum number of gradient descent iterations to run.
    tol : float (default=1e-5)
        The gradient descent iterations will converge when the gradient
        norm is less than this.
    learning_rate : float (default=0.01)
        The step-size for the gradient descent updates.
    random_state : np.random.random_state or int or None (default=None)
        A numpy random state object or a seed for a numpy random state object.
    """

    def __init__(
        self, max_iter=1000, tol=1e-5, learning_rate=0.01, random_state=None
    ):
        """Initialise a logistic regression instance.

        The ``__init__`` method of scikit-learn estimators should not do any
        logic or input validation. This is all taken care of in the ``fit``
        method. 

        Parameters
        ----------
        max_iter : int (default=1000)
            Maximum number of gradient descent iterations to run.
        tol : float (default=1e-5)
            The gradient descent iterations will converge when the gradient
            norm is less than this.
        learning_rate : float (default=0.01)
            The step-size for the gradient descent updates.
        random_state : np.random.random_state or int or None (default=None)
            A numpy random state object or a seed for a numpy random state object.
        """
        # Your code here
        pass

    def _has_converged(self, coef, X, y):
        r"""Whether the gradient descent algorithm has converged.

        Returns True if the norm of the gradient is smaller than ``self.tol``,
        mathematically, that is

        .. math::

            ||\nabla_w L(\mathbf{w}^{(k)}; X, \mathbf{y})|| < T

        where :math:`\nabla_w L` is the gradient of the loss function,
        :math:`|| \mathbf{v} ||` is the norm of the vector :math:`\mathbf{v}`,
        :math:`\mathbf{w}^{(k)}` is the weights at iteration ``k``, and
        :math:`T` is the convergence tolerance (``self.tol``).

        Parameters
        ----------
        coef : np.ndarray(shape=(r,))
            The weight vector, :math:`\mathbf{w}^{(k)}`
        X : np.ndarray(shape=(n, r))
            The data matrix (aka design or measurement matrix)
        y : np.ndarray(shape=(n,))
            The true class labels for each data point.

        Returns
        -------
        has_converged : bool
            True if the convergence criteria above is met, False otherwise.
        """
        # Your code here
        pass

    def _fit_gradient_descent(self, coef, X, y):
        r"""Fit the logisitc regression model to the data given initial weights

        Gradient descent works by iteratively applying the following update
        rule

        .. math::

            \mathbf{w}^{(k)} \gets \mathbf{w}^{(k-1)} - \eta \nabla L(\mathbf{w}^{(k-1)}; X, \mathbf{y}),

        where :math:`\mathbf{w}^{(k)}` is the coefficient vector at iteration 
        ``k``, :math:`\mathbf{w}^{(k-1)}` is the coefficient vector at 
        iteration k-1, :math:`\eta` is the learning rate and 
        :math:`\nabla L(\mathbf{w}^{(k-1)}; X, \mathbf{y})` is the gradient of
        the loss function at iteration k-1.

        The iterative algorithm should be performed for at most
        ``self.max_iter`` iterations, or untill the convergence criteria is
        reached.

        Parameters
        ----------
        coef : np.ndarray(shape=(r,))
            The initial guess for the coefficient vector.
            May be modified inplace by the method.
        X : np.ndarray(shape=(n, r))
            The data matrix
        y : np.ndarray(shape=(n,))
            The target vector

        Returns
        -------
        coef : np.ndarray(shape=(n,))
            The logistic regression weights
        """
        # Your code here
        pass

    def fit(self, X, y):
        """Fit a logistic regression model to the data.

        Parameters
        ----------
        X : np.ndarray(shape=(n, r))
            The data matrix
        y : np.ndarray(shape=(n,))
            The observed classes for each data point in X.
        """
        # This function ensures that X and y has acceptable data types
        # and flattens y to have shape (n,) if it has shape (n, 1)
        X, y = check_X_y(X, y, order="C")

        if any((y < 0) | (y > 1)):
            raise ValueError("Only y-values between 0 and 1 are accepted.")

        # A random state is a random number generator, akin to those
        # you made in earlier coursework. It has all functions of
        # np.ranom, but its sequence of random numbers is not affected
        # by calls to np.random.
        random_state = check_random_state(self.random_state)
        coef = random_state.standard_normal(X.shape[1])

        self.coef_ = self._fit_gradient_descent(coef, X, y)
        return self

    def predict_proba(self, X):
        """Estimate the class probabilities.

        This function returns the probability that each datapoint belongs to
        the positive class.

        Parameters
        ----------
        X : np.ndarray
            The data matrix.

        Returns
        -------
        p : np.ndarray
            A vector of probabilities. The i-th entry is the probability for
            the i-th data point belonging to the positive class.
        """
        if not hasattr(self, "coef_"):
            raise NotFittedError("Call fit before prediction")
        return predict_proba(self.coef_, X)

    def predict_log_proba(self, X):
        """Estimate the class log probabilities.

        This function returns the probability that each datapoint belongs to
        the positive class.

        Parameters
        ----------
        X : np.ndarray
            The data matrix.

        Returns
        -------
        lp : np.ndarray
            A vector of log probabilities. The i-th entry is the log
            probability for the i-th data point belonging to the positive
            class.
        """
        return np.log(self.predict_proba(X))

    def predict(self, X):
        """Predict whether each data point in X belongs to the positive class

        Parameters
        ----------
        X : np.ndarray
            Data matrix

        Returns
        -------
        yhat : np.ndarray
            Predicted classes for the input data matrix. len(yhat) == len(X)
        """
        return self.predict_proba(X) >= 0.5


if __name__ == "__main__":
    # Simulate a random dataset
    X = np.random.standard_normal((100, 5))
    coef = np.random.standard_normal(5)
    y = predict_proba(coef, X) > 0.5

    # Fit a logistic regression model to the X and y vector
    # Fill in your code here.
    # Create a logistic regression object and fit it to the dataset

    # Print performance information
    print(f"Accuracy: {lr_model.score(X, y)}")
    print(f"True coefficients: {coef}")
    print(f"Learned coefficients: {lr_model.coef_}")
