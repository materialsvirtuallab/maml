"""
This module implements more robust optmization 
using the cvxpy package
"""
from typing import Optional, List, Dict, Union

from monty.dev import requires
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import get_scorer
from scipy.linalg import lstsq


try:
    import cvxpy as cp
    Expression = cp.expressions.expression.Expression
except ImportError:
    cp = None
    Expression = "Expression"


class BaseSelectorCP:
    """
    Base selector using cvxpy (CP)
    """

    @requires(cp is not None, "cvxpy is not present.")
    def __init__(self, coef_thres: float = 1e-6, method: Optional[str] = None):
        """
        Base selector
        Args:
            coef_thres (float): threshold to discard certain coefficents
            method (str): solver for cvxpy problem, if None, then ECOS will be used
        """
        self.coef_thres = coef_thres
        self.is_fitted = False
        self.coef_ = None
        self.method = method

    def select(self, x: np.ndarray, y: np.ndarray,
               options: Optional[Dict] = None) -> np.ndarray:
        """
        Select feature indices from x

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            options (dict): kwargs for cp.Problem.solve
        Returns: list of int indices
        """
        options = options or {}
        n, p = x.shape
        beta = cp.Variable(p)
        objective = cp.Minimize(self.construct_loss(x, y, beta))
        constraints = self.construct_constraints(x, y, beta)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.method, **options)
        self.coef_ = beta.value
        return np.where(np.abs(self.coef_) > self.coef_thres)[0]

    def construct_constraints(self, x: np.ndarray, y: np.ndarray, beta: cp.Variable) \
            -> Optional[List[Expression]]:  # type: ignore
        """
        Get constraints dictionary from data, e.g.,
        {"func": lambda beta: fun(x, y, beta), "type": "ineq"}

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta: (cp.Variable): target variable for optimization

        Returns: dict of constraints

        """
        return None

    def construct_loss(self, x: np.ndarray, y: np.ndarray, beta: cp.Variable) \
            -> Expression:  # type: ignore
        """
        Get loss function from data and tentative coefficients beta

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): N coefficients

        Returns: loss value

        """
        raise NotImplementedError

    def evaluate(self, x: np.ndarray, y: np.ndarray,
                 metric: str = 'neg_mean_absolute_error') -> float:
        """
        Evaluate the linear model using x, and y test data

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            metric (str): scorer function, used with
                sklearn.metrics.get_scorer

        Returns:

        """
        metric_func = get_scorer(metric)
        lr = LinearRegression(fit_intercept=False)
        lr.coef_ = self.coef_
        lr.intercept_ = 0
        return metric_func(lr, x, y)

    def get_coef(self) -> np.ndarray:
        """
        Get coefficients
        Returns: the coefficients array

        """
        return self.coef_


class DantzigSelectorCP(BaseSelectorCP):
    """
    Equation 11 in
    https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf

    and reference in https://projecteuclid.org/download/pdfview_1/euclid.aos/1201012958
    """
    @requires(cp is not None, "cvxpy is not present.")
    def __init__(self, lambd, sigma=1.0, **kwargs):
        """
        Dantzig selector

        Args:
            lamb: tunable parameter
            sigma: standard deviation of the error
        """
        self.lambd = lambd
        self.sigma = sigma
        super().__init__(**kwargs)

    def construct_loss(self, x: np.ndarray, y: np.ndarray, beta: cp.Variable) \
            -> Expression:  # type: ignore
        """
        L1 loss
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (cp.Variable): dimension N vector for optimization

        Returns: loss expression

        """
        return cp.norm1(beta)

    def construct_constraints(self, x: np.ndarray, y: np.ndarray, beta: cp.Variable) \
            -> Optional[List[Expression]]:  # type: ignore
        """
        Dantzig selector constraints

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (cp.Variable): dimension N vector for optimization
        Returns: List of constraints
        """
        return [cp.norm_inf(x.T @ (y - x @ beta)) <= self.lambd * self.sigma]


class PenalizedLeastSquaresCP(BaseSelectorCP):
    """
    Penalized least squares. In addition to minimizing the sum of squares loss,
    it adds an additional penalty to the coefficients
    """

    def construct_loss(self, x: np.ndarray, y: np.ndarray, beta: cp.Variable) \
            -> Expression:  # type: ignore
        """
        L1 loss
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (cp.Variable): dimension N vector for optimization

        Returns: loss expression

        """
        n = x.shape[0]
        se = 1. / (2 * n) * cp.sum_squares(y - x @ beta) + self.penalty(beta, x=x, y=y)
        return se

    def penalty(self, beta: cp.Variable, x: Optional[np.ndarray] = None,
                y: Optional[np.ndarray] = None) -> Union[Expression, float]:  # type: ignore
        """
        Calculate the penalty from input x, output y and coefficient beta

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): N coefficients

        Returns: penalty value

        """
        return 0.


class LassoCP(PenalizedLeastSquaresCP):
    """
    Simple Lasso regression
    """
    @requires(cp is not None, "cvxpy not installed")
    def __init__(self, lambd, **kwargs):
        """
        Lasso regression with lambda * norm_1(beta) as penalty
        Args:
            lambd (float): weights for the penalty
            **kwargs:
        """
        self.lambd = lambd
        super().__init__(**kwargs)

    def penalty(self, beta: cp.Variable, x: Optional[np.ndarray] = None,
                y: Optional[np.ndarray] = None) -> Union[Expression, float]:  # type: ignore
        """
        Calculate the penalty from input x, output y and coefficient beta

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): N coefficients

        Returns: penalty value

        """
        beta_abs = cp.norm1(beta)
        return self.lambd * beta_abs


class AdaptiveLassoCP(PenalizedLeastSquaresCP):
    """
    Adaptive lasso regression using OLS coefficients
    as the root-n estimator coefficients
    """
    @requires(cp is not None, "cvxpy not installed")
    def __init__(self, lambd, gamma, **kwargs):
        """
        Adaptive lasso regression

        Args:
            lambd (float or list of floats):
            gamma (float): exponential for hat(beta)
            **kwargs:
        """
        self.lambd = lambd
        self.gamma = gamma
        self.w = 1
        super().__init__(**kwargs)

    def select(self, x: np.ndarray, y: np.ndarray,
               options: Optional[Dict] = None) -> np.ndarray:
        """
        Select feature indices from x

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            options (dict): options in the cp.Problem.solve
        Returns: array int indices
        """
        self.w = self.get_w(x, y)
        return super().select(x, y, options)

    def get_w(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get adaptive weights from data

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets

        Returns: coefficients array

        """
        beta_hat = lstsq(x, y)[0]
        w = 1. / np.abs(beta_hat) ** self.gamma
        return w

    def penalty(self, beta: cp.Variable, x: Optional[np.ndarray] = None,
                y: Optional[np.ndarray] = None) -> Union[Expression, float]:  # type: ignore
        """
        Calculate the penalty from input x, output y and coefficient beta

        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            beta (np.ndarray): N coefficients

        Returns: penalty value

        """
        return cp.sum(self.lambd * cp.multiply(self.w, cp.abs(beta)))
