"""
Sure Independence Screening

https://orfe.princeton.edu/~jqfan/papers/06/SIS.pdf

"""
from typing import Optional, Dict, List
import logging

import numpy as np

from ._selectors import BaseSelector, DantzigSelector

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SIS:
    """
    Sure independence screening method.
    The method consists of two steps:
        1. Screen
        2. Select

    """

    def __init__(self, gamma=0.1, selector: Optional[BaseSelector] = None,
                 verbose: bool = True):
        """
        Sure independence screening

        Args:
            gamma (float): ratio between selected features and original feature sizes
            selector (BaseSelector): selector after the screening
            verbose (bool): whether to output information along the way

        """
        self.gamma = gamma
        self.selector = selector
        self.verbose = verbose

    def run(self, x, y, select_options=None):
        """
        Run the SIS with selector
        Args:
            x (np.ndarray): MxN input data array
            y (np.ndarray): M output targets
            select_options (dict): options in the optimizations provided
                to scipy.optimize.minimize. If the selector is using cvxpy
                optimization package, this option is fed into cp.Problem.solve

        Returns: selected feature indices

        """
        screened_indices = self.screen(x, y)
        if self.verbose:
            logger.info(f"After the screening step, {len(screened_indices)}/{x.shape[1]} features remains")
        x_screen = x[:, screened_indices]

        final_selected = self.select(x_screen, y, select_options)

        if self.verbose:
            logger.info(f"After the selection step, {len(final_selected)}/{x.shape[1]} features remains")
        return screened_indices[final_selected]

    def screen(self, x, y):
        """
        Simple screening method by comparing the correlation between features
        and the target

        Args:
            x (np.ndarray): input array
            y (np.ndarray): target array

        Returns: top indices

        """
        n = x.shape[1]
        omega = x.T.dot(y)
        sorted_omega = np.argsort(omega)[::-1]
        d = int(n * self.gamma)
        top_indices = sorted_omega[:d]
        return top_indices

    def select(self, x, y, options=None):
        """
        Select features using selectors
        Args:
            x (np.ndarray): input array
            y (np.ndarray): target array
            options (dict): options for the optimization

        Returns:

        """
        return self.selector.select(x, y, options)

    def compute_residual(self, x, y):
        """
        Compute residual
        Args:
            x (np.ndarray): input array
            y (np.ndarray): target array

        Returns: residual vector

        """
        return self.selector.compute_residual(x, y)

    def set_selector(self, selector: BaseSelector):
        """
        Set new selector
        Args:
            selector (BaseSelector): a feature selector

        Returns:

        """
        self.selector = selector

    def set_gamma(self, gamma):
        """
        Set gamma

        Args:
            gamma(float): new gamma value

        """
        self.gamma = gamma

    def update_gamma(self, step: float = 0.5):
        """
        Update the sis object so that sis.select
        return at least one feature

        Args:
            step(float): ratio to update the parameters

        """
        self.set_gamma(self.gamma * (1 + step))


class ISIS:
    """Iterative SIS"""

    def __init__(self,
                 sis: SIS = SIS(gamma=0.1, selector=DantzigSelector(0.1))):
        """

        Args:
            sis(SIS): sis object
        """
        self.sis = sis
        self.selector = sis.selector

    def run(self, x, y, max_p: int = 10,
            options: Optional[Dict] = None,
            step: float = 0.5) -> np.array:
        """
        Run the ISIS
        Args:
            x:
            y:
            max_p(int): Number of feature desired
            options:
            step(float): update gamma with

        Returns: np.array of index of selected features

        """
        assert max_p <= x.shape[1]
        findex = np.array(np.arange(0, x.shape[1]))
        find_sel = self.sis.select(x, y, options)
        if len(find_sel) >= max_p:
            return find_sel[:max_p]
        new_findex = np.array(list(set(findex) - set(find_sel)))
        new_y = self.sis.compute_residual(x, y)
        new_x = x[:, new_findex]
        while len(find_sel) < max_p:
            find_sel_new: List[int] = []
            try:
                find_sel_new = self.sis.run(new_x, new_y, options)
            except ValueError:
                while len(find_sel_new) == 0:
                    self.sis.update_gamma(step)
                    find_sel_new = self.sis.run(new_x, new_y)
            find_sel = np.append(find_sel, new_findex[find_sel_new])
            new_findex = np.array(list(set(findex) - set(find_sel)))
            new_y = self.sis.compute_residual(new_x, new_y)
            new_x = x[:, new_findex]
        return find_sel
