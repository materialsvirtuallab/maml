"""
Simple numba utility.
Some functions can excelerated substantially with numba.
"""
from __future__ import annotations

from typing import Callable

try:
    from numba import njit  # type: ignore
except ModuleNotFoundError:

    def njit(func: Callable) -> Callable:
        """
        Dummy decorator, returns the original function
        Args:
            func (Callable): function to be wrapped.

        Returns: decorated function

        """
        return func
