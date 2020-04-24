"""
Feature Generator
"""
import math
from functools import partial
from itertools import combinations_with_replacement
from typing import Optional, Callable

import numpy as np
import pandas as pd


class FeatureGenerator:
    """FeatureGenerator class for feature augmentation before selection """

    def __init__(self, feature_df: pd.DataFrame, operators: list):
        """

        Args:
            feature_df(pd.DataFrame): original features
            operators(list): list of operators(str)
        """
        self.fdf = feature_df
        self.operators = operators

    def _generate_feature(self, feature_df: pd.DataFrame, operators: list) -> pd.DataFrame:
        """
        Generate new features by applying operators to columns in feature_df

        Args:
            feature_df(pd.DataFrame): dataframe of original features
            operators(list): list of str of operators (check Operator.support_op_rep for reference)

        Returns: dataframe of augmented features

        """
        fdf = feature_df.copy()
        if not np.all(o in Operator.support_op_rep for o in operators):
            raise ValueError("Contain unsupported operators, check Operator.supported_op_rep")
        ops = [Operator.from_str(o) for o in operators]
        sop = [op for op in ops if op.is_unary]
        bop = [op for op in ops if op.is_binary]
        for fn1, fn2 in combinations_with_replacement(fdf.columns, r=2):
            if fn1 == fn2:
                if not sop:
                    continue
                else:
                    for op in sop:
                        self._update_df(fdf, op, fn1)
            else:
                if not bop:
                    continue
                else:
                    for op in bop:
                        self._update_df(fdf, op, fn1, fn2)
                        if not op.commutative:
                            self._update_df(fdf, op, fn2, fn1)
        return fdf

    def _update_df(self, df, op, fn1, fn2=None):
        """Helper function to update the dataframe with new generated feature array"""
        if op.is_unary:
            new_fname = op.gen_name(fn1)
            df[new_fname] = df[fn1].apply(op)
        elif op.is_binary:
            new_fname = op.gen_name(fn1, fn2)
            df[new_fname] = op(df[fn1], df[fn2])

    def augment(self):
        """Augment features"""
        return self._generate_feature(self.fdf, self.operators)

    @staticmethod
    def generate_feature(feature_df: pd.DataFrame, operators: list) -> pd.DataFrame:
        """
        Generate new features by applying operators to columns in feature_df

        Args:
            feature_df(pd.DataFrame): dataframe of original features
            operators(list): list of str of operators (check Operator.support_op_rep for reference)

        Returns: dataframe of augmented features

        """
        return FeatureGenerator(feature_df, operators).augment()


class Operator:
    """
    Operator class. Wrap math operators with more attributes including check
    is_unary, is_binary, and is_commutative, and generate name string
    for the output.
    """
    support_op_rep = ['^2', '^3', 'sqrt', 'sqrtabs', 'cbrt', 'exp', 'abs', 'log10',
                      '+', '-', '*', '/', '|+|', '|-|', 'sum_power_2', 'sum_exp']

    def __init__(self, operation: Callable, rep: str, unary: bool, commutative: bool):
        """
        Args:
            operation(Callable): operation function
            rep(str): representations of the operator
            unary(bool): whether it is a unary operator
            commutative(bool): whether it is a commutative operator
        """
        self.opt = operation
        self.rep = rep
        self.unary = unary
        self.commutative = commutative

    @property
    def is_unary(self) -> bool:
        """
        Returns: True if the operator takes one argument else False

        """
        if not self.unary:
            if self.rep in ['^2', '^3', 'sqrt', 'sqrtabs', 'cbrt', 'exp', 'abs', 'log10']:
                self.unary = True
            elif self.rep in ['+', '-', '*', '/', '|+|', '|-|', 'sum_power_2', 'sum_exp']:
                self.unary = False
        return self.unary

    @property
    def is_binary(self) -> bool:
        """
        Returns: True if the operator takes two arguments else False

        """
        return False if self.is_unary else True

    @property
    def is_commutative(self) -> bool:
        """
        Returns: True if the operator is commutative else False

        """
        if not self.commutative:
            if self.is_unary:
                self.commutative = True
            elif self.rep in ['-', '/']:
                self.commutative = False
            else:
                self.commutative = True
        return self.commutative

    def compute(self, i1: np.ndarray, i2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the results
        Args:
            i1(np.ndarray): first input array
            i2(np.ndarray): second input array (for binary operators)

        Returns: array of computed results
        """
        if self.is_binary and not np.all(i2):
            raise ValueError("Please provide the second input for binary operator {}".format(self.rep))
        if self.is_unary:
            return self.opt(i1)
        else:
            return self.opt(i1, i2)

    def __call__(self, i1: np.ndarray, i2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the results
        Args:
            i1(np.ndarray): first input array
            i2(np.ndarray): second input array (for binary operators)

        Returns: array of computed results

        """
        return self.compute(i1, i2)

    def __str__(self) -> str:
        return self.rep

    def gen_name(self, f1: str, f2: Optional[str] = None) -> str:
        """
        Generate string representation for output
        Args:
            f1(str): name of the first input array
            f2(str): name of the second input array

        Returns: name of the output

        """
        if self.rep.startswith('^'):
            return ('({}){}'.format(f1, self.rep))
        if self.rep in ['sqrt', 'cbrt', 'exp', 'abs']:
            return ('{}({})'.format(self.rep, f1))
        if self.rep.startswith('log'):
            return ('{}({})'.format(self.rep, f1))
        if self.rep == 'sqrtabs':
            return ('sqrt(|{}|)'.format(f1))
        if self.rep in ['+', '-', '*', '/']:
            assert f2
            return ('(({}) {} ({}))'.format(f1, self.rep, f2))
        if self.rep in ['|+|', '|-|']:
            assert f2
            return ('|({}) {} ({})|'.format(f1, self.rep[1], f2))
        if self.rep == 'sum_power_2':
            assert f2
            return ('(({}) + ({}))^2'.format(f1, f2))
        if self.rep == 'sum_exp':
            assert f2
            return ('exp(({}) + ({}))'.format(f1, f2))
        else:
            if f2:
                return ('{}(({}) ({}))'.format(self.rep, f1, f2))
            else:
                return ('{}({})'.format(self.rep, f1))

    @classmethod
    def from_str(cls, op_name: str):
        """
        Operator from name of the operator
        Args:
            op_name(str): string representation of the operator,
            check Operator.support_op_rep for reference

        Returns: Operator

        """
        if op_name == 'sqrtabs':
            opt = _my_abs_sqrt
            return cls(operation=opt, rep=op_name, unary=True, commutative=False)

        if op_name.startswith('^'):
            n = int(op_name[1:])
            opt = partial(_my_power, n=n)
            return cls(operation=opt, rep=op_name, unary=True, commutative=False)

        if op_name == 'sqrt':
            opt = partial(_my_power, n=1 / 2)
            return cls(operation=opt, rep=op_name, unary=True, commutative=False)

        if op_name == 'cbrt':
            opt = partial(_my_power, n=1 / 3)
            return cls(operation=opt, rep=op_name, unary=True, commutative=False)

        if op_name == 'exp':
            opt = _my_exp
            return cls(operation=opt, rep=op_name, unary=True, commutative=False)

        if op_name == 'abs':
            opt = abs
            return cls(operation=opt, rep=op_name, unary=True, commutative=False)

        if op_name == 'log10':
            opt = np.log10
            return cls(operation=opt, rep=op_name, unary=True, commutative=False)

        if op_name == '+':
            opt = _my_sum
            return cls(operation=opt, rep=op_name, unary=False, commutative=True)

        if op_name == '|+|':
            opt = _my_abs_sum
            return cls(operation=opt, rep=op_name, unary=False, commutative=True)

        if op_name == '*':
            opt = _my_mul
            return cls(operation=opt, rep=op_name, unary=False, commutative=True)

        if op_name == '-':
            opt = _my_diff
            return cls(operation=opt, rep=op_name, unary=False, commutative=False)

        if op_name == '|-|':
            opt = _my_abs_diff
            return cls(operation=opt, rep=op_name, unary=False, commutative=True)

        if op_name == '/':
            opt = _my_div
            return cls(operation=opt, rep=op_name, unary=False, commutative=False)

        if op_name == 'sum_power_2':
            opt = _my_sum_power_2
            return cls(operation=opt, rep=op_name, unary=False, commutative=True)

        if op_name == 'sum_exp':
            opt = _my_sum_exp
            return cls(operation=opt, rep=op_name, unary=False, commutative=True)


def _my_power(x: float, n: int) -> float:
    return pow(x, n)


def _my_abs_sqrt(x):
    return math.sqrt(abs(x))


def _my_exp(x):
    return math.exp(x)


def _my_exp_power_2(x):
    return math.exp(pow(x, 2))


def _my_exp_power_3(x):
    return math.exp(pow(x, 3))


def _my_sum(x, y):
    return x + y


def _my_abs_sum(x, y):
    return abs(x + y)


def _my_mul(x, y):
    return x * y


def _my_diff(x, y):
    return x - y


def _my_abs_diff(x, y):
    return abs(x - y)


def _my_div(x, y):
    return x / y


def _my_sum_power_2(x, y):
    return pow((x + y), 2)


def _my_sum_power_3(x, y):
    return pow((x + y), 3)


def _my_sum_exp(x, y):
    return math.exp(x + y)


def _my_sum_exp_power_2(x, y):
    return math.exp(pow(x + y, 2))


def _my_sum_exp_power_3(x, y):
    return math.exp(pow(x + y, 3))
