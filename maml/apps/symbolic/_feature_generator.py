from itertools import combinations_with_replacement
import numpy as np
import pandas as pd
import math
from functools import partial
from typing import Optional

def features_generator(df: pd.DataFrame, operators: list) -> pd.DataFrame:
    """
    Generate new features by applying operators to columns in df

    Args:
        df: dataframe of original features
        operators: list of str of operators (check Operator.support_op_rep for reference)

    Returns: dataframe of new features

    """
    if not np.all(o in Operator.support_op_rep for o in operators):
        raise ValueError("Contain unsupported operators, check Operator.supported_op_rep")
    ops = [Operator.from_str(o) for o in operators]
    sop = [op for op in ops if op.is_singular]
    bop = [op for op in ops if op.is_binary]
    for fn1, fn2 in combinations_with_replacement(df.columns, r=2):
        if fn1 == fn2:
            if not sop:
                continue
            else:
                for op in sop:
                    new_fname = op.gen_name(fn1)
                    df[new_fname] = df[fn1].apply(op)
        else:
            if not bop:
                continue
            else:
                for op in bop:
                    new_fname = op.gen_name(fn1, fn2)
                    df[new_fname] = op(df[fn1], df[fn2])
                    if not op.commutative:
                        new_fname = op.gen_name(fn2, fn1)
                        df[new_fname] = op(df[fn2], df[fn1])
    return df


class Operator:
    support_op_rep = ['^2', '^3', 'sqrt', 'sqrtabs', 'cbrt', 'exp', 'abs', 'log10',
                      '+', '-', '*', '/', '|+|', '|-|', 'sum_power_2', 'sum_exp']

    def __init__(self, operation, rep: str, singular: bool, commutative: bool):
        """
        Operator class
        Args:
            operation: operation function
            rep: representations of the operator
            singular: whether it is a singular operator
            binary: whether it is a bianry operator
            commutative: whether it is a commutative operator
        """
        self.opt = operation
        self.rep = rep
        self.singular = singular
        self.commutative = commutative

    @property
    def is_singular(self) -> bool:
        if not self.singular:
            if self.rep in ['^2', '^3', 'sqrt', 'sqrtabs', 'cbrt', 'exp', 'abs', 'log10']:
                self.singular = True
            elif self.rep in ['+', '-', '*', '/', '|+|', '|-|', 'sum_power_2', 'sum_exp']:
                self.singular = False
        return self.singular

    @property
    def is_binary(self) -> bool:
        return False if self.is_singular else True

    @property
    def is_commutative(self) -> bool:
        if not self.commutative:
            if self.is_singular:
                self.commutative = True
            elif self.rep in ['-', '/']:
                self.commutative = False
            else:
                self.commutative = True
        return self.commutative

    def compute(self, i1: np.ndarray, i2: Optional[np.ndarray] = None) -> np.ndarray:
        if self.is_singular:
            return self.opt(i1)
        else:
            if np.any(i2):
                return self.opt(i1, i2)
            else:
                raise ValueError("Please provide the second input for binary operator {}".format(self.rep))

    def __call__(self, i1: np.ndarray, i2: Optional[np.ndarray] = None) -> np.ndarray:
        return self.compute(i1, i2)

    def __str__(self) -> str:
        return self.rep

    def gen_name(self, f1: str, f2: Optional[str] = None) -> str:
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
        if op_name.startswith('^'):
            n = int(op_name[1:])
            opt = partial(_my_power, n=n)
            return cls(operation=opt, rep=op_name, singular=True, commutative=False)

        if op_name == 'sqrt':
            opt = partial(_my_power, n=1 / 2)
            return cls(operation=opt, rep=op_name, singular=True, commutative=False)

        if op_name == 'sqrtabs':
            opt = _my_abs_sqrt
            return cls(operation=opt, rep=op_name, singular=True, commutative=False)

        if op_name == 'cbrt':
            opt = partial(_my_power, n=1 / 3)
            return cls(operation=opt, rep=op_name, singular=True, commutative=False)

        if op_name == 'exp':
            opt = _my_exp
            return cls(operation=opt, rep=op_name, singular=True, commutative=False)

        if op_name == 'abs':
            opt = abs
            return cls(operation=opt, rep=op_name, singular=True, commutative=False)

        if op_name.startswith('log'):
            n = op_name[3:]
            opt = partial(math.log, base=n)
            return cls(operation=opt, rep=op_name, singular=True, commutative=False)

        if op_name == '+':
            opt = _my_sum
            return cls(operation=opt, rep=op_name, singular=False, commutative=True)

        if op_name == '|+|':
            opt = _my_abs_sum
            return cls(operation=opt, rep=op_name, singular=False, commutative=True)

        if op_name == '*':
            opt = _my_mul
            return cls(operation=opt, rep=op_name, singular=False, commutative=True)

        if op_name == '-':
            opt = _my_diff
            return cls(operation=opt, rep=op_name, singular=False, commutative=False)

        if op_name == '|-|':
            opt = _my_abs_diff
            return cls(operation=opt, rep=op_name, singular=False, commutative=True)

        if op_name == '/':
            opt = _my_div
            return cls(operation=opt, rep=op_name, singular=False, commutative=False)

        if op_name == 'sum_power_2':
            opt = _my_sum_power_2
            return cls(operation=opt, rep=op_name, singular=False, commutative=True)

        if op_name == 'sum_exp':
            opt = _my_sum_exp
            return cls(operation=opt, rep=op_name, singular=False, commutative=True)


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

