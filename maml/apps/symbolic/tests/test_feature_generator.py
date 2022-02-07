import unittest
import warnings
from functools import partial

import numpy as np
import pandas as pd
from pymatgen.util.testing import PymatgenTest

from maml.apps.symbolic import FeatureGenerator, Operator
from maml.apps.symbolic._feature_generator import (
    _my_abs_diff,
    _my_abs_sqrt,
    _my_abs_sum,
    _my_diff,
    _my_div,
    _my_exp,
    _my_exp_power_2,
    _my_exp_power_3,
    _my_mul,
    _my_power,
    _my_sum,
    _my_sum_exp,
    _my_sum_power_2,
    _my_sum_power_3,
    _update_df,
    generate_feature,
    operation_dict,
)

pow2 = Operator.from_str("^2")
pow3 = Operator.from_str("^3")
sqrt = Operator.from_str("sqrt")
cbrt = Operator.from_str("cbrt")
exp = Operator.from_str("exp")
log10 = Operator.from_str("log10")
my_abs = Operator.from_str("abs")
add = Operator.from_str("+")
sub = Operator.from_str("-")
mul = Operator.from_str("*")
div = Operator.from_str("/")
sum_power_2 = Operator.from_str("sum_power_2")
sum_exp = Operator.from_str("sum_exp")


class TestOperator(PymatgenTest):
    def setUp(self):
        x1 = np.array([2, 4, 6, 8, 10])
        x2 = np.array([1, 4, 9, 16, 25])
        x3 = np.array([1, 8, 27, 64, 125])
        x4 = np.array([1, -8, 27, -64, 125])
        x5 = np.array([1, -4, 9, -16, 25])
        self.df = pd.DataFrame({"i1": x1, "i2": x2, "i3": x3, "i4": x4, "i5": x5})

    def testFuncitons(self):
        self.assertArrayEqual(_my_power(self.df["i1"], n=2), np.array([4, 16, 36, 64, 100]))
        self.assertArrayEqual(_my_abs_sqrt(self.df["i5"]), np.array([1, 2, 3, 4, 5]))
        self.assertArrayEqual(_my_exp(self.df["i1"]), np.exp([2, 4, 6, 8, 10]))
        self.assertArrayEqual(_my_exp_power_2(self.df["i1"]), np.exp(self.df["i1"].pow(2)).values)
        self.assertArrayEqual(_my_exp_power_3(self.df["i1"]), np.exp(self.df["i1"].pow(3)).values)
        self.assertArrayEqual(_my_sum(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).values)
        self.assertArrayEqual(_my_abs_sum(self.df["i1"], self.df["i5"]), abs((self.df["i1"] + self.df["i5"]).values))
        self.assertArrayEqual(_my_mul(self.df["i1"], self.df["i2"]), (self.df["i1"] * self.df["i2"]).values)
        self.assertArrayEqual(_my_diff(self.df["i1"], self.df["i2"]), (self.df["i1"] - self.df["i2"]).values)
        self.assertArrayEqual(_my_abs_diff(self.df["i1"], self.df["i2"]), abs(self.df["i1"] - self.df["i2"]).values)
        self.assertArrayEqual(_my_div(self.df["i1"], self.df["i2"]), (self.df["i1"] / self.df["i2"]).values)
        self.assertArrayEqual(
            _my_sum_power_2(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).pow(2).values
        )
        self.assertArrayEqual(
            _my_sum_power_3(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).pow(3).values
        )
        self.assertArrayEqual(
            _my_sum_exp(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).apply(np.exp).values
        )

    def testSingularOperators(self):
        pow2_2 = Operator(operation=partial(_my_power, n=2), rep="^2", unary=True, commutative=False)
        pow2_3 = Operator(**operation_dict["^2"]["kwgs"])
        self.assertArrayEqual(pow2_2(self.df["i1"]), np.array([4, 16, 36, 64, 100]))
        self.assertTrue(pow2_2.is_unary)
        self.assertFalse(pow2_2.is_binary)
        self.assertFalse(pow2_2.is_commutative)
        self.assertArrayEqual(pow2_3(self.df["i1"]), np.array([4, 16, 36, 64, 100]))
        self.assertTrue(pow2_3.is_unary)
        self.assertFalse(pow2_3.is_binary)
        self.assertFalse(pow2_3.is_commutative)

        self.assertArrayEqual(pow2(self.df["i1"]), np.array([4, 16, 36, 64, 100]))
        self.assertArrayEqual(pow2.compute(self.df["i1"]), np.array([4, 16, 36, 64, 100]))
        self.assertArrayEqual(self.df["i1"].apply(pow2), np.array([4, 16, 36, 64, 100]))
        self.assertEqual(pow2.__str__(), pow2.rep)
        self.assertEqual(pow2.gen_name("i1"), "(i1)^2")
        self.assertTrue(pow2.is_unary)
        self.assertFalse(pow2.is_binary)
        self.assertFalse(pow2.is_commutative)

        self.assertArrayEqual(pow3(self.df["i1"]), np.array([8, 64, 36 * 6, 64 * 8, 1000]))
        self.assertEqual(pow3.gen_name("i1"), "(i1)^3")
        self.assertTrue(pow3.is_unary)
        self.assertFalse(pow3.is_binary)
        self.assertFalse(pow3.is_commutative)

        self.assertArrayEqual(sqrt(self.df["i2"]), np.array([1, 2, 3, 4, 5]))
        self.assertEqual(sqrt.gen_name("i1"), "sqrt(i1)")
        self.assertTrue(sqrt.is_unary)
        self.assertFalse(sqrt.is_binary)
        self.assertFalse(sqrt.is_commutative)

        self.assertArrayAlmostEqual(cbrt(self.df["i3"]), np.array([1, 2, 3, 4, 5]))
        self.assertEqual(cbrt.gen_name("i3"), "cbrt(i3)")
        self.assertTrue(cbrt.is_unary)
        self.assertFalse(cbrt.is_binary)
        self.assertFalse(cbrt.is_commutative)

        self.assertArrayEqual(log10(self.df["i1"]), self.df["i1"].apply(np.log10).values)
        self.assertEqual(log10.gen_name("i1"), "log10(i1)")
        self.assertTrue(log10.is_unary)
        self.assertFalse(log10.is_binary)
        self.assertFalse(log10.is_commutative)

        self.assertArrayEqual(exp(self.df["i1"]), self.df["i1"].apply(np.exp).values)
        self.assertEqual(exp.gen_name("i1"), "exp(i1)")
        self.assertTrue(exp.is_unary)
        self.assertFalse(exp.is_binary)
        self.assertFalse(exp.is_commutative)

        self.assertArrayEqual(my_abs(self.df["i4"]), self.df["i3"].values)
        self.assertEqual(my_abs.gen_name("i4"), "abs(i4)")
        self.assertTrue(my_abs.is_unary)
        self.assertFalse(my_abs.is_binary)
        self.assertFalse(my_abs.is_commutative)

        self.assertArrayEqual(add(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).values)
        self.assertEqual(add.gen_name("i1", "i2"), "((i1) + (i2))")
        self.assertFalse(add.is_unary)
        self.assertTrue(add.is_binary)
        self.assertTrue(add.is_commutative)

        self.assertArrayEqual(mul(self.df["i1"], self.df["i2"]), (self.df["i1"] * self.df["i2"]).values)
        self.assertEqual(mul.gen_name("i1", "i2"), "((i1) * (i2))")
        self.assertFalse(mul.is_unary)
        self.assertTrue(mul.is_binary)
        self.assertTrue(mul.is_commutative)

        self.assertArrayEqual(sub(self.df["i1"], self.df["i2"]), (self.df["i1"] - self.df["i2"]).values)
        self.assertEqual(sub.gen_name("i1", "i2"), "((i1) - (i2))")
        self.assertFalse(sub.is_unary)
        self.assertTrue(sub.is_binary)
        self.assertFalse(sub.is_commutative)

        self.assertArrayEqual(div(self.df["i1"], self.df["i2"]), (self.df["i1"] / self.df["i2"]).values)
        self.assertEqual(div.gen_name("i1", "i2"), "((i1) / (i2))")
        self.assertFalse(div.is_unary)
        self.assertTrue(div.is_binary)
        self.assertFalse(div.is_commutative)

        self.assertArrayEqual(sum_power_2(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).pow(2).values)
        self.assertEqual(sum_power_2.gen_name("i1", "i2"), "((i1) + (i2))^2")
        self.assertFalse(sum_power_2.is_unary)
        self.assertTrue(sum_power_2.is_binary)
        self.assertTrue(sum_power_2.is_commutative)

        self.assertArrayEqual(
            sum_exp(self.df["i1"], self.df["i2"]), (self.df["i1"] + self.df["i2"]).apply(np.exp).values
        )
        self.assertEqual(sum_exp.gen_name("i1", "i2"), "exp((i1) + (i2))")
        self.assertFalse(sum_exp.is_unary)
        self.assertTrue(sum_exp.is_binary)
        self.assertTrue(sum_exp.is_commutative)

    def testBinaryOperators(self):
        self.assertArrayEqual(add(self.df["i1"], self.df["i2"]), np.array([3, 8, 15, 24, 35]))
        self.assertTrue(add.__str__(), "+")
        self.assertEqual(add.gen_name("i1", "i2"), "((i1) + (i2))")
        self.assertFalse(add.is_unary)
        self.assertTrue(add.is_binary)
        self.assertTrue(add.is_commutative)

        self.assertArrayEqual(sub(self.df["i1"], self.df["i2"]), np.array([1, 0, -3, -8, -15]))
        self.assertTrue(sub.__str__(), "-")
        self.assertEqual(sub.gen_name("i1", "i2"), "((i1) - (i2))")
        self.assertFalse(sub.is_unary)
        self.assertTrue(sub.is_binary)
        self.assertFalse(sub.is_commutative)


class TestFeatureGenerator(PymatgenTest):
    def setUp(self):
        x1 = np.array([2, 4, 6, 8, 10])
        x2 = np.array([1, 4, 9, 16, 25])
        x3 = np.array([1, 8, 27, 64, 125])
        self.df = pd.DataFrame({"i1": x1, "i2": x2, "i3": x3})

    def testUpdateDf(self):
        test_df = self.df.copy()
        _update_df(test_df, pow2, "i1")
        self.assertTrue("(i1)^2" in test_df.columns)
        self.assertArrayEqual(test_df["(i1)^2"], test_df["i1"].pow(2).values)

        # Test duplicate won't happen
        _update_df(test_df, pow2, "i1")
        self.assertArrayEqual(test_df.columns, np.array(["i1", "i2", "i3", "(i1)^2"]))

        _update_df(test_df, add, "i1", "i2")
        self.assertTrue("((i1) + (i2))" in test_df.columns)
        self.assertArrayEqual(test_df["((i1) + (i2))"], (test_df["i1"] + test_df["i2"]).values)

        _update_df(test_df, add, "i1", "i2")
        self.assertArrayEqual(test_df.columns, np.array(["i1", "i2", "i3", "(i1)^2", "((i1) + (i2))"]))

        # Test negative with sqrt and log10
        test_df["i4"] = np.array([1, -8, 27, -64, 125])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _update_df(test_df, sqrt, "i4")
            self.assertEqual(len(w), 1)
            self.assertTrue("abssqrt" in str(w[-1].message))

    def testGenerateFeature(self):
        ops = ["^2", "+", "-"]
        nf_df = generate_feature(self.df, ops)
        new_columns = [
            "i1",
            "i2",
            "i3",
            "(i1)^2",
            "((i1) + (i2))",
            "((i1) - (i2))",
            "((i2) - (i1))",
            "((i1) + (i3))",
            "((i1) - (i3))",
            "((i3) - (i1))",
            "(i2)^2",
            "((i2) + (i3))",
            "((i2) - (i3))",
            "((i3) - (i2))",
            "(i3)^2",
        ]
        self.assertArrayEqual(nf_df.columns, np.array(new_columns))
        self.assertArrayEqual(nf_df["(i1)^2"].values, self.df["i1"].pow(2).values)
        self.assertArrayEqual(nf_df["((i1) + (i2))"].values, (self.df["i1"] + self.df["i2"]).values)
        self.assertArrayEqual(nf_df["((i1) - (i2))"].values, (self.df["i1"] - self.df["i2"]).values)
        self.assertArrayEqual(nf_df["((i2) - (i1))"].values, (self.df["i2"] - self.df["i1"]).values)

        ops = ["^2", "abcd"]
        self.assertRaises(ValueError, generate_feature, self.df, ops)

    def testFeatureGenerator(self):
        ops = ["^2", "+", "-"]
        fg = FeatureGenerator(self.df, ops)
        nf_df = fg.augment(n=1)
        new_columns = [
            "i1",
            "i2",
            "i3",
            "(i1)^2",
            "((i1) + (i2))",
            "((i1) - (i2))",
            "((i2) - (i1))",
            "((i1) + (i3))",
            "((i1) - (i3))",
            "((i3) - (i1))",
            "(i2)^2",
            "((i2) + (i3))",
            "((i2) - (i3))",
            "((i3) - (i2))",
            "(i3)^2",
        ]
        self.assertArrayEqual(nf_df.columns, np.array(new_columns))
        self.assertArrayEqual(nf_df["(i1)^2"].values, self.df["i1"].pow(2).values)
        self.assertArrayEqual(nf_df["((i1) + (i2))"].values, (self.df["i1"] + self.df["i2"]).values)
        self.assertArrayEqual(nf_df["((i1) - (i2))"].values, (self.df["i1"] - self.df["i2"]).values)
        self.assertArrayEqual(nf_df["((i2) - (i1))"].values, (self.df["i2"] - self.df["i1"]).values)

        df = self.df[["i1", "i2"]]
        fg = FeatureGenerator(df, ["^2", "+"])
        new_f1 = ["i1", "i2", "(i1)^2", "((i1) + (i2))", "(i2)^2"]
        new_f2 = [
            "i1",
            "i2",
            "(i1)^2",
            "((i1) + (i2))",
            "(i2)^2",
            "((i1) + ((i1)^2))",
            "((i1) + (((i1) + (i2))))",
            "((i1) + ((i2)^2))",
            "((i2) + ((i1)^2))",
            "((i2) + (((i1) + (i2))))",
            "((i2) + ((i2)^2))",
            "((i1)^2)^2",
            "(((i1)^2) + (((i1) + (i2))))",
            "(((i1)^2) + ((i2)^2))",
            "(((i1) + (i2)))^2",
            "((((i1) + (i2))) + ((i2)^2))",
            "((i2)^2)^2",
        ]
        nf1_df = fg.augment(n=1)
        nf2_df = fg.augment(n=2)
        self.assertArrayEqual(nf1_df.columns, np.array(new_f1))
        self.assertArrayEqual(nf2_df.columns, np.array(new_f2))


if __name__ == "__main__":
    unittest.main()
