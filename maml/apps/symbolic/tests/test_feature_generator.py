import unittest
import pandas as pd
import numpy as np

from pymatgen.util.testing import PymatgenTest

from maml.apps.symbolic import FeatureGenerator, Operator


pow2 = Operator.from_str('^2')
pow3 = Operator.from_str('^3')
sqrt = Operator.from_str('sqrt')
cbrt = Operator.from_str('cbrt')
exp = Operator.from_str('exp')
log10 = Operator.from_str('log10')
my_abs = Operator.from_str('abs')
add = Operator.from_str('+')
sub = Operator.from_str('-')
mul = Operator.from_str('*')
div = Operator.from_str('/')
sum_power_2 = Operator.from_str('sum_power_2')
sum_exp = Operator.from_str('sum_exp')

class TestOperator(PymatgenTest):
    def setUp(self):
        x1 = np.array([2, 4, 6, 8, 10])
        x2 = np.array([1, 4, 9, 16, 25])
        x3 = np.array([1, 8, 27, 64, 125])
        x4 = np.array([1, -8, 27, -64, 125])
        self.df = pd.DataFrame({"i1":x1, "i2":x2, "i3": x3, 'i4': x4})

    def testSingularOperators(self):
        self.assertArrayEqual(pow2(self.df["i1"]), np.array([4, 16, 36, 64, 100]))
        self.assertArrayEqual(self.df["i1"].apply(pow2), np.array([4, 16, 36, 64, 100]))
        self.assertEqual(pow2.gen_name('i1'), '(i1)^2')
        self.assertTrue(pow2.is_unary)
        self.assertFalse(pow2.is_binary)
        self.assertTrue(pow2.is_commutative)

        self.assertArrayEqual(pow3(self.df["i1"]), np.array([8, 64, 36*6, 64*8, 1000]))
        self.assertEqual(pow3.gen_name('i1'), '(i1)^3')
        self.assertTrue(pow3.is_unary)
        self.assertFalse(pow3.is_binary)
        self.assertTrue(pow3.is_commutative)

        self.assertArrayEqual(sqrt(self.df["i2"]), np.array([1, 2, 3, 4, 5]))
        self.assertEqual(sqrt.gen_name('i1'), 'sqrt(i1)')
        self.assertTrue(sqrt.is_unary)
        self.assertFalse(sqrt.is_binary)
        self.assertTrue(sqrt.is_commutative)

        self.assertArrayAlmostEqual(cbrt(self.df["i3"]), np.array([1, 2, 3, 4, 5]))
        self.assertEqual(cbrt.gen_name('i3'), 'cbrt(i3)')
        self.assertTrue(cbrt.is_unary)
        self.assertFalse(cbrt.is_binary)
        self.assertTrue(cbrt.is_commutative)

        self.assertArrayEqual(my_abs(self.df["i4"]), self.df['i3'].values)
        self.assertEqual(my_abs.gen_name('i4'), 'abs(i4)')
        self.assertTrue(my_abs.is_unary)
        self.assertFalse(my_abs.is_binary)
        self.assertTrue(my_abs.is_commutative)

        self.assertArrayEqual(log10(self.df["i1"]), self.df['i1'].apply(np.log10).values)
        self.assertEqual(log10.gen_name('i1'), 'log10(i1)')
        self.assertTrue(log10.is_unary)
        self.assertFalse(log10.is_binary)
        self.assertTrue(log10.is_commutative)

    def testBinaryOperators(self):
        self.assertArrayEqual(add(self.df['i1'], self.df['i2']), np.array([3, 8, 15, 24, 35]))
        self.assertEqual(add.gen_name("i1", "i2"), "((i1) + (i2))")
        self.assertFalse(add.is_unary)
        self.assertTrue(add.is_binary)
        self.assertTrue(add.commutative)

        self.assertArrayEqual(sub(self.df['i1'], self.df['i2']), np.array([1, 0, -3, -8, -15]))
        self.assertEqual(sub.gen_name("i1", "i2"), "((i1) - (i2))")
        self.assertFalse(sub.is_unary)
        self.assertTrue(sub.is_binary)
        self.assertFalse(sub.commutative)

class TestFeatureGenerator(PymatgenTest):
    def setUp(self):
        x1 = np.array([2, 4, 6, 8, 10])
        x2 = np.array([1, 4, 9, 16, 25])
        x3 = np.array([1, 8, 27, 64, 125])
        self.df = pd.DataFrame({"i1": x1, "i2": x2, "i3": x3})

    def testFeatureGenerator(self):
        ops = ['^2', '+', '-']
        fg = FeatureGenerator(self.df, ops)
        nf_df1 = fg.augment()
        nf_df2 = FeatureGenerator.generate_feature(self.df, ops)
        new_columns = ['i1', 'i2', 'i3', '(i1)^2', '((i1) + (i2))', '((i1) - (i2))',
                       '((i2) - (i1))', '((i1) + (i3))', '((i1) - (i3))', '((i3) - (i1))',
                       '(i2)^2', '((i2) + (i3))', '((i2) - (i3))', '((i3) - (i2))', '(i3)^2']
        self.assertTrue(nf_df1.equals(nf_df2))
        self.assertArrayEqual(nf_df1.columns, np.array(new_columns))
        self.assertArrayEqual(nf_df1['(i1)^2'].values, self.df['i1'].pow(2).values)
        self.assertArrayEqual(nf_df1['((i1) + (i2))'].values, (self.df['i1'] + self.df['i2']).values)
        self.assertArrayEqual(nf_df1['((i1) - (i2))'].values, (self.df['i1'] - self.df['i2']).values)
        self.assertArrayEqual(nf_df1['((i2) - (i1))'].values, (self.df['i2'] - self.df['i1']).values)

if __name__ == '__main__':
    unittest.main()