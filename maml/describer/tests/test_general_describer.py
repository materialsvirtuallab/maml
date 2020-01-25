# coding: utf-8

import unittest
import json

import numpy as np
import pandas as pd
from pymatgen.util.testing import PymatgenTest

from maml.describer.general import FuncGenerator, MultiDescriber
from maml.describer.structural_describer import DistinctSiteProperty


class GeneratorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = np.random.rand(100, 3) * 10 - 5
        cls.df = pd.DataFrame(cls.data, columns=["x", "y", "z"])
        func_dict = {"sin": "np.sin",
                     "sum": "lambda d: d.sum(axis=1)",
                     "nest": "lambda d: np.log(np.exp(d['x']))"}
        cls.generator = FuncGenerator(func_dict=func_dict)

    def test_describe(self):
        results = self.generator.describe(self.df)
        np.testing.assert_array_equal(np.sin(self.data),
                                      results[["sin x", "sin y", "sin z"]])
        np.testing.assert_array_equal(np.sum(self.data, axis=1),
                                      results["sum"])
        np.testing.assert_array_almost_equal(self.data[:, 0],
                                             results["nest"])

    def test_serialize(self):
        json_str = json.dumps(self.generator.as_dict())
        recover = FuncGenerator.from_dict(json.loads(json_str))


class MultiDescriberTest(PymatgenTest):

    def test_describe(self):
        li2o = self.get_structure("Li2O")
        na2o = li2o.copy()
        na2o["Li+"] = "Na+"
        d1 = DistinctSiteProperty(['2c', '1a'], ["Z", "atomic_radius"])
        d2 = FuncGenerator(func_dict={"exp": "np.exp"}, append=False)
        d = MultiDescriber([d1, d2])

        results = d.describe(li2o)
        self.assertAlmostEqual(results.iloc[0]["exp 2c-Z"], np.exp(3))
        self.assertAlmostEqual(results.iloc[0]["exp 2c-atomic_radius"],
                               np.exp(1.45))

        df = d.describe_all([li2o, na2o])
        self.assertAlmostEqual(df.iloc[0]["exp 2c-Z"], np.exp(3))
        self.assertAlmostEqual(df.iloc[1]["exp 2c-Z"], np.exp(11))

if __name__ == "__main__":
    unittest.main()
