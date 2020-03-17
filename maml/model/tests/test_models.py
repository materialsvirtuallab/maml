# coding: utf-8

import unittest
import os
import shutil
import tempfile

from monty.tempfile import ScratchDir
import numpy as np
import pandas as pd
from pymatgen.util.testing import PymatgenTest
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor

from maml import BaseDescriber, ModelWithSklearn
from maml.describer._structure import DistinctSiteProperty
from maml.model._neural_network import MultiLayerPerceptron


class NeuralNetTest(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()

    def setUp(self):
        self.nn = MultiLayerPerceptron(describer=DistinctSiteProperty(['2c'], ["Z"]),
                                       hidden_layer_sizes=[25, 5], input_dim=1)
        self.nn2 = MultiLayerPerceptron(describer=DistinctSiteProperty(['2c'], ["Z"]),
                                        hidden_layer_sizes=[25, 5], input_dim=1)
        self.li2o = self.get_structure("Li2O")
        self.na2o = self.li2o.copy()
        self.na2o["Li+"] = "Na+"
        self.structures = [self.li2o] * 100 + [self.na2o] * 100
        self.energies = np.array([3] * 100 + [4] * 100)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.this_dir)
        shutil.rmtree(cls.test_dir)

    def test_fit_predict(self):
        self.nn.train(objs=self.structures, targets=self.energies, epochs=2)
        self.assertTrue(self.nn.predict_objs([self.na2o]).shape == (1, 1))

    def test_model_save_load(self):
        self.nn.train(objs=self.structures, targets=self.energies, epochs=2)
        with ScratchDir('.'):
            self.nn.save("test.h5")
            self.nn2.load("test.h5")
        self.assertEqual(self.nn.predict_objs([self.na2o])[0][0],
                         self.nn2.predict_objs([self.na2o])[0][0])


class LinearModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x_train = np.random.rand(10, 2)
        cls.coef = np.random.rand(2)
        cls.intercept = np.random.rand()
        cls.y_train = cls.x_train.dot(cls.coef) + cls.intercept

    def setUp(self):
        class DummyDescriber(BaseDescriber):
            def transform_one(self, obj):
                pass

            def transform(self, objs):
                return pd.DataFrame(objs)

        self.lm = ModelWithSklearn(model=LinearRegression(), describer=DummyDescriber())
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_fit_predict(self):
        self.lm.fit(features=self.x_train, targets=self.y_train)
        x_test = np.random.rand(10, 2)
        y_test = x_test.dot(self.coef) + self.intercept
        y_pred = self.lm._predict(x_test)
        np.testing.assert_array_almost_equal(y_test, y_pred)
        np.testing.assert_array_almost_equal(self.coef, self.lm.model.coef_)
        self.assertAlmostEqual(self.intercept, self.lm.model.intercept_)

    def model_save_load(self):
        with ScratchDir('.'):
            self.lm.save('test_lm.save')
            ori = self.lm.model.coef_
            self.lm.load('test_lm.save')
            loaded = self.lm.model.coef_
            self.assertAlmostEqual(ori, loaded)


class GaussianProcessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_dir = tempfile.mkdtemp()

    def setUp(self):
        self.x_train = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        self.y_train = (self.x_train * np.sin(self.x_train)).ravel()

        class DummyDescriber(BaseDescriber):
            def transform_one(self, obj):
                pass

            def transform(self, objs):
                return pd.DataFrame(objs)

        self.gpr = ModelWithSklearn(model=GaussianProcessRegressor(), describer=DummyDescriber())


    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.this_dir)
        shutil.rmtree(cls.test_dir)

    def test_fit_predict(self):
        self.gpr.fit(features=self.x_train, targets=self.y_train)
        x_test = np.atleast_2d(np.linspace(0, 9, 1000)).T
        y_test = x_test * np.sin(x_test)
        y_pred, sigma = self.gpr._predict(x_test, return_std=True)
        upper_bound = y_pred + 1.96 * sigma
        lower_bound = y_pred - 1.96 * sigma
        self.assertTrue(np.all([l < y and y < u
                                for u, y, l in zip(upper_bound, y_test, lower_bound)]))


if __name__ == "__main__":
    unittest.main()
