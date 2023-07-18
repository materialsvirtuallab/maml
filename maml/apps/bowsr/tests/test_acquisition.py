from __future__ import annotations

import unittest

import numpy as np
from scipy.special import erfc
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

from maml.apps.bowsr.acquisition import AcquisitionFunction, ensure_rng, predict_mean_std, propose_query_point
from maml.apps.bowsr.preprocessing import DummyScaler, StandardScaler


class AcquisitionFunctionTest(unittest.TestCase):
    def setUp(self):
        self.acq_ucb = AcquisitionFunction(acq_type="ucb", kappa=1.0, xi=0.0)
        self.acq_ei = AcquisitionFunction(acq_type="ei", kappa=1.0, xi=0.1)
        self.acq_poi = AcquisitionFunction(acq_type="poi", kappa=1.0, xi=0.1)
        self.acq_gpucb = AcquisitionFunction(acq_type="gp-ucb", kappa=1.0, xi=0.1)
        self.standardscaler = StandardScaler()
        self.dummyscaler = DummyScaler()

        self.X = np.array(
            [
                [-0.99, -0.99],
                [-0.65, -0.99],
                [-0.30, -0.99],
                [0.00, -0.99],
                [0.35, -0.65],
                [0.65, -0.65],
                [0.99, -0.65],
                [0.65, -0.35],
                [0.99, -0.35],
                [0.35, 0.00],
                [0.65, 0.00],
                [0.99, 0.35],
                [0.35, 0.65],
                [0.65, 0.65],
                [0.99, 0.65],
                [0.99, 0.99],
            ]
        )
        self.test_func = lambda x: -((x[:, 0] - 0.6) ** 3) - (x[:, 1] + 0.3) ** 2
        self.y_max = np.max(self.test_func(self.X))
        self.matern_gpr = GaussianProcessRegressor(kernel=Matern(length_scale=1.0))
        self.rq_gpr = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=1.0))

    def test_attributes(self):
        self.assertRaises(NotImplementedError, AcquisitionFunction, acq_type="other", kappa=1.0, xi=0.1)
        assert self.acq_ucb.acq_type == "ucb"
        assert self.acq_ucb.kappa == 1.0
        assert self.acq_ucb.xi == 0.0
        assert self.acq_ei.acq_type == "ei"
        assert self.acq_ei.kappa == 1.0
        assert self.acq_ei.xi == 0.1
        assert self.acq_poi.acq_type == "poi"
        assert self.acq_poi.kappa == 1.0
        assert self.acq_poi.xi == 0.1
        assert self.acq_gpucb.acq_type == "gp-ucb"
        assert self.acq_gpucb.kappa == 1.0
        assert self.acq_gpucb.xi == 0.1

    def test_propose(self):
        self.matern_gpr.fit(self.X, self.test_func(self.X))
        propose_ucb = propose_query_point(
            acquisition=self.acq_ucb.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert -1 <= propose_ucb[0] <= 1
        assert -1 <= propose_ucb[1] <= 1

        propose_ucb = propose_query_point(
            acquisition=self.acq_gpucb.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert -1 <= propose_ucb[0] <= 1
        assert -1 <= propose_ucb[1] <= 1

        propose_ei = propose_query_point(
            acquisition=self.acq_ei.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[0, 2], [0, 2]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert 0 <= propose_ei[0] <= 2
        assert 0 <= propose_ei[1] <= 2

        propose_poi = propose_query_point(
            acquisition=self.acq_poi.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[-2, 0], [-2, 0]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert -2 <= propose_poi[0] <= 0
        assert -2 <= propose_poi[1] <= 0

        self.standardscaler.fit(self.X)
        self.rq_gpr.fit(self.standardscaler.transform(self.X), self.test_func(self.X))
        propose_ei = propose_query_point(
            acquisition=self.acq_ei.calculate,
            scaler=self.standardscaler,
            gpr=self.rq_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[0, 2], [0, 2]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert 0 <= propose_ei[0] <= 2
        assert 0 <= propose_ei[1] <= 2

        noise = 0.02
        self.matern_gpr.fit(self.X, self.test_func(self.X) + np.random.uniform(-1, 1, len(self.X)) * noise)
        y_max = np.max(self.matern_gpr.predict(self.X))
        propose_ucb = propose_query_point(
            acquisition=self.acq_ucb.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=y_max,
            noise=noise,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert -1 <= propose_ucb[0] <= 1
        assert -1 <= propose_ucb[1] <= 1

        propose_ei = propose_query_point(
            acquisition=self.acq_ei.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=y_max,
            noise=noise,
            bounds=np.array([[0, 2], [0, 2]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert 0 <= propose_ei[0] <= 2
        assert 0 <= propose_ei[1] <= 2

        propose_poi = propose_query_point(
            acquisition=self.acq_poi.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=y_max,
            noise=noise,
            bounds=np.array([[-2, 0], [-2, 0]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert -2 <= propose_poi[0] <= 0
        assert -2 <= propose_poi[1] <= 0

        self.standardscaler.fit(self.X)
        self.rq_gpr.fit(
            self.standardscaler.transform(self.X),
            self.test_func(self.X) + np.random.uniform(-1, 1, len(self.X)) * noise,
        )
        y_max = np.max(self.rq_gpr.predict(self.X))
        propose_ei = propose_query_point(
            acquisition=self.acq_ei.calculate,
            scaler=self.standardscaler,
            gpr=self.rq_gpr,
            y_max=y_max,
            noise=noise,
            bounds=np.array([[0, 2], [0, 2]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert 0 <= propose_ei[0] <= 2
        assert 0 <= propose_ei[1] <= 2

    def test_predict_mean_std(self):
        self.matern_gpr.fit(self.X, self.test_func(self.X))
        self.rq_gpr.fit(self.X, self.test_func(self.X))
        mean1, std1 = predict_mean_std(self.X, self.matern_gpr, noise=0.0)
        mean2, std2 = self.matern_gpr.predict(self.X, return_std=True)
        assert np.all(abs(mean1 - mean2) < 1e-05)
        assert np.all(abs(std1 - std2) < 1e-05)
        mean3, std3 = predict_mean_std(self.X, self.rq_gpr, noise=0.0)
        mean4, std4 = self.matern_gpr.predict(self.X, return_std=True)
        assert np.all(abs(mean3 - mean4) < 1e-05)
        assert np.all(abs(std3 - std4) < 1e-05)

    def test_distribution(self):
        z = np.random.random(5)
        pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        cdf = 0.5 * erfc(-z / np.sqrt(2))
        assert np.all(abs(pdf - norm.pdf(z)) < 0.0001)
        assert np.all(abs(cdf - norm.cdf(z)) < 0.0001)

    def test_calculate(self):
        epislon = 1e-2
        self.matern_gpr.fit(self.X, self.test_func(self.X))
        self.rq_gpr.fit(self.X, self.test_func(self.X))
        mesh = np.dstack(np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))).reshape(-1, 2)

        y_mesh_ucb = self.acq_ucb.calculate(mesh, gpr=self.matern_gpr, y_max=self.y_max, noise=0.0)
        calc_ucb = mesh[np.argmax(y_mesh_ucb)]
        propose_ucb = propose_query_point(
            self.acq_ucb.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_ucb - propose_ucb) < epislon)

        y_mesh_ei = self.acq_ei.calculate(mesh, gpr=self.matern_gpr, y_max=self.y_max, noise=0.0)
        calc_ei = mesh[np.argmax(y_mesh_ei)]
        propose_ei = propose_query_point(
            self.acq_ei.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_ei - propose_ei) < epislon)

        y_mesh_poi = self.acq_poi.calculate(mesh, gpr=self.matern_gpr, y_max=self.y_max, noise=0.0)
        calc_poi = mesh[np.argmax(y_mesh_poi)]
        propose_poi = propose_query_point(
            self.acq_poi.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_poi - propose_poi) < epislon)

        y_mesh_ucb = self.acq_ucb.calculate(mesh, gpr=self.rq_gpr, y_max=self.y_max, noise=0.0)
        calc_ucb = mesh[np.argmax(y_mesh_ucb)]
        propose_ucb = propose_query_point(
            self.acq_ucb.calculate,
            scaler=self.dummyscaler,
            gpr=self.rq_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_ucb - propose_ucb) < epislon)

        y_mesh_ei = self.acq_ei.calculate(mesh, gpr=self.rq_gpr, y_max=self.y_max, noise=0.0)
        calc_ei = mesh[np.argmax(y_mesh_ei)]
        propose_ei = propose_query_point(
            self.acq_ei.calculate,
            scaler=self.dummyscaler,
            gpr=self.rq_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_ei - propose_ei) < epislon)

        y_mesh_poi = self.acq_poi.calculate(mesh, gpr=self.rq_gpr, y_max=self.y_max, noise=0.0)
        calc_poi = mesh[np.argmax(y_mesh_poi)]
        propose_poi = propose_query_point(
            self.acq_poi.calculate,
            scaler=self.dummyscaler,
            gpr=self.rq_gpr,
            y_max=self.y_max,
            noise=0.0,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_poi - propose_poi) < epislon)

        noise = 0.02
        self.matern_gpr.fit(self.X, self.test_func(self.X) + np.random.uniform(-1, 1, len(self.X)) * noise)
        self.rq_gpr.fit(self.X, self.test_func(self.X) + np.random.uniform(-1, 1, len(self.X)) * noise)
        matern_y_max = np.max(self.matern_gpr.predict(self.X))
        rq_y_max = np.max(self.rq_gpr.predict(self.X))

        y_mesh_ucb = self.acq_ucb.calculate(mesh, gpr=self.matern_gpr, y_max=matern_y_max, noise=noise)
        calc_ucb = mesh[np.argmax(y_mesh_ucb)]
        propose_ucb = propose_query_point(
            self.acq_ucb.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=matern_y_max,
            noise=noise,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_ucb - propose_ucb) < epislon)

        y_mesh_ei = self.acq_ei.calculate(mesh, gpr=self.matern_gpr, y_max=matern_y_max, noise=noise)
        calc_ei = mesh[np.argmax(y_mesh_ei)]
        propose_ei = propose_query_point(
            self.acq_ei.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=matern_y_max,
            noise=noise,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_ei - propose_ei) < epislon)

        y_mesh_poi = self.acq_poi.calculate(mesh, gpr=self.matern_gpr, y_max=matern_y_max, noise=noise)
        calc_poi = mesh[np.argmax(y_mesh_poi)]
        propose_poi = propose_query_point(
            self.acq_poi.calculate,
            scaler=self.dummyscaler,
            gpr=self.matern_gpr,
            y_max=matern_y_max,
            noise=noise,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_poi - propose_poi) < epislon)

        y_mesh_ucb = self.acq_ucb.calculate(mesh, gpr=self.rq_gpr, y_max=rq_y_max, noise=noise)
        calc_ucb = mesh[np.argmax(y_mesh_ucb)]
        propose_ucb = propose_query_point(
            self.acq_ucb.calculate,
            scaler=self.dummyscaler,
            gpr=self.rq_gpr,
            y_max=rq_y_max,
            noise=noise,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_ucb - propose_ucb) < epislon)

        y_mesh_ei = self.acq_ei.calculate(mesh, gpr=self.rq_gpr, y_max=rq_y_max, noise=noise)
        calc_ei = mesh[np.argmax(y_mesh_ei)]
        propose_ei = propose_query_point(
            self.acq_ei.calculate,
            scaler=self.dummyscaler,
            gpr=self.rq_gpr,
            y_max=rq_y_max,
            noise=noise,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_ei - propose_ei) < epislon)

        y_mesh_poi = self.acq_poi.calculate(mesh, gpr=self.rq_gpr, y_max=rq_y_max, noise=noise)
        calc_poi = mesh[np.argmax(y_mesh_poi)]
        propose_poi = propose_query_point(
            self.acq_poi.calculate,
            scaler=self.dummyscaler,
            gpr=self.rq_gpr,
            y_max=rq_y_max,
            noise=noise,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_poi - propose_poi) < epislon)

        y_mesh_gpucb = self.acq_gpucb.calculate(mesh, gpr=self.rq_gpr, y_max=rq_y_max, noise=noise)
        calc_gpucb = mesh[np.argmax(y_mesh_gpucb)]
        propose_gpucb = propose_query_point(
            self.acq_gpucb.calculate,
            scaler=self.dummyscaler,
            gpr=self.rq_gpr,
            y_max=rq_y_max,
            noise=noise,
            bounds=np.array([[-1, 1], [-1, 1]]),
            sampler="lhs",
            random_state=ensure_rng(0),
        )
        assert all(abs(calc_gpucb - propose_gpucb) < epislon)


if __name__ == "__main__":
    unittest.main()
