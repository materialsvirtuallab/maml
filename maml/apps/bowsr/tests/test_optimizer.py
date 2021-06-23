import os
import unittest
import numpy as np

from pymatgen import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

from maml.apps.bowsr.model.megnet import MEGNet
from maml.apps.bowsr.acquisition import AcquisitionFunction
from maml.apps.bowsr.perturbation import get_standardized_structure
from maml.apps.bowsr.preprocessing import StandardScaler, DummyScaler
from maml.apps.bowsr.optimizer import BayesianOptimizer, struct2perturbation

test_lfpo = Structure.from_file(os.path.join(os.path.dirname(__file__), "test_structures", "test_lfpo.cif"))
test_lco = Structure.from_file(os.path.join(os.path.dirname(__file__), "test_structures", "test_lco.cif"))


class BayesianOptimizerTest(unittest.TestCase):
    def setUp(self):
        self.test_lfpo = test_lfpo
        self.test_lco = test_lco
        self.test_refined_lco = get_standardized_structure(self.test_lco)
        model = MEGNet()
        self.optimizer_fixed_latt_lfpo = BayesianOptimizer(
            model=model, structure=self.test_lfpo, relax_coords=True, relax_lattice=False, use_scaler=False
        )
        self.optimizer_relaxed_latt_lfpo = BayesianOptimizer(
            model=model, structure=self.test_lfpo, relax_coords=True, relax_lattice=True, use_scaler=False
        )
        self.optimizer_scaler_lfpo = BayesianOptimizer(
            model=model, structure=self.test_lfpo, relax_coords=True, relax_lattice=True, use_scaler=True
        )
        self.optimizer_fixed_position_lfpo = BayesianOptimizer(
            model=model, structure=self.test_lfpo, relax_coords=False, relax_lattice=True, use_scaler=True
        )

    def test_struct2perturbation(self):

        lfpo_wps, lfpo_indices, lfpo_mapping, lfpo_lp = struct2perturbation(self.test_lfpo)
        self.assertTrue(len(lfpo_wps) == len(lfpo_indices))
        self.assertTrue(len(lfpo_wps) == 6)
        self.assertTrue(len(lfpo_mapping) == self.test_lfpo.num_sites)
        self.assertTrue(lfpo_lp._lattice == self.test_lfpo.lattice)
        self.assertTrue(lfpo_lp.crys_system == "orthorhombic")
        self.assertTrue(lfpo_lp.spg_int_symbol == 62)
        self.assertTrue(lfpo_lp.dims == (3, 0))

        lco_wps, lco_indices, lco_mapping, lco_lp = struct2perturbation(self.test_refined_lco)
        self.assertTrue(len(lco_wps) == len(lco_indices))
        self.assertTrue(len(lco_wps) == 3)
        self.assertTrue(len(lco_mapping) == self.test_refined_lco.num_sites)
        self.assertTrue(lco_lp._lattice == self.test_refined_lco.lattice)
        self.assertTrue(lco_lp.crys_system == "rhombohedral")
        self.assertTrue(lco_lp.spg_int_symbol == 166)
        self.assertTrue(lco_lp.dims == (2, 0))

    def test_attributes(self):
        self.assertFalse(self.optimizer_fixed_latt_lfpo.relax_lattice)
        self.assertTrue(self.optimizer_fixed_latt_lfpo.noisy)
        self.assertTrue(self.optimizer_relaxed_latt_lfpo.relax_lattice)
        self.assertTrue(isinstance(self.optimizer_fixed_latt_lfpo.scaler, DummyScaler))
        self.assertTrue(isinstance(self.optimizer_relaxed_latt_lfpo.scaler, DummyScaler))
        self.assertTrue(isinstance(self.optimizer_scaler_lfpo.scaler, StandardScaler))
        self.assertListEqual(self.optimizer_fixed_latt_lfpo.wyckoff_dims, [0, 2, 2, 2, 2, 3])
        self.assertListEqual(self.optimizer_relaxed_latt_lfpo.wyckoff_dims, [0, 2, 2, 2, 2, 3])
        self.assertListEqual(self.optimizer_scaler_lfpo.wyckoff_dims, [0, 2, 2, 2, 2, 3])
        self.assertListEqual(self.optimizer_fixed_position_lfpo.wyckoff_dims, [0, 2, 2, 2, 2, 3])
        self.assertEqual(self.optimizer_fixed_latt_lfpo.abc_dim, 3)
        self.assertEqual(self.optimizer_fixed_latt_lfpo.angles_dim, 0)
        self.assertEqual(self.optimizer_fixed_latt_lfpo.space.dim, 11)
        self.assertEqual(self.optimizer_relaxed_latt_lfpo.space.dim, 14)
        self.optimizer_fixed_latt_lfpo.set_bounds()
        self.assertTrue(np.all(abs(self.optimizer_fixed_latt_lfpo.space.bounds) <= 0.2))

        self.assertTrue(isinstance(self.optimizer_fixed_latt_lfpo.gpr.kernel, RationalQuadratic))
        self.optimizer_fixed_latt_lfpo.set_gpr_params(kernel=Matern(length_scale=1.0))
        self.assertTrue(isinstance(self.optimizer_fixed_latt_lfpo.gpr.kernel, Matern))
        self.assertTrue(self.optimizer_fixed_latt_lfpo.gpr.kernel.length_scale == 1.0)
        self.optimizer_fixed_latt_lfpo.set_gpr_params(kernel=RationalQuadratic(length_scale=5.0))
        self.assertTrue(self.optimizer_fixed_latt_lfpo.gpr.kernel.length_scale == 5.0)

    def test_get_derived_struct(self):
        matcher = StructureMatcher()
        x0 = np.array([0] * 11)
        derived_struct = self.optimizer_fixed_latt_lfpo.get_derived_structure(x0)
        self.assertTrue(matcher.fit(derived_struct, self.test_lfpo))
        x0 = np.array([0] * 14)
        derived_struct = self.optimizer_relaxed_latt_lfpo.get_derived_structure(x0)
        self.assertTrue(matcher.fit(derived_struct, self.test_lfpo))
        x1 = np.array([0] * 11 + [1] * 3)
        derived_struct = self.optimizer_scaler_lfpo.get_derived_structure(x1)
        abc = tuple(a + 1 for a in self.test_lfpo.lattice.abc)
        self.assertTrue(sorted(abc) == sorted(derived_struct.lattice.abc))
        x2 = np.array([0.05] * 11)
        x3 = np.array([0.05] * 11 + [0] * 3)
        self.assertTrue(
            matcher.fit(
                self.optimizer_fixed_latt_lfpo.get_derived_structure(x2),
                self.optimizer_relaxed_latt_lfpo.get_derived_structure(x3),
            )
        )

    def test_get_formation_energy(self):
        x0 = np.array([0] * 11)
        formation_energy_per_atom = self.optimizer_fixed_latt_lfpo.get_formation_energy(x0)
        self.assertAlmostEqual(formation_energy_per_atom, 2.5418098, places=5)
        x0 = np.array([0] * 14)
        formation_energy_per_atom = self.optimizer_relaxed_latt_lfpo.get_formation_energy(x0)
        self.assertAlmostEqual(formation_energy_per_atom, 2.5418098, places=5)
        x1 = np.array([0] * 11 + [1] * 3)
        formation_energy_per_atom = self.optimizer_relaxed_latt_lfpo.get_formation_energy(x1)
        angles = self.test_lfpo.lattice.angles
        abc = tuple(a + 1 for a in self.test_lfpo.lattice.abc)
        struct = Structure(
            lattice=Lattice.from_parameters(*(abc + angles)),
            species=self.test_lfpo.species,
            coords=self.test_lfpo.frac_coords,
        )
        self.assertAlmostEqual(
            formation_energy_per_atom, -self.optimizer_fixed_latt_lfpo.model.predict_energy(struct), places=5
        )
        x2 = np.array([0.05] * 11)
        x3 = np.array([0.05] * 11 + [0] * 3)
        self.assertAlmostEqual(
            self.optimizer_fixed_latt_lfpo.get_formation_energy(x2),
            self.optimizer_relaxed_latt_lfpo.get_formation_energy(x3),
            places=5,
        )

    def test_add_query(self):
        self.optimizer_fixed_latt_lfpo.set_bounds()
        self.optimizer_relaxed_latt_lfpo.set_bounds()
        self.optimizer_fixed_position_lfpo.set_bounds()
        self.optimizer_scaler_lfpo.set_bounds()
        mean = np.random.random(self.optimizer_scaler_lfpo.space.dim)
        std = np.random.random(self.optimizer_scaler_lfpo.space.dim)
        scaler = StandardScaler(mean=mean, std=std)
        self.optimizer_scaler_lfpo.scaler = scaler
        self.optimizer_scaler_lfpo.space.scaler = scaler

        for i in np.arange(0, 10, 2):
            sample = self.optimizer_fixed_latt_lfpo.space.uniform_sample()
            target1 = self.optimizer_fixed_latt_lfpo.add_query(sample)
            self.assertEqual(len(self.optimizer_fixed_latt_lfpo.space), i + 1)
            target2 = self.optimizer_fixed_latt_lfpo.add_query(sample)
            self.assertEqual(len(self.optimizer_fixed_latt_lfpo.space), i + 2)
            self.assertAlmostEqual(target1, target2)
        self.optimizer_fixed_latt_lfpo.set_space_empty()
        self.assertEqual(len(self.optimizer_fixed_latt_lfpo.space), 0)

        orig_x = np.empty(shape=(0, self.optimizer_scaler_lfpo.space.dim))
        for i in range(10):
            sample = self.optimizer_scaler_lfpo.space.uniform_sample()
            orig_x = np.concatenate([orig_x, sample.reshape(1, -1)])
            target1 = self.optimizer_scaler_lfpo.add_query(sample)
            self.assertAlmostEqual(target1, self.optimizer_scaler_lfpo.get_formation_energy(sample))
            self.assertEqual(len(self.optimizer_scaler_lfpo.space), i + 1)
        self.assertTrue(np.all(abs(scaler.transform(orig_x) - self.optimizer_scaler_lfpo.space.params) < 1e-3))

    def test_propose(self):
        self.optimizer_fixed_latt_lfpo.set_bounds()
        self.optimizer_fixed_latt_lfpo.optimize(5, 0)
        self.optimizer_relaxed_latt_lfpo.set_bounds(element_wise_wyckoff_bounds={"Li": 0.4, "Fe": 0.4, "P": 0.4})
        self.optimizer_relaxed_latt_lfpo.optimize(5, 0)
        self.optimizer_scaler_lfpo.set_bounds()
        self.optimizer_scaler_lfpo.optimize(5, 0)

        acq_ucb = AcquisitionFunction(acq_type="ucb", kappa=1.0, xi=0)
        acq_ei = AcquisitionFunction(acq_type="ei", kappa=1.0, xi=0.1)
        acq_poi = AcquisitionFunction(acq_type="poi", kappa=1.0, xi=0.1)
        x_next_ucb = self.optimizer_fixed_latt_lfpo.propose(acq_ucb, n_warmup=1000, sampler="lhs")
        self.assertEqual(len(x_next_ucb), self.optimizer_fixed_latt_lfpo.space.dim)
        self.assertTrue(np.all(self.optimizer_fixed_latt_lfpo.space.bounds[:, 0] <= x_next_ucb))
        self.assertTrue(np.all(x_next_ucb <= self.optimizer_fixed_latt_lfpo.space.bounds[:, 1]))
        x_next_ei = self.optimizer_fixed_latt_lfpo.propose(acq_ei, n_warmup=1000, sampler="lhs")
        self.assertEqual(len(x_next_ei), self.optimizer_fixed_latt_lfpo.space.dim)
        self.assertTrue(np.all(self.optimizer_fixed_latt_lfpo.space.bounds[:, 0] <= x_next_ei))
        self.assertTrue(np.all(x_next_ei <= self.optimizer_fixed_latt_lfpo.space.bounds[:, 1]))
        x_next_relaxed_latt = self.optimizer_relaxed_latt_lfpo.propose(acq_ei, n_warmup=1000, sampler="uniform")
        self.assertEqual(len(x_next_relaxed_latt), self.optimizer_relaxed_latt_lfpo.space.dim)
        self.assertTrue(np.all(self.optimizer_relaxed_latt_lfpo.space.bounds[:, 0] <= x_next_relaxed_latt))
        self.assertTrue(np.all(x_next_relaxed_latt <= self.optimizer_relaxed_latt_lfpo.space.bounds[:, 1]))
        x_next_scaler = self.optimizer_scaler_lfpo.propose(acq_ei, n_warmup=1000, sampler="lhs")
        self.assertEqual(len(x_next_scaler), self.optimizer_scaler_lfpo.space.dim)
        self.assertTrue(np.all(self.optimizer_scaler_lfpo.space.bounds[:, 0] <= x_next_scaler))
        self.assertTrue(np.all(x_next_scaler <= self.optimizer_scaler_lfpo.space.bounds[:, 1]))

    def test_optimize(self):
        self.optimizer_fixed_latt_lfpo.set_bounds()
        self.optimizer_relaxed_latt_lfpo.set_bounds()
        self.optimizer_scaler_lfpo.set_bounds()

        self.optimizer_fixed_latt_lfpo.optimize(n_init=4, n_iter=4, acq_type="ucb", kappa=1.0, xi=0)
        self.assertEqual(len(self.optimizer_fixed_latt_lfpo.space), 9)
        self.optimizer_fixed_latt_lfpo.optimize(n_init=2, n_iter=2, acq_type="ucb", kappa=1.0, xi=0)
        self.assertEqual(len(self.optimizer_fixed_latt_lfpo.space), 14)
        self.optimizer_fixed_latt_lfpo.optimize(n_init=0, n_iter=3, acq_type="ucb", kappa=1.0, xi=0)
        self.assertEqual(len(self.optimizer_fixed_latt_lfpo.space), 17)
        self.optimizer_relaxed_latt_lfpo.optimize(n_init=3, n_iter=3, acq_type="ei", kappa=1.0, xi=0.1)
        self.assertEqual(len(self.optimizer_relaxed_latt_lfpo.space), 7)
        self.optimizer_relaxed_latt_lfpo.optimize(n_init=1, n_iter=1, acq_type="ei", kappa=1.0, xi=0.1)
        self.assertEqual(len(self.optimizer_relaxed_latt_lfpo.space), 10)
        self.optimizer_relaxed_latt_lfpo.optimize(n_init=0, n_iter=2, acq_type="ei", kappa=1.0, xi=0.1)
        self.assertEqual(len(self.optimizer_relaxed_latt_lfpo.space), 12)
        self.optimizer_scaler_lfpo.optimize(n_init=3, n_iter=3, acq_type="ei", kappa=1.0, xi=0.1)
        self.optimizer_scaler_lfpo.optimize(n_init=0, n_iter=3, acq_type="ei", kappa=1.0, xi=0.1)
        self.assertEqual(len(self.optimizer_scaler_lfpo.space), 10)

    def test_as_dict(self):
        self.optimizer_fixed_latt_lfpo.set_bounds()
        self.optimizer_relaxed_latt_lfpo.set_bounds(abc_bound=1.5)
        self.optimizer_scaler_lfpo.set_bounds()

        self.optimizer_fixed_latt_lfpo.optimize(n_init=3, n_iter=3, acq_type="ucb", kappa=1.0, xi=0)
        optimizer_fixed_latt_lfpo_dict = self.optimizer_fixed_latt_lfpo.as_dict()
        self.assertTrue(
            Structure.from_dict(optimizer_fixed_latt_lfpo_dict["structure"]) == self.optimizer_fixed_latt_lfpo.structure
        )
        self.assertTrue(optimizer_fixed_latt_lfpo_dict["noisy"] == self.optimizer_fixed_latt_lfpo.noisy)
        self.assertTrue(optimizer_fixed_latt_lfpo_dict["gpr"]["kernel"]["name"] == "RationalQuadratic")
        self.assertTrue(
            optimizer_fixed_latt_lfpo_dict["gpr"]["kernel"]["params"]
            == self.optimizer_fixed_latt_lfpo.gpr.kernel.get_params()
        )
        self.assertTrue(
            optimizer_fixed_latt_lfpo_dict["gpr"]["kernel"]["opt_params"]
            == self.optimizer_fixed_latt_lfpo.gpr.kernel_.get_params()
        )
        self.assertTrue(
            np.all(optimizer_fixed_latt_lfpo_dict["space"]["bounds"] == self.optimizer_fixed_latt_lfpo.space.bounds)
        )
        self.assertTrue(
            np.all(optimizer_fixed_latt_lfpo_dict["space"]["params"] == self.optimizer_fixed_latt_lfpo.space.params)
        )
        self.assertTrue(
            np.all(optimizer_fixed_latt_lfpo_dict["space"]["target"] == self.optimizer_fixed_latt_lfpo.space.target)
        )
        self.assertTrue(
            np.all(
                optimizer_fixed_latt_lfpo_dict["space"]["random_state"][1]
                == self.optimizer_fixed_latt_lfpo.space.random_state.get_state()[1]
            )
        )
        self.assertEqual(optimizer_fixed_latt_lfpo_dict["scaler"]["@class"], "DummyScaler")

        self.optimizer_relaxed_latt_lfpo.optimize(n_init=3, n_iter=3, acq_type="ucb", kappa=1.0, xi=0)
        optimizer_relaxed_latt_lfpo_dict = self.optimizer_relaxed_latt_lfpo.as_dict()
        self.assertTrue(
            Structure.from_dict(optimizer_relaxed_latt_lfpo_dict["structure"])
            == self.optimizer_relaxed_latt_lfpo.structure
        )
        self.assertTrue(optimizer_relaxed_latt_lfpo_dict["noisy"] == self.optimizer_relaxed_latt_lfpo.noisy)
        self.assertTrue(
            np.all(optimizer_relaxed_latt_lfpo_dict["space"]["bounds"] == self.optimizer_relaxed_latt_lfpo.space.bounds)
        )
        self.assertTrue(
            np.all(optimizer_relaxed_latt_lfpo_dict["space"]["params"] == self.optimizer_relaxed_latt_lfpo.space.params)
        )
        self.assertTrue(
            np.all(optimizer_relaxed_latt_lfpo_dict["space"]["target"] == self.optimizer_relaxed_latt_lfpo.space.target)
        )
        self.assertTrue(
            np.all(
                optimizer_relaxed_latt_lfpo_dict["space"]["random_state"][1]
                == self.optimizer_relaxed_latt_lfpo.space.random_state.get_state()[1]
            )
        )
        self.assertEqual(optimizer_relaxed_latt_lfpo_dict["scaler"]["@class"], "DummyScaler")

        self.optimizer_scaler_lfpo.optimize(n_init=3, n_iter=3, acq_type="ei", kappa=1.0, xi=0.1)
        optimizer_scaler_lfpo_dict = self.optimizer_scaler_lfpo.as_dict()
        self.assertEqual(optimizer_scaler_lfpo_dict["scaler"]["@class"], "StandardScaler")
        self.assertTrue(
            np.all(
                abs(
                    np.array(optimizer_scaler_lfpo_dict["scaler"]["params"]["mean"])
                    - self.optimizer_scaler_lfpo.scaler.mean
                )
                < 1e-4
            )
        )
        self.assertTrue(
            np.all(
                abs(
                    np.array(optimizer_scaler_lfpo_dict["scaler"]["params"]["std"])
                    - self.optimizer_scaler_lfpo.scaler.std
                )
                < 1e-4
            )
        )

    def test_from_dict(self):
        self.optimizer_fixed_latt_lfpo.set_bounds()
        self.optimizer_relaxed_latt_lfpo.set_bounds(abc_bound=1.5)
        self.optimizer_scaler_lfpo.set_bounds()

        self.optimizer_fixed_latt_lfpo.optimize(n_init=3, n_iter=3, acq_type="ucb", kappa=1.0, xi=0)
        optimizer_fixed_latt_lfpo_dict = self.optimizer_fixed_latt_lfpo.as_dict()
        optimizer1 = BayesianOptimizer.from_dict(optimizer_fixed_latt_lfpo_dict)
        self.assertTrue(optimizer1.gpr.kernel == self.optimizer_fixed_latt_lfpo.gpr.kernel)
        self.assertTrue(optimizer1.gpr.kernel_ == self.optimizer_fixed_latt_lfpo.gpr.kernel_)
        self.assertTrue(np.all(optimizer1.space.bounds == self.optimizer_fixed_latt_lfpo.space.bounds))
        self.assertTrue(np.all(optimizer1.space.params == self.optimizer_fixed_latt_lfpo.space.params))
        self.assertTrue(np.all(optimizer1.space.target == self.optimizer_fixed_latt_lfpo.space.target))

        self.optimizer_relaxed_latt_lfpo.optimize(n_init=3, n_iter=3, acq_type="ucb", kappa=1.0, xi=0)
        optimizer_relaxed_latt_lfpo_dict = self.optimizer_relaxed_latt_lfpo.as_dict()
        optimizer2 = BayesianOptimizer.from_dict(optimizer_relaxed_latt_lfpo_dict)
        self.assertTrue(optimizer2.gpr.kernel == self.optimizer_relaxed_latt_lfpo.gpr.kernel)
        self.assertTrue(optimizer2.gpr.kernel_ == self.optimizer_relaxed_latt_lfpo.gpr.kernel_)
        self.assertTrue(np.all(optimizer2.space.bounds == self.optimizer_relaxed_latt_lfpo.space.bounds))
        self.assertTrue(np.all(optimizer2.space.params == self.optimizer_relaxed_latt_lfpo.space.params))
        self.assertTrue(np.all(optimizer2.space.target == self.optimizer_relaxed_latt_lfpo.space.target))

        self.optimizer_scaler_lfpo.optimize(n_init=3, n_iter=3, acq_type="ei", kappa=1.0, xi=0.1)
        optimizer_scaler_lfpo_dict = self.optimizer_scaler_lfpo.as_dict()
        optimizer5 = BayesianOptimizer.from_dict(optimizer_scaler_lfpo_dict)
        self.assertTrue(optimizer5.gpr.kernel == self.optimizer_scaler_lfpo.gpr.kernel)
        self.assertTrue(optimizer5.gpr.kernel_ == self.optimizer_scaler_lfpo.gpr.kernel_)
        self.assertTrue(np.all(optimizer5.space.bounds == self.optimizer_scaler_lfpo.space.bounds))
        self.assertTrue(np.all(optimizer5.space.params == self.optimizer_scaler_lfpo.space.params))
        self.assertTrue(np.all(optimizer5.space.target == self.optimizer_scaler_lfpo.space.target))
        self.assertTrue(np.all(abs(optimizer5.scaler.mean - self.optimizer_scaler_lfpo.scaler.mean) < 1e-4))
        self.assertTrue(np.all(abs(optimizer5.scaler.std - self.optimizer_scaler_lfpo.scaler.std) < 1e-4))


if __name__ == "__main__":
    unittest.main()
