import unittest
import numpy as np
from maml.apps.bowsr.preprocessing import StandardScaler, DummyScaler


class ScalerTest(unittest.TestCase):
    def setUp(self):
        self.target = np.random.random((12, 6))
        self.scaler1 = StandardScaler()
        self.scaler2 = DummyScaler()

    def test_attributes(self):
        self.assertIsNone(self.scaler1.mean)
        self.assertIsNone(self.scaler1.std)

    def test_fit(self):
        self.scaler1.fit(self.target)
        self.scaler2.fit(self.target)

        self.assertTrue(np.all(abs(self.scaler1.mean - np.mean(self.target, axis=0)) < 1e-4))
        self.assertTrue(np.all(abs(self.scaler1.std - np.std(self.target, axis=0)) < 1e-4))

    def test_transform(self):
        self.assertRaises(ValueError, self.scaler1.transform, self.target)
        self.scaler1.fit(self.target)
        self.scaler2.fit(self.target)

        transformed_target1 = self.scaler1.transform(self.target)
        self.assertTrue(transformed_target1.shape == self.target.shape)
        transformed_target2 = (self.target - np.mean(self.target, axis=0)) / np.std(self.target, axis=0)
        self.assertTrue(np.all(abs(transformed_target1 - transformed_target2) < 1e-5))

        transformed_target3 = self.scaler2.transform(self.target)
        self.assertTrue(np.all(abs(transformed_target3 - self.target) < 1e-5))

    def test_inverse_transform(self):
        self.assertRaises(ValueError, self.scaler1.inverse_transform, self.target)
        self.scaler1.fit(self.target)
        self.scaler2.fit(self.target)

        transformed_target1 = self.scaler1.transform(self.target)
        target1 = self.scaler1.inverse_transform(transformed_target1)
        self.assertTrue(np.all(abs(target1 - self.target) < 1e-4))

        target2 = self.scaler2.inverse_transform(self.target)
        self.assertTrue(np.all(abs(target2 - self.target) < 1e-4))

    def test_as_dict(self):
        self.scaler1.fit(self.target)
        scaler1_dict = self.scaler1.as_dict()
        self.assertEqual(scaler1_dict["@class"], "StandardScaler")
        self.assertTrue(np.all(abs(scaler1_dict["params"]["mean"] - np.mean(self.target, axis=0)) < 1e-4))
        self.assertTrue(np.all(abs(scaler1_dict["params"]["std"] - np.std(self.target, axis=0)) < 1e-4))

        scaler2_dict = self.scaler2.as_dict()
        self.assertEqual(scaler2_dict["@class"], "DummyScaler")

    def test_from_dict(self):
        self.scaler1.fit(self.target)
        scaler1_dict = self.scaler1.as_dict()
        scaler1 = StandardScaler.from_dict(scaler1_dict)
        self.assertTrue(np.all(abs(scaler1.mean - self.scaler1.mean) < 1e-4))
        self.assertTrue(np.all(abs(scaler1.std - self.scaler1.std) < 1e-4))

        scaler2_dict = self.scaler2.as_dict()
        scaler2 = DummyScaler.from_dict(scaler2_dict)
        transformed_target2 = scaler2.transform(self.target)
        target2 = scaler2.inverse_transform(self.target)
        self.assertTrue(np.all(abs(transformed_target2 - self.target) < 1e-4))
        self.assertTrue(np.all(abs(target2 - self.target) < 1e-4))


if __name__ == "__main__":
    unittest.main()
