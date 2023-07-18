from __future__ import annotations

import unittest

from maml.utils import ConstantValue, LinearProfile


class TestValueProfile(unittest.TestCase):
    def test_constant(self):
        constant_profile = ConstantValue(0.1)
        constant_profile.increment_step()
        assert constant_profile.step == 1
        self.assertAlmostEqual(constant_profile.get_value(), 0.1)

    def test_linearprofile(self):
        linear = LinearProfile(value_start=100, value_end=1, max_steps=99)

        self.assertAlmostEqual(linear.rate, -1)
        for _ in range(99):
            linear.increment_step()
        self.assertAlmostEqual(linear.get_value(), 1)


if __name__ == "__main__":
    unittest.main()
