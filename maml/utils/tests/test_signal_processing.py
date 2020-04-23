import os
import unittest

import numpy as np
from scipy import signal

from maml.utils import get_sp_method, fft_magnitude, cwt, spectrogram

CWD = os.path.join(os.path.dirname(__file__))


class TestSP(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.x = np.load(os.path.join(CWD, "test_x.npy"))

    def test_fft(self):
        fft_mag = fft_magnitude(self.x)
        self.assertTrue(fft_mag.shape == (100, ))
        fft_mag = get_sp_method('fft_magnitude')(self.x)
        self.assertTrue(fft_mag.shape == (100,))
        fft_mag = get_sp_method(fft_magnitude)(self.x)
        self.assertTrue(fft_mag.shape == (100,))

    def test_spectrogram(self):
        spec = spectrogram(self.x)
        self.assertTrue(spec.shape == (129, 8))

        freq, time, spec = spectrogram(self.x, return_time_freq=True)
        self.assertTrue(freq.shape == (129, ))
        self.assertTrue(time.shape == (8, ))
        self.assertTrue(spec.shape == (129, 8))

    def test_cwt(self):
        cwt_res = cwt(self.x, np.arange(1, 31), 'ricker')
        self.assertTrue(cwt_res.shape == (30, 100))
        cwt_res = cwt(self.x, np.arange(1, 31), signal.ricker)
        self.assertTrue(cwt_res.shape == (30, 100))


if __name__ == "__main__":
    unittest.main()
