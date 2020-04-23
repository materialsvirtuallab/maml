import os
import unittest

import numpy as np
from scipy import signal

from maml.utils import get_sp_method, fft_magnitude, cwt, spectrogram, wvd

CWD = os.path.join(os.path.dirname(__file__))

try:
    import tftb
except ImportError:
    tftb = None


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
        self.assertRaises(KeyError, get_sp_method, sp_method="wrong_method")

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

    @unittest.skipIf(tftb is None)
    def test_wvd(self):
        wvd_res = wvd(self.x)
        self.assertTrue(wvd_res.shape == (100, 100))


if __name__ == "__main__":
    unittest.main()
