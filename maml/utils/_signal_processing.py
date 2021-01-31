"""Signal processing utils"""
from math import ceil, floor
from typing import Union, Tuple, Callable

import numpy as np

from scipy import fft
from scipy import signal

from monty.dev import requires

try:
    import tftb
except ImportError:
    tftb = None


def fft_magnitude(z: np.ndarray) -> np.ndarray:
    """
    Dicrete Fourier Transform the signal z and return
    the magnitude of the  coefficents
    Args:
        z (np.ndarray): 1D signal array
    Returns: 1D magnitude
    """
    return np.absolute(fft.fft(z))


def spectrogram(z: np.ndarray, return_time_freq: bool = False) -> Union[Tuple, np.ndarray]:
    """
    The spectrogram of the signal
    Args:
        z (np.ndarray): 1D signal array
        return_time_freq (bool): whether to return time and frequency
    Returns: 2D spectrogram
    """
    nx = len(z)
    nsc = floor(nx / 4.5)  # use matlab values
    nov = floor(nsc / 2)
    nfft = max(256, 2 ** ceil(np.log2(nsc)))
    freq, time, s = signal.spectrogram(z, window=signal.get_window("hann", nsc), noverlap=nov, nfft=nfft)
    if return_time_freq:
        return freq, time, s
    return s


def cwt(z: np.ndarray, widths: np.ndarray, wavelet: Union[str, Callable] = "morlet2", **kwargs) -> np.ndarray:
    """
    The scalogram of the signal
    Args:
        z (np.ndarray): 1D signal array
        widths (np.ndarray): wavelet widths
        wavelet (str): wavelet name
    Returns: 2D scalogram
    """
    if isinstance(wavelet, str):
        wavelet_func = getattr(signal, wavelet)
    else:
        wavelet_func = wavelet
    return np.absolute(signal.cwt(z, wavelet_func, widths=widths, **kwargs))


@requires(tftb is not None, "Requires installation of tftb package")
def wvd(z: np.ndarray, return_all: bool = False) -> Union[Tuple, np.ndarray]:
    """
    Wigner Ville Distribution calculator
    Args:
        z (np.ndarray): signal 1D
        return_all (bool): whether to return time and freq info, default
            only return the wvd information
    Returns: NxN wvd matrix
    """
    tfr = tftb.processing.WignerVilleDistribution(z)
    (
        res,
        f1,
        f2,
    ) = tfr.run()
    if return_all:
        return res, f1, f2
    return res


AVAILABLE_SP_METHODS = {"fft_magnitude": fft_magnitude, "spectrogram": spectrogram, "cwt": cwt, "wvd": wvd}


def get_sp_method(sp_method: Union[str, Callable]) -> Callable:  # type: ignore
    """
    Providing a signal processing method name return the callable
    Args:
        sp_method (str): name of the sp function
    Returns: callable for signal processing
    """
    if isinstance(sp_method, str):
        try:
            return AVAILABLE_SP_METHODS[sp_method]
        except KeyError:
            raise KeyError(f"{sp_method} is not in available methods: " f"{AVAILABLE_SP_METHODS.keys()}")
    else:
        return sp_method
