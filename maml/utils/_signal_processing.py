"""Signal processing utils"""
from typing import Union, Tuple, Callable
from math import ceil, floor

import numpy as np
from scipy import fft
from scipy import signal


def fft_magnitude(z: np.ndarray) -> np.ndarray:
    """
    Dicrete Fourier Transform the signal z and return
    the magnitude of the  coefficents
    Args:
        z (np.ndarray): 1D signal array

    Returns: 1D magnitude
    """
    return np.absolute(fft.fft(z))


def spectrogram(z: np.ndarray, return_time_freq: bool = False) \
        -> Union[Tuple, np.ndarray]:
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
    freq, time, s = signal.spectrogram(z, window=signal.get_window('hann', nsc),
                                       noverlap=nov, nfft=nfft)
    if return_time_freq:
        return freq, time, s
    return s


def cwt(z: np.ndarray, widths: np.ndarray,
        wavelet: Union[str, Callable] = 'morlet', **kwargs) -> np.ndarray:
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
    return np.absolute(signal.cwt(
        z, wavelet_func, widths=widths, **kwargs))


def get_sp_method(sp_method: Union[str, Callable]) -> Callable:  # type: ignore
    """
    Providing a signal processing method name return the callable
    Args:
        sp_method (str): name of the sp function
    Returns: callable for signal processing
    """

    if isinstance(sp_method, Callable):  # type: ignore
        return sp_method  # type: ignore

    if isinstance(sp_method, str):
        return globals()[sp_method]
