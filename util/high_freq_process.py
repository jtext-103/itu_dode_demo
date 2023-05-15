# @Time : 2023/5/10 21:55 
# @Author : zhongyu 
# @File : high_freq_process.py
from jddb.processor import Signal, BaseProcessor
import numpy as np
from copy import deepcopy


def psd2(x, fs=1.0, window='hanning', nperseg=None, noverlap=None, nfft=None,
         detrend='constant', show=False, ax=None, scales='linear', xlim=None,
         units='V'):
    from scipy import signal, integrate

    if not nperseg:
        nperseg = np.ceil(len(x) / 2)
    f, P = signal.welch(x, fs, window, nperseg, noverlap, nfft, detrend)
    Area = integrate.cumtrapz(P, f, initial=0)
    Ptotal = Area[-1]
    mpf = integrate.trapz(f * P, f) / Ptotal  # mean power frequency
    fmax = f[np.argmax(P)]
    fmin = f[np.argmin(P)]
    # frequency percentiles
    inds = [0]
    Area = 100 * Area / Ptotal  # + 10 * np.finfo(np.float).eps
    for i in range(1, 101):
        inds.append(np.argmax(Area[inds[-1]:] >= i) + inds[-1])
    fpcntile = f[inds]

    return mpf, fmax, fmin, Ptotal


class MainFreq(BaseProcessor):

    def __init__(self, ):
        super().__init__()

    def transform(self, signal: Signal) -> Signal:
        new_signal = deepcopy(signal)
        mpf, fmax, fmin, Ptotal = psd2(new_signal.data, new_signal.attributes["SampleRate"], window='hanning',
                                       nperseg=None, noverlap=None, nfft=None,
                                       detrend='constant', show=True)
        main_freq = fmax
        main_freq_attribute = deepcopy(signal.attributes)
        return Signal(data=main_freq, attributes=main_freq_attribute)
