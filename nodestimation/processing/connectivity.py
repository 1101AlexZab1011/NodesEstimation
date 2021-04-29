from typing import *

from nodestimation.processing.timewindow import sliding_window
import numpy as np


@sliding_window(size=400, overlap=0.5)
def do_nothing(sig):
    """does exactly what its name says

    :param sig: signal or set of signals
    :type sig: |inp.ndarray|_
    :return: the given signal/set of signals
    :rtype: np.ndarray_

    .. _np.ndarray:
    .. _inp.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html

    .. |inp.ndarray| replace:: *np.ndarray*
    """
    return sig


@sliding_window(400, 0.5)
def pearson(signals: np.ndarray) -> np.ndarray:
    """computes `Pearson's correlation coefficients <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_ map used in
        :func:`nodestimation.processing.timewindow.sliding_window` for 400-dots with 50% overlapping

    :param signals: set of signals
    :type signals: |inp.ndarray|_
    :return: signal-to-signal pearson`s correlations map
    :rtype: np.ndarray_
    """

    nsigmals, lsignals = signals.shape
    out = np.zeros((nsigmals, nsigmals))

    for i in range(nsigmals):
        for j in range(nsigmals):

            if i == j:
                out[i, j] = 0
                continue

            out[i, j] = np.corrcoef(signals[i, :], signals[j, :])[0, 1]

    return out


def pearson_ts(label_ts: List[np.ndarray]) -> np.ndarray:
    """computes `Pearson's correlation coefficients <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_ map for list of signals

    :param label_ts: list of signals
    :type label_ts: |ilist|_ *of* |inp.ndarray|_
    :return: signal-to-signal pearson`s correlations map
    :rtype: np.ndarray_

    .. _ilist: https://docs.python.org/3/library/stdtypes.html#list

    .. |ilist| replace:: *list*
    """

    out = list()
    for signals in label_ts:
        n, m = signals.shape
        lout = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                lout[i, j] = np.corrcoef(signals[i, :], signals[j, :])[0, 1]
        out.append(lout)
    return np.mean(np.array(out), axis=0)


@sliding_window(400, 0.5)
def phase_locking_value(signals: np.ndarray) -> np.ndarray:
    """computes `Phase Locking Value coefficients <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3674231/>`_ map used in
        :func:`nodestimation.processing.timewindow.sliding_window` for 400-dots with 50% overlapping

    :param signals: set of signals
    :type signals: |inp.ndarray|_
    :return: signal-to-signal pearson`s correlations map
    :rtype: np.ndarray_
    """

    nsigmals, lsignals = signals.shape
    out = np.zeros((nsigmals, nsigmals, lsignals))

    for i in range(nsigmals):
        for j in range(nsigmals):

            sig1_fourier = np.fft.fft(signals[i])
            sig2_fourier = np.fft.fft(signals[j])
            plv_1_2 = []

            for k in range(lsignals):
                plv_1_2.append(sig1_fourier[k] * np.conj(sig2_fourier[k]) /
                               (np.abs(sig1_fourier[k]) * np.abs(sig2_fourier[k])))

            out[i, j, :] = plv_1_2

    return np.array(out)
