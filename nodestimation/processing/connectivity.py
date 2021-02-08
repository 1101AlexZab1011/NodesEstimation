from nodestimation.processing.timewindow import sliding_window
import numpy as np


@sliding_window(size=400, overlap=0.5)
def do_nothing(sig):
    return sig


@sliding_window(400, 0.5)
def pearson(signals):
    # computes Pearson's correlation coefficients map for 400-dots time windows with 50% overlapping

    nsigmals, lsignals = signals.shape
    out = np.zeros((nsigmals, nsigmals))

    for i in range(nsigmals):
        for j in range(nsigmals):

            if i == j:
                out[i, j] = 0
                continue

            out[i, j] = np.corrcoef(signals[i, :], signals[j, :])[0, 1]

    return out


def pearson_ts(label_ts):
    # computes Pearson's correlation coefficients map for list of data

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
def phase_locking_value(signals):
    # computes Phase Locking Value coefficients map for 400-dots time windows with 50% overlapping

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
