from typing import *
from functools import wraps
import numpy as np


class TimeWindow(object):
    """Class representing time window of signal

        :param data: supposed to be a signal or set of signals inside the current time window
        :type data: |inp.ndarray|_
        :param start: timepoint of time window start
        :type start: |iint|_ *or* |ifloat|_
        :param end: timepoint of time window end
        :type end: |iint|_ *or* |ifloat|_

        .. _iint: https://docs.python.org/3/library/functions.html#int

        .. |iint| replace:: *int*
    """

    def __init__(self, data: np.ndarray, start: Union[int, float], end: Union[int, float]):
        self.data = data
        self.start = start
        self.end = end
        self.__len = end - start

    @property
    def end(self):
        return self

    @property
    def data(self):
        return self

    @property
    def start(self):
        return self

    @data.setter
    def data(self, data: np.ndarray):
        self._data = data

    @data.getter
    def data(self):
        """supposed to be a signal or set of signals inside the current time window"""
        return self._data

    @start.setter
    def start(self, t_start):
        self._start = t_start
        self._end = t_start + self.__len

    @start.getter
    def start(self):
        """timepoint of time window start"""
        return self._start

    @end.setter
    def end(self, value):
        self._end = value

    @end.getter
    def end(self):
        """timepoint of time window end"""
        return self._end


def sliding_window(size: int, overlap: float):
    """divides data given to wrapped function (1st argument) into slices of given size with given overlap and calls it

    :param size: size (in points) of time window to divide data
    :type size: int
    :param overlap: time windows overlapping
    :type: float
    :return: time windows with processed data
    :rtype: :class:`nodestimation.processing.timewindow.TimeWindow`
    """

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs):

            f = args[0]
            sig_len = f.shape[-1]
            out = []
            now = 0
            i = 0
            not_end = True

            def resize(args, start, end):

                dim = len(args[0].shape)

                if dim == 1:
                    largs = list(args)
                    largs[0] = f[start:end]
                    args = tuple(largs)

                    return args

                elif dim == 2:
                    largs = list(args)
                    largs[0] = f[:, start:end]
                    args = tuple(largs)

                    return args

                elif dim == 3:
                    largs = list(args)
                    largs[0] = f[:, :, start:end]
                    args = tuple(largs)

                    return args

            while not_end:
                i += 1
                start = now
                end = start + size

                if end > sig_len - 1:
                    end = sig_len - 1
                    args = resize(args, start, end)
                    t_w = TimeWindow(func(*args, **kwargs), start, end)
                    out.append(t_w)
                    not_end = not not_end

                elif end <= sig_len - 1:
                    args = resize(args, start, end)
                    t_w = TimeWindow(func(*args, **kwargs), start, end)
                    out.append(t_w)
                    now = int(end - overlap * size)

            return out

        return wrapper

    return decorator


def mean_across_tw(twlist: List[TimeWindow]) -> np.ndarray:
    """computes mean for data inside the given list of time windows

    :param twlist: windowed data
    :type twlist: |ilist|_ *of* :class:`nodestimation.processing.timewindow.TimeWindow`
    :return: mean for windowed data
    :rtype: np.ndarray_
    """

    if len(twlist[0].data.shape) == 2:
        l, w = twlist[0].data.shape
        voxel = voxel_from_tw(twlist)
        out = np.zeros((l, w))
        for i in range(l):
            for j in range(w):
                out[i, j] = np.mean(voxel[i, j, :])

        return out

    elif len(twlist[0].data.shape) == 3:
        l, w, h = twlist[0].data.shape
        voxel = voxel_from_tw(twlist)
        out = np.zeros((l, w, h))

        for i in range(l):
            for j in range(w):
                for k in range(h):
                    out[i, j, k] = np.mean(voxel[i, j, k, :])

        return out

    else:
        raise ValueError('Can not work with dimension less than two and higher than four')


def voxel_from_tw(twlist: List[TimeWindow]) -> np.ndarray:
    """creates n+1-dimensional voxel from the given time windows of n-dimensional data (n supposed to be 2 or 3)

    :param twlist: windowed data
    :type twlist: |ilist|_ *of* :class:`nodestimation.processing.timewindow.TimeWindow`
    :return: time windows collected in a voxel
    :rtype: np.ndarray
    """

    if len(twlist[0].data.shape) == 2:
        l, w = twlist[0].data.shape
        h = len(twlist)
        voxel = np.zeros((l, w, h))

        for i in range(h):
            voxel[:, :, i] = twlist[i].data

        return voxel

    elif len(twlist[0].data.shape) == 3:
        l, w, h = twlist[0].data.shape
        d = len(twlist)
        voxel = np.zeros((l, w, h, d))

        for i in range(d):
            voxel[0:twlist[i].data.shape[0],
            0:twlist[i].data.shape[1],
            0:twlist[i].data.shape[2], i] = twlist[i].data

        return voxel

    else:
        raise ValueError('Can not work with dimension less than two and higher than four')
