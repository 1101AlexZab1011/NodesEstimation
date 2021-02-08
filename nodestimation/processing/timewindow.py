import numpy as np


class TimeWindow(object):

    def __init__(self, data, start, end):
        self.data = data
        self.start = start
        self.end = end
        self.len = end - start

    def set_data(self, data):
        self.data = data

    def set_size(self, t_start):
        self.start = t_start
        self.end = t_start + self.len


def sliding_window(size, overlap):
    # does computation dividing given data into slices of given size with given overlap

    def decorator(func):

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


def mean_across_tw(twlist):
    # computes mean value for the given list of time windows

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


def voxel_from_tw(twlist):
    # creates n+1-dimensional voxel from the given time windows of n-dimensional data (n supposed to be 2 or 3)

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
