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