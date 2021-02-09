def by_default(value=None):
    # if the wrapped function has None as the first argument, it ignores the wrapped function and returns specified value or None

    def decorator(func):

        def wrapper(*args, **kwargs):

            if args[0] is None:
                out = value
            else:
                out = func(*args, **kwargs)

            return out

        return wrapper

    return decorator


def get_ith(func):
    # if wrapped function returns list, it ignores all except priority_ value
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)

        if not isinstance(out, list):
            return out

        elif 'priority_' in kwargs:
            return out[kwargs['priority_']]

        else:
            return out[0]

    return wrapper
