from typing import *
from functools import wraps


def by_default(default_values: tuple, default_result: Optional[Any] = None) -> Any:
    """if the wrapped function called with parameters as specified, it ignores the wrapped function and returns a specified value or None

    :param default_values: supposed values of a wrapped function to return a specified result instead of calling it
    :type default_values: tuple
    :param default_result: output to give if supposed values the same as given to a wrapped function, default None
    :type default_result: any, optional
    :return: wrapped function or default_result
    :rtype: `callable <https://docs.python.org/3/library/typing.html#typing.Callable>`_ or any
    """

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs):

            if all([default_value == arg for default_value, arg in zip(default_values, args)]):
                out = default_result
            else:
                out = func(*args, **kwargs)

            return out

        return wrapper

    return decorator


def get_ith(func: Callable) -> Callable:
    """if wrapped function returns list, it ignores all except 0th output or,
        if wrapped function has an argument "_priority" of type `int <https://docs.python.org/3/library/functions.html#int>`_,
        it returns output with number specified in "_priority" argument

        :param func: a wrapped function
        :type func: callable
        :return: specified output of a wrapped function
        :rtype: any
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)

        if not isinstance(out, list):
            return out

        elif '_priority' in kwargs and isinstance(kwargs['_priority'], int):
            return out[kwargs['_priority']]

        else:
            return out[0]

    return wrapper
