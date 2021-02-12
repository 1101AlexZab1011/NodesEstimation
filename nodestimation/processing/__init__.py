from typing import *


def by_default(default_values: tuple, default_result: Optional[Union[tuple, Any]] = None) -> Any:
    # if the wrapped function has None as the first argument, it ignores the wrapped function and returns specified nodes or None

    def decorator(func: Callable) -> Callable:

        def wrapper(*args, **kwargs):

            if all([default_value == arg for default_value, arg in zip(default_values, args)]):
                out = default_result
            else:
                out = func(*args, **kwargs)

            return out

        return wrapper

    return decorator


def get_ith(func: Callable) -> Callable:
    # if wrapped function returns list, it ignores all except priority_ nodes
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)

        if not isinstance(out, list):
            return out

        elif 'priority_' in kwargs:
            return out[kwargs['priority_']]

        else:
            return out[0]

    return wrapper
