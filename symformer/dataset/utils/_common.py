import errno
import math
import os
import signal
import time
from functools import partial, wraps
from itertools import product
from typing import Tuple

import numpy as np


def find_furtherest_points(points: np.array):
    pairwise_dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    indexes = np.unravel_index(np.argmax(pairwise_dist, axis=None), pairwise_dist.shape)
    return points[list(indexes)], np.max(pairwise_dist)


def generate_all_possible_ranges(num_variables: int, left: float, right: float):
    """
    Returns all possible ranges from the largest to the small
    :param num_variables: number of variables
    :param left: left barrier
    :param right: right barrier
    :return: all ranges
    """
    points_ranges = np.array(list(product([left, right, 0], repeat=2 * num_variables)))
    points_ranges = map(lambda x: (x[:num_variables], x[num_variables:]), points_ranges)
    points_ranges = list(filter(lambda x: np.all(x[0] < x[1]), points_ranges))
    return sorted(points_ranges, key=lambda x: (x[0] - x[1]).sum())


def generate_all_possible_extrapolation_ranges(
    old_range: Tuple[float, ...], barrier: float, new_barrier: float
):
    new_range = []
    for value in old_range:
        if value == barrier:
            new_range.append(new_barrier)
        else:
            new_range.append(value)

    return new_range


class TimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


if __name__ == "__main__":
    new_ranges = generate_all_possible_ranges(1, -5, 5)
    for n_range in new_ranges:
        print(n_range)
        print(generate_all_possible_extrapolation_ranges(n_range[0], -5, -6))
        print(generate_all_possible_extrapolation_ranges(n_range[1], 5, 6))
        print()
