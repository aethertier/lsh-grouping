from collections import defaultdict

class Incr:
    """
    Factory for generating sequential integers, starting from an initial value.

    This class is typically used in conjunction with `defaultdict` to lazily assign
    unique IDs or indices to new keys as they are encountered. Each call to an `Inc`
    instance returns the current value and increments the internal counter.

    Examples
    --------
    >>> inc = Incr()
    >>> inc()
    0
    >>> inc()
    1

    >>> from collections import defaultdict
    >>> auto_index = defaultdict(Incr())
    >>> auto_index['a'], auto_index['b'], auto_index['a']
    (0, 1, 0)

    Parameters
    ----------
    initial_value : int, optional
        The starting value of the counter (default is 0).

    Returns
    -------
    int
        The next value in the sequence.
    """

    def __init__(self, initial_value: int = 0):
        self.value = initial_value

    def __call__(self) -> int:
        v, self.value = self.value, self.value + 1
        return v


def indexdict(start_value: int=0):
    '''
    Create a defaultdict that assigns incremental integer indices to new keys.

    This factory function returns a defaultdict where each missing key is automatically
    assigned a unique integer index, starting from `start`. It is useful for creating
    reverse indices, lookup tables, or encoding categorical values.

    Parameters
    ----------
    start_value : int, optional
        The starting value for the incremental index (default is 0).

    Returns
    -------
    defaultdict
        A defaultdict[int] that maps each new key to an incrementing integer.
    '''
    return defaultdict(Incr(start_value))