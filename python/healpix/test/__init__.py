import numpy as np


def assert_shape(a, shape):
    s = np.shape(a)
    if shape is None:
        assert s == (), f'expected scalar, got shape {s}'
    else:
        if not isinstance(shape, tuple):
            shape = (shape,)
        assert s == shape, f'expected shape {shape}, got shape {s}'
