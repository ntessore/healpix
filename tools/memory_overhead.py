import argparse
import tracemalloc
import numpy as np
import healpix


TEST_FUNCTIONS = [
    'ang2vec',
    'vec2ang',
    'ang2pix',
    'pix2ang',
    'vec2pix',
    'pix2vec',
]


def get_nbytes(obj):
    if isinstance(obj, tuple):
        return sum(map(get_nbytes, obj))
    else:
        return obj.nbytes


def params_add(params, *args, **kwargs):
    return [((*a, *args), {**kw, **kwargs}) for a, kw in params]


def params_mul(params, *args, **kwargs):
    if args:
        params = [((*a, *a_), kw) for a, kw in params for a_ in args]
    if kwargs:
        kwargs = [{k: v} for k in kwargs for v in kwargs[k]]
        params = [(a, {**kw, **kw_}) for a, kw in params for kw_ in kwargs]
    return params


def printargs(*args, **kwargs):
    def fmt(x):
        return '[...]' if np.ndim(x) > 0 else repr(x)

    args_ = ', '.join(fmt(a) for a in args)
    kwargs_ = ', '.join(f'{k}={fmt(v)}' for k, v in kwargs.items())
    return ', '.join(filter(None, [args_, kwargs_]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--nside', type=int, default=1024,
                        help='NSIDE parameter, sets array sizes')
    parser.add_argument('--healpy', action='store_true',
                        help='check memory overhead of healpy functions')
    args = parser.parse_args()

    nside = args.nside
    use_healpy = args.healpy

    size = 12*nside**2

    if use_healpy:
        healpix = __import__('healpy')

    print('')
    print('using %s functions with NSIDE=%d' % ('healpix' if not use_healpy else 'healpy', nside))
    print('')
    print(f'{"Function":60}  {"Overhead":8}')
    print(f'{"-"*60:}--{"-"*8:}')

    for name in TEST_FUNCTIONS:

        base_params = [((), {})]

        if 'ang' in name:
            base_params = params_mul(base_params, lonlat=[False, True])

        for base_args, base_kwargs in base_params:
            params = [(base_args, base_kwargs)]
            if 'pix' in name:
                params = params_add(params, nside)
                params = params_mul(params, nest=[False, True])
            if name.startswith('pix2'):
                ipix = np.arange(size)
                params = params_add(params, ipix)
            elif name.startswith('ang2'):
                if base_kwargs['lonlat']:
                    lon = np.random.uniform(0, 360, size=size)
                    lat = np.random.uniform(-90, 90, size=size)
                    params = params_add(params, lon, lat)
                else:
                    theta = np.random.uniform(0, np.pi, size=size)
                    phi = np.random.uniform(0, 2*np.pi, size=size)
                    params = params_add(params, theta, phi)
            elif name.startswith('vec2'):
                if use_healpy and name == 'vec2ang':
                    xyz = np.random.randn(size, 3)
                    params = params_add(params, xyz)
                else:
                    x, y, z = np.random.randn(3, size)
                    params = params_add(params, x, y, z)

            func = getattr(healpix, name)

            for args, kwargs in params:
                label = f'{name}({printargs(*args, **kwargs)})'
                try:
                    tracemalloc.start()
                    result = func(*args, **kwargs)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                except Exception as e:
                    raise RuntimeError(label) from e
                output = get_nbytes(result)
                assert current - output < 1_000_000
                overhead = peak/current - 1
                print(f'{label:60}  {overhead:-8.0%}')

    print('')
