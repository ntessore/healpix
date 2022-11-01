import pytest
import numpy as np

np.random.seed(12345)


def is_power_of_two(n):
    return (n & (n-1)) == 0


def fixture_get(request, name, default):
    if name in request.fixturenames:
        return request.getfixturevalue(name)
    else:
        return default


@pytest.fixture(params=[1, 15, 16, 128])
def nside(request):
    nside = request.param
    if (fixture_get(request, 'nest', False) or request.node.get_closest_marker('nest')) \
            and not is_power_of_two(nside):
        pytest.skip(reason='nside is not power of two')
    return nside


@pytest.fixture(params=[True, False])
def nest(request):
    return request.param


@pytest.fixture(params=[True, False])
def lonlat(request):
    return request.param


@pytest.fixture(params=[None, (), 1, 1000, (7, 11)])
def size(request):
    return request.param


@pytest.fixture
def base_pixel_ang(request):
    theta = np.repeat(np.arccos([2/3, 0, -2/3]), 4)
    phi = np.pi/4*np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])
    if fixture_get(request, 'lonlat', False):
        lon = np.rad2deg(phi)
        lat = np.rad2deg(np.pi/2 - theta)
        theta, phi = lon, lat
    return theta, phi


@pytest.fixture
def base_pixel_vec(request):
    a, b = (5/18)**0.5, 2/3
    x = np.array([a, -a, -a, a, 1, 0, -1, 0, a, -a, -a, a])
    y = np.array([a, a, -a, -a, 0, 1, 0, -1, a, a, -a, -a])
    z = np.array([b, b, b, b, 0, 0, 0, 0, -b, -b, -b, -b])
    return x, y, z


def pytest_configure(config):
    config.addinivalue_line(
            'markers', 'nest: mark test as using the NEST scheme')
