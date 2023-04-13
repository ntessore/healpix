import pytest
import numpy as np
import numpy.testing as npt


def test_ang2vec():
    from chealpix import ang2vec
    theta = np.arccos(np.random.uniform(-1, 1, size=100))
    phi = np.random.uniform(0, 2*np.pi, size=100)
    x, y, z = ang2vec(theta, phi)
    npt.assert_allclose(x, np.sin(theta)*np.cos(phi))
    npt.assert_allclose(y, np.sin(theta)*np.sin(phi))
    npt.assert_allclose(z, np.cos(theta))


def test_vec2ang():
    from chealpix import vec2ang
    x, y, z = np.random.randn(3, 100)
    theta, phi = vec2ang(x, y, z)
    npt.assert_allclose(theta, np.arctan2(np.hypot(x, y), z))
    npt.assert_allclose(phi, np.arctan2(y, x))


@pytest.mark.nest
def test_ang_nest(nside):
    from chealpix import ang2nest, nest2ang
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(ang2nest(nside, *nest2ang(nside, ipix)), ipix)


def test_ang_ring(nside):
    from chealpix import ang2ring, ring2ang
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(ang2ring(nside, *ring2ang(nside, ipix)), ipix)


@pytest.mark.nest
def test_vec_nest(nside):
    from chealpix import vec2nest, nest2vec
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(vec2nest(nside, *nest2vec(nside, ipix)), ipix)


def test_vec_ring(nside):
    from chealpix import vec2ring, ring2vec
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(vec2ring(nside, *ring2vec(nside, ipix)), ipix)


@pytest.mark.nest
def test_ang_nest_uv(nside):
    from chealpix import ang2nest_uv, nest2ang_uv
    ipix = np.arange(12*nside**2)
    u, v = np.random.rand(2, ipix.size)
    result = ang2nest_uv(nside, *nest2ang_uv(nside, ipix, u, v))
    npt.assert_array_equal(result[0], ipix)
    npt.assert_allclose(result[1], u)
    npt.assert_allclose(result[2], v)


def test_ang_ring_uv(nside):
    from chealpix import ang2ring_uv, ring2ang_uv
    ipix = np.arange(12*nside**2)
    u, v = np.random.rand(2, ipix.size)
    result = ang2ring_uv(nside, *ring2ang_uv(nside, ipix, u, v))
    npt.assert_array_equal(result[0], ipix)
    npt.assert_allclose(result[1], u)
    npt.assert_allclose(result[2], v)


@pytest.mark.nest
def test_vec_nest_uv(nside):
    from chealpix import vec2nest_uv, nest2vec_uv
    ipix = np.arange(12*nside**2)
    u, v = np.random.rand(2, ipix.size)
    result = vec2nest_uv(nside, *nest2vec_uv(nside, ipix, u, v))
    npt.assert_array_equal(result[0], ipix)
    npt.assert_allclose(result[1], u)
    npt.assert_allclose(result[2], v)


def test_vec_ring_uv(nside):
    from chealpix import vec2ring_uv, ring2vec_uv
    ipix = np.arange(12*nside**2)
    u, v = np.random.rand(2, ipix.size)
    result = vec2ring_uv(nside, *ring2vec_uv(nside, ipix, u, v))
    npt.assert_array_equal(result[0], ipix)
    npt.assert_allclose(result[1], u)
    npt.assert_allclose(result[2], v)


@pytest.mark.nest
def test_nest_ring(nside):
    from chealpix import nest2ring, ring2nest
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(nest2ring(nside, ring2nest(nside, ipix)), ipix)
    npt.assert_array_equal(ring2nest(nside, nest2ring(nside, ipix)), ipix)


def test_nside2npix(nside):
    from chealpix import nside2npix
    assert nside2npix(nside) == 12*nside**2


def test_npix2nside(nside):
    from chealpix import npix2nside
    assert npix2nside(12*nside**2) == nside


def test_npix2nside_invalid():
    from chealpix import npix2nside
    assert npix2nside(7) == -1
    assert npix2nside(49) == -1


@pytest.mark.nest
def test_uniq_nest(nside):
    from chealpix import uniq2nest, nest2uniq
    ipix = np.arange(12*nside**2)
    nside_out, ipix_out = uniq2nest(nest2uniq(nside, ipix))
    npt.assert_array_equal(nside_out, nside)
    npt.assert_array_equal(ipix_out, ipix)


@pytest.mark.nest
def test_uniq_ring(nside):
    from chealpix import uniq2ring, ring2uniq
    ipix = np.arange(12*nside**2)
    nside_out, ipix_out = uniq2ring(ring2uniq(nside, ipix))
    npt.assert_array_equal(nside_out, nside)
    npt.assert_array_equal(ipix_out, ipix)
