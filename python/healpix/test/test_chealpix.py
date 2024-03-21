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


def test_ang_nest(nside):
    from chealpix import ang2nest, nest2ang
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(ang2nest(nside, *nest2ang(nside, ipix)), ipix)


def test_ang_ring(nside_dirty):
    from chealpix import ang2ring, ring2ang
    ipix = np.arange(12*nside_dirty**2)
    npt.assert_array_equal(ang2ring(nside_dirty, *ring2ang(nside_dirty, ipix)), ipix)


def test_vec_nest(nside):
    from chealpix import vec2nest, nest2vec
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(vec2nest(nside, *nest2vec(nside, ipix)), ipix)


def test_vec_ring(nside_dirty):
    from chealpix import vec2ring, ring2vec
    ipix = np.arange(12*nside_dirty**2)
    npt.assert_array_equal(vec2ring(nside_dirty, *ring2vec(nside_dirty, ipix)), ipix)


def test_ang_nest_uv(nside):
    from chealpix import ang2nest_uv, nest2ang_uv
    ipix = np.arange(12*nside**2)
    u, v = np.random.rand(2, ipix.size)
    result = ang2nest_uv(nside, *nest2ang_uv(nside, ipix, u, v))
    npt.assert_array_equal(result[0], ipix)
    npt.assert_allclose(result[1], u, rtol=1e-6)
    npt.assert_allclose(result[2], v, rtol=1e-6)


def test_ang_ring_uv(nside_dirty):
    from chealpix import ang2ring_uv, ring2ang_uv
    ipix = np.arange(12*nside_dirty**2)
    u, v = np.random.rand(2, ipix.size)
    result = ang2ring_uv(nside_dirty, *ring2ang_uv(nside_dirty, ipix, u, v))
    npt.assert_array_equal(result[0], ipix)
    npt.assert_allclose(result[1], u, rtol=1e-6)
    npt.assert_allclose(result[2], v, rtol=1e-6)


def test_vec_nest_uv(nside):
    from chealpix import vec2nest_uv, nest2vec_uv
    ipix = np.arange(12*nside**2)
    u, v = np.random.rand(2, ipix.size)
    result = vec2nest_uv(nside, *nest2vec_uv(nside, ipix, u, v))
    npt.assert_array_equal(result[0], ipix)
    npt.assert_allclose(result[1], u, rtol=1e-6)
    npt.assert_allclose(result[2], v, rtol=1e-6)


def test_vec_ring_uv(nside_dirty):
    from chealpix import vec2ring_uv, ring2vec_uv
    ipix = np.arange(12*nside_dirty**2)
    u, v = np.random.rand(2, ipix.size)
    result = vec2ring_uv(nside_dirty, *ring2vec_uv(nside_dirty, ipix, u, v))
    npt.assert_array_equal(result[0], ipix)
    npt.assert_allclose(result[1], u, rtol=1e-6)
    npt.assert_allclose(result[2], v, rtol=1e-6)


def test_nest_ring(nside):
    from chealpix import nest2ring, ring2nest
    ipix = np.arange(12*nside**2)
    npt.assert_array_equal(nest2ring(nside, ring2nest(nside, ipix)), ipix)
    npt.assert_array_equal(ring2nest(nside, nest2ring(nside, ipix)), ipix)


def test_nside2npix(nside_dirty):
    from chealpix import nside2npix
    assert nside2npix(nside_dirty) == 12*nside_dirty**2


def test_npix2nside(nside_dirty):
    from chealpix import npix2nside
    assert npix2nside(12*nside_dirty**2) == nside_dirty


def test_npix2nside_invalid():
    from chealpix import npix2nside
    assert npix2nside(7) == -1
    assert npix2nside(49) == -1


def test_nside2order(nside, order):
    from chealpix import nside2order
    assert nside2order(nside) == order


def test_nside2order_invalid():
    from chealpix import nside2order
    assert nside2order(-1) == -1
    assert nside2order(3) == -1


def test_order2nside(nside, order):
    from chealpix import order2nside
    assert order2nside(order) == nside


def test_order2nside_invalid():
    from chealpix import order2nside
    assert order2nside(-1) == -1


def test_uniq_nest(order):
    from chealpix import uniq2nest, nest2uniq
    ipix = np.arange(12 * (1 << 2*order))
    order_out, ipix_out = uniq2nest(nest2uniq(order, ipix))
    npt.assert_array_equal(order_out, order)
    npt.assert_array_equal(ipix_out, ipix)


def test_uniq_ring(order):
    from chealpix import uniq2ring, ring2uniq
    ipix = np.arange(12 * (1 << 2*order))
    order_out, ipix_out = uniq2ring(ring2uniq(order, ipix))
    npt.assert_array_equal(order_out, order)
    npt.assert_array_equal(ipix_out, ipix)
