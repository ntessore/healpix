#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/random/distributions.h>
#include <math.h>

#include "healpix.h"


// ----------------------------------------------------------------------------
// constants

static const double TWOPI = 6.28318530717958647692528676655900576839433880;
static const double DEGREE = 0.0174532925199432957692369076848861271344287189;


// ----------------------------------------------------------------------------
// array operations


// maximum number of operands for array operations
#define ARR_MAXOP 4


typedef void* t_args;


#define args_ptr(type, args, i) ((type*)args[i])
#define args_val(type, args, i) (*args_ptr(type, args, i))


typedef struct {
    char* ptr[ARR_MAXOP];
    npy_intp num;
    npy_intp size;
    npy_intp* strides;
} t_arr;


#define arr_ptr(type, arr, i) ((type*)arr->ptr[i])
#define arr_val(type, arr, i) (*arr_ptr(type, arr, i))


static inline bool arrnext(t_arr* arr) {
    arr->size -= 1;
    if (arr->size == 0)
        return false;
    for (npy_intp i = 0; i < arr->num; ++i)
        arr->ptr[i] += arr->strides[i];
    return true;
}


typedef void (*t_arrfunc)(t_args*, t_arr*);


bool map_array(t_arrfunc func, t_args* args,
               npy_intp nin, PyObject** in, int* in_types,
               npy_intp nout, PyObject** out, int* out_types)
{
    npy_intp i, nop;
    NpyIter* iter;
    PyArrayObject* op[ARR_MAXOP] = {0};
    npy_uint32 flags;
    npy_uint32 op_flags[ARR_MAXOP];
    PyArray_Descr* op_dtypes[ARR_MAXOP] = {0};
    PyArrayObject** outarr;

    nop = nin + nout;

    if (nop > ARR_MAXOP) {
        PyErr_SetString(PyExc_RuntimeError,
                        "healpix internal error: increase ARR_MAXOP");
        return false;
    }

    for (i = 0; i < nin; ++i) {
        op[i] = (PyArrayObject*)PyArray_FromAny(in[i], NULL, 0, 0, 0, NULL);
        op_flags[i] = NPY_ITER_READONLY | NPY_ITER_NBO;
        op_dtypes[i] = PyArray_DescrFromType(in_types[i]);
        if (!op[i] || !op_dtypes[i])
            goto fail;
    }

    for (; i < nop; ++i) {
        op[i] = NULL;
        op_flags[i] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
        op_dtypes[i] = PyArray_DescrFromType(out_types[i-nin]);
        if (!op_dtypes[i])
            goto fail;
    }

    flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED | NPY_ITER_GROWINNER |
            NPY_ITER_ZEROSIZE_OK;

    iter = NpyIter_MultiNew(nop, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, op_dtypes);
    if (!iter)
        goto fail;

    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, NULL);
        npy_intp* strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp* sizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        char** dataptr = NpyIter_GetDataPtrArray(iter);
        t_arr arr = {.num = nop, .strides = strides};

        do {
            for (npy_intp i = 0; i < nop; ++i)
                arr.ptr[i] = dataptr[i];
            arr.size = *sizeptr;
            func(args, &arr);
        } while (iternext(iter));
    }

    outarr = &NpyIter_GetOperandArray(iter)[nin];
    for (i = 0; i < nout; ++i) {
        Py_INCREF(outarr[i]);
        out[i] = PyArray_Return(outarr[i]);
    }

    if (NpyIter_Deallocate(iter) != NPY_SUCCEED)
        goto fail;

    for (i = 0; i < nin; ++i) {
        Py_DECREF(op[i]);
        Py_DECREF(op_dtypes[i]);
    }
    for (; i < nop; ++i) {
        Py_DECREF(op_dtypes[i]);
    }

    return true;

fail:
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        Py_XDECREF(op_dtypes[i]);
    }
    for (i = 0; i < nout; ++i) {
        Py_XDECREF(out[i]);
    }
    return false;
}


// ----------------------------------------------------------------------------
// Python random number generation


PyObject* default_rng() {
    PyObject* np_random = NULL;
    PyObject* fromlist = NULL;
    PyObject* default_rng = NULL;
    PyObject* rng = NULL;

    fromlist = Py_BuildValue("[s]", "default_rng");
    if (!fromlist)
        goto fail;

    np_random = PyImport_ImportModuleEx("numpy.random", NULL, NULL, fromlist);
    if (!np_random)
        goto fail;

    default_rng = PyObject_GetAttrString(np_random, "default_rng");
    if (!default_rng) {
        PyErr_SetString(PyExc_ImportError,
                        "cannot import 'default_rng' from 'numpy.random'");
        goto fail;
    }

    rng = PyObject_CallObject(default_rng, NULL);
    if (!rng)
        goto fail;

    Py_DECREF(fromlist);
    Py_DECREF(np_random);
    Py_DECREF(default_rng);

    return rng;

fail:
    Py_XDECREF(fromlist);
    Py_XDECREF(np_random);
    Py_XDECREF(default_rng);
    Py_XDECREF(rng);
    return NULL;
}


bitgen_t* bitgen_from_rng(PyObject* rng) {
    PyObject* bit_generator = NULL;
    PyObject* capsule = NULL;
    bitgen_t* bitgen;

    bit_generator = PyObject_GetAttrString(rng, "bit_generator");
    if (!bit_generator) {
        PyErr_SetString(PyExc_AttributeError,
                        "'rng' has no attribute 'bit_generator'");
        goto fail;
    }

    capsule = PyObject_GetAttrString(bit_generator, "capsule");
    if (!capsule) {
        PyErr_SetString(PyExc_AttributeError,
                        "'rng.bit_generator' has no attribute 'capsule'");
        goto fail;
    }

    bitgen = (bitgen_t*)PyCapsule_GetPointer(capsule, "BitGenerator");
    if(!bitgen)
        goto fail;

    Py_XDECREF(bit_generator);
    Py_XDECREF(capsule);

    return bitgen;

fail:
    Py_XDECREF(bit_generator);
    Py_XDECREF(capsule);
    return NULL;
}


// ----------------------------------------------------------------------------
// HEALPix utility functions


static bool check_nside(int64_t nside, bool nest) {
    if (nside < 1 || nside > NSIDE_MAX) {
        PyErr_Format(PyExc_ValueError,
                     "nside must be a positive integer N with 1 <= N <= 2^%d",
                     (int)log2(NSIDE_MAX));
        return false;
    }
    if (nest && (nside & (nside-1)) != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "nside must be power of two in NEST scheme");
        return false;
    }
    return true;
}


static inline void setang(t_ang ang, double* x, double* y, bool lonlat) {
    if (lonlat) {
        *x = ang.phi/DEGREE;
        while (*x < 0)
            *x += 360;
        *y = 90 - ang.theta/DEGREE;
    } else {
        *x = ang.theta;
        *y = ang.phi;
        while (*y < 0)
            *y += TWOPI;
    }
}


static inline t_ang getang(double x, double y, bool lonlat) {
    if (lonlat) {
        return (t_ang){(90 - y)*DEGREE, x*DEGREE};
    } else {
        return (t_ang){x, y};
    }
}


#define arr_ang(arr, i, j, lonlat) getang(arr_val(double, arr, i), \
                                          arr_val(double, arr, j), \
                                          lonlat)


static inline void setvec(t_vec vec, double* x, double* y, double* z) {
    *x = vec.x;
    *y = vec.y;
    *z = vec.z;
}


static inline t_vec getvec(double x, double y, double z) {
    return (t_vec){x, y, z};
}


#define arr_vec(arr, i, j, k) getvec(arr_val(double, arr, i), \
                                     arr_val(double, arr, j), \
                                     arr_val(double, arr, k))


// ----------------------------------------------------------------------------
// ang2pix


static void ang2nest_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    int lonlat = args_val(int, args, 1);
    do {
        arr_val(int64_t, arr, 2) = ang2nest(nside, arr_ang(arr, 0, 1, lonlat));
    } while (arrnext(arr));
}


static void ang2ring_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    int lonlat = args_val(int, args, 1);
    do {
        arr_val(int64_t, arr, 2) = ang2ring(nside, arr_ang(arr, 0, 1, lonlat));
    } while (arrnext(arr));
}


static PyObject* ang2pix_(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* keywords[] = {"nside", "theta_lon", "phi_lat", "nest",
                               "lonlat", NULL};

    Py_ssize_t nside;
    PyObject* in[] = {NULL, NULL};
    int in_types[] = {NPY_DOUBLE, NPY_DOUBLE};
    int nest = false;
    int lonlat = false;
    PyObject* out[] = {NULL};
    int out_types[] = {NPY_INT64};
    t_arrfunc func;
    void* margs[] = {&nside, &lonlat};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nOO|pp:ang2pix", keywords,
                                     &nside, &in[0], &in[1], &nest, &lonlat)) {
        return NULL;
    }

    if (!check_nside(nside, nest))
        return NULL;

    func = nest ? ang2nest_arr : ang2ring_arr;

    if (!map_array(func, margs, 2, in, in_types, 1, out, out_types))
        return NULL;

    return out[0];
}


// ----------------------------------------------------------------------------
// pix2ang


static void nest2ang_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    int lonlat = args_val(int, args, 1);
    do {
        setang(nest2ang(nside, arr_val(int64_t, arr, 0)),
               arr_ptr(double, arr, 1),
               arr_ptr(double, arr, 2),
               lonlat);
    } while (arrnext(arr));
}


static void ring2ang_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    int lonlat = args_val(int, args, 1);
    do {
        setang(ring2ang(nside, arr_val(int64_t, arr, 0)),
               arr_ptr(double, arr, 1),
               arr_ptr(double, arr, 2),
               lonlat);
    } while (arrnext(arr));
}


PyDoc_STRVAR(pix2ang_doc, "\
pix2ang(nside, ipix, nest=False, lonlat=False)\n\
--\n\
\n\
Return the centres of the given HEALPix pixels as spherical coordinates.\n\
\n\
This function produces the spherical coordinates of each HEALPix pixel\n\
listed in `ipix`, which can be scalar or array-like.  The pixel indices use\n\
the `nside` resolution parameter and either the RING scheme (if `nest` is\n\
false, the default) or the NEST scheme (if `nest` is true).\n\
\n\
Returns either a tuple `theta, phi` of mathematical coordinates in radians\n\
if `lonlat` is False (the default), or a tuple `lon, lat` of longitude and\n\
latitude in degrees if `lonlat` is True.  The output is of the same shape as\n\
the input.\n\
");


static PyObject* pix2ang_(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* keywords[] = {"nside", "ipix", "nest", "lonlat", NULL};

    Py_ssize_t nside;
    PyObject* in[] = {NULL};
    int in_types[] = {NPY_INT64};
    int nest = false;
    int lonlat = false;
    PyObject* out[] = {NULL, NULL};
    int out_types[] = {NPY_DOUBLE, NPY_DOUBLE};
    PyObject* ret;
    t_arrfunc func;
    void* margs[] = {&nside, &lonlat};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nO|pp:pix2ang", keywords,
                                     &nside, &in[0], &nest, &lonlat)) {
        return NULL;
    }

    if (!check_nside(nside, nest))
        return NULL;

    func = nest ? nest2ang_arr : ring2ang_arr;

    if (!map_array(func, margs, 1, in, in_types, 2, out, out_types))
        return NULL;

    ret = Py_BuildValue("NN", out[0], out[1]);
    if (!ret) {
        Py_DECREF(out[0]);
        Py_DECREF(out[1]);
        return NULL;
    }

    return ret;
}


// ----------------------------------------------------------------------------
// vec2pix


static void vec2nest_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    do {
        arr_val(int64_t, arr, 3) = vec2nest(nside, arr_vec(arr, 0, 1, 2));
    } while (arrnext(arr));
}


static void vec2ring_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    do {
        arr_val(int64_t, arr, 3) = vec2ring(nside, arr_vec(arr, 0, 1, 2));
    } while (arrnext(arr));
}


static PyObject* vec2pix_(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* keywords[] = {"nside", "x", "y", "z", "nest", NULL};

    Py_ssize_t nside;
    PyObject* in[] = {NULL, NULL, NULL};
    int in_types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
    int nest = false;
    PyObject* out[] = {NULL};
    int out_types[] = {NPY_INT64};
    t_arrfunc func;
    void* margs[] = {&nside};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nOOO|p:vec2pix", keywords,
                                     &nside, &in[0], &in[1], &in[2], &nest)) {
        return NULL;
    }

    if (!check_nside(nside, nest))
        return NULL;

    func = nest ? vec2nest_arr : vec2ring_arr;;

    if (!map_array(func, margs, 3, in, in_types, 1, out, out_types))
        return NULL;

    return out[0];
}


// ----------------------------------------------------------------------------
// pix2vec


static void nest2vec_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    do {
        setvec(nest2vec(nside, arr_val(int64_t, arr, 0)),
               arr_ptr(double, arr, 1),
               arr_ptr(double, arr, 2),
               arr_ptr(double, arr, 3));
    } while(arrnext(arr));
}


static void ring2vec_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    do {
        setvec(ring2vec(nside, arr_val(int64_t, arr, 0)),
               arr_ptr(double, arr, 1),
               arr_ptr(double, arr, 2),
               arr_ptr(double, arr, 3));
    } while(arrnext(arr));
}


PyDoc_STRVAR(pix2vec_doc, "\
pix2vec(nside, ipix, nest=False)\n\
--\n\
\n\
Return the centres of the given HEALPix pixels as unit vectors.\n\
\n\
This function produces the unit vector of each HEALPix pixel listed in\n\
`ipix`, which can be scalar or array-like.  The pixel indices use the \n\
`nside` resolution parameter and either the RING scheme (if `nest` is false,\n\
the default) or the NEST scheme (if `nest` is true).\n\
\n\
Returns a tuple `x, y, z` of normalised vector components with the same \n\
shape as the input.\n\
");


static PyObject* pix2vec_(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* keywords[] = {"nside", "ipix", "nest", NULL};

    Py_ssize_t nside;
    PyObject* in[] = {NULL};
    int in_types[] = {NPY_INT64};
    int nest = false;
    PyObject* out[] = {NULL, NULL, NULL};
    int out_types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
    PyObject* ret;
    t_arrfunc func;
    void* margs[] = {&nside};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nO|p:pix2vec", keywords,
                                     &nside, &in[0], &nest)) {
        return NULL;
    }

    if (!check_nside(nside, nest))
        return NULL;

    func = nest ? nest2vec_arr : ring2vec_arr;

    if (!map_array(func, margs, 1, in, in_types, 3, out, out_types))
        return NULL;

    ret = Py_BuildValue("NNN", out[0], out[1], out[2]);
    if (!ret) {
        Py_DECREF(out[0]);
        Py_DECREF(out[1]);
        Py_DECREF(out[2]);
        return NULL;
    }

    return ret;
}


// ----------------------------------------------------------------------------
// nside2npix


static PyObject* nside2npix_(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    int64_t npix;

    if (!PyArg_ParseTuple(args, "n:nside2npix", &nside)) {
        return NULL;
    }

    npix = nside2npix(nside);

    return Py_BuildValue("n", (Py_ssize_t)npix);
}


// ----------------------------------------------------------------------------
// npix2nside


static PyObject* npix2nside_(PyObject* self, PyObject* args) {
    Py_ssize_t npix;
    int64_t nside;

    if (!PyArg_ParseTuple(args, "n:npix2nside", &npix)) {
        return NULL;
    }

    nside = npix2nside(npix);
    if (nside == -1) {
        return PyErr_Format(PyExc_ValueError,
                            "%zu is not a valid npix value", npix);
    }

    return Py_BuildValue("n", (Py_ssize_t)nside);
}


// ----------------------------------------------------------------------------
// randang


static void randang_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    int nest = args_val(int, args, 1);
    int lonlat = args_val(int, args, 2);
    bitgen_t* bitgen = args_val(bitgen_t*, args, 3);
    double u[2];
    do {
        random_standard_uniform_fill(bitgen, 2, u);
        setang(randang(nside, arr_val(int64_t, arr, 0), u[0], u[1], nest),
               arr_ptr(double, arr, 1),
               arr_ptr(double, arr, 2),
               lonlat);
    } while (arrnext(arr));
}


PyDoc_STRVAR(randang_doc, "\
randang(nside, ipix, nest=False, lonlat=False, rng=None)\n\
--\n\
\n\
Sample random spherical coordinates from the given HEALPix pixels.\n\
\n\
This function produces one pair of random spherical coordinates from each\n\
HEALPix pixel listed in `ipix`, which can be scalar or array-like.  The\n\
indices use the `nside` resolution parameter and either the RING scheme\n\
(if `nest` is false, the default) or the NEST scheme (if `nest` is true).\n\
\n\
Returns either a tuple `theta, phi` of mathematical coordinates in radians\n\
if `lonlat` is False (the default), or a tuple `lon, lat` of longitude and\n\
latitude in degrees if `lonlat` is True.  The output is of the same shape as\n\
the input.\n\
\n\
An optional numpy random number generator can be provided using `rng`;\n\
otherwise, a new numpy.random.default_rng() is used.\n\
");


static PyObject* randang_(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* keywords[] = {"nside", "ipix", "nest", "lonlat", "rng", NULL};

    Py_ssize_t nside;
    PyObject* in[] = {NULL};
    int in_types[] = {NPY_INT64};
    int nest = false;
    int lonlat = false;
    PyObject* rng = Py_None;
    bitgen_t* bitgen;
    PyObject* out[] = {NULL, NULL};
    int out_types[] = {NPY_DOUBLE, NPY_DOUBLE};
    PyObject* ret;
    void* margs[] = {&nside, &nest, &lonlat, &bitgen};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nO|ppO:randang", keywords,
                                     &nside, &in[0], &nest, &lonlat, &rng)) {
        return NULL;
    }

    if (!check_nside(nside, nest))
        return NULL;

    if (rng == Py_None) {
        rng = default_rng();
        if (!rng)
            return NULL;
    } else {
        // incref here so that we can unconditionally decref later
        Py_INCREF(rng);
    }
    bitgen = bitgen_from_rng(rng);

    if (!map_array(randang_arr, margs, 1, in, in_types, 2, out, out_types))
        goto fail;

    ret = Py_BuildValue("NN", out[0], out[1]);
    if (!ret)
        goto fail;

    Py_DECREF(rng);

    return ret;

fail:
    Py_XDECREF(rng);
    Py_XDECREF(out[0]);
    Py_XDECREF(out[1]);
    return NULL;
}


// ----------------------------------------------------------------------------
// randvec


static void randvec_arr(t_args* args, t_arr* arr) {
    int64_t nside = args_val(int64_t, args, 0);
    int nest = args_val(int, args, 1);
    bitgen_t* bitgen = args_val(bitgen_t*, args, 2);
    double u[2];
    do {
        random_standard_uniform_fill(bitgen, 2, u);
        setvec(randvec(nside, arr_val(int64_t, arr, 0), u[0], u[1], nest),
               arr_ptr(double, arr, 1),
               arr_ptr(double, arr, 2),
               arr_ptr(double, arr, 3));
    } while (arrnext(arr));
}


PyDoc_STRVAR(randvec_doc, "\
randvec(nside, ipix, nest=False, rng=None)\n\
--\n\
\n\
Sample random unit vectors from the given HEALPix pixels.\n\
\n\
This function produces one random unit vector from each HEALPix pixel listed\n\
in `ipix`, which can be scalar or array-like.  The pixel indices use the \n\
`nside` resolution parameter and either the RING scheme (if `nest` is false,\n\
the default) or the NEST scheme (if `nest` is true).\n\
\n\
Returns a tuple `x, y, z` of normalised vector components with the same \n\
shape as the input.\n\
\n\
An optional numpy random number generator can be provided using `rng`;\n\
otherwise, a new numpy.random.default_rng() is used.\n\
");


static PyObject* randvec_(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* keywords[] = {"nside", "ipix", "nest", "rng", NULL};

    Py_ssize_t nside;
    PyObject* in[] = {NULL};
    int in_types[] = {NPY_INT64};
    int nest = false;
    PyObject* rng = Py_None;
    bitgen_t* bitgen;
    PyObject* out[] = {NULL, NULL, NULL};
    int out_types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
    PyObject* ret;
    void* margs[] = {&nside, &nest, &bitgen};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nO|pO:randvec", keywords,
                                     &nside, &in[0], &nest, &rng)) {
        return NULL;
    }

    if (!check_nside(nside, nest))
        return NULL;

    if (rng == Py_None) {
        rng = default_rng();
        if (!rng)
            return NULL;
    } else {
        // incref here so that we can unconditionally decref later
        Py_INCREF(rng);
    }
    bitgen = bitgen_from_rng(rng);

    if (!map_array(randvec_arr, margs, 1, in, in_types, 3, out, out_types))
        goto fail;

    ret = Py_BuildValue("NNN", out[0], out[1], out[2]);
    if (!ret)
        goto fail;

    Py_DECREF(rng);

    return ret;

fail:
    Py_XDECREF(rng);
    Py_XDECREF(out[0]);
    Py_XDECREF(out[1]);
    Py_XDECREF(out[2]);
    return NULL;
}


// ----------------------------------------------------------------------------
// Python module definition


static PyMethodDef methods[] = {
    {"ang2pix", (PyCFunction)ang2pix_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pix2ang", (PyCFunction)pix2ang_, METH_VARARGS | METH_KEYWORDS,
     pix2ang_doc},
    {"vec2pix", (PyCFunction)vec2pix_, METH_VARARGS | METH_KEYWORDS, NULL},
    {"pix2vec", (PyCFunction)pix2vec_, METH_VARARGS | METH_KEYWORDS,
     pix2vec_doc},
    {"nside2npix", nside2npix_, METH_VARARGS, NULL},
    {"npix2nside", npix2nside_, METH_VARARGS, NULL},
    {"randang", (PyCFunction)randang_, METH_VARARGS | METH_KEYWORDS,
     randang_doc},
    {"randvec", (PyCFunction)randvec_, METH_VARARGS | METH_KEYWORDS,
     randvec_doc},
    {NULL, NULL}
};


static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "chealpix",
    PyDoc_STR("healpix C library bindings"),
    -1,
    methods
};


PyMODINIT_FUNC PyInit_chealpix(void) {
    PyObject* module = PyModule_Create(&module_def);
    if (!module)
        return NULL;
    import_array();
    if (PyErr_Occurred())
        return NULL;
    return module;
}
