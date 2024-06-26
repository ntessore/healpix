// author: Nicolas Tessore <n.tessore@ucl.ac.uk>
// license: BSD-3-Clause

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#include "healpix.h"

// maximum number of operands for vectorized operations
#ifndef VEC_MAXOP
#define VEC_MAXOP 6
#endif


static void setang(t_ang ang, double* theta, double* phi) {
    *theta = ang.theta;
    *phi = ang.phi;
}


static void setvec(t_vec vec, double* x, double* y, double* z) {
    *x = vec.x;
    *y = vec.y;
    *z = vec.z;
}


static void setpix(t_pix pix, npy_int8 *order, npy_int64 *ipix) {
    *order = pix.order;
    *ipix = pix.ipix;
}


typedef void (*vecfunc)(void*, npy_intp, char **, npy_intp *);


PyObject* vectorize(vecfunc func, void* args, npy_intp nin, npy_intp nout,
                    PyObject** op, int* types) {
    npy_intp i, nop;
    NpyIter* iter;
    PyArrayObject* op_[VEC_MAXOP] = {0};
    npy_uint32 flags;
    npy_uint32 op_flags[VEC_MAXOP];
    PyArray_Descr* op_dtypes[VEC_MAXOP] = {0};
    PyArrayObject** out;
    PyObject* ret = NULL;

    nop = nin + nout;

    if (nop > VEC_MAXOP) {
        PyErr_Format(PyExc_RuntimeError,
                     "chealpix internal error: increase VEC_MAXOP to %llu",
                     nop);
        return NULL;
    }

    for (i = 0; i < nin; ++i) {
        op_[i] = (PyArrayObject*)PyArray_FromAny(op[i], NULL, 0, 0, 0, NULL);
        op_flags[i] = NPY_ITER_READONLY | NPY_ITER_NBO;
        op_dtypes[i] = PyArray_DescrFromType(types[i]);
        if (!op_[i] || !op_dtypes[i])
            goto fail;
    }

    for (; i < nop; ++i) {
        if (op[i] && op[i] != Py_None) {
            op_[i] = (PyArrayObject*)PyArray_FromAny(op[i], NULL, 0, 0, 0,
                                                     NULL);
            if (!op_[i])
                goto fail;
        }
        op_flags[i] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_NBO;
        op_dtypes[i] = PyArray_DescrFromType(types[i]);
        if (!op_dtypes[i])
            goto fail;
    }

    flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED | NPY_ITER_GROWINNER |
            NPY_ITER_ZEROSIZE_OK;

    iter = NpyIter_MultiNew(nop, op_, flags, NPY_KEEPORDER, NPY_SAME_KIND_CASTING,
                            op_flags, op_dtypes);
    if (iter == NULL)
        goto fail;

    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        npy_intp *sizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        npy_intp *stride = NpyIter_GetInnerStrideArray(iter);
        char **data = NpyIter_GetDataPtrArray(iter);
        NPY_BEGIN_THREADS_DEF
        if (!iternext)
            goto iterfail;
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter))
        do {
            func(args, *sizeptr, data, stride);
        } while (iternext(iter));
        NPY_END_THREADS
    }

    out = &NpyIter_GetOperandArray(iter)[nin];
    if (nout == 0) {
        ret = Py_None;
        Py_INCREF(ret);
    } else if (nout == 1) {
        Py_INCREF(out[0]);
        ret = PyArray_Return(out[0]);
    } else {
        ret = PyTuple_New(nout);
        if (!ret)
            goto iterfail;
        for (i = 0; i < nout; ++i) {
            Py_INCREF(out[i]);
            PyTuple_SET_ITEM(ret, i, PyArray_Return(out[i]));
        }
    }

    if (NpyIter_Deallocate(iter) != NPY_SUCCEED)
        goto fail;

    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op_[i]);
        Py_DECREF(op_dtypes[i]);
    }

    return ret;

iterfail:
    NpyIter_Deallocate(iter);

fail:
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op_[i]);
        Py_XDECREF(op_dtypes[i]);
    }
    Py_XDECREF(ret);
    return NULL;
}


static void vang2vec(void *args, npy_intp size, char **data, npy_intp *stride) {
    while (size--) {
        setvec(
            ang2vec((t_ang){
                *(double *)data[0],
                *(double *)data[1],
            }),
            (double *)data[2],
            (double *)data[3],
            (double *)data[4]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
    }
}


PyDoc_STRVAR(cang2vec_doc,
"ang2vec(nside, theta, phi, x=None, y=None, z=None, /)\n"
"--\n"
"\n");


static PyObject* cang2vec(PyObject* self, PyObject* args) {
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "OO|OOO:ang2vec",
                          &op[0], &op[1], &op[2], &op[3], &op[4])) {
        return NULL;
    }

    return vectorize(vang2vec, NULL, 2, 3, op, types);
}


static void vvec2ang(void *args, npy_intp size, char **data, npy_intp *stride) {
    while (size--) {
        setang(
            vec2ang((t_vec){
                *(double *)data[0],
                *(double *)data[1],
                *(double *)data[2],
            }),
            (double *)data[3],
            (double *)data[4]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
    }
}


PyDoc_STRVAR(cvec2ang_doc,
"vec2ang(nside, x, y, z, theta=None, phi=None, /)\n"
"--\n"
"\n");


static PyObject* cvec2ang(PyObject* self, PyObject* args) {
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "OOO|OO:vec2ang",
                          &op[0], &op[1], &op[2], &op[3], &op[4])) {
        return NULL;
    }

    return vectorize(vvec2ang, NULL, 3, 2, op, types);
}


static void vang2nest(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[2] = ang2nest(
            nside,
            (t_ang){
                *(double *)data[0],
                *(double *)data[1],
            }
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
    }
}


PyDoc_STRVAR(cang2nest_doc,
"ang2nest(nside, theta, phi, ipix=None, /)\n"
"--\n"
"\n");


static PyObject* cang2nest(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_INT64};

    if (!PyArg_ParseTuple(args, "nOO|O:ang2nest", &nside,
                          &op[0], &op[1], &op[2])) {
        return NULL;
    }

    return vectorize(vang2nest, &nside, 2, 1, op, types);
}


static void vang2ring(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[2] = ang2ring(
            nside,
            (t_ang){
                *(double *)data[0],
                *(double *)data[1],
            }
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
    }
}


PyDoc_STRVAR(cang2ring_doc,
"ang2ring(nside, theta, phi, ipix=None, /)\n"
"--\n"
"\n");


static PyObject* cang2ring(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_INT64};

    if (!PyArg_ParseTuple(args, "nOO|O:ang2ring", &nside,
                          &op[0], &op[1], &op[2])) {
        return NULL;
    }

    return vectorize(vang2ring, &nside, 2, 1, op, types);
}


static void vnest2ang(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        setang(
            nest2ang(nside, *(npy_int64 *)data[0]),
            (double *)data[1],
            (double *)data[2]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
    }
}


PyDoc_STRVAR(cnest2ang_doc,
"nest2ang(nside, ipix, theta=None, phi=None, /)\n"
"--\n"
"\n");


static PyObject* cnest2ang(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nO|OO:nest2ang", &nside,
                          &op[0], &op[1], &op[2])) {
        return NULL;
    }

    return vectorize(vnest2ang, &nside, 1, 2, op, types);
}


static void vring2ang(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        setang(
            ring2ang(nside, *(npy_int64 *)data[0]),
            (double *)data[1],
            (double *)data[2]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
    }
}


PyDoc_STRVAR(cring2ang_doc,
"ring2ang(nside, ipix, theta=None, phi=None, /)\n"
"--\n"
"\n");


static PyObject* cring2ang(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nO|OO:ring2ang", &nside,
                          &op[0], &op[1], &op[2]))
        return NULL;

    return vectorize(vring2ang, &nside, 1, 2, op, types);
}


static void vvec2nest(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[3] = vec2nest(
            nside,
            (t_vec){
                *(double *)data[0],
                *(double *)data[1],
                *(double *)data[2],
            }
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
    }
}


PyDoc_STRVAR(cvec2nest_doc,
"vec2nest(nside, x, y, z, ipix=None, /)\n"
"--\n"
"\n");


static PyObject* cvec2nest(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_INT64};

    if (!PyArg_ParseTuple(args, "nOOO|O:vec2nest", &nside,
                          &op[0], &op[1], &op[2], &op[3])) {
        return NULL;
    }

    return vectorize(vvec2nest, &nside, 3, 1, op, types);
}


static void vvec2ring(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[3] = vec2ring(
            nside,
            (t_vec){
                *(double *)data[0],
                *(double *)data[1],
                *(double *)data[2],
            }
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
    }
}


PyDoc_STRVAR(cvec2ring_doc,
"vec2ring(nside, x, y, z, ipix=None, /)\n"
"--\n"
"\n");


static PyObject* cvec2ring(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_INT64};

    if (!PyArg_ParseTuple(args, "nOOO|O:vec2ring", &nside,
                          &op[0], &op[1], &op[2], &op[3])) {
        return NULL;
    }

    return vectorize(vvec2ring, &nside, 3, 1, op, types);
}


static void vnest2vec(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        setvec(
            nest2vec(nside, *(npy_int64 *)data[0]),
            (double *)data[1],
            (double *)data[2],
            (double *)data[3]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
    }
}


PyDoc_STRVAR(cnest2vec_doc,
"nest2vec(nside, ipix, x=None, y=None, z=None, /)\n"
"--\n"
"\n");


static PyObject* cnest2vec(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nO|OOO:nest2vec", &nside,
                          &op[0], &op[1], &op[2], &op[3])) {
        return NULL;
    }

    return vectorize(vnest2vec, &nside, 1, 3, op, types);
}


static void vring2vec(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        setvec(
            ring2vec(nside, *(npy_int64 *)data[0]),
            (double *)data[1],
            (double *)data[2],
            (double *)data[3]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
    }
}


PyDoc_STRVAR(cring2vec_doc,
"ring2vec(nside, ipix, x=None, y=None, z=None, /)\n"
"--\n"
"\n");


static PyObject* cring2vec(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nO|OOO:ring2vec", &nside,
                          &op[0], &op[1], &op[2], &op[3])) {
        return NULL;
    }

    return vectorize(vring2vec, &nside, 1, 3, op, types);
}


static void vang2nest_uv(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[2] = ang2nest_uv(
            nside,
            (t_ang){
                *(double *)data[0],
                *(double *)data[1],
            },
            (double *)data[3],
            (double *)data[4]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
    }
}


PyDoc_STRVAR(cang2nest_uv_doc,
"ang2nest_uv(nside, theta, phi, ipix=None, u=None, v=None, /)\n"
"--\n"
"\n");


static PyObject* cang2nest_uv(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_INT64, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nOO|OOO:ang2nest_uv", &nside,
                          &op[0], &op[1], &op[2], &op[3], &op[4])) {
        return NULL;
    }

    return vectorize(vang2nest_uv, &nside, 2, 3, op, types);
}


static void vang2ring_uv(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[2] = ang2ring_uv(
            nside,
            (t_ang){
                *(double *)data[0],
                *(double *)data[1],
            },
            (double *)data[3],
            (double *)data[4]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
    }
}


PyDoc_STRVAR(cang2ring_uv_doc,
"ang2ring_uv(nside, theta, phi, ipix=None, u=None, v=None, /)\n"
"--\n"
"\n");


static PyObject* cang2ring_uv(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_INT64, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nOO|OOO:ang2ring_uv", &nside,
                          &op[0], &op[1], &op[2], &op[3], &op[4])) {
        return NULL;
    }

    return vectorize(vang2ring_uv, &nside, 2, 3, op, types);
}


static void vnest2ang_uv(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        setang(
            nest2ang_uv(
                nside,
                *(npy_int64 *)data[0],
                *(double *)data[1],
                *(double *)data[2]
            ),
            (double *)data[3],
            (double *)data[4]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
    }
}


PyDoc_STRVAR(cnest2ang_uv_doc,
"nest2ang_uv(nside, ipix, u, v, theta=None, phi=None, /)\n"
"--\n"
"\n");


static PyObject* cnest2ang_uv(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nOOO|OO:nest2ang_uv", &nside,
                          &op[0], &op[1], &op[2], &op[3], &op[4])) {
        return NULL;
    }

    return vectorize(vnest2ang_uv, &nside, 3, 2, op, types);
}


static void vring2ang_uv(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        setang(
            ring2ang_uv(
                nside,
                *(npy_int64 *)data[0],
                *(double *)data[1],
                *(double *)data[2]
            ),
            (double *)data[3],
            (double *)data[4]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
    }
}


PyDoc_STRVAR(cring2ang_uv_doc,
"ring2ang_uv(nside, ipix, u, v, theta=None, phi=None, /)\n"
"--\n"
"\n");


static PyObject* cring2ang_uv(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nOOO|OO:ring2ang_uv", &nside,
                          &op[0], &op[1], &op[2], &op[3], &op[4]))
        return NULL;

    return vectorize(vring2ang_uv, &nside, 3, 2, op, types);
}


static void vvec2nest_uv(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[3] = vec2nest_uv(
            nside,
            (t_vec){
                *(double *)data[0],
                *(double *)data[1],
                *(double *)data[2],
            },
            (double *)data[4],
            (double *)data[5]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
        data[5] += stride[5];
    }
}


PyDoc_STRVAR(cvec2nest_uv_doc,
"vec2nest_uv(nside, x, y, z, ipix=None, u=None, v=None, /)\n"
"--\n"
"\n");


static PyObject* cvec2nest_uv(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_INT64, NPY_DOUBLE,
                   NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nOOO|OOO:vec2nest_uv", &nside,
                          &op[0], &op[1], &op[2], &op[3], &op[4], &op[5])) {
        return NULL;
    }

    return vectorize(vvec2nest_uv, &nside, 3, 3, op, types);
}


static void vvec2ring_uv(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[3] = vec2ring_uv(
            nside,
            (t_vec){
                *(double *)data[0],
                *(double *)data[1],
                *(double *)data[2],
            },
            (double *)data[4],
            (double *)data[5]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
        data[5] += stride[5];
    }
}


PyDoc_STRVAR(cvec2ring_uv_doc,
"vec2ring_uv(nside, x, y, z, ipix=None, u=None, v=None, /)\n"
"--\n"
"\n");


static PyObject* cvec2ring_uv(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_INT64, NPY_DOUBLE,
                   NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nOOO|OOO:vec2ring_uv", &nside,
                          &op[0], &op[1], &op[2], &op[3], &op[4], &op[5])) {
        return NULL;
    }

    return vectorize(vvec2ring_uv, &nside, 3, 3, op, types);
}


static void vnest2vec_uv(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        setvec(
            nest2vec_uv(
                nside,
                *(npy_int64 *)data[0],
                *(double *)data[1], 
                *(double *)data[2]
            ),
            (double *)data[3],
            (double *)data[4],
            (double *)data[5]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
        data[5] += stride[5];
    }
}


PyDoc_STRVAR(cnest2vec_uv_doc,
"nest2vec_uv(nside, ipix, u, v, x=None, y=None, z=None, /)\n"
"--\n"
"\n");


static PyObject* cnest2vec_uv(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                   NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nOOO|OOO:nest2vec_uv", &nside,
                          &op[0], &op[1], &op[2], &op[3], &op[4], &op[5])) {
        return NULL;
    }

    return vectorize(vnest2vec_uv, &nside, 3, 3, op, types);
}


static void vring2vec_uv(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        setvec(
            ring2vec_uv(
                nside,
                *(npy_int64 *)data[0],
                *(double *)data[1], 
                *(double *)data[2]
            ),
            (double *)data[3],
            (double *)data[4],
            (double *)data[5]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
        data[3] += stride[3];
        data[4] += stride[4];
        data[5] += stride[5];
    }
}


PyDoc_STRVAR(cring2vec_uv_doc,
"ring2vec_uv(nside, ipix, u, v, x=None, y=None, z=None, /)\n"
"--\n"
"\n");


static PyObject* cring2vec_uv(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL, NULL, NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                   NPY_DOUBLE};

    if (!PyArg_ParseTuple(args, "nOOO|OOO:ring2vec_uv", &nside,
                          &op[0], &op[1], &op[2], &op[3], &op[4], &op[5])) {
        return NULL;
    }

    return vectorize(vring2vec_uv, &nside, 3, 3, op, types);
}


static void vring2nest(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[1] = ring2nest(nside, *(npy_int64 *)data[0]);
        data[0] += stride[0];
        data[1] += stride[1];
    }
}


PyDoc_STRVAR(cring2nest_doc,
"ring2nest(nside, ipring, ipnest=None, /)\n"
"--\n"
"\n");


static PyObject* cring2nest(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL};
    int types[] = {NPY_INT64, NPY_INT64};

    if (!PyArg_ParseTuple(args, "nO|O:ring2nest", &nside, &op[0], &op[1]))
        return NULL;

    return vectorize(vring2nest, &nside, 1, 1, op, types);
}


static void vnest2ring(void *args, npy_intp size, char **data, npy_intp *stride) {
    Py_ssize_t nside = *(Py_ssize_t *)args;
    while (size--) {
        *(npy_int64 *)data[1] = nest2ring(nside, *(npy_int64 *)data[0]);
        data[0] += stride[0];
        data[1] += stride[1];
    }
}


PyDoc_STRVAR(cnest2ring_doc,
"nest2ring(nside, ipnest, ipring=None, /)\n"
"--\n"
"\n");


static PyObject* cnest2ring(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    PyObject* op[] = {NULL, NULL};
    int types[] = {NPY_INT64, NPY_INT64};

    if (!PyArg_ParseTuple(args, "nO|O:nest2ring", &nside, &op[0], &op[1]))
        return NULL;

    return vectorize(vnest2ring, &nside, 1, 1, op, types);
}


PyDoc_STRVAR(cnside2npix_doc, "nside2npix(nside, /)\n--\n\n");


static PyObject* cnside2npix(PyObject* self, PyObject* args) {
    Py_ssize_t nside, npix;

    if (!PyArg_ParseTuple(args, "n:nside2npix", &nside))
        return NULL;

    npix = nside2npix(nside);

    return Py_BuildValue("n", npix);
}


PyDoc_STRVAR(cnpix2nside_doc, "npix2nside(npix, /)\n--\n\n");


static PyObject* cnpix2nside(PyObject* self, PyObject* args) {
    Py_ssize_t npix, nside;

    if (!PyArg_ParseTuple(args, "n:npix2nside", &npix))
        return NULL;

    nside = npix2nside(npix);

    return Py_BuildValue("n", nside);
}


PyDoc_STRVAR(cnside2order_doc, "nside2order(nside, /)\n--\n\n");


static PyObject* cnside2order(PyObject* self, PyObject* args) {
    Py_ssize_t nside;
    int order;

    if (!PyArg_ParseTuple(args, "n:nside2order", &nside))
        return NULL;

    order = nside2order(nside);

    return Py_BuildValue("i", order);
}


PyDoc_STRVAR(corder2nside_doc, "order2nside(order, /)\n--\n\n");


static PyObject* corder2nside(PyObject* self, PyObject* args) {
    int order;
    Py_ssize_t nside;

    if (!PyArg_ParseTuple(args, "i:order2nside", &order))
        return NULL;

    nside = order2nside(order);

    return Py_BuildValue("n", nside);
}


static void vuniq2nest(void *args, npy_intp size, char **data, npy_intp *stride) {
    while (size--) {
        setpix(
            uniq2nest(*(npy_int64 *)data[0]),
            (npy_int8 *)data[1],
            (npy_int64 *)data[2]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
    }
}


PyDoc_STRVAR(cuniq2nest_doc,
"uniq2nest(uniq, order=None, ipix=None, /)\n"
"--\n"
"\n");


static PyObject* cuniq2nest(PyObject* self, PyObject* args) {
    PyObject* op[] = {NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_INT8, NPY_INT64};

    if (!PyArg_ParseTuple(args, "O|OO:uniq2nest", &op[0], &op[1], &op[2]))
        return NULL;

    return vectorize(vuniq2nest, NULL, 1, 2, op, types);
}


static void vuniq2ring(void *args, npy_intp size, char **data, npy_intp *stride) {
    while (size--) {
        setpix(
            uniq2ring(*(npy_int64 *)data[0]),
            (npy_int8 *)data[1],
            (npy_int64 *)data[2]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
    }
}


PyDoc_STRVAR(cuniq2ring_doc,
"uniq2ring(uniq, order=None, ipix=None, /)\n"
"--\n"
"\n");


static PyObject* cuniq2ring(PyObject* self, PyObject* args) {
    PyObject* op[] = {NULL, NULL, NULL};
    int types[] = {NPY_INT64, NPY_INT8, NPY_INT64};

    if (!PyArg_ParseTuple(args, "O|OO:uniq2ring", &op[0], &op[1], &op[2]))
        return NULL;

    return vectorize(vuniq2ring, NULL, 1, 2, op, types);
}


static void vnest2uniq(void *args, npy_intp size, char **data, npy_intp *stride) {
    while (size--) {
        *(npy_int64 *)data[2] = nest2uniq(
            *(npy_int8 *)data[0],
            *(npy_int64 *)data[1]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
    }
}


PyDoc_STRVAR(cnest2uniq_doc,
"nest2uniq(order, ipix, uniq=None, /)\n"
"--\n"
"\n");


static PyObject* cnest2uniq(PyObject* self, PyObject* args) {
    PyObject* op[] = {NULL, NULL, NULL};
    int types[] = {NPY_INT8, NPY_INT64, NPY_INT64};

    if (!PyArg_ParseTuple(args, "OO|O:nest2uniq", &op[0], &op[1], &op[2]))
        return NULL;

    return vectorize(vnest2uniq, NULL, 2, 1, op, types);
}


static void vring2uniq(void *args, npy_intp size, char **data, npy_intp *stride) {
    while (size--) {
        *(npy_int64 *)data[2] = ring2uniq(
            *(npy_int8 *)data[0],
            *(npy_int64 *)data[1]
        );
        data[0] += stride[0];
        data[1] += stride[1];
        data[2] += stride[2];
    }
}


PyDoc_STRVAR(cring2uniq_doc,
"ring2uniq(order, ipix, uniq=None, /)\n"
"--\n"
"\n");


static PyObject* cring2uniq(PyObject* self, PyObject* args) {
    PyObject* op[] = {NULL, NULL, NULL};
    int types[] = {NPY_INT8, NPY_INT64, NPY_INT64};

    if (!PyArg_ParseTuple(args, "OO|O:ring2uniq", &op[0], &op[1], &op[2]))
        return NULL;

    return vectorize(vring2uniq, NULL, 2, 1, op, types);
}


static const char* version = "2024.1";


static PyMethodDef methods[] = {
    {"ang2vec", cang2vec, METH_VARARGS, cang2vec_doc},
    {"vec2ang", cvec2ang, METH_VARARGS, cvec2ang_doc},
    {"ang2nest", cang2nest, METH_VARARGS, cang2nest_doc},
    {"ang2ring", cang2ring, METH_VARARGS, cang2ring_doc},
    {"nest2ang", cnest2ang, METH_VARARGS, cnest2ang_doc},
    {"ring2ang", cring2ang, METH_VARARGS, cring2ang_doc},
    {"vec2nest", cvec2nest, METH_VARARGS, cvec2nest_doc},
    {"vec2ring", cvec2ring, METH_VARARGS, cvec2ring_doc},
    {"nest2vec", cnest2vec, METH_VARARGS, cnest2vec_doc},
    {"ring2vec", cring2vec, METH_VARARGS, cring2vec_doc},
    {"ang2nest_uv", cang2nest_uv, METH_VARARGS, cang2nest_uv_doc},
    {"ang2ring_uv", cang2ring_uv, METH_VARARGS, cang2ring_uv_doc},
    {"nest2ang_uv", cnest2ang_uv, METH_VARARGS, cnest2ang_uv_doc},
    {"ring2ang_uv", cring2ang_uv, METH_VARARGS, cring2ang_uv_doc},
    {"vec2nest_uv", cvec2nest_uv, METH_VARARGS, cvec2nest_uv_doc},
    {"vec2ring_uv", cvec2ring_uv, METH_VARARGS, cvec2ring_uv_doc},
    {"nest2vec_uv", cnest2vec_uv, METH_VARARGS, cnest2vec_uv_doc},
    {"ring2vec_uv", cring2vec_uv, METH_VARARGS, cring2vec_uv_doc},
    {"ring2nest", cring2nest, METH_VARARGS, cring2nest_doc},
    {"nest2ring", cnest2ring, METH_VARARGS, cnest2ring_doc},
    {"nside2npix", cnside2npix, METH_VARARGS, cnside2npix_doc},
    {"npix2nside", cnpix2nside, METH_VARARGS, cnpix2nside_doc},
    {"nside2order", cnside2order, METH_VARARGS, cnside2order_doc},
    {"order2nside", corder2nside, METH_VARARGS, corder2nside_doc},
    {"uniq2nest", cuniq2nest, METH_VARARGS, cuniq2nest_doc},
    {"uniq2ring", cuniq2ring, METH_VARARGS, cuniq2ring_doc},
    {"nest2uniq", cnest2uniq, METH_VARARGS, cnest2uniq_doc},
    {"ring2uniq", cring2uniq, METH_VARARGS, cring2uniq_doc},
    {NULL, NULL}
};


static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "chealpix",
    PyDoc_STR("healpix C library interface"),
    -1,
    methods
};


PyMODINIT_FUNC PyInit_chealpix(void) {
    PyObject* module = PyModule_Create(&module_def);
    if (!module)
        return NULL;
    PyModule_AddStringConstant(module, "__version__", version);
    PyModule_AddIntConstant(module, "NSIDE_MAX", NSIDE_MAX);
    import_array();
    if (PyErr_Occurred())
        return NULL;
    return module;
}
