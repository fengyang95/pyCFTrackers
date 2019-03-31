#include <Python.h>
#include <numpy/arrayobject.h>

#include "gradient.hpp"

/* Docstrings */
static char module_docstring[] = 
	"This module provides an interface for caculating gradMag using C.";

static char gradMag_docstring[] = 
	"Caculate the gradient mag of images.";
// static char gradMagNorm_docstring[] = 
//	"Caculate the magnitute fo the gradient.";
static char fhog_docstring[] =
	"Caculate the gradient hist.";
// static char convTri_docstring[] =
//	"Caculate the conv tri."

/* Available function */
static PyObject *_gradient_gradMag(PyObject *self, PyObject *args);
// static PyObject *_gradient_gradMagNorm(PyObject *self, PyObject *args);
static PyObject *_gradient_fhog(PyObject *self, PyObject *args);

/* Module specification */
static PyMethodDef module_methods[] = {
	{"gradMag", _gradient_gradMag, METH_VARARGS, gradMag_docstring},
	// {"gradMagNorm", _gradient_gradMagNorm, METH_VARARGS, gradMagNorm_docstring},
	{"fhog", _gradient_fhog, METH_VARARGS, fhog_docstring},
	// {"convTri", _gradient_convTri, METH_VARARGS, convTri_docstring}
	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT, 
	"_gradient",
	module_docstring,
	-1, 
	module_methods,
	NULL,
	NULL,
	NULL,
	NULL,
};


/* Initialize the module */
extern "C" PyObject* PyInit__gradient(void){
	PyObject *module = PyModule_Create(&moduledef);
	// PyErr_SetString(PyExc_RuntimeError, "Coundn't build output array.");
	if (module == NULL)
		return NULL;
	import_array();

	return module;
}

#else

// extern "C" PyMODINIT_FUNC init_gradient(void){
PyMODINIT_FUNC init_gradient(void){
	PyObject *m = Py_InitModule3("_gradient", module_methods, module_docstring);
	if (m == NULL)
		return;
	import_array();
}
#endif

static PyObject *_gradient_gradMag(PyObject *self, PyObject *args){
	
	int c;
	bool full;
	int full_;
	PyObject *I_obj=NULL;

	if(!PyArg_ParseTuple(args, "Oii", &I_obj, &c, &full_))
		return NULL;

	full = (bool)full_;
	
	PyObject *I_array = PyArray_FROM_OTF(I_obj, NPY_FLOAT, NPY_ARRAY_F_CONTIGUOUS);
	
	if (I_array == NULL){
		PyErr_SetString(PyExc_RuntimeError, "Coundn't get array.");
		Py_XDECREF(I_array);
		return NULL;
	}

	int ndim = PyArray_NDIM(I_array);
	int h = (int)PyArray_DIM(I_array, 0);
	int w = (int)PyArray_DIM(I_array, 1);
	int d = 1;
	if (ndim == 3)
		d = (int)PyArray_DIM(I_array, 2);

	int nd = 2;
	npy_intp dims[] = {h, w};
	PyObject *M_array = PyArray_ZEROS(nd, dims, NPY_FLOAT, 1);
	PyObject *O_array = PyArray_ZEROS(nd, dims, NPY_FLOAT, 1);
	
	if (M_array == NULL || O_array == NULL){
		PyErr_SetString(PyExc_RuntimeError, "Coundn't build output array.");
		Py_DECREF(I_array);
		Py_XDECREF(M_array);
		Py_XDECREF(O_array);
		return NULL;
	}
	
	float *I_data = (float*)PyArray_DATA(I_array);
	float *M_data = (float*)PyArray_DATA(M_array);
	float *O_data = (float*)PyArray_DATA(O_array);
	if (c > 0 && c <= d){
		I_data += h * w * c;
		d = 1;
	}

	gradMag(I_data, M_data, O_data, h, w, d, full);
	
	Py_DECREF(I_array);
	
	PyObject *rlst = PyTuple_New(2);
	PyTuple_SetItem(rlst, 0, M_array);
	PyTuple_SetItem(rlst, 1, O_array);
	return rlst;
}

static PyObject *_gradient_fhog(PyObject *self, PyObject *args){
	PyObject *M_obj=NULL, *O_obj;
	int binSize, nOrients, softBin;
	float clip;

	if(!PyArg_ParseTuple(args, "OOiiif", &M_obj, &O_obj, 
				&binSize, &nOrients, &softBin, &clip))
		return NULL;
	
	PyObject *M_array = PyArray_FROM_OTF(M_obj, NPY_FLOAT, NPY_ARRAY_F_CONTIGUOUS);
	PyObject *O_array = PyArray_FROM_OTF(O_obj, NPY_FLOAT, NPY_ARRAY_F_CONTIGUOUS);
	
	if (M_array == NULL || O_array == NULL){
		PyErr_SetString(PyExc_RuntimeError, "Coundn't get array.");
		Py_XDECREF(M_array);
		Py_XDECREF(O_array);
		return NULL;
	}

	int h = (int)PyArray_DIM(M_array, 0);
	int w = (int)PyArray_DIM(M_array, 1);

	int nd = 3;
	npy_intp dims[] = {h / binSize, w / binSize, nOrients * 3 + 5};
	PyObject *H_array = PyArray_ZEROS(nd, dims, NPY_FLOAT, 1);
	
	if (H_array == NULL){
		PyErr_SetString(PyExc_RuntimeError, "Coundn't build output array.");
		Py_DECREF(M_array);
		Py_DECREF(O_array);
		Py_XDECREF(H_array);
		return NULL;
	}
	
	float *M_data = (float*)PyArray_DATA(M_array);
	float *O_data = (float*)PyArray_DATA(O_array);
	float *H_data = (float*)PyArray_DATA(H_array);
	
	fhog(M_data, O_data, H_data, h, w, binSize, nOrients, softBin, clip);

	Py_DECREF(M_array);
	Py_DECREF(O_array);
	
	return H_array;
}
