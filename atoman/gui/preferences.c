
/*******************************************************************************
 ** Set parameters/preferences for use in C libraries
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include "preferences.h"

static PyObject* setNumThreads(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"setNumThreads", setNumThreads, METH_VARARGS, "Set the number of OpenMP threads to use."},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_preferences(void)
{
    (void)Py_InitModule("_preferences", methods);
}

/*******************************************************************************
 ** Set the number of OpenMP threads
 *******************************************************************************/
static PyObject*
setNumThreads(PyObject *self, PyObject *args)
{
    int numThreads;
    
    
    /* parse arguments from Python */
    if (!PyArg_ParseTuple(args, "i", &numThreads))
        return NULL;
    
    /* set the number of threads */
    prefs_numThreads = numThreads;
    
    /* return None on success */
    Py_RETURN_NONE;
}
