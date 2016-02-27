
/*******************************************************************************
 ** Set parameters/preferences for use in C libraries
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include "preferences.h"

#if PY_MAJOR_VERSION >= 3
    #define MOD_ERROR_VAL NULL
    #define MOD_SUCCESS_VAL(val) val
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
        static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);
#else
    #define MOD_ERROR_VAL
    #define MOD_SUCCESS_VAL(val)
    #define MOD_INIT(name) void init##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
        ob = Py_InitModule3(name, methods, doc);
#endif

static PyObject* setNumThreads(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef module_methods[] = {
    {"setNumThreads", setNumThreads, METH_VARARGS, "Set the number of OpenMP threads to use."},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
MOD_INIT(_preferences)
{
    PyObject *mod;
    
    MOD_DEF(mod, "_preferences", "Preferences for C extensions", module_methods)
    if (mod == NULL)
        return MOD_ERROR_VAL;
    
    return MOD_SUCCESS_VAL(mod);
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
