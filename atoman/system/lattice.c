
/*******************************************************************************
 ** Lattice functions in C to improve performance
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "array_utils.h"

static PyObject* wrapAtoms(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"wrapAtoms", wrapAtoms, METH_VARARGS, "Wrap atoms that have left the periodic cell"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_lattice(void)
{
    (void)Py_InitModule("_lattice", methods);
    import_array();
}

static PyObject*
wrapAtoms(PyObject *self, PyObject *args)
{
    int numAtoms;
    PyObject *result = NULL;
    PyArrayObject *pos = NULL;
    PyArrayObject *cellDims = NULL;
    PyArrayObject *pbc = NULL;
    
    /* parse and check arguments from Python */
    if (PyArg_ParseTuple(args, "iO!O!O!", &numAtoms, &PyArray_Type, &pos, &PyArray_Type, &cellDims, &PyArray_Type, &pbc))
    {
        int i;
        
        #pragma omp parallel for
        for (i = 0; i < numAtoms; i++)
        {
            int i3 = 3 * i;
            
            if (IIND1(pbc, 0)) DIND1(pos, i3    ) = DIND1(pos, i3    ) - floor(DIND1(pos, i3    ) / DIND1(cellDims, 0)) * DIND1(cellDims, 0);
            if (IIND1(pbc, 1)) DIND1(pos, i3 + 1) = DIND1(pos, i3 + 1) - floor(DIND1(pos, i3 + 1) / DIND1(cellDims, 1)) * DIND1(cellDims, 1);
            if (IIND1(pbc, 2)) DIND1(pos, i3 + 2) = DIND1(pos, i3 + 2) - floor(DIND1(pos, i3 + 2) / DIND1(cellDims, 2)) * DIND1(cellDims, 2);
        }
        
        Py_INCREF(Py_None);
        result = Py_None;
    }
    
    return result;
}
