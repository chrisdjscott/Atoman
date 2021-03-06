
/*******************************************************************************
 ** Tests for boxeslib.c
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "visclibs/boxeslib.h"
#include "visclibs/array_utils.h"

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

static PyObject* test_boxes(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef module_methods[] = {
    {"test_boxes", test_boxes, METH_VARARGS, "The boxes functionality"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
MOD_INIT(_test_boxeslib)
{
    PyObject *mod;
    
    MOD_DEF(mod, "_test_boxeslib", "Boxeslib tests", module_methods)
    if (mod == NULL)
        return MOD_ERROR_VAL;
    
    import_array();
    
    return MOD_SUCCESS_VAL(mod);
}

/*******************************************************************************
 ** Test boxes
 *******************************************************************************/
static PyObject*
test_boxes(PyObject *self, PyObject *args)
{
    double approxWidth;
    PyObject *result=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *pbcIn=NULL;
    PyArrayObject *numBoxes=NULL;
    PyArrayObject *cellLengths=NULL;
    
    
    /* parse and check arguments from Python */
    if (PyArg_ParseTuple(args, "O!O!O!dO!O!", &PyArray_Type, &posIn, &PyArray_Type, &cellDimsIn, &PyArray_Type, &pbcIn,
            &approxWidth, &PyArray_Type, &numBoxes, &PyArray_Type, &cellLengths))
    {
        int i, numAtoms, *pbc;
        double *pos, *cellDims;
        struct Boxes *boxes;
        
        /* convert numpy arrays to C pointers */
        if (not_doubleVector(posIn)) return NULL;
        pos = pyvector_to_Cptr_double(posIn);
        numAtoms = (int) PyArray_DIM(posIn, 0);
        
        if (not_doubleVector(cellDimsIn)) return NULL;
        cellDims = pyvector_to_Cptr_double(cellDimsIn);
        
        if (not_intVector(pbcIn)) return NULL;
        pbc = pyvector_to_Cptr_int(pbcIn);
        
        /* begin testing... */
        
        /* setup boxes */
        boxes = setupBoxes(approxWidth, pbc, cellDims);
        if (boxes == NULL) return Py_BuildValue("i", 1);
        
        /* store box values for checking in Python */
        for (i = 0; i < 3; i++)
        {
            IIND1(numBoxes, i) = boxes->NBoxes[i];
            DIND1(cellLengths, i) = boxes->boxWidth[i];
        }
        
        
        
        
        /* free boxes memory */
        freeBoxes(boxes);
        
        /* build success result */
        result = Py_BuildValue("i", 0);
    }
    
    return result;
}
