
/*******************************************************************************
 ** Vector operations written in C to improve performance
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "visclibs/utilities.h"
#include "visclibs/array_utils.h"


static PyObject* eliminatePBCFlicker(PyObject*, PyObject*);
static PyObject* separationVector(PyObject*, PyObject*);
static PyObject* separationMagnitude(PyObject*, PyObject*);
static PyObject* magnitude(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"eliminatePBCFlicker", eliminatePBCFlicker, METH_VARARGS, "Eliminate flicker across PBCs between two configurations"},
    {"separationVector", separationVector, METH_VARARGS, "Calculate separation vector between two vectors"},
    {"separationMagnitude", separationMagnitude, METH_VARARGS, "Calculate magnitude of separation between two vectors"},
    {"magnitude", magnitude, METH_VARARGS, "Calculate magnitude of a vector"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_vectors(void)
{
    (void)Py_InitModule("_vectors", methods);
    import_array();
}

/*******************************************************************************
 * eliminate pbc flicker
 *******************************************************************************/
static PyObject*
eliminatePBCFlicker(PyObject *self, PyObject *args)
{
    int NAtoms, *pbc;
    double *pos, *previousPos, *cellDims;
    PyArrayObject *posIn=NULL;
    PyArrayObject *previousPosIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *pbcIn=NULL;
    
    int i, j, count;
    double sep, absSep, halfDims[3];
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "iO!O!O!O!", &NAtoms, &PyArray_Type, &posIn, &PyArray_Type, &previousPosIn, 
            &PyArray_Type, &cellDimsIn, &PyArray_Type, &pbcIn))
        return NULL;
    
    if (not_intVector(pbcIn)) return NULL;
    pbc = pyvector_to_Cptr_int(pbcIn);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(previousPosIn)) return NULL;
    previousPos = pyvector_to_Cptr_double(previousPosIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    /* half of cellDims */
    halfDims[0] = cellDims[0] * 0.5;
    halfDims[1] = cellDims[1] * 0.5;
    halfDims[2] = cellDims[2] * 0.5;
    
    /* loop over atoms */
    count = 0;
    for (i = 0; i < NAtoms; i++)
    {
        sep = atomicSeparation2(pos[3*i], pos[3*i+1], pos[3*i+2], previousPos[3*i], previousPos[3*i+1], previousPos[3*i+2], 
                                cellDims[0], cellDims[1], cellDims[2], pbc[0], pbc[1], pbc[2]);
        
        for (j = 0; j < 3; j++)
        {
            if (sep < 1.0 && pos[3*i+j] < 1.0)
            {
                absSep = fabs(pos[3*i+j] - previousPos[3*i+j]);
                if (absSep > halfDims[j])
                {
                    pos[3*i+j] += cellDims[j];
                    count++;
                }
            }
            
            else if (sep < 1.0 && fabs(cellDims[j] - pos[3*i+j]) < 1.0)
            {
                absSep = fabs(pos[3*i+j] - previousPos[3*i+j]);
                if (absSep > halfDims[j])
                {
                    pos[3*i+j] -= cellDims[j];
                    count++;
                }
            }
        }
    }
    
    return Py_BuildValue("i", count);
}


/*******************************************************************************
 * Find separation vector between two pos vectors
 *******************************************************************************/
static PyObject*
separationVector(PyObject *self, PyObject *args)
{
    int length, *PBC;
    double *returnVector, *pos1, *pos2, *cellDims;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *returnVectorIn=NULL;
    PyArrayObject *pos1In=NULL;
    PyArrayObject *pos2In=NULL;
    PyArrayObject *cellDimsIn=NULL;
    int i;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyArray_Type, &returnVectorIn, &PyArray_Type, &pos1In, 
            &PyArray_Type, &pos2In, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn))
        return NULL;
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(pos1In)) return NULL;
    pos1 = pyvector_to_Cptr_double(pos1In);
    length = ((int) PyArray_DIM(pos1In, 0)) / 3;
    
    if (not_doubleVector(pos2In)) return NULL;
    pos2 = pyvector_to_Cptr_double(pos2In);
    
    if (not_doubleVector(returnVectorIn)) return NULL;
    returnVector = pyvector_to_Cptr_double(returnVectorIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    /* loop */
    for (i = 0; i < length; i++)
    {
        double atomSepVec[3];
        
        atomSeparationVector(atomSepVec, pos1[3*i], pos1[3*i+1], pos1[3*i+2], pos2[3*i], pos2[3*i+1], pos2[3*i+2], 
                             cellDims[0], cellDims[4], cellDims[8], PBC[0], PBC[1], PBC[2]);
        
        returnVector[3*i] = atomSepVec[0];
        returnVector[3*i+1] = atomSepVec[1];
        returnVector[3*i+2] = atomSepVec[2];
    }
    
    return Py_BuildValue("i", 0);
}


/*******************************************************************************
 * return magnitude of separation vector between two pos vectors
 *******************************************************************************/
static PyObject*
separationMagnitude(PyObject *self, PyObject *args)
{
    int length, *PBC;
    double *pos1, *pos2, *cellDims;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *pos1In=NULL;
    PyArrayObject *pos2In=NULL;
    PyArrayObject *cellDimsIn=NULL;
    
    int i;
    double sum;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &pos1In, &PyArray_Type, &pos2In, 
            &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn))
        return NULL;
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(pos1In)) return NULL;
    pos1 = pyvector_to_Cptr_double(pos1In);
    length = ((int) PyArray_DIM(pos1In, 0)) / 3;
    
    if (not_doubleVector(pos2In)) return NULL;
    pos2 = pyvector_to_Cptr_double(pos2In);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    /* loop */
    sum = 0;
    for (i = 0; i < length; i++ )
    {
        double r2;
        
        r2 = atomicSeparation2( pos1[3*i], pos1[3*i+1], pos1[3*i+2], pos2[3*i], pos2[3*i+1], pos2[3*i+2], cellDims[0], cellDims[4], cellDims[8], PBC[0], PBC[1], PBC[2]);
        
        sum += r2;
    }
    
    return Py_BuildValue("d", sqrt(sum));
}


/*******************************************************************************
 * Return the magnitude of given vector
 *******************************************************************************/
static PyObject*
magnitude(PyObject *self, PyObject *args)
{
    int length;
    double *pos;
    PyArrayObject *posIn=NULL;
    
    int i;
    double sum;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &posIn))
        return NULL;
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    length = (int) PyArray_DIM(posIn, 0);
    
    /* compute */
    sum = 0.0;
    for (i = 0; i < length; i++)
        sum += pos[i] * pos[i];
    
    return Py_BuildValue("d", sqrt(sum));
}
