
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** Rendering routines written in C to improve performance
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include "array_utils.h"


static PyObject* splitVisAtomsBySpecie(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"splitVisAtomsBySpecie", splitVisAtomsBySpecie, METH_VARARGS, "Create specie pos and scalar arrays"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_rendering(void)
{
    (void)Py_InitModule("_rendering", methods);
    import_array();
}

/*******************************************************************************
 ** Split visible atoms by specie (position and scalar)
 *******************************************************************************/
static PyObject*
splitVisAtomsBySpecie(PyObject *self, PyObject *args)
{
    int NVisible, *visibleAtoms, NSpecies, *specieArray, *specieCount, scalarType, heightAxis;
    double *pos, *PE, *KE, *charge, *scalars;  
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *specieArrayIn=NULL;
    PyArrayObject *specieCountIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *PEIn=NULL;
    PyArrayObject *KEIn=NULL;
    PyArrayObject *chargeIn=NULL;
    PyArrayObject *scalarsIn=NULL;
    
    int i, j, index, specie, count;
    int numpyDims[1], numpyDims2[2];
//    double *speciePos, *specieScalars;
    double scalar;
    PyObject *list=NULL;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!iO!O!O!O!O!O!O!ii", &PyArray_Type, &visibleAtomsIn, &NSpecies, &PyArray_Type, &specieArrayIn, 
            &PyArray_Type, &specieCountIn, &PyArray_Type, &posIn, &PyArray_Type, &PEIn, &PyArray_Type, &KEIn, &PyArray_Type, 
            &chargeIn, &PyArray_Type, &scalarsIn, &scalarType, &heightAxis))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisible = (int) visibleAtomsIn->dimensions[0];
    
    if (not_intVector(specieArrayIn)) return NULL;
    specieArray = pyvector_to_Cptr_int(specieArrayIn);
    
    if (not_intVector(specieCountIn)) return NULL;
    specieCount = pyvector_to_Cptr_int(specieCountIn);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(PEIn)) return NULL;
    PE = pyvector_to_Cptr_double(PEIn);
    
    if (not_doubleVector(KEIn)) return NULL;
    KE = pyvector_to_Cptr_double(KEIn);
    
    if (not_doubleVector(chargeIn)) return NULL;
    charge = pyvector_to_Cptr_double(chargeIn);
    
    if (not_doubleVector(scalarsIn)) return NULL;
    scalars = pyvector_to_Cptr_double(scalarsIn);
    
    /* first pass to get counters, assume counter zeroed before */
    for (i = 0; i < NVisible; i++)
    {
        index = visibleAtoms[i];
        specie = specieArray[index];
        specieCount[specie]++;
    }
    
    /* create list for returning */
    list = PyList_New(NSpecies);
    
    /* loop over species */
    for (i = 0; i < NSpecies; i++)
    {
        PyArrayObject *speciePos = NULL;
        PyArrayObject *specieScalars = NULL;
        PyObject *tuple = NULL;
        
        
        /* allocate position array */
        numpyDims2[0] = specieCount[i];
        numpyDims2[1] = 3;
//        speciePos = (double*) allocator("", 2, numpyDims2, 'd');
        speciePos = (PyArrayObject *) PyArray_FromDims(2, numpyDims2, NPY_FLOAT64);
        
        /* allocate position array */
        numpyDims[0] = specieCount[i];
//        specieScalars = (double*) allocator("", 1, numpyDims, 'd');
        specieScalars = (PyArrayObject *) PyArray_FromDims(1, numpyDims, NPY_FLOAT64);
        
        /* loop over atoms */
        count = 0;
        for (j = 0; j < NVisible; j++)
        {
            index = visibleAtoms[j];
            specie = specieArray[index];
            
            if (specie == i)
            {
                /* position */
//                speciePos[3*count+0] = pos[3*index+0];
//                speciePos[3*count+1] = pos[3*index+1];
//                speciePos[3*count+2] = pos[3*index+2];
                DIND2(speciePos, count, 0) = pos[3*index+0];
                DIND2(speciePos, count, 1) = pos[3*index+1];
                DIND2(speciePos, count, 2) = pos[3*index+2];
                
                /* scalar */
                if (scalarType == 0) scalar = specie;
                else if (scalarType == 1) scalar = pos[3*index+heightAxis];
                else if (scalarType == 2) scalar = KE[index];
                else if (scalarType == 3) scalar = PE[index];
                else if (scalarType == 4) scalar = charge[index];
                else scalar = scalars[j];
                
//                specieScalars[count] = scalar;
                DIND1(specieScalars, count) = scalar;
                
                count++;
            }
        }
        
        /* create tuple (setItem steals ownership of array objects!!??) */
        tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyArray_Return(speciePos));
        PyTuple_SetItem(tuple, 1, PyArray_Return(specieScalars));
        
        /* store in list (setItem steals ownership of tuple!?) */
        PyList_SetItem(list, i, tuple);
        
        /* deref */
//        speciePos = NULL;
//        specieScalars = NULL;
    }
    
    return list;
}
