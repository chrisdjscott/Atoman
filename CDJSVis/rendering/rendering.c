
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** Rendering routines written in C to improve performance
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include "array_utils.h"


static PyObject* splitVisAtomsBySpecie(PyObject*, PyObject*);
static PyObject* makeVisibleRadiusArray(PyObject*, PyObject*);
static PyObject* makeVisibleScalarArray(PyObject*, PyObject*);
static PyObject* makeVisiblePointsArray(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"splitVisAtomsBySpecie", splitVisAtomsBySpecie, METH_VARARGS, "Create specie pos and scalar arrays"},
    {"makeVisibleRadiusArray", makeVisibleRadiusArray, METH_VARARGS, "Create radius array for visible atoms"},
    {"makeVisibleScalarArray", makeVisibleScalarArray, METH_VARARGS, "Create scalars array for visible atoms"},
    {"makeVisiblePointsArray", makeVisiblePointsArray, METH_VARARGS, "Create points array for visible atoms"},
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
    int NVisible, *visibleAtoms, NSpecies, *specieArray, *specieCount, scalarType, heightAxis, vectorsLen;
    double *pos, *PE, *KE, *charge, *scalars;  
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *specieArrayIn=NULL;
    PyArrayObject *specieCountIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *PEIn=NULL;
    PyArrayObject *KEIn=NULL;
    PyArrayObject *chargeIn=NULL;
    PyArrayObject *scalarsIn=NULL;
    PyArrayObject *vectors=NULL;
    
    int i, j, index, specie, count;
    int numpyDims[1], numpyDims2[2];
    double scalar;
    PyObject *list=NULL;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!iO!O!O!O!O!O!O!iiO!", &PyArray_Type, &visibleAtomsIn, &NSpecies, &PyArray_Type, &specieArrayIn, 
            &PyArray_Type, &specieCountIn, &PyArray_Type, &posIn, &PyArray_Type, &PEIn, &PyArray_Type, &KEIn, &PyArray_Type, 
            &chargeIn, &PyArray_Type, &scalarsIn, &scalarType, &heightAxis, &PyArray_Type, &vectors))
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
    
    if (not_doubleVector(vectors)) return NULL;
    vectorsLen = (int) vectors->dimensions[0];
    
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
        PyArrayObject *specieVectors = NULL;
        PyObject *tuple = NULL;
        
        /* allocate position array */
        numpyDims2[0] = specieCount[i];
        numpyDims2[1] = 3;
        speciePos = (PyArrayObject *) PyArray_FromDims(2, numpyDims2, NPY_FLOAT64);
        
        /* allocate position array */
        numpyDims[0] = specieCount[i];
        specieScalars = (PyArrayObject *) PyArray_FromDims(1, numpyDims, NPY_FLOAT64);
        
        if (vectorsLen > 0)
        {
            /* allocate vectors array */
            numpyDims2[0] = specieCount[i];
            numpyDims2[1] = 3;
            specieVectors = (PyArrayObject *) PyArray_FromDims(2, numpyDims2, NPY_FLOAT64);
        }
        else
        {
            numpyDims[0] = 0;
            specieVectors = (PyArrayObject *) PyArray_FromDims(1, numpyDims, NPY_FLOAT64);
        }
        
        /* loop over atoms */
        count = 0;
        for (j = 0; j < NVisible; j++)
        {
            index = visibleAtoms[j];
            specie = specieArray[index];
            
            if (specie == i)
            {
                /* position */
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
                
                DIND1(specieScalars, count) = scalar;
                
                if (vectorsLen > 0)
                {
                    /* vector */
                    DIND2(specieVectors, count, 0) = DIND2(vectors, index, 0);
                    DIND2(specieVectors, count, 1) = DIND2(vectors, index, 1);
                    DIND2(specieVectors, count, 2) = DIND2(vectors, index, 2);
                }
                
                count++;
            }
        }
        
        /* create tuple (setItem steals ownership of array objects!!??) */
        tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, PyArray_Return(speciePos));
        PyTuple_SetItem(tuple, 1, PyArray_Return(specieScalars));
        PyTuple_SetItem(tuple, 2, PyArray_Return(specieVectors));
        
        /* store in list (setItem steals ownership of tuple!?) */
        PyList_SetItem(list, i, tuple);
    }
    
    return list;
}

/*******************************************************************************
 ** Make radius array for visible atoms
 *******************************************************************************/
static PyObject*
makeVisibleRadiusArray(PyObject *self, PyObject *args)
{
    int NVisible;
    PyArrayObject *visibleAtoms=NULL;
    PyArrayObject *specie=NULL;
    PyArrayObject *specieCovalentRadius=NULL;

    int i, numpydims[1];
    PyArrayObject *radius=NULL;

    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &visibleAtoms, &PyArray_Type, &specie,
            &PyArray_Type, &specieCovalentRadius))
        return NULL;

    if (not_intVector(visibleAtoms)) return NULL;
    NVisible = (int) visibleAtoms->dimensions[0];

    if (not_intVector(specie)) return NULL;

    if (not_doubleVector(specieCovalentRadius)) return NULL;

    /* create radius array */
    numpydims[0] = NVisible;
    radius = (PyArrayObject *) PyArray_FromDims(1, numpydims, NPY_FLOAT64);

    /* populate array */
    for (i = 0; i < NVisible; i++)
    {
        int index, specieIndex;

        index = IIND1(visibleAtoms, i);
        specieIndex = IIND1(specie, index);

        DIND1(radius, i) = DIND1(specieCovalentRadius, specieIndex);
    }

    return PyArray_Return(radius);
}

/*******************************************************************************
 ** Make scalars array for visible atoms
 *******************************************************************************/
static PyObject*
makeVisibleScalarArray(PyObject *self, PyObject *args)
{
    int NVisible;
    PyArrayObject *visibleAtoms=NULL;
    PyArrayObject *scalarsFull=NULL;

    int i, numpydims[1];
    PyArrayObject *scalars=NULL;

    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &visibleAtoms, &PyArray_Type, &scalarsFull))
        return NULL;

    if (not_intVector(visibleAtoms)) return NULL;
    NVisible = (int) visibleAtoms->dimensions[0];

    if (not_doubleVector(scalarsFull)) return NULL;

    /* create radius array */
    numpydims[0] = NVisible;
    scalars = (PyArrayObject *) PyArray_FromDims(1, numpydims, NPY_FLOAT64);

    /* populate array */
    for (i = 0; i < NVisible; i++)
    {
        int index;

        index = IIND1(visibleAtoms, i);
        DIND1(scalars, i) = DIND1(scalarsFull, index);
    }

    return PyArray_Return(scalars);
}

/*******************************************************************************
 ** Make points array for visible atoms
 *******************************************************************************/
static PyObject*
makeVisiblePointsArray(PyObject *self, PyObject *args)
{
    int NVisible;
    PyArrayObject *visibleAtoms=NULL;
    PyArrayObject *pos=NULL;

    int i, numpydims[2];
    PyArrayObject *visiblePos=NULL;

    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &visibleAtoms, &PyArray_Type, &pos))
        return NULL;

    if (not_intVector(visibleAtoms)) return NULL;
    NVisible = (int) visibleAtoms->dimensions[0];

    if (not_doubleVector(pos)) return NULL;

    /* create radius array */
    numpydims[0] = NVisible;
    numpydims[1] = 3;
    visiblePos = (PyArrayObject *) PyArray_FromDims(2, numpydims, NPY_FLOAT64);

    /* populate array */
    for (i = 0; i < NVisible; i++)
    {
        int index, index3, j;

        index = IIND1(visibleAtoms, i);
        index3 = index * 3;

        for (j = 0; j < 3; j++)
            DIND2(visiblePos, i, j) = DIND1(pos, index3 + j);
    }

    return PyArray_Return(visiblePos);
}
