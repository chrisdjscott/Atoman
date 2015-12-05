
/*******************************************************************************
 ** Rendering routines written in C to improve performance
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include "visclibs/array_utils.h"


static PyObject* splitVisAtomsBySpecie(PyObject*, PyObject*);
static PyObject* makeVisibleRadiusArray(PyObject*, PyObject*);
static PyObject* makeVisibleScalarArray(PyObject*, PyObject*);
static PyObject* makeVisiblePointsArray(PyObject*, PyObject*);
static PyObject* countVisibleBySpecie(PyObject *, PyObject *);
static PyObject* countAntisitesBySpecie(PyObject *, PyObject *);
static PyObject* countSplitIntsBySpecie(PyObject *, PyObject *);
static PyObject* makeBondsArrays(PyObject *, PyObject *);
static PyObject* makeDisplacementVectorBondsArrays(PyObject *, PyObject *);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"splitVisAtomsBySpecie", splitVisAtomsBySpecie, METH_VARARGS, "Create specie pos and scalar arrays"},
    {"makeVisibleRadiusArray", makeVisibleRadiusArray, METH_VARARGS, "Create radius array for visible atoms"},
    {"makeVisibleScalarArray", makeVisibleScalarArray, METH_VARARGS, "Create scalars array for visible atoms"},
    {"makeVisiblePointsArray", makeVisiblePointsArray, METH_VARARGS, "Create points array for visible atoms"},
    {"countVisibleBySpecie", countVisibleBySpecie, METH_VARARGS, "Count the number of visible atoms of each specie"},
    {"countAntisitesBySpecie", countAntisitesBySpecie, METH_VARARGS, "Count the number of antisites of each specie"},
    {"countSplitIntsBySpecie", countSplitIntsBySpecie, METH_VARARGS, "Count the number of split interstitials by specie"},
    {"makeBondsArrays", makeBondsArrays, METH_VARARGS, "Create bonds arrays for rendering"},
    {"makeDisplacementVectorBondsArrays", makeDisplacementVectorBondsArrays, METH_VARARGS, "Create bonds arrays for rendering displacement vectors"},
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
 ** Make arrays for rendering bonds for displacment vectors
 *******************************************************************************/
static PyObject*
makeDisplacementVectorBondsArrays(PyObject *self, PyObject *args)
{
    int numBonds;
    PyArrayObject *visibleAtoms=NULL;
    PyArrayObject *scalarsArray=NULL;
    PyArrayObject *pos=NULL;
    PyArrayObject *drawBondArray=NULL;
    PyArrayObject *bondVectorsArray=NULL;
    PyObject *result=NULL;
    
    /* parse arguments from Python */
    if (PyArg_ParseTuple(args, "iO!O!O!O!O!", &numBonds, &PyArray_Type, &visibleAtoms, &PyArray_Type, &scalarsArray,
            &PyArray_Type, &pos, &PyArray_Type, &drawBondArray, &PyArray_Type, &bondVectorsArray))
    {
        int i, count, numVisible;
        npy_intp numpydims[2];
        PyArrayObject *bondCoords = NULL;
        PyArrayObject *bondScalars = NULL;
        PyArrayObject *bondVectors = NULL;
        
        /* check arguments */
        if (not_intVector(visibleAtoms)) return NULL;
        numVisible = (int) PyArray_DIM(visibleAtoms, 0);
        if (not_doubleVector(scalarsArray)) return NULL;
        if (not_doubleVector(pos)) return NULL;
        if (not_intVector(drawBondArray)) return NULL;
        if (not_doubleVector(bondVectorsArray)) return NULL;
        
        /* create array for bond coordinates */
        numpydims[0] = (npy_intp) numBonds;
        numpydims[1] = 3;
        bondCoords = (PyArrayObject *) PyArray_SimpleNew(2, numpydims, NPY_FLOAT64);
        if (bondCoords == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate bondCoords");
            return NULL;
        }
        
        /* create array for bond vectors */
        numpydims[0] = (npy_intp) numBonds;
        numpydims[1] = 3;
        bondVectors = (PyArrayObject *) PyArray_SimpleNew(2, numpydims, NPY_FLOAT64);
        if (bondVectors == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate bondVectors");
            Py_DECREF(bondCoords);
            return NULL;
        }
        
        /* create array for bond scalars */
        numpydims[0] = (npy_intp) numBonds;
        bondScalars = (PyArrayObject *) PyArray_SimpleNew(1, numpydims, NPY_FLOAT64);
        if (bondScalars == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate bondScalars");
            Py_DECREF(bondCoords);
            Py_DECREF(bondVectors);
            return NULL;
        }
        
        /* loop over visible atoms */
        count = 0;
        for (i = 0; i < numVisible; i++)
        {
            /* check if we should be drawing this bond */
            if (IIND1(drawBondArray, i))
            {
                int j, i3 = 3 * i;
                int index3 = 3 * IIND1(visibleAtoms, i);
                
                /* store bond values */
                for (j = 0; j < 3; j++)
                {
                    DIND2(bondCoords, count, j) = DIND1(pos, index3 + j);
                    DIND2(bondVectors, count, j) = DIND1(bondVectorsArray, i3 + j);
                }
                DIND1(bondScalars, count) = DIND1(scalarsArray, i);
                count++;
            }
        }
        
        /* tuple for result */
        result = PyTuple_New(3);
        PyTuple_SetItem(result, 0, PyArray_Return(bondCoords));
        PyTuple_SetItem(result, 1, PyArray_Return(bondVectors));
        PyTuple_SetItem(result, 2, PyArray_Return(bondScalars));
    }
    
    return result;
}

/*******************************************************************************
 ** Make arrays for rendering bonds
 *******************************************************************************/
static PyObject*
makeBondsArrays(PyObject *self, PyObject *args)
{
    PyArrayObject *visibleAtoms=NULL;
    PyArrayObject *scalarsArray=NULL;
    PyArrayObject *pos=NULL;
    PyArrayObject *numBondsArray=NULL;
    PyArrayObject *bondsArray=NULL;
    PyArrayObject *bondVectorsArray=NULL;
    PyObject *result=NULL;
    
    /* parse arguments from Python */
    if (PyArg_ParseTuple(args, "O!O!O!O!O!O!", &PyArray_Type, &visibleAtoms, &PyArray_Type, &scalarsArray,
            &PyArray_Type, &pos, &PyArray_Type, &numBondsArray, &PyArray_Type, &bondsArray, &PyArray_Type,
            &bondVectorsArray))
    {
        int i, numVisible, numBonds, count, bondCount;
        npy_intp numpydims[2];
        PyArrayObject *bondCoords = NULL;
        PyArrayObject *bondScalars = NULL;
        PyArrayObject *bondVectors = NULL;
        
        /* check arguments */
        if (not_intVector(visibleAtoms)) return NULL;
        numVisible = (int) PyArray_DIM(visibleAtoms, 0);
        if (not_doubleVector(scalarsArray)) return NULL;
        if (not_doubleVector(pos)) return NULL;
        if (not_intVector(numBondsArray)) return NULL;
        if (not_intVector(bondsArray)) return NULL;
        if (not_doubleVector(bondVectorsArray)) return NULL;
        
        /* calculate the number of bonds */
        numBonds = 0;
        for (i = 0; i < numVisible; i++) numBonds += IIND1(numBondsArray, i);
        /* multiply by two as there are two "bonds" drawn per real bond */
        numBonds *= 2;
        
        /* create array for bond coordinates */
        numpydims[0] = (npy_intp) numBonds;
        numpydims[1] = 3;
        bondCoords = (PyArrayObject *) PyArray_SimpleNew(2, numpydims, NPY_FLOAT64);
        if (bondCoords == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate bondCoords");
            return NULL;
        }
        
        /* create array for bond vectors */
        numpydims[0] = (npy_intp) numBonds;
        numpydims[1] = 3;
        bondVectors = (PyArrayObject *) PyArray_SimpleNew(2, numpydims, NPY_FLOAT64);
        if (bondVectors == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate bondVectors");
            Py_DECREF(bondCoords);
            return NULL;
        }
        
        /* create array for bond scalars */
        numpydims[0] = (npy_intp) numBonds;
        bondScalars = (PyArrayObject *) PyArray_SimpleNew(1, numpydims, NPY_FLOAT64);
        if (bondScalars == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate bondScalars");
            Py_DECREF(bondCoords);
            Py_DECREF(bondVectors);
            return NULL;
        }
        
        /* loop over visible atoms */
        count = 0;
        bondCount = 0;
        for (i = 0; i < numVisible; i++)
        {
            int j;
            int indexa = IIND1(visibleAtoms, i);
            int indexa3 = 3 * indexa;
            int numBondsForAtom = IIND1(numBondsArray, i);
            double xposa, yposa, zposa, scalara;
            
            /* atom position and scalar */
            xposa = DIND1(pos, indexa3    );
            yposa = DIND1(pos, indexa3 + 1);
            zposa = DIND1(pos, indexa3 + 2);
            scalara = DIND1(scalarsArray, i);
            
            /* loop over this atoms bonds */
            for (j = 0; j < numBondsForAtom; j++)
            {
                int cnt3 = count * 3;
                int visIndex = IIND1(bondsArray, count);
                int indexb3 = 3 * IIND1(visibleAtoms, visIndex);
                double xposb = DIND1(pos, indexb3    );
                double yposb = DIND1(pos, indexb3 + 1);
                double zposb = DIND1(pos, indexb3 + 2);
                double scalarb = DIND1(scalarsArray, visIndex);
                double bvecx = DIND1(bondVectorsArray, cnt3    );
                double bvecy = DIND1(bondVectorsArray, cnt3 + 1);
                double bvecz = DIND1(bondVectorsArray, cnt3 + 2);
                
                /* partial bond from atom a towards b */
                DIND2(bondCoords, bondCount, 0) = xposa;
                DIND2(bondCoords, bondCount, 1) = yposa;
                DIND2(bondCoords, bondCount, 2) = zposa;
                DIND2(bondVectors, bondCount, 0) = bvecx;
                DIND2(bondVectors, bondCount, 1) = bvecy;
                DIND2(bondVectors, bondCount, 2) = bvecz;
                DIND1(bondScalars, bondCount) = scalara;
                bondCount++;
                
                /* partial bond from atom b towards a */
                DIND2(bondCoords, bondCount, 0) = xposb;
                DIND2(bondCoords, bondCount, 1) = yposb;
                DIND2(bondCoords, bondCount, 2) = zposb;
                DIND2(bondVectors, bondCount, 0) = -1.0 * bvecx;
                DIND2(bondVectors, bondCount, 1) = -1.0 * bvecy;
                DIND2(bondVectors, bondCount, 2) = -1.0 * bvecz;
                DIND1(bondScalars, bondCount) = scalarb;
                bondCount++;
                
                count++;
            }
        }
        
        /* tuple for result */
        result = PyTuple_New(3);
        PyTuple_SetItem(result, 0, PyArray_Return(bondCoords));
        PyTuple_SetItem(result, 1, PyArray_Return(bondVectors));
        PyTuple_SetItem(result, 2, PyArray_Return(bondScalars));
    }
    
    return result;
}
 
/*******************************************************************************
 ** Count split interstitials by species
 *******************************************************************************/
static PyObject*
countSplitIntsBySpecie(PyObject *self, PyObject *args)
{
    int NSpecies;
    PyArrayObject *splitInts=NULL;
    PyArrayObject *specie=NULL;
    PyObject *result=NULL;
    
    /* parse and check arguments from Python */
    if (PyArg_ParseTuple(args, "O!iO!", &PyArray_Type, &splitInts, &NSpecies, &PyArray_Type, &specie))
    {
        int i, nsplit;
        npy_intp dims[2];
        PyArrayObject *specieCount=NULL;
        
        
        if (not_intVector(splitInts)) return NULL;
        nsplit = (int) (PyArray_DIM(splitInts, 0) / 3);
        if (not_intVector(specie)) return NULL;
        
        /* specie counter array */
        dims[0] = (npy_intp) NSpecies;
        dims[1] = (npy_intp) NSpecies;
        specieCount = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_INT32);
        for (i = 0; i < NSpecies; i++) 
        {
            int j;
            for (j = 0; j < NSpecies; j++)
                IIND2(specieCount, i, j) = 0;
        }
        
        /* loop over antistes, incrementing specie counter */
        for (i = 0; i < nsplit; i++)
        {
            int i3 = 3 * i;
            int index, index2, specieIndex, specieIndex2;
            
            index = IIND1(splitInts, i3 + 1);
            specieIndex = IIND1(specie, index);
            
            index2 = IIND1(splitInts, i3 + 2);
            specieIndex2 = IIND1(specie, index2);
            
            IIND2(specieCount, specieIndex, specieIndex2) = IIND2(specieCount, specieIndex, specieIndex2) + 1;
            if (specieIndex != specieIndex2)
                IIND2(specieCount, specieIndex2, specieIndex) = IIND2(specieCount, specieIndex2, specieIndex) + 1;
        }
        
        /* store result */
        result = PyArray_Return(specieCount);
    }
    
    return result;
}

/*******************************************************************************
 ** Count antisites by species
 *******************************************************************************/
static PyObject*
countAntisitesBySpecie(PyObject *self, PyObject *args)
{
    int NSpeciesInput, NSpeciesRef;
    PyArrayObject *antisites=NULL;
    PyArrayObject *onAntisites=NULL;
    PyArrayObject *specieRef=NULL;
    PyArrayObject *specieInput=NULL;
    PyObject *result=NULL;
    
    /* parse and check arguments from Python */
    if (PyArg_ParseTuple(args, "O!iO!O!iO!", &PyArray_Type, &antisites, &NSpeciesRef, &PyArray_Type, &specieRef,
            &PyArray_Type, &onAntisites, &NSpeciesInput, &PyArray_Type, &specieInput))
    {
        int i, nants;
        npy_intp dims[2];
        PyArrayObject *specieCount=NULL;
        
        
        if (not_intVector(antisites)) return NULL;
        nants = (int) PyArray_DIM(antisites, 0);
        if (not_intVector(specieRef)) return NULL;
        if (not_intVector(specieInput)) return NULL;
        
        /* specie counter array */
        dims[0] = (npy_intp) NSpeciesRef;
        dims[1] = (npy_intp) NSpeciesInput;
        specieCount = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_INT32);
        for (i = 0; i < NSpeciesRef; i++) 
        {
            int j;
            for (j = 0; j < NSpeciesInput; j++)
                IIND2(specieCount, i, j) = 0;
        }
        
        /* loop over antistes, incrementing specie counter */
        for (i = 0; i < nants; i++)
        {
            int index, index2, specieIndex, specieIndex2;
            
            index = IIND1(antisites, i);
            specieIndex = IIND1(specieRef, index);
            
            index2 = IIND1(onAntisites, i);
            specieIndex2 = IIND1(specieInput, index2);
            
            IIND2(specieCount, specieIndex, specieIndex2) = IIND2(specieCount, specieIndex, specieIndex2) + 1;
        }
        
        /* store result */
        result = PyArray_Return(specieCount);
    }
    
    return result;
}

/*******************************************************************************
 ** Split visible atoms by specie (position and scalar)
 *******************************************************************************/
static PyObject*
countVisibleBySpecie(PyObject *self, PyObject *args)
{
    int i;
    npy_intp dims[1];
    int NSpecies, NVisible;
    PyArrayObject *visibleAtoms=NULL;
    PyArrayObject *specieArray=NULL;
    PyArrayObject *specieCount=NULL;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!iO!", &PyArray_Type, &visibleAtoms, &NSpecies, &PyArray_Type, &specieArray))
        return NULL;
    
    if (not_intVector(visibleAtoms)) return NULL;
    NVisible = (int) PyArray_DIM(visibleAtoms, 0);
    if (not_intVector(specieArray)) return NULL;
    
    /* specie counter array */
    dims[0] = (npy_intp) NSpecies;
    specieCount = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    for (i = 0; i < NSpecies; i++) IIND1(specieCount, i) = 0;
    
    /* loop over visible atoms, incrementing specie counter */
    for (i = 0; i < NVisible; i++)
    {
        int index, specieIndex;
        
        index = IIND1(visibleAtoms, i);
        specieIndex = IIND1(specieArray, index);
        IIND1(specieCount, specieIndex) = IIND1(specieCount, specieIndex) + 1;
    }
    
    return PyArray_Return(specieCount);
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
    npy_intp numpyDims[2];
    double scalar;
    PyObject *list=NULL;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!iO!O!O!O!O!O!O!iiO!", &PyArray_Type, &visibleAtomsIn, &NSpecies, &PyArray_Type, &specieArrayIn, 
            &PyArray_Type, &specieCountIn, &PyArray_Type, &posIn, &PyArray_Type, &PEIn, &PyArray_Type, &KEIn, &PyArray_Type, 
            &chargeIn, &PyArray_Type, &scalarsIn, &scalarType, &heightAxis, &PyArray_Type, &vectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisible = (int) PyArray_DIM(visibleAtomsIn, 0);
    
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
    vectorsLen = (int) PyArray_DIM(vectors, 0);
    
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
        numpyDims[0] = (npy_intp) specieCount[i];
        numpyDims[1] = 3;
        speciePos = (PyArrayObject *) PyArray_SimpleNew(2, numpyDims, NPY_FLOAT64);
        
        /* allocate position array */
        numpyDims[0] = (npy_intp) specieCount[i];
        specieScalars = (PyArrayObject *) PyArray_SimpleNew(1, numpyDims, NPY_FLOAT64);
        
        if (vectorsLen > 0)
        {
            /* allocate vectors array */
            numpyDims[0] = (npy_intp) specieCount[i];
            numpyDims[1] = 3;
            specieVectors = (PyArrayObject *) PyArray_SimpleNew(2, numpyDims, NPY_FLOAT64);
        }
        else
        {
            numpyDims[0] = 0;
            specieVectors = (PyArrayObject *) PyArray_SimpleNew(1, numpyDims, NPY_FLOAT64);
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

    int i;
    npy_intp numpydims[1];
    PyArrayObject *radius=NULL;

    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &visibleAtoms, &PyArray_Type, &specie,
            &PyArray_Type, &specieCovalentRadius))
        return NULL;

    if (not_intVector(visibleAtoms)) return NULL;
    NVisible = (int) PyArray_DIM(visibleAtoms, 0);

    if (not_intVector(specie)) return NULL;

    if (not_doubleVector(specieCovalentRadius)) return NULL;

    /* create radius array */
    numpydims[0] = (npy_intp) NVisible;
    radius = (PyArrayObject *) PyArray_SimpleNew(1, numpydims, NPY_FLOAT64);

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

    int i;
    npy_intp numpydims[1];
    PyArrayObject *scalars=NULL;

    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &visibleAtoms, &PyArray_Type, &scalarsFull))
        return NULL;

    if (not_intVector(visibleAtoms)) return NULL;
    NVisible = (int) PyArray_DIM(visibleAtoms, 0);

    if (not_doubleVector(scalarsFull)) return NULL;

    /* create radius array */
    numpydims[0] = (npy_intp) NVisible;
    scalars = (PyArrayObject *) PyArray_SimpleNew(1, numpydims, NPY_FLOAT64);

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

    int i;
    npy_intp numpydims[2];
    PyArrayObject *visiblePos=NULL;

    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &visibleAtoms, &PyArray_Type, &pos))
        return NULL;

    if (not_intVector(visibleAtoms)) return NULL;
    NVisible = (int) PyArray_DIM(visibleAtoms, 0);

    if (not_doubleVector(pos)) return NULL;

    /* create radius array */
    numpydims[0] = (npy_intp) NVisible;
    numpydims[1] = 3;
    visiblePos = (PyArrayObject *) PyArray_SimpleNew(2, numpydims, NPY_FLOAT64);

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
