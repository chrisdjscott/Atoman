
/*******************************************************************************
 ** Generate FCC lattice
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include <unistd.h>
#include "visclibs/array_utils.h"


static PyObject* generatePu3GaLattice(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"generatePu3GaLattice", generatePu3GaLattice, METH_VARARGS, "Generate a Pu-Ga lattice (Pu3Ga method)"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_lattice_gen_pu3ga(void)
{
    (void)Py_InitModule("_lattice_gen_pu3ga", methods);
    import_array();
}

/*******************************************************************************
 ** Generate lattice
 *******************************************************************************/
static PyObject*
generatePu3GaLattice(PyObject *self, PyObject *args)
{
    int *numCells, *PBC;
    double latticeConstant, gaConc;
    PyArrayObject *numCellsIn=NULL;
    PyArrayObject *PBCIn=NULL;
    
    const int unitCellNAtoms = 4;
    int i, loopStop[3], count, numAtoms;
    npy_intp numpyDims[1];
    int unitCellSpecie[unitCellNAtoms], countGa;
    int *gaIndexes, numGaRemove, requiredNGa;
    int numRemoved, counter[2];
    double cellDims[3], a1;
    double unitCellPos[3 * unitCellNAtoms];
    double unitCellCharge[unitCellNAtoms];
#ifdef DEBUG
    double currentGaConc;
#endif
    const double epsilon = 1e-4;
    PyArrayObject *pos=NULL;
    PyArrayObject *specie=NULL;
    PyArrayObject *charge=NULL;
    PyArrayObject *specieCount=NULL;
    PyObject *tuple=NULL;
    
#ifdef DEBUG
    printf("Generate Pu3Ga lattice (CLIB)\n");
#endif
   
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!dd", &PyArray_Type, &numCellsIn, &PyArray_Type, &PBCIn, &latticeConstant, &gaConc))
        return NULL;
    
    if (not_intVector(numCellsIn)) return NULL;
    numCells = pyvector_to_Cptr_int(numCellsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (gaConc < 0.0 || gaConc > 25.0)
    {
        PyErr_SetString(PyExc_RuntimeError, "Ga concentration must be between 0 and 25 percent.");
        return NULL;
    }
    
    /* dimensions/handle PBCs */
    for (i = 0; i < 3; i++)
    {
        cellDims[i] = latticeConstant * numCells[i];
        loopStop[i] = (PBC[i]) ? numCells[i] : numCells[i] + 1;
    }
    
    /* define primitive cell */
    a1 = latticeConstant / 2.0;
    unitCellPos[0] = 0.0; unitCellPos[1] = 0.0; unitCellPos[2] = 0.0;
    unitCellPos[3] = 0.0; unitCellPos[4] = a1; unitCellPos[5] = a1;
    unitCellPos[6] = a1; unitCellPos[7] = 0.0; unitCellPos[8] = a1;
    unitCellPos[9] = a1; unitCellPos[10] = a1; unitCellPos[11] = 0.0;
    unitCellSpecie[0] = 0;
    unitCellSpecie[1] = 0;
    unitCellSpecie[2] = 0;
    unitCellSpecie[3] = 1;
    unitCellCharge[0] = 0.0;
    unitCellCharge[1] = 0.0;
    unitCellCharge[2] = 0.0;
    unitCellCharge[3] = 0.0;
    
    /* first pass to get number of atoms (not required with PBCs...) */
    count = 0;
    countGa = 0;
    for (i = 0; i < loopStop[0]; i++)
    {
        int j;
        double ilatt = i * latticeConstant;
        
        for (j = 0; j < loopStop[1]; j++)
        {
            int k;
            double jlatt = j * latticeConstant;
            
            for (k = 0; k < loopStop[2]; k++)
            {
                int m;
                
                for (m = 0; m < unitCellNAtoms; m++)
                {
                    double atomPos[3];
                    
                    atomPos[0] = unitCellPos[3 * m    ] + ilatt;
                    atomPos[1] = unitCellPos[3 * m + 1] + jlatt;
                    atomPos[2] = unitCellPos[3 * m + 2] + k * latticeConstant;
                    
                    if (atomPos[0] > cellDims[0] + epsilon || atomPos[1] > cellDims[1] + epsilon || atomPos[2] > cellDims[2] + epsilon)
                        continue;
                    
                    /* add atom */
                    count++;
                    
                    /* count Ga atoms */
                    if (unitCellSpecie[m] == 1) countGa++;
                }
            }
        }
    }
    
    /* allocate arrays */
    gaIndexes = malloc(countGa * sizeof(int));
    if (gaIndexes == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate gaIndexes array");
        return NULL;
    }
    
    numpyDims[0] = (npy_intp) (3 * count);
    pos = (PyArrayObject *) PyArray_SimpleNew(1, numpyDims, NPY_FLOAT64);
    
    numpyDims[0] = (npy_intp) count;
    charge = (PyArrayObject *) PyArray_SimpleNew(1, numpyDims, NPY_FLOAT64);
    
    numpyDims[0] = (npy_intp) count;
    specie = (PyArrayObject *) PyArray_SimpleNew(1, numpyDims, NPY_INT32);
    
    numpyDims[0] = 2;
    specieCount = (PyArrayObject *) PyArray_SimpleNew(1, numpyDims, NPY_INT32);
    for (i = 0; i < 2; i++) IIND1(specieCount, i) = 0;
    
    /* second pass to fill the arrays */
    count = 0;
    countGa = 0;
    for (i = 0; i < loopStop[0]; i++)
    {
        int j;
        double ilatt = i * latticeConstant;
        
        for (j = 0; j < loopStop[1]; j++)
        {
            int k;
            double jlatt = j * latticeConstant;
            
            for (k = 0; k < loopStop[2]; k++)
            {
                int m;
                
                for (m = 0; m < unitCellNAtoms; m++)
                {
                    int c3;
                    double atomPos[3];
                    
                    atomPos[0] = unitCellPos[3 * m    ] + ilatt;
                    atomPos[1] = unitCellPos[3 * m + 1] + jlatt;
                    atomPos[2] = unitCellPos[3 * m + 2] + k * latticeConstant;
                    
                    if (atomPos[0] > cellDims[0] + epsilon || atomPos[1] > cellDims[1] + epsilon || atomPos[2] > cellDims[2] + epsilon)
                        continue;
                    
                    /* add atom */
                    /* position */
                    c3 = count * 3;
                    DIND1(pos, c3    ) = atomPos[0];
                    DIND1(pos, c3 + 1) = atomPos[1];
                    DIND1(pos, c3 + 2) = atomPos[2];
                    
                    /* charge */
                    DIND1(charge, count) = unitCellCharge[m];
                    
                    /* specie */
                    IIND1(specie, count) = unitCellSpecie[m];
                    
                    /* store Ga indexes */
                    if (unitCellSpecie[m] == 1) gaIndexes[countGa++] = count;
                    
                    count++;
                }
            }
        }
    }
    
    numAtoms = count;
    
    /* correct Ga concentration */
    requiredNGa = (int) (gaConc * 0.01 * numAtoms);
    numGaRemove = countGa - requiredNGa;
    
#ifdef DEBUG
    currentGaConc = ((double) countGa) / ((double) numAtoms);
    printf("Current Ga concentration = %lf (N = %d (/ %d))\n", currentGaConc, countGa, numAtoms);
    printf("Required num Ga = %d\n", requiredNGa);
    printf("Removing %d Ga atoms\n", numGaRemove);
#endif
    
    /* random seed */
    srand(((time(NULL) * 181) * ((getpid() - 83) * 359)) % 104729);
    
    /* remove Ga */
    numRemoved = 0;
    while (numRemoved < numGaRemove)
    {
        int randindex, index;
        
        /* random number */
        randindex = (int) (rand() % countGa);
        index = gaIndexes[randindex];
        if (index < 0) continue;
        
#ifdef DEBUG
        printf("  Removing Ga %d\n", index);
#endif
        
        IIND1(specie, index) = 0;
        gaIndexes[randindex] = -1;
        numRemoved++;
    }
    
    free(gaIndexes);
    
    /* verify */
    counter[0] = 0;
    counter[1] = 0;
    for (i = 0; i < numAtoms; i++) counter[IIND1(specie, i)]++;
    
    if (counter[1] != requiredNGa)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not correct the Ga concentration.");
        Py_XDECREF(specie);
        Py_XDECREF(pos);
        Py_XDECREF(charge);
        Py_XDECREF(specieCount);
        return NULL;
    }
    
    /* specie count */
    IIND1(specieCount, 0) = counter[0];
    IIND1(specieCount, 1) = counter[1];
    
#ifdef DEBUG
    currentGaConc = ((double) counter[1]) / ((double) numAtoms);
    printf("Current Ga concentration = %lf (N = %d (/ %d))\n", currentGaConc, counter[1], numAtoms);
#endif
    
    /* create tuple (setItem steals ownership??) */
    tuple = PyTuple_New(5);
    PyTuple_SetItem(tuple, 0, Py_BuildValue("i", numAtoms));
    PyTuple_SetItem(tuple, 1, PyArray_Return(specie));
    PyTuple_SetItem(tuple, 2, PyArray_Return(pos));
    PyTuple_SetItem(tuple, 3, PyArray_Return(charge));
    PyTuple_SetItem(tuple, 4, PyArray_Return(specieCount));
    
    return tuple;
}
