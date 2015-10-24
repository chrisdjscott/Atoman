
/*******************************************************************************
 ** Generate FCC lattice
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "array_utils.h"


static PyObject* generateFCCLattice(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"generateFCCLattice", generateFCCLattice, METH_VARARGS, "Generate a FCC lattice"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_lattice_gen_fcc(void)
{
    (void)Py_InitModule("_lattice_gen_fcc", methods);
    import_array();
}

/*******************************************************************************
 ** Generate lattice
 *******************************************************************************/
static PyObject*
generateFCCLattice(PyObject *self, PyObject *args)
{
    int *numCells, *PBC;
    double latticeConstant;
    PyArrayObject *numCellsIn=NULL;
    PyArrayObject *PBCIn=NULL;
    
    const int unitCellNAtoms = 4;
    int i, loopStop[3], count;
    npy_intp numpyDims[1];
    int unitCellSpecie[unitCellNAtoms];
    double cellDims[3], a1;
    double unitCellPos[3 * unitCellNAtoms];
    double unitCellCharge[unitCellNAtoms];
    const double epsilon = 1e-4;
    PyArrayObject *pos=NULL;
    PyArrayObject *specie=NULL;
    PyArrayObject *charge=NULL;
    PyObject *tuple=NULL;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &numCellsIn, &PyArray_Type, &PBCIn, &latticeConstant))
        return NULL;
    
    if (not_intVector(numCellsIn)) return NULL;
    numCells = pyvector_to_Cptr_int(numCellsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
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
    unitCellSpecie[3] = 0;
    unitCellCharge[0] = 0.0;
    unitCellCharge[1] = 0.0;
    unitCellCharge[2] = 0.0;
    unitCellCharge[3] = 0.0;
    
    /* first pass to get number of atoms (not required with PBCs...) */
    count = 0;
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
                }
            }
        }
    }
    
    /* allocate arrays */
    numpyDims[0] = (npy_intp) (3 * count);
    pos = (PyArrayObject *) PyArray_SimpleNew(1, numpyDims, NPY_FLOAT64);
    
    numpyDims[0] = (npy_intp) count;
    charge = (PyArrayObject *) PyArray_SimpleNew(1, numpyDims, NPY_FLOAT64);
    
    numpyDims[0] = (npy_intp) count;
    specie = (PyArrayObject *) PyArray_SimpleNew(1, numpyDims, NPY_INT32);
    
    /* second pass to fill the arrays */
    count = 0;
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
                    
                    count++;
                }
            }
        }
    }
    
    /* create tuple (setItem steals ownership??) */
    tuple = PyTuple_New(4);
    PyTuple_SetItem(tuple, 0, Py_BuildValue("i", count));
    PyTuple_SetItem(tuple, 1, PyArray_Return(specie));
    PyTuple_SetItem(tuple, 2, PyArray_Return(pos));
    PyTuple_SetItem(tuple, 3, PyArray_Return(charge));
    
    return tuple;
}
