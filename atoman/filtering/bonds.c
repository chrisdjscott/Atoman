/*******************************************************************************
 ** Functions associated with bonds
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "boxeslib.h"
#include "utilities.h"
#include "array_utils.h"

static PyObject* calculateBonds(PyObject*, PyObject*);
static PyObject* calculateDisplacementVectors(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"calculateBonds", calculateBonds, METH_VARARGS, "Find bonds between visible atoms"},
    {"calculateDisplacementVectors", calculateDisplacementVectors, METH_VARARGS, "Calculate atom displacement vectors"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
initbonds(void)
{
    (void)Py_InitModule("bonds", methods);
    import_array();
}

/*******************************************************************************
 * Calculate bonds
 *******************************************************************************/
static PyObject*
calculateBonds(PyObject *self, PyObject *args)
{
    int NVisible, *visibleAtoms, *specie, NSpecies, maxBondsPerAtom, *PBC, *bondArray, *NBondsArray, *bondSpecieCounter;
    double *pos, *bondMinArray, *bondMaxArray, approxBoxWidth, *cellDims, *bondVectorArray;   
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *bondArrayIn=NULL;
    PyArrayObject *NBondsArrayIn=NULL;
    PyArrayObject *bondSpecieCounterIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *bondMinArrayIn=NULL;
    PyArrayObject *bondMaxArrayIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *bondVectorArrayIn=NULL;
    
    int i, j, k, index, index2, visIndex;
    int speca, specb, count, boxstat;
    int boxIndex, boxNebList[27];
    double *visiblePos, sep2, sep;
    double sepVec[3];
    struct Boxes *boxes;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!iO!O!diO!O!O!O!O!O!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, &PyArray_Type, &specieIn,
            &NSpecies, &PyArray_Type, &bondMinArrayIn, &PyArray_Type, &bondMaxArrayIn, &approxBoxWidth, &maxBondsPerAtom, &PyArray_Type, &cellDimsIn,
            &PyArray_Type, &PBCIn, &PyArray_Type, &bondArrayIn, &PyArray_Type, &NBondsArrayIn, &PyArray_Type, &bondVectorArrayIn, &PyArray_Type,
            &bondSpecieCounterIn))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisible = (int) PyArray_DIM(visibleAtomsIn, 0);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    
    if (not_doubleVector(bondMinArrayIn)) return NULL;
    bondMinArray = pyvector_to_Cptr_double(bondMinArrayIn);
    
    if (not_doubleVector(bondMaxArrayIn)) return NULL;
    bondMaxArray = pyvector_to_Cptr_double(bondMaxArrayIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_intVector(bondArrayIn)) return NULL;
    bondArray = pyvector_to_Cptr_int(bondArrayIn);
    
    if (not_intVector(NBondsArrayIn)) return NULL;
    NBondsArray = pyvector_to_Cptr_int(NBondsArrayIn);
    
    if (not_intVector(bondSpecieCounterIn)) return NULL;
    bondSpecieCounter = pyvector_to_Cptr_int(bondSpecieCounterIn);
    
    if (not_doubleVector(bondVectorArrayIn)) return NULL;
    bondVectorArray = pyvector_to_Cptr_double(bondVectorArrayIn);
    
#ifdef DEBUG
    printf("BONDS CLIB\n");
    printf("N VIS: %d\n", NVisible);
    
    for (i = 0; i < NSpecies; i++)
    {
        for (j = i; j < NSpecies; j++)
            printf("%d - %d: %lf -> %lf\n", i, j, bondMinArray[i*NSpecies+j], bondMaxArray[i*NSpecies+j]);
    }
#endif
    
    /* construct visible pos array */
    visiblePos = malloc(3 * NVisible * sizeof(double));
    if (visiblePos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate visiblePos");
        return NULL;
    }
    
    for (i=0; i<NVisible; i++)
    {
        int index = visibleAtoms[i];
        int ind3 = 3 * index;
        int i3 = 3 * i;
        visiblePos[i3    ] = pos[ind3    ];
        visiblePos[i3 + 1] = pos[ind3 + 1];
        visiblePos[i3 + 2] = pos[ind3 + 2];
    }
    
    /* box visible atoms */
    boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
    if (boxes == NULL)
    {
        free(visiblePos);
        return NULL;
    }
    boxstat = putAtomsInBoxes(NVisible, visiblePos, boxes);
    free(visiblePos);
    if (boxstat) return NULL;
    
    /* loop over visible atoms */
    count = 0;
    for (i = 0; i < NVisible; i++)
    {
        int boxNebListSize;
        
        index = visibleAtoms[i];
        
        speca = specie[index];
        
        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(pos[3*index], pos[3*index+1], pos[3*index+2], boxes);
        if (boxIndex < 0)
        {
            freeBoxes(boxes);
            return NULL;
        }
        
        /* find neighbouring boxes */
        boxNebListSize = getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over box neighbourhood */
        for (j = 0; j < boxNebListSize; j++)
        {
            boxIndex = boxNebList[j];
            
            for (k = 0; k < boxes->boxNAtoms[boxIndex]; k++)
            {
                visIndex = boxes->boxAtoms[boxIndex][k];
                index2 = visibleAtoms[visIndex];
                
                if (index >= index2) continue;
                
                specb = specie[index2];
                
                if (bondMinArray[speca*NSpecies+specb] == 0.0 && bondMaxArray[speca*NSpecies+specb] == 0.0)
                    continue;
                
                /* atomic separation */
                sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                         pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                
                sep = sqrt(sep2);
                
                /* check if these atoms are bonded */
                if (sep >= bondMinArray[speca*NSpecies+specb] && sep <= bondMaxArray[speca*NSpecies+specb])
                {
                    if (NBondsArray[i] + 1 == maxBondsPerAtom)
                    {
                        printf("ERROR: maxBondsPerAtom exceeded\n");
                        return Py_BuildValue("i", 1);
                    }
                    
                    bondArray[count] = visIndex;
                    NBondsArray[i]++;
                    
                    /* separation vector */
                    atomSeparationVector(sepVec, pos[3*index], pos[3*index+1], pos[3*index+2], 
                                         pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                    
                    bondVectorArray[3*count] = sepVec[0] / 2.0;
                    bondVectorArray[3*count+1] = sepVec[1] / 2.0;
                    bondVectorArray[3*count+2] = sepVec[2] / 2.0;
                    
                    bondSpecieCounter[speca*NSpecies+specb]++;
                    
                    count++;
                }
            }
        }
    }
    
#ifdef DEBUG
    printf("  N BONDS TOT: %d\n", count);
#endif
    
    /* free */
    freeBoxes(boxes);
    
    return Py_BuildValue("i", 0);
}

/*******************************************************************************
 * Calculate displacement vectors
 *******************************************************************************/
static PyObject*
calculateDisplacementVectors(PyObject *self, PyObject *args)
{
    int NVisible, *visibleAtoms, *PBC;
    double *pos, *refPos, *cellDims, *bondVectorArray;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *refPosIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *bondVectorArrayIn=NULL;
    PyArrayObject *drawBondVector=NULL;
    int i, numBonds;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, &PyArray_Type, &refPosIn, 
            &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &PyArray_Type, &bondVectorArrayIn, &PyArray_Type, &drawBondVector))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisible = (int) PyArray_DIM(visibleAtomsIn, 0);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(refPosIn)) return NULL;
    refPos = pyvector_to_Cptr_double(refPosIn);
    
    if (not_doubleVector(bondVectorArrayIn)) return NULL;
    bondVectorArray = pyvector_to_Cptr_double(bondVectorArrayIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_intVector(drawBondVector)) return NULL;
    
    /* main loop */
    numBonds = 0;
    for (i = 0; i < NVisible; i++)
    {
        int index;
        double sepVec[3], sep2;
        
        index = visibleAtoms[i];
        
        /* separation vector */
        atomSeparationVector(sepVec, pos[3*index], pos[3*index+1], pos[3*index+2], 
                             refPos[3*index], refPos[3*index+1], refPos[3*index+2], 
                             cellDims[0], cellDims[1], cellDims[2], 
                             PBC[0], PBC[1], PBC[2]);
        
        bondVectorArray[3*i] = sepVec[0];
        bondVectorArray[3*i+1] = sepVec[1];
        bondVectorArray[3*i+2] = sepVec[2];
        
        sep2 = sepVec[0] * sepVec[0] + sepVec[1] * sepVec[1] + sepVec[2] * sepVec[2];
        if (sep2 < 0.04) // don't show displacements smaller than 0.2
        {
            IIND1(drawBondVector, i) = 0;
        }
        else 
        {
            IIND1(drawBondVector, i) = 1;
            numBonds++;
        }
    }
    
    return Py_BuildValue("i", numBonds);
}
