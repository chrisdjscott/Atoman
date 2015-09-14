/*******************************************************************************
 ** Copyright Chris Scott 2015
 ** Bubbles C functions
 *******************************************************************************/

//#define DEBUG

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "boxeslib.h"
#include "neb_list.h"
#include "array_utils.h"


static PyObject* putBubbleAtomsInClusters(PyObject*, PyObject*);
static int compare_two_nebs(const void *, const void *);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"putBubbleAtomsInClusters", putBubbleAtomsInClusters, METH_VARARGS, "Associate bubble atoms with a vacancy cluster"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_bubbles(void)
{
    (void)Py_InitModule("_bubbles", methods);
    import_array();
}

/*******************************************************************************
 ** Associate bubble atoms with a vacancy cluster
 *******************************************************************************/
static PyObject*
putBubbleAtomsInClusters(PyObject *self, PyObject *args)
{
    int NAtoms, NRefAtoms, driftCompensation, NBubbleAtoms, NVacancies, NVacClusters;
    double vacBubbleRad;
    PyArrayObject *posIn=NULL;
    PyArrayObject *refPosIn=NULL;
    PyArrayObject *driftVectorIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *pbcIn=NULL;
    PyArrayObject *bubbleAtomIndexesIn=NULL;
    PyArrayObject *vacanciesIn=NULL;
    PyArrayObject *vacClusterIndexesIn=NULL;
    PyObject *result=NULL;
    
    
#ifdef DEBUG
    printf("BUBBLESC: Putting bubble atoms into vacancy clusters\n");
#endif
    
    /* parse arguments */
    if (PyArg_ParseTuple(args, "iO!iO!iO!O!O!iO!iO!iO!d", &NAtoms, &PyArray_Type, &posIn, &NRefAtoms, &PyArray_Type, &refPosIn,
            &driftCompensation, &PyArray_Type, &driftVectorIn, &PyArray_Type, &cellDimsIn, &PyArray_Type, &pbcIn, &NBubbleAtoms,
            &PyArray_Type, &bubbleAtomIndexesIn, &NVacancies, &PyArray_Type, &vacanciesIn, &NVacClusters, &PyArray_Type,
            &vacClusterIndexesIn, &vacBubbleRad))
    {
        int *pbc, *bubbleAtomIndexes, *vacancies, *vacClusterIndexes;
        double *pos, *refPos, *refPosTmp, *cellDims, *driftVector;
        
        /* C pointers to Python arrays and checking types */
        if (not_doubleVector(posIn)) return NULL;
        pos = pyvector_to_Cptr_double(posIn);
        
        if (not_doubleVector(refPosIn)) return NULL;
        refPosTmp = pyvector_to_Cptr_double(refPosIn);
        
        if (not_doubleVector(driftVectorIn)) return NULL;
        driftVector = pyvector_to_Cptr_double(driftVectorIn);
        
        if (not_intVector(bubbleAtomIndexesIn)) return NULL;
        bubbleAtomIndexes = pyvector_to_Cptr_int(bubbleAtomIndexesIn);
        
        if (not_intVector(vacanciesIn)) return NULL;
        vacancies = pyvector_to_Cptr_int(vacanciesIn);
        
        if (not_intVector(vacClusterIndexesIn)) return NULL;
        vacClusterIndexes = pyvector_to_Cptr_int(vacClusterIndexesIn);
        
        if (not_doubleVector(cellDimsIn)) return NULL;
        cellDims = pyvector_to_Cptr_double(cellDimsIn);
        
        if (not_intVector(pbcIn)) return NULL;
        pbc = pyvector_to_Cptr_int(pbcIn);
        
        /* drift compensation - modify reference positions */
        if (driftCompensation)
        {
            int i;
            
#ifdef DEBUG
            printf("BUBBLESC:   Drift compensation: %lf %lf %lf\n", driftVector[0], driftVector[1], driftVector[2]);
#endif
            
            refPos = malloc(3 * NRefAtoms * sizeof(double));
            if (refPos == NULL)
            {
                PyErr_SetString(PyExc_MemoryError, "Could not allocate refPos");
                return NULL;
            }
            
            for (i = 0; i < NRefAtoms; i++)
            {
                int i3 = 3 * i;
                refPos[i3    ] = refPosTmp[i3    ] + driftVector[0];
                refPos[i3 + 1] = refPosTmp[i3 + 1] + driftVector[1];
                refPos[i3 + 2] = refPosTmp[i3 + 2] + driftVector[2];
            }
        }
        else refPos = refPosTmp;
        
        /* only continue if there is something to do */
        if (NVacancies > 0 && NBubbleAtoms > 0)
        {
            int i, numBubbles, listCount;
            npy_intp dims[1];
            int *atomCluster, *clusterCounter;
            PyObject *tuple=NULL;
            PyObject *bubbleList=NULL;
            PyArrayObject *bubbleMapper=NULL;
            double *vacPos, *bubbleAtomPos;
            struct NeighbourList2 *nebList;
            
#ifdef DEBUG
            printf("BUBBLESC:   Constructing neighbour lists for bubble atoms\n");
#endif
            
            /* vacancy positions */
            vacPos = malloc(3 * NVacancies * sizeof(double));
            if (vacPos == NULL)
            {
                if (driftCompensation) free(refPos);
                PyErr_SetString(PyExc_MemoryError, "Could not allocate vacPos");
                return NULL;
            }
            for (i = 0; i < NVacancies; i++)
            {
                int i3 = 3 * i;
                int indx3 = 3 * vacancies[i];
                vacPos[i3    ] = refPos[indx3    ];
                vacPos[i3 + 1] = refPos[indx3 + 1];
                vacPos[i3 + 2] = refPos[indx3 + 2];
            }
            
            /* bubble atom positions */
            bubbleAtomPos = malloc(3 * NBubbleAtoms * sizeof(double));
            if (bubbleAtomPos == NULL)
            {
                if (driftCompensation) free(refPos);
                free(vacPos);
                PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleAtomPos");
                return NULL;
            }
            for (i = 0; i < NBubbleAtoms; i++)
            {
                int i3 = 3 * i;
                int indx3 = 3 * bubbleAtomIndexes[i];
                bubbleAtomPos[i3    ] = pos[indx3    ];
                bubbleAtomPos[i3 + 1] = pos[indx3 + 1];
                bubbleAtomPos[i3 + 2] = pos[indx3 + 2];
            }
            
            /* we build a neighbour list for each bubble atom, containing neighbouring vacancies */
            nebList = constructNeighbourList2DiffPos(NBubbleAtoms, bubbleAtomPos, NVacancies, vacPos, cellDims, pbc, vacBubbleRad);
            free(vacPos);
            free(bubbleAtomPos);
            if (nebList == NULL) 
            {
                if (driftCompensation) free(refPos);
                return NULL;
            }
            
#ifdef DEBUG
            printf("BUBBLESC:   Sorting neighbour lists by separation\n");
#endif
            
            /* sort the neighbour list by separation */
            for (i = 0; i < NBubbleAtoms; i++)
                qsort(nebList[i].neighbour, nebList[i].neighbourCount, sizeof(struct Neighbour), compare_two_nebs);
            
#ifdef DEBUG
            printf("BUBBLESC:   Allocating arrays for results\n");
#endif
            
            /* allocate numpy array for storing cluster indexes of bubble atoms */
            atomCluster = malloc(NBubbleAtoms * sizeof(int));
            if (atomCluster == NULL)
            {
                PyErr_SetString(PyExc_MemoryError, "Could not allocate atomCluster");
                if (driftCompensation) free(refPos);
                return NULL;
            }
            for (i = 0; i < NBubbleAtoms; i++) atomCluster[i] = -1;
            
            /* counter for num bubble atoms in each cluster */
            clusterCounter = calloc(NVacClusters, sizeof(int));
            if (clusterCounter == NULL)
            {
                PyErr_SetString(PyExc_MemoryError, "Could not allocate clusterCounter");
                if (driftCompensation) free(refPos);
                free(atomCluster);
                return NULL;
            }
            
#ifdef DEBUG
            printf("BUBBLESC:   Associating bubble atoms with clusters\n");
#endif
            
            /* loop over bubble atoms and put in clusters (note it is possible for a bubble atom not to belong to a cluster!) */
            for (i = 0; i < NBubbleAtoms; i++)
            {
                if (nebList[i].neighbourCount > 0)
                {
                    struct Neighbour neb = nebList[i].neighbour[0];
                    if (neb.separation < vacBubbleRad)
                    {
                        int vacIndex = neb.index;
                        int clusterIndex = vacClusterIndexes[vacIndex];
                        atomCluster[i] = clusterIndex;
                        clusterCounter[clusterIndex]++;
                    }
                }
            }
            
            /* free */
            freeNeighbourList2(nebList, NBubbleAtoms);
            
            /* number of bubbles */
            numBubbles = 0;
            for (i = 0; i < NVacClusters; i++) if (clusterCounter[i] > 0) numBubbles++;
#ifdef DEBUG
            printf("BUBBLESC:   Number of bubbles: %d\n", numBubbles);
#endif
            
            /* list of numpy arrays containing the indices of the bubbles atoms */
            bubbleList = PyList_New(numBubbles);
            if (bubbleList == NULL)
            {
                free(atomCluster);
                free(clusterCounter);
                if (driftCompensation) free(refPos);
                return NULL;
            }
            
            /* which bubble is linked to which cluster (of vacancies) */
            dims[0] = numBubbles;
            bubbleMapper = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
            if (bubbleMapper == NULL)
            {
                free(atomCluster);
                free(clusterCounter);
                Py_DECREF(bubbleList);
                if (driftCompensation) free(refPos);
                return NULL;
            }
            
#ifdef DEBUG
            printf("BUBBLESC:   Constructing the result\n");
#endif
            
            /* construct the result */
            listCount = 0;
            for (i = 0; i < NVacClusters; i++)
            {
                if (clusterCounter[i] > 0)
                {
                    int j, count;
                    PyArrayObject *bubbleIndices=NULL;
                    
                    /* allocate array */
                    dims[0] = clusterCounter[i];
                    bubbleIndices = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
                    if (bubbleIndices == NULL)
                    {
                        free(atomCluster);
                        free(clusterCounter);
                        Py_DECREF(bubbleList);
                        Py_DECREF(bubbleMapper);
                        if (driftCompensation) free(refPos);
                        return NULL;
                    }
                    
                    /* add bubble atom indices */
                    count = 0;
                    for (j = 0; j < NBubbleAtoms; j++)
                    {
                        if (atomCluster[j] == i)
                        {
                            IIND1(bubbleIndices, count) = bubbleAtomIndexes[j];
                            count++;
                        }
                    }
                    
                    /* add to list */
                    PyList_SetItem(bubbleList, listCount, PyArray_Return(bubbleIndices));
                    
                    /* set mapper */
                    IIND1(bubbleMapper, listCount) = i;
                    
                    listCount++;
                }
            }
            
            /* free */
            free(atomCluster);
            free(clusterCounter);
            
            /* result */
            tuple = PyTuple_New(2);
            if (tuple == NULL)
            {
                if (driftCompensation) free(refPos);
                Py_DECREF(bubbleList);
                Py_DECREF(bubbleMapper);
                return NULL;
            }
            PyTuple_SetItem(tuple, 0, bubbleList);
            PyTuple_SetItem(tuple, 1, PyArray_Return(bubbleMapper));
            result = tuple;
        }
        else
        {
            printf("Add empty result...........\n");
            Py_INCREF(Py_None);
            result = Py_None;
        }
        
        /* free */
        if (driftCompensation) free(refPos);
    }
    
#ifdef DEBUG
    printf("BUBBLESC: End\n");
#endif
    
    return result;
}

/*******************************************************************************
 ** Function that compares two elements in a neighbour list
 *******************************************************************************/
static int compare_two_nebs(const void * a, const void * b)
{
    const struct Neighbour *n1 = a;
    const struct Neighbour *n2 = b;
    
    if (n1->separation < n2->separation) return -1;
    else if (n1->separation > n2->separation) return 1;
    else return 0;
}
