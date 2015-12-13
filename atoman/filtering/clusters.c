/*******************************************************************************
 ** Find clusters of atoms and associated functions
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "visclibs/boxeslib.h"
#include "visclibs/utilities.h"
#include "visclibs/array_utils.h"


static PyObject* findClusters(PyObject*, PyObject *);
static PyObject* prepareClusterToDrawHulls(PyObject*, PyObject*);
static int findNeighbours(int, int, int, int *, double *, double, struct Boxes *, double *, int *);
static int findNeighboursUnapplyPBC(int, int, int, int, int *, double *, double, double *, int *, int *);
static void setAppliedPBCs(int *, int *);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"findClusters", findClusters, METH_VARARGS, "Find clusters of (visible) atoms"},
    {"prepareClusterToDrawHulls", prepareClusterToDrawHulls, METH_VARARGS, "Prepare clusters for drawing convex hulls"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_clusters(void)
{
    (void)Py_InitModule("_clusters", methods);
    import_array();
}

/*******************************************************************************
 * Find clusters
 *******************************************************************************/
static PyObject*
findClusters(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, *clusterArray, *PBC, minClusterSize, maxClusterSize, *results, NScalars, NVectors;
    double *pos, neighbourRad, *cellDims, *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *clusterArrayIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *resultsIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, j, index, NClusters, numInCluster;
    int maxNumInCluster, boxstat;
    double nebRad2, approxBoxWidth;
    double *visiblePos;
    struct Boxes *boxes;
    int *NAtomsCluster, clusterIndex;
    int *NAtomsClusterNew;
    int NVisible, count;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!dO!O!iiO!iO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, &PyArray_Type, &clusterArrayIn,
            &neighbourRad, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &minClusterSize, &maxClusterSize, &PyArray_Type, &resultsIn,
			&NScalars, &PyArray_Type, &fullScalarsIn, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) PyArray_DIM(visibleAtomsIn, 0);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_intVector(clusterArrayIn)) return NULL;
    clusterArray = pyvector_to_Cptr_int(clusterArrayIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_intVector(resultsIn)) return NULL;
    results = pyvector_to_Cptr_int(resultsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    /* construct visible pos array */
    visiblePos = malloc(3 * NVisibleIn * sizeof(double));
    if (visiblePos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate visiblePos");
        return NULL;
    }
    
    for (i = 0; i < NVisibleIn; i++)
    {
        int index = visibleAtoms[i];
        int ind3 = 3 * index;
        int i3 = 3 * i;
        visiblePos[i3    ] = pos[ind3    ];
        visiblePos[i3 + 1] = pos[ind3 + 1];
        visiblePos[i3 + 2] = pos[ind3 + 2];
    }
    
    /* box visible atoms */
    approxBoxWidth = neighbourRad;
    boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
    if (boxes == NULL)
    {
        free(visiblePos);
        return NULL;
    }
    boxstat = putAtomsInBoxes(NVisibleIn, visiblePos, boxes);
    if (boxstat)
    {
        free(visiblePos);
        return NULL;
    }
    
    nebRad2 = neighbourRad * neighbourRad;
    
    /* initialise clusters array */
    for (i = 0; i < NVisibleIn; i++) clusterArray[i] = -1;
    
    /* allocate NAtomsCluster */
    NAtomsCluster = calloc(NVisibleIn, sizeof(int));
    if (NAtomsCluster == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate NAtomsCluster");
        freeBoxes(boxes);
        free(visiblePos);
        return NULL;
    }
    
    /* loop over atoms */
    maxNumInCluster = -9999;
    NClusters = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        /* skip atom if already allocated */
        if (clusterArray[i] == -1)
        {
            clusterArray[i] = NClusters;
            numInCluster = 1;
            
            /* recursive search for cluster atoms */
            numInCluster = findNeighbours(i, clusterArray[i], numInCluster, clusterArray, visiblePos, nebRad2, boxes, cellDims, PBC);
            if (numInCluster < 0)
            {
                free(NAtomsCluster);
                freeBoxes(boxes);
                free(visiblePos);
            }
            maxNumInCluster = (numInCluster > maxNumInCluster) ? numInCluster : maxNumInCluster;
            NAtomsCluster[NClusters++] = numInCluster;
        }
    }
    
    free(visiblePos);
    
    NAtomsClusterNew = calloc(NClusters, sizeof(int));
    if (NAtomsClusterNew == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate NAtomsClusterNew");
        free(NAtomsCluster);
        freeBoxes(boxes);
        return NULL;
    }
    
    /* loop over visible atoms to see if their cluster has more than the min number */
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        clusterIndex = clusterArray[i];
        index = visibleAtoms[i];
        
        numInCluster = NAtomsCluster[clusterIndex];
        if (numInCluster < minClusterSize) continue;
        if (maxClusterSize >= minClusterSize && numInCluster > maxClusterSize) continue;
        
        visibleAtoms[NVisible] = index;
        clusterArray[NVisible] = clusterIndex;
        NAtomsClusterNew[clusterIndex]++;
        
        /* handle full scalars array */
        for (j = 0; j < NScalars; j++)
            fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
        
        for (j = 0; j < NVectors; j++)
        {
            int addr = NVisibleIn * j;
            int addr1 = addr + NVisible;
            int addr2 = addr + i;
            DIND2(fullVectors, addr1, 0) = DIND2(fullVectors, addr2, 0);
            DIND2(fullVectors, addr1, 1) = DIND2(fullVectors, addr2, 1);
            DIND2(fullVectors, addr1, 2) = DIND2(fullVectors, addr2, 2);
        }
        
        NVisible++;
    }
    
    /* how many clusters now */
    count = 0;
    for (i = 0; i < NClusters; i++) if (NAtomsClusterNew[i] > 0) count++;
    
    /* store results */
    results[0] = NVisible;
    results[1] = count;
    
    free(NAtomsClusterNew);
    free(NAtomsCluster);
    freeBoxes(boxes);
    
    return Py_BuildValue("i", 0);
}


/*******************************************************************************
 * recursive search for neighbouring defects
 *******************************************************************************/
static int findNeighbours(int index, int clusterID, int numInCluster, int* atomCluster, double *pos, double maxSep2, 
                          struct Boxes *boxes, double *cellDims, int *PBC)
{
    int i, j, index2, boxNebListSize;
    int boxIndex, boxNebList[27];
    double sep2;
    
    
    /* box of primary atom */
    boxIndex = boxIndexOfAtom(pos[3*index], pos[3*index+1], pos[3*index+2], boxes);
    if (boxIndex < 0) return -1;
    
    /* find neighbouring boxes */
    boxNebListSize = getBoxNeighbourhood(boxIndex, boxNebList, boxes);
    
    /* loop over neighbouring boxes */
    for (i = 0; i < boxNebListSize; i++)
    {
        boxIndex = boxNebList[i];
        
        for (j=0; j<boxes->boxNAtoms[boxIndex]; j++)
        {
            index2 = boxes->boxAtoms[boxIndex][j];
            
            /* skip itself or if already searched */
            if ((index == index2) || (atomCluster[index2] != -1))
                continue;
            
            /* calculate separation */
            sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                    cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
            
            /* check if neighbours */
            if (sep2 < maxSep2)
            {
                atomCluster[index2] = clusterID;
                numInCluster++;
                
                /* search for neighbours to this new cluster atom */
                numInCluster = findNeighbours(index2, clusterID, numInCluster, atomCluster, pos, maxSep2, boxes, cellDims, PBC);
                if (numInCluster < 0) return -1;
            }
        }
    }
    
    return numInCluster;
}


/*******************************************************************************
 * Prepare cluster to draw hulls (ie unapply PBCs)
 *******************************************************************************/
static PyObject*
prepareClusterToDrawHulls(PyObject *self, PyObject *args)
{
    int N, *PBC, *appliedPBCs;
    double *pos, *cellDims, neighbourRadius;
    PyArrayObject *posIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *appliedPBCsIn=NULL;
    
    int i, numInCluster, NClusters;
    int *clusterArray;
    double nebRad2;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "iO!O!O!O!d", &N, &PyArray_Type, &posIn, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, 
            &PyArray_Type, &appliedPBCsIn, &neighbourRadius))
        return NULL;
    
    if (not_intVector(appliedPBCsIn)) return NULL;
    appliedPBCs = pyvector_to_Cptr_int(appliedPBCsIn);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    /* is there any point boxing this? don't think so */
    
    nebRad2 = neighbourRadius * neighbourRadius;
    
    /* initialise clusters array */
    clusterArray = malloc(N * sizeof(int));
    if (clusterArray == NULL)
    {
        printf("ERROR: could not allocate cluster array\n");
        exit(50);
    }
    for (i = 0; i < N; i++) clusterArray[i] = -1;
    
    NClusters = 0;
    numInCluster = 0;
    for (i = 0; i < N; i++)
    {
        /* skip atom if already allocated */
        if (clusterArray[i] == -1)
        {
            clusterArray[i] = NClusters;
            
            numInCluster = 1;

            /* recursive search for cluster atoms */
            numInCluster = findNeighboursUnapplyPBC(N, i, clusterArray[i], numInCluster, clusterArray, pos, 
                                                    nebRad2, cellDims, PBC, appliedPBCs);
            
            NClusters++;
        }
    }
    
    if (numInCluster != N)
    {
        printf("ERROR: SOME CLUSTER ATOMS ARE MISSING: %d %d (%d)\n", N, numInCluster, NClusters);
    }
    
    free(clusterArray);
    
    return Py_BuildValue("i", 0);
}


/*******************************************************************************
 * recursive search for neighbouring defects unapplying PBCs as going along
 *******************************************************************************/
static int findNeighboursUnapplyPBC(int NAtoms, int index, int clusterID, int numInCluster, int* atomCluster, double *pos, double maxSep2, 
                                    double *cellDims, int *PBC, int *appliedPBCs)
{
    int j, index2;
    double sep2;
    int localPBCsApplied[3];
    

    /* loop over atoms */
    for (index2=0; index2<NAtoms; index2++)
    {
        /* skip itself or if already searched */
        if ((index == index2) || (atomCluster[index2] != -1))
            continue;
        
        /* calculate separation */
        sep2 = atomicSeparation2PBCCheck(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                         pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2],
                                         localPBCsApplied);
        
        /* check if neighbours */
        if (sep2 < maxSep2)
        {
            for (j=0; j<3; j++)
            {
                if (localPBCsApplied[j])
                {
                    if (pos[3*index2+j] < 0.5 * cellDims[j])
                    {
                        pos[3*index2+j] += cellDims[j];
                    }
                    else
                    {
                        pos[3*index2+j] -= cellDims[j];
                    }
                }
            }
            
            setAppliedPBCs(localPBCsApplied, appliedPBCs);
            
            atomCluster[index2] = clusterID;
            numInCluster++;
            
            /* search for neighbours to this new cluster atom */
            numInCluster = findNeighboursUnapplyPBC(NAtoms, index2, clusterID, numInCluster, atomCluster, pos, maxSep2, cellDims, PBC, appliedPBCs);
        }
    }
    
    return numInCluster;
}


/*******************************************************************************
 * set applied PBC
 *******************************************************************************/
static void setAppliedPBCs(int *PBC, int *appliedPBCs)
{
    if (PBC[0] == 1 && PBC[1] == 0 && PBC[2] == 0)
    {
        appliedPBCs[0] = 1;
    }
    else if (PBC[0] == 0 && PBC[1] == 1 && PBC[2] == 0)
    {
        appliedPBCs[1] = 1;
    }
    else if (PBC[0] == 0 && PBC[1] == 0 && PBC[2] == 1)
    {
        appliedPBCs[2] = 1;
    }
    else if (PBC[0] == 1 && PBC[1] == 1 && PBC[2] == 0)
    {
        appliedPBCs[3] = 1;
    }
    else if (PBC[0] == 1 && PBC[1] == 0 && PBC[2] == 1)
    {
        appliedPBCs[4] = 1;
    }
    else if (PBC[0] == 0 && PBC[1] == 1 && PBC[2] == 1)
    {
        appliedPBCs[5] = 1;
    }
    else if (PBC[0] == 1 && PBC[1] == 1 && PBC[2] == 1)
    {
        appliedPBCs[6] = 1;
    }
}
