
/*******************************************************************************
 ** Adaptive Common Neighbour Analysis
 ** A. Stutowski. Modelling Simul. Mater. Sci. Eng. 20 (2012) 045021
 ** Adapted from http://asa.ovito.org/
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "visclibs/boxeslib.h"
#include "visclibs/neb_list.h"
#include "visclibs/utilities.h"
#include "visclibs/array_utils.h"
#include "filtering/atom_structure.h"
#include "visclibs/constants.h"
#include "gui/preferences.h"

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

#define MAX_REQUIRED_NEBS 14
#define MIN_REQUIRED_NEBS 12


/* function prototypes */
static PyObject* adaptiveCommonNeighbourAnalysis(PyObject*, PyObject*);
static int compare_two_nebs(const void *, const void *);
static int analyseAtom(int, struct NeighbourList2 *);
static int checkForNeighbourBond(int, int, struct NeighbourList2 *, double);
static void setNeighbourBond(unsigned int *, int, int, int);
static int findCommonNeighbours(unsigned int *, int, unsigned int *);
static int findNeighbourBonds(unsigned int *, unsigned int, int, unsigned int *);
static int calcMaxChainLength(unsigned int *, int);
static int getAdjacentBonds(unsigned int, unsigned int *, int *, unsigned int *, unsigned int *);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef module_methods[] = {
    {"adaptiveCommonNeighbourAnalysis", adaptiveCommonNeighbourAnalysis, METH_VARARGS, "Run Adaptive Common Neighbour Analysis"},
    {NULL, NULL, 0, NULL}
};


/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
MOD_INIT(_acna)
{
    PyObject *mod;
    
    MOD_DEF(mod, "_acna", "Adaptive common neighbour analysis", module_methods)
    if (mod == NULL)
        return MOD_ERROR_VAL;
    
    import_array();
    
    return MOD_SUCCESS_VAL(mod);
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

/*******************************************************************************
 ** perform adaptive common neighbour analysis
 *******************************************************************************/
static PyObject*
adaptiveCommonNeighbourAnalysis(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, *PBC, NScalars, *counters, filteringEnabled;
    int *structureVisibility, NVectors;
    double *pos, *scalars, *cellDims, *fullScalars, maxBondDistance;
    PyArrayObject *posIn=NULL;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *countersIn=NULL;
    PyArrayObject *structureVisibilityIn=NULL;
    PyArrayObject *scalarsIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, NVisible, status;
    double *visiblePos, approxBoxWidth, maxSep2;
    struct Boxes *boxes;
    struct NeighbourList2 *nebList;
    
/* parse and check arguments from Python */
    
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iO!dO!iO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, &PyArray_Type, &scalarsIn,
            &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &NScalars, &PyArray_Type, &fullScalarsIn, &maxBondDistance, &PyArray_Type,
            &countersIn, &filteringEnabled, &PyArray_Type, &structureVisibilityIn, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) PyArray_DIM(visibleAtomsIn, 0);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(scalarsIn)) return NULL;
    scalars = pyvector_to_Cptr_double(scalarsIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_intVector(countersIn)) return NULL;
    counters = pyvector_to_Cptr_int(countersIn);
    
    if (not_intVector(structureVisibilityIn)) return NULL;
    structureVisibility = pyvector_to_Cptr_int(structureVisibilityIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
/* first we construct neighbour list for each atom, containing indexes and separations */
    
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
    approxBoxWidth = maxBondDistance;
    maxSep2 = maxBondDistance * maxBondDistance;
    boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
    if (boxes == NULL)
    {
        free(visiblePos);
        return NULL;
    }
    status = putAtomsInBoxes(NVisibleIn, visiblePos, boxes);
    if (status)
    {
        free(visiblePos);
        return NULL;
    }
    
    /* create neighbour list */
    nebList = constructNeighbourList2(NVisibleIn, visiblePos, boxes, cellDims, PBC, maxSep2);
    
    /* only required for building neb list */
    freeBoxes(boxes);
    free(visiblePos);
    
    if (nebList == NULL) return NULL;
    
/* now we order the neighbour lists by separation */
    
    /* if less than min neighbours, mark as disordered!!! */
    
    
    /* sort neighbours by distance */
    #pragma omp parallel for num_threads(prefs_numThreads)
    for (i = 0; i < NVisibleIn; i++)
        qsort(nebList[i].neighbour, nebList[i].neighbourCount, sizeof(struct Neighbour), compare_two_nebs);
    
/* classify atoms */
    
    #pragma omp parallel for num_threads(prefs_numThreads)
    for (i = 0; i < NVisibleIn; i++)
    {
        int atomStructure;
        
        atomStructure = analyseAtom(i, nebList);
        scalars[i] = (double) atomStructure;
        #pragma omp atomic
        counters[atomStructure]++;
    }
    
/* there should be option to only show atoms of specific structure type */
    
    if (filteringEnabled)
    {
        NVisible = 0;
        for (i = 0; i < NVisibleIn; i++)
        {
            int atomStructure;
            
            atomStructure = (int) scalars[i];
            if (structureVisibility[atomStructure])
            {
                int j;
                
                visibleAtoms[NVisible] = visibleAtoms[i];
                scalars[NVisible++] = scalars[i];
                
                /* handle full scalars array */
                for (j = 0; j < NScalars; j++)
                {
                    fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
                }
                
                for (j = 0; j < NVectors; j++)
                {
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
                }
            }
        }
    }
    else NVisible = NVisibleIn;
    
/* tidy up */
    
    freeNeighbourList2(nebList, NVisibleIn);
    
    return Py_BuildValue("i", NVisible);
}

/*******************************************************************************
 ** classify atom
 *******************************************************************************/
static int analyseAtom(int mainIndex, struct NeighbourList2 *nebList)
{
    int i, j, nn, ok, visInd1, visInd2;
    double localScaling, localCutoff;
    double localScalingSum;
    
    
    /* check we have the minimum number of neighbours */
    if (nebList[mainIndex].neighbourCount < MIN_REQUIRED_NEBS)
        return ATOM_STRUCTURE_DISORDERED;
    
/* first we test for FCC, HCP, Icosohedral (12 1NN) */
    
    /* number of neighbours to test for */
    nn = 12;
    
    /* check enough nebs */
    if (nebList[mainIndex].neighbourCount < nn)
        return ATOM_STRUCTURE_DISORDERED;
    
    /* compute local cutoff */
    localScaling = 0.0;
    for (i = 0; i < nn; i++)
    {
        localScaling += nebList[mainIndex].neighbour[i].separation;
    }
    localScaling /= nn;
    localCutoff = localScaling * (1.0 + CONST_SQRT2) / 2.0;
    
    /* at this point I feel like we should check that the 12 NN are within localCutoff ????? */
    ok = 1;
    for (i = 0; i < nn; i++)
    {
        if (nebList[mainIndex].neighbour[i].separation > localCutoff)
        {
            ok = 0;
            break;
        }
    }
    
    if (ok)
    {
        int n421;
        int n422;
        int n555;
        unsigned int neighbourArray[MAX_REQUIRED_NEBS] = {0};
        
        /* determine bonding between neighbours, based on local cutoff */
        for (i = 0; i < nn; i++)
        {
            visInd1 = nebList[mainIndex].neighbour[i].index;
            setNeighbourBond(neighbourArray, i, i, 0);
            for (j = i + 1; j < nn; j++)
            {
                visInd2 = nebList[mainIndex].neighbour[j].index;
                setNeighbourBond(neighbourArray, i, j, checkForNeighbourBond(visInd1, visInd2, nebList, localCutoff));
            }
        }
        
        n421 = 0;
        n422 = 0;
        n555 = 0;
        for (i = 0; i < nn; i++)
        {
            int numCommonNeighbours;
            int numNeighbourBonds;
            int maxChainLength;
            unsigned int commonNeighbours;
            unsigned int neighbourBonds[MAX_REQUIRED_NEBS*MAX_REQUIRED_NEBS] = {0};
            
            /* number of common neighbours */
            numCommonNeighbours = findCommonNeighbours(neighbourArray, i, &commonNeighbours);
            if (numCommonNeighbours != 4 && numCommonNeighbours != 5)
                break;
            
            /* number of bonds among common neighbours */
            numNeighbourBonds = findNeighbourBonds(neighbourArray, commonNeighbours, nn, neighbourBonds);
            if (numNeighbourBonds != 2 && numNeighbourBonds != 5)
                break;
            
            /* number of bonds in the longest continuous chain */
            maxChainLength = calcMaxChainLength(neighbourBonds, numNeighbourBonds);
            if (numCommonNeighbours == 4 && numNeighbourBonds == 2)
            {
                if (maxChainLength == 1) n421++;
                else if (maxChainLength == 2) n422++;
                else break;
            }
            else if (numCommonNeighbours == 5 && numNeighbourBonds == 5 && maxChainLength == 5) n555++;
            else break;
        }
        if (n421 == 12) return ATOM_STRUCTURE_FCC;
        else if (n421 == 6 && n422 == 6) return ATOM_STRUCTURE_HCP;
        else if (n555 == 12) return ATOM_STRUCTURE_ICOSAHEDRAL;
    }
    
/* next we test for BCC (8 1NN + 6 2NN) */
    
    /* number of neighbours to test for */
    nn = 14;
    
    /* check enough nebs */
    if (nebList[mainIndex].neighbourCount < nn)
        return ATOM_STRUCTURE_DISORDERED;
    
    /* compute local cutoff */
    localScaling = 0.0;
    for (i = 0; i < 8; i++)
    {
        localScaling += nebList[mainIndex].neighbour[i].separation;
    }
    localScaling /= 8.0;
    
    localScalingSum = 0.0;
    for (i = 8; i < 14; i++)
    {
        localScalingSum += nebList[mainIndex].neighbour[i].separation;
    }
    localScalingSum /= 6.0;
    
    localCutoff = (1.0 + CONST_SQRT2) / 4.0 * (2.0 / CONST_SQRT3 * localScaling + localScalingSum); // divide by 4 not 2 as in the paper
    
    /* at this point I feel like we should check that the 12 NN are within localCutoff ????? */
    ok = 1;
    for (i = 0; i < nn; i++)
    {
        if (nebList[mainIndex].neighbour[i].separation > localCutoff)
        {
            ok = 0;
            break;
        }
    }
    
    if (ok)
    {
        int n444;
        int n666;
        unsigned int neighbourArray[MAX_REQUIRED_NEBS] = {0};
        
        /* determine bonding between neighbours, based on local cutoff */
        for (i = 0; i < nn; i++)
        {
            visInd1 = nebList[mainIndex].neighbour[i].index;
            setNeighbourBond(neighbourArray, i, i, 0);
            for (j = i + 1; j < nn; j++)
            {
                visInd2 = nebList[mainIndex].neighbour[j].index;
                setNeighbourBond(neighbourArray, i, j, checkForNeighbourBond(visInd1, visInd2, nebList, localCutoff));
            }
        }
        
        n444 = 0;
        n666 = 0;
        for (i = 0; i < nn; i++)
        {
            int numCommonNeighbours;
            int numNeighbourBonds;
            int maxChainLength;
            unsigned int commonNeighbours;
            unsigned int neighbourBonds[MAX_REQUIRED_NEBS*MAX_REQUIRED_NEBS] = {0};
            
            /* number of common neighbours */
            numCommonNeighbours = findCommonNeighbours(neighbourArray, i, &commonNeighbours);
            if (numCommonNeighbours != 4 && numCommonNeighbours != 6)
                break;
            
            /* number of bonds among common neighbours */
            numNeighbourBonds = findNeighbourBonds(neighbourArray, commonNeighbours, nn, neighbourBonds);
            if (numNeighbourBonds != 4 && numNeighbourBonds != 6)
                break;
            
            /* number of bonds in the longest continuous chain */
            maxChainLength = calcMaxChainLength(neighbourBonds, numNeighbourBonds);
            if (numCommonNeighbours == 4 && numNeighbourBonds == 4 && maxChainLength == 4) n444++;
            else if (numCommonNeighbours == 6 && numNeighbourBonds == 6 && maxChainLength == 6) n666++;
            else break;
        }
        if (n666 == 8 && n444 == 6) return ATOM_STRUCTURE_BCC;
    }
    
    return ATOM_STRUCTURE_DISORDERED;
}

/*******************************************************************************
 ** find all chains of bonds between common neighbours and determine the length
 ** of the longest continuous chain
 *******************************************************************************/
static int calcMaxChainLength(unsigned int *neighbourBonds, int numBonds)
{
    int maxChainLength;
    
    
    maxChainLength = 0;
    while (numBonds)
    {
        int clusterSize;
        unsigned int atomsToProcess, atomsProcessed;
        
        
        /* make a new cluster starting with the first remaining bond to be processed */
        numBonds--;
        
        /* initialise some variables */
        atomsToProcess = neighbourBonds[numBonds];
        atomsProcessed = 0;
        clusterSize = 1;
        
        do 
        {
            unsigned int nextAtom;
            /* determine number of trailing 0-bits in atomsToProcess
             * starting with least significant bit position
             */
#if defined(__GNUC__)
            int nextAtomIndex = __builtin_ctz(atomsToProcess);
#elif defined(_MSC_VER)
            unsigned long nextAtomIndex;
            _BitScanForward(&nextAtomIndex, atomsToProcess);
#else
            #error "Your C compiler is not supported."
#endif
            if (nextAtomIndex < 0 || nextAtomIndex >= 32)
            {
                printf("nextAtomIndex error (%d)\n", nextAtomIndex);
                exit(98);
            }
            
            nextAtom = 1 << nextAtomIndex;
            atomsProcessed |= nextAtom;
            atomsToProcess &= ~nextAtom;
            clusterSize += getAdjacentBonds(nextAtom, neighbourBonds, &numBonds, &atomsToProcess, &atomsProcessed);
        }
        while (atomsToProcess);
        
        if (clusterSize > maxChainLength)
            maxChainLength = clusterSize;
    }
    
    return maxChainLength;
}

/*******************************************************************************
 ** find all chains of bonds
 *******************************************************************************/
static int getAdjacentBonds(unsigned int atom, unsigned int *bondsToProcess, int *numBonds, unsigned int *atomsToProcess, unsigned int *atomsProcessed)
{
    int adjacentBonds, b;
    
    
    adjacentBonds = 0;
    for (b = *numBonds - 1; b >= 0; b--)
    {
        if (atom & *bondsToProcess)
        {
            ++adjacentBonds;
//            *atomsToProcess |= *bondsToProcess & (~*atomsProcessed);
            *atomsToProcess = *atomsToProcess | (*bondsToProcess & (~*atomsProcessed));
            memmove(bondsToProcess, bondsToProcess + 1, sizeof(unsigned int) * b);
            *numBonds = *numBonds - 1;
        }
        else ++bondsToProcess;
    }
    
    return adjacentBonds;
}


/*******************************************************************************
 ** find bonds between common nearest neighbours
 *******************************************************************************/
static int findNeighbourBonds(unsigned int *neighbourArray, unsigned int commonNeighbours, int numNeighbours, unsigned int *neighbourBonds)
{
    int ni1, n;
    int numBonds;
    int neb_size;
    unsigned int nib[MAX_REQUIRED_NEBS] = {0};
    int nibn;
    unsigned int ni1b;
    unsigned int b;
    
    
    neb_size = MAX_REQUIRED_NEBS * MAX_REQUIRED_NEBS;
    
    numBonds = 0;
    nibn = 0;
    ni1b = 1;
    for (ni1 = 0; ni1 < numNeighbours; ni1++, ni1b <<= 1)
    {
        if (commonNeighbours & ni1b)
        {
            b = commonNeighbours & neighbourArray[ni1];
            for (n = 0; n < nibn; n++)
            {
                if (b & nib[n])
                {
                    if (numBonds > neb_size)
                    {
                        printf("ERROR: num bonds exceeds limit (findNeighbourBonds)\n");
                        exit(58);
                    }
                    neighbourBonds[numBonds++] = ni1b | nib[n];
                }
            }
            
            nib[nibn++] = ni1b;
        }
    }
    
    return numBonds;
}

/*******************************************************************************
 ** find common neighbours
 *******************************************************************************/
static int findCommonNeighbours(unsigned int *neighbourArray, int neighbourIndex, unsigned int *commonNeighbours)
{
#ifdef __GNUC__
    *commonNeighbours = neighbourArray[neighbourIndex];
    
    /* Count the number of bits set in neighbor bit field. */
    return __builtin_popcount(*commonNeighbours); // GNU g++ specific
#else
    unsigned int v;
    
    *commonNeighbours = neighbourArray[neighbourIndex];
    
    /* Count the number of bits set in neighbor bit field. */
    v = *commonNeighbours - ((*commonNeighbours >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
}

/*******************************************************************************
 ** check if two neighbours are bonded
 *******************************************************************************/
static int checkForNeighbourBond(int visInd1, int visInd2, struct NeighbourList2 *nebList, double cutoff)
{
    int i, bonded;
    
    
    bonded = 0;
    for (i = 0; i < nebList[visInd1].neighbourCount; i++)
    {
        if (nebList[visInd1].neighbour[i].index == visInd2 && nebList[visInd1].neighbour[i].separation <= cutoff)
        {
            bonded = 1;
            break;
        }
    }
    
    return bonded;
}

/*******************************************************************************
 ** set neighbour bond
 *******************************************************************************/
static void setNeighbourBond(unsigned int *neighbourArray, int index1, int index2, int bonded)
{
    if (bonded)
    {
        neighbourArray[index1] |= (1<<index2);
        neighbourArray[index2] |= (1<<index1);
    }
    else
    {
        neighbourArray[index1] &= ~(1<<index2);
        neighbourArray[index2] &= ~(1<<index1);
    }
}
