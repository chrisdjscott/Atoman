/*******************************************************************************
 ** Bubbles C functions
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//#define DEBUG

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "visclibs/utilities.h"
#include "visclibs/boxeslib.h"
#include "visclibs/neb_list.h"
#include "visclibs/array_utils.h"


static PyObject* identifyBubbles(PyObject*, PyObject*);
static PyObject* putBubbleAtomsInClusters(PyObject*, PyObject*);
static int classifyVacsAndInts(double, int, double*, int, double*, int*, double*, int*, int*, int*, int, int*);
static int identifySplitInterstitialsNew(int, int*, int, int*, int*, double*, double*, int*, double*, int*, double);
static int refineVacancies(int, int*, int, int*, int, int*, int, int*, double*, double*, int*, double*, int*, double, double);
static int findVacancyClusters(int, int*, double*, int*, int*, double, double*, int*, int*);
static int findDefectNeighbours(int, int, int, int*, double*, struct Boxes*, double, double*, int*);
static int getClusterIndexForBubbleAtoms(int, int*, double*, int, int*, double*, int*, int*, double*, int*, double, int, int*, int*);
static PyObject* constructBubbleResult(int, int, int*, int*, int*, int*, int, int*, int, int*);
static int compare_two_nebs(const void*, const void*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"identifyBubbles", identifyBubbles, METH_VARARGS, "Identify bubbles"},
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
 ** Find bubbles:
 **   - First identify vacancies and interstitials
 **   - Apply ACNA (if required)
 **   - Identify Splits
 **   - Associate H with vacs (associated if within vacancyBubbleRadius)
 **   - If an unassociated vac is within 'x' of an interstitial (splits included), remove from vacancies
 **   - Find clusters of vacancies using modified list
 **   - Associate H with vacs to identify bubbles... (as before...)
 *******************************************************************************/
static PyObject*
identifyBubbles(PyObject *self, PyObject *args)
{
    int NAtoms, NRefAtoms, driftCompensation, NBubbleAtoms;
    double vacBubbleRad, vacancyRadius, vacNebRad, vacIntRad;
    PyArrayObject *posIn=NULL;
    PyArrayObject *refPosIn=NULL;
    PyArrayObject *driftVectorIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *pbcIn=NULL;
    PyArrayObject *bubbleAtomIndexesIn=NULL;
    PyArrayObject *acnaArrayIn=NULL;
    PyObject *result=NULL;
    
    
#ifdef DEBUG
    printf("BUBBLESC: Identifying bubbles\n");
#endif
    
    /* parse arguments */
    if (PyArg_ParseTuple(args, "iO!iO!iO!O!O!iO!dO!ddd", &NAtoms, &PyArray_Type, &posIn, &NRefAtoms, &PyArray_Type, &refPosIn,
            &driftCompensation, &PyArray_Type, &driftVectorIn, &PyArray_Type, &cellDimsIn, &PyArray_Type, &pbcIn, &NBubbleAtoms,
            &PyArray_Type, &bubbleAtomIndexesIn, &vacBubbleRad, &PyArray_Type, &acnaArrayIn, &vacancyRadius, &vacNebRad, &vacIntRad))
    {
        int *pbc, *bubbleAtomIndexes, acnaArrayDim, status, counters[3];
        int *vacancies, *interstitials, *splitInterstitials=NULL;
        int NVacancies, NInterstitials, NSplitInterstitials;
        int *vacancyCluster, *bubbleAtomCluster, *NBubbleAtomsCluster;
        int *NVacanciesCluster, NVacancyClusters, NBubbles;
        double *pos, *refPos, *refPosTmp, *cellDims, *driftVector, *acnaArray;
        
#ifdef DEBUG
        printf("BUBBLESC:   Vacancy radius is %lf\n", vacancyRadius);
        printf("BUBBLESC:   Vacancy bubble radius is %lf\n", vacBubbleRad);
        printf("BUBBLESC:   Vacancy neighbour radius is %lf\n", vacNebRad);
        printf("BUBBLESC:   Vacancy-interstitial association radius is %lf\n", vacIntRad);
#endif
        
        /* C pointers to Python arrays and checking types */
        if (not_doubleVector(posIn)) return NULL;
        pos = pyvector_to_Cptr_double(posIn);
        
        if (not_doubleVector(refPosIn)) return NULL;
        refPosTmp = pyvector_to_Cptr_double(refPosIn);
        
        if (not_doubleVector(driftVectorIn)) return NULL;
        driftVector = pyvector_to_Cptr_double(driftVectorIn);
        
        if (not_intVector(bubbleAtomIndexesIn)) return NULL;
        bubbleAtomIndexes = pyvector_to_Cptr_int(bubbleAtomIndexesIn);
        
        if (not_doubleVector(cellDimsIn)) return NULL;
        cellDims = pyvector_to_Cptr_double(cellDimsIn);
        
        if (not_intVector(pbcIn)) return NULL;
        pbc = pyvector_to_Cptr_int(pbcIn);
        
        if (not_doubleVector(acnaArrayIn)) return NULL;
        acnaArray = pyvector_to_Cptr_double(acnaArrayIn);
        acnaArrayDim = (int) PyArray_DIM(acnaArrayIn, 0);
        
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
        
        /* allocate defect arrays */
        vacancies = malloc(NRefAtoms * sizeof(int));
        if (vacancies == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate vacancies");
            if (driftCompensation) free(refPos);
            return NULL;
        }
        
        interstitials = malloc(NAtoms * sizeof(int));
        if (interstitials == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate interstitials");
            if (driftCompensation) free(refPos);
            free(vacancies);
            return NULL;
        }
        
        /* basic defect classification (vacancies and interstitials) */
        status = classifyVacsAndInts(vacancyRadius, NAtoms, pos, NRefAtoms, refPos, pbc, cellDims, counters,
                vacancies, interstitials, NBubbleAtoms, bubbleAtomIndexes);
        if (status)
        {
            if (driftCompensation) free(refPos);
            free(vacancies);
            free(interstitials);
            return NULL;
        }
        
        /* unpack counters */
        NVacancies = counters[0];
        NInterstitials = counters[1];
        
#ifdef DEBUG
        printf("BUBBLESC:   Basic defect classification: %d vacancies; %d interstitials\n", NVacancies, NInterstitials);
#endif
        
        /* TODO: add ACNA stuff here! */
        
        
        
        /* Identify split interstitials */
        if (NVacancies > 0 && NInterstitials > 1)
        {
            /* allocate array */
            splitInterstitials = malloc(3 * NVacancies * sizeof(int));
            if (splitInterstitials == NULL)
            {
                PyErr_SetString(PyExc_MemoryError, "Could not allocate splitInterstitials");
                if (driftCompensation) free(refPos);
                free(vacancies);
                free(interstitials);
                return NULL;
            }
            
            /* identify splits */
            status = identifySplitInterstitialsNew(NVacancies, vacancies, NInterstitials, interstitials, splitInterstitials, pos, refPos, pbc,
                    cellDims, counters, vacancyRadius);
            if (status)
            {
                if (driftCompensation) free(refPos);
                free(vacancies);
                free(interstitials);
                free(splitInterstitials);
                return NULL;
            }
            
            /* unpack counters */
            NVacancies = counters[0];
            NInterstitials = counters[1];
            NSplitInterstitials = counters[2];
            
            /* if no splits, free the memory here */
            if (NSplitInterstitials == 0) free(splitInterstitials);
            
#ifdef DEBUG
            printf("BUBBLESC:   Defect count after split interstitial identification: %d vacancies; %d interstitials; %d split interstitials\n", 
                    NVacancies, NInterstitials, NSplitInterstitials);
#endif
        }
        else NSplitInterstitials = 0;
        
        /* Associate vacancies with bubble atoms */
        status = refineVacancies(NBubbleAtoms, bubbleAtomIndexes, NVacancies, vacancies, NInterstitials, interstitials,
                NSplitInterstitials, splitInterstitials, pos, refPos, counters, cellDims, pbc, vacBubbleRad, vacIntRad);
        if (status)
        {
            if (driftCompensation) free(refPos);
            free(vacancies);
            free(interstitials);
            if (NSplitInterstitials) free(splitInterstitials);
            return NULL;
        }
        
        NVacancies = counters[0];
        
#ifdef DEBUG
        printf("BUBBLESC:     Number of vacancies after refining: %d\n", NVacancies);
#endif        
        
        /* allocate array for storing cluster index */
        vacancyCluster = malloc(NVacancies * sizeof(int));
        if (vacancyCluster == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate vacancyCluster");
            if (driftCompensation) free(refPos);
            free(vacancies);
            free(interstitials);
            if (NSplitInterstitials) free(splitInterstitials);
            return NULL;
        }
        
        NVacanciesCluster = malloc(NVacancies * sizeof(int));
        if (NVacanciesCluster == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate NBubbleAtomsCluster");
            if (driftCompensation) free(refPos);
            free(vacancies);
            free(interstitials);
            if (NSplitInterstitials) free(splitInterstitials);
            free(vacancyCluster);
            return NULL;
        }
        
        /* clusters of vacancies */
        status = findVacancyClusters(NVacancies, vacancies, refPos, vacancyCluster, NVacanciesCluster, vacNebRad, cellDims, pbc, counters);
        if (status)
        {
            if (driftCompensation) free(refPos);
            free(vacancies);
            free(interstitials);
            if (NSplitInterstitials) free(splitInterstitials);
            free(vacancyCluster);
            free(NVacanciesCluster);
            return NULL;
        }
        NVacancyClusters = counters[0];
        
        /* Reallocate array */
        NVacanciesCluster = realloc(NVacanciesCluster, NVacancyClusters * sizeof(int));
        if (NVacanciesCluster == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not reallocate NVacanciesCluster");
            if (driftCompensation) free(refPos);
            free(vacancies);
            free(interstitials);
            if (NSplitInterstitials) free(splitInterstitials);
            free(vacancyCluster);
            return NULL;
        }
        
        /* allocate arrays for storing cluster indices of bubble atoms and counters */
        bubbleAtomCluster = malloc(NBubbleAtoms * sizeof(int));
        if (bubbleAtomCluster == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleAtomCluster");
            if (driftCompensation) free(refPos);
            free(vacancies);
            free(interstitials);
            if (NSplitInterstitials) free(splitInterstitials);
            free(vacancyCluster);
            free(NVacanciesCluster);
            return NULL;
        }
        
        NBubbleAtomsCluster = calloc(NVacancyClusters, sizeof(int));
        if (NBubbleAtomsCluster == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate NBubbleAtomsCluster");
            if (driftCompensation) free(refPos);
            free(vacancies);
            free(interstitials);
            if (NSplitInterstitials) free(splitInterstitials);
            free(vacancyCluster);
            free(bubbleAtomCluster);
            free(NVacanciesCluster);
            return NULL;
        }
        
        /* cluster index for bubble atoms */
        status = getClusterIndexForBubbleAtoms(NBubbleAtoms, bubbleAtomIndexes, pos, NVacancies, vacancies, refPos, bubbleAtomCluster,
                NBubbleAtomsCluster, cellDims, pbc, vacBubbleRad, NVacancyClusters, vacancyCluster, counters);
        if (status)
        {
            if (driftCompensation) free(refPos);
            free(vacancies);
            free(interstitials);
            if (NSplitInterstitials) free(splitInterstitials);
            free(vacancyCluster);
            free(NVacanciesCluster);
            free(bubbleAtomCluster);
            free(NBubbleAtomsCluster);
            return NULL;
        }
        NBubbles = counters[0];
        
        /* make the bubbles */
        result = constructBubbleResult(NBubbles, NVacancyClusters, vacancyCluster, NVacanciesCluster, bubbleAtomCluster, NBubbleAtomsCluster,
                NVacancies, vacancies, NBubbleAtoms, bubbleAtomIndexes);
        
        /* free */
        free(vacancies);
        free(interstitials);
        if (NSplitInterstitials) free(splitInterstitials);
        if (driftCompensation) free(refPos);
        free(vacancyCluster);
        free(NVacanciesCluster);
        free(bubbleAtomCluster);
        free(NBubbleAtomsCluster);
    }
    
#ifdef DEBUG
    printf("BUBBLESC: End\n");
#endif    
    return result;
}

/*******************************************************************************
 * Construct the result
 *******************************************************************************/
static PyObject*
constructBubbleResult(int NBubbles, int NClusters, int *vacancyCluster, int *NVacanciesCluster, int *bubbleAtomCluster, int *NBubbleAtomsCluster,
        int NVacancies, int *vacancies, int NBubbleAtoms, int *bubbleAtomIndices)
{
    int i, count, status;
    int *bubbleIndexMapper, *bubbleVacCount, *bubbleAtomCount;
    PyObject *result=NULL;
    
    PyObject *bubbleAtomList=NULL;
    PyObject *bubbleVacList=NULL;
    PyObject *bubbleVacIndexList=NULL;
    
    
#ifdef DEBUG
    printf("BUBBLESC:   Constructing result...\n");
#endif
    
    /* array for mapping cluster indices to bubble indices */
    bubbleIndexMapper = malloc(NClusters * sizeof(int));
    if (bubbleIndexMapper == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleIndexMapper");
        return NULL;
    }
    bubbleVacCount = malloc(NBubbles * sizeof(int));
    if (bubbleVacCount == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleVacCount");
        free(bubbleIndexMapper);
        return NULL;
    }
    bubbleAtomCount = malloc(NBubbles * sizeof(int));
    if (bubbleAtomCount == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleAtomCount");
        free(bubbleIndexMapper);
        free(bubbleVacCount);
        return NULL;
    }
    count = 0;
    for (i = 0; i < NClusters; i++)
    {
        if (NBubbleAtomsCluster[i] > 0)
        {
            bubbleVacCount[count] = NVacanciesCluster[i];
            bubbleAtomCount[count] = NBubbleAtomsCluster[i];
            bubbleIndexMapper[i] = count++;
#ifdef DEBUG
            printf("BUBBLESC:     Mapper: cluster %d -> bubble %d (%d vac, %d atm)\n", i, bubbleIndexMapper[i], NVacanciesCluster[i], NBubbleAtomsCluster[i]);
#endif
        }
        else
        {   
            bubbleIndexMapper[i] = -1;
#ifdef DEBUG
            printf("BUBBLESC:     Mapper: cluster %d -> not a bubble\n", i);
#endif
        }
    }
    
    /* make lists and arrays */
    bubbleVacList = PyList_New(NBubbles);
    if (bubbleVacList == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleVacList");
        free(bubbleIndexMapper);
        free(bubbleVacCount);
        free(bubbleAtomCount);
        return NULL;
    }
    bubbleAtomList = PyList_New(NBubbles);
    if (bubbleAtomList == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleAtomList");
        free(bubbleIndexMapper);
        free(bubbleVacCount);
        free(bubbleAtomCount);
        Py_DECREF(bubbleVacList);
        return NULL;
    }
    bubbleVacIndexList = PyList_New(NBubbles);
    if (bubbleVacIndexList == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleVacIndexList");
        free(bubbleIndexMapper);
        free(bubbleVacCount);
        free(bubbleAtomCount);
        Py_DECREF(bubbleVacList);
        Py_DECREF(bubbleAtomList);
        return NULL;
    }
    for (i = 0; i < NBubbles; i++)
    {
        npy_intp dims[1];
        PyArrayObject *vacIndices=NULL;
        PyArrayObject *atomIndices=NULL;
        PyArrayObject *vacAsIndexes=NULL;
        
        /* allocate numpy arrays for storing the vacancy and atom indices of this bubble */
        dims[0] = bubbleVacCount[i];
        vacIndices = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
        if (vacIndices == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate vacIndices");
            free(bubbleIndexMapper);
            free(bubbleVacCount);
            free(bubbleAtomCount);
            Py_DECREF(bubbleVacList);
            Py_DECREF(bubbleAtomList);
            Py_DECREF(bubbleVacIndexList);
            return NULL;
        }
        dims[0] = bubbleAtomCount[i];
        atomIndices = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
        if (vacIndices == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate vacIndices");
            free(bubbleIndexMapper);
            free(bubbleVacCount);
            free(bubbleAtomCount);
            Py_DECREF(bubbleVacList);
            Py_DECREF(bubbleAtomList);
            Py_DECREF(vacIndices);
            Py_DECREF(bubbleVacIndexList);
            return NULL;
        }
        dims[0] = bubbleVacCount[i];
        vacAsIndexes = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
        if (vacAsIndexes == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate vacAsIndexes");
            free(bubbleIndexMapper);
            free(bubbleVacCount);
            free(bubbleAtomCount);
            Py_DECREF(bubbleVacList);
            Py_DECREF(bubbleAtomList);
            Py_DECREF(vacIndices);
            Py_DECREF(atomIndices);
            Py_DECREF(bubbleVacIndexList);
            return NULL;
        }
        
        /* put numpy arrays into lists (lists steal ownership) */
        status = PyList_SetItem(bubbleVacList, i, PyArray_Return(vacIndices));
        if (status)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not set item in bubbleVacList");
            free(bubbleIndexMapper);
            free(bubbleVacCount);
            free(bubbleAtomCount);
            Py_DECREF(bubbleVacList);
            Py_DECREF(bubbleAtomList);
            Py_DECREF(vacIndices);
            Py_DECREF(atomIndices);
            Py_DECREF(vacAsIndexes);
            Py_DECREF(bubbleVacIndexList);
            return NULL;
        }
        status = PyList_SetItem(bubbleAtomList, i, PyArray_Return(atomIndices));
        if (status)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not set item in bubbleVacList");
            free(bubbleIndexMapper);
            free(bubbleVacCount);
            free(bubbleAtomCount);
            Py_DECREF(bubbleVacList);
            Py_DECREF(bubbleAtomList);
            Py_DECREF(atomIndices);
            Py_DECREF(vacAsIndexes);
            Py_DECREF(bubbleVacIndexList);
            return NULL;
        }
        status = PyList_SetItem(bubbleVacIndexList, i, PyArray_Return(vacAsIndexes));
        if (status)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not set item in bubbleVacIndexList");
            free(bubbleIndexMapper);
            free(bubbleVacCount);
            free(bubbleAtomCount);
            Py_DECREF(bubbleVacList);
            Py_DECREF(bubbleAtomList);
            Py_DECREF(vacAsIndexes);
            Py_DECREF(bubbleVacIndexList);
            return NULL;
        }
    }
    
    /* loop over vacancies, putting into bubbles */
    for (i = 0; i < NBubbles; i++) bubbleVacCount[i] = 0;
    for (i = 0; i < NVacancies; i++)
    {
        int clusterIndex = vacancyCluster[i];
        int bubbleIndex = bubbleIndexMapper[clusterIndex];
        
        if (bubbleIndex > -1)
        {
            PyArrayObject *bubbleVacIndices=NULL;
            PyArrayObject *bubbleVacAsIndexes=NULL;
            
            /* get the numpy array of vacancy indices for this bubble */
            bubbleVacIndices = (PyArrayObject *) PyList_GetItem(bubbleVacList, bubbleIndex);
            if (bubbleVacIndices == NULL)
            {
                free(bubbleIndexMapper);
                free(bubbleVacCount);
                free(bubbleAtomCount);
                Py_DECREF(bubbleVacList);
                Py_DECREF(bubbleAtomList);
                Py_DECREF(bubbleVacIndexList);
                return NULL;
            }
            bubbleVacAsIndexes = (PyArrayObject *) PyList_GetItem(bubbleVacIndexList, bubbleIndex);
            if (bubbleVacAsIndexes == NULL)
            {
                free(bubbleIndexMapper);
                free(bubbleVacCount);
                free(bubbleAtomCount);
                Py_DECREF(bubbleVacList);
                Py_DECREF(bubbleAtomList);
                Py_DECREF(bubbleVacIndexList);
                return NULL;
            }
            
            /* store this vacancy */
            IIND1(bubbleVacIndices, bubbleVacCount[bubbleIndex]) = vacancies[i];
            IIND1(bubbleVacAsIndexes, bubbleVacCount[bubbleIndex]) = i;
            bubbleVacCount[bubbleIndex]++;
        }
    }
    free(bubbleVacCount);
    
    /* loop over bubble atoms, putting into bubbles */
    for (i = 0; i < NBubbles; i++) bubbleAtomCount[i] = 0;
    for (i = 0; i < NBubbleAtoms; i++)
    {
        int clusterIndex = bubbleAtomCluster[i];
        
        if (clusterIndex > -1)
        {
            int bubbleIndex = bubbleIndexMapper[clusterIndex];
        
            if (bubbleIndex > -1)
            {
                PyArrayObject *bubbleIndices=NULL;
            
                /* get the numpy array of vacancy indices for this bubble */
                bubbleIndices = (PyArrayObject *) PyList_GetItem(bubbleAtomList, bubbleIndex);
                if (bubbleIndices == NULL)
                {
                    free(bubbleIndexMapper);
                    free(bubbleAtomCount);
                    Py_DECREF(bubbleVacList);
                    Py_DECREF(bubbleAtomList);
                    Py_DECREF(bubbleVacIndexList);
                    return NULL;
                }
            
                /* store this vacancy */
                IIND1(bubbleIndices, bubbleAtomCount[bubbleIndex]) = bubbleAtomIndices[i];
                bubbleAtomCount[bubbleIndex]++;
            }
        }
    }
    free(bubbleAtomCount);
    free(bubbleIndexMapper);
    
    /* result tuple */
    result = PyTuple_New(4);
    if (result == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate result");
        Py_DECREF(bubbleVacList);
        Py_DECREF(bubbleAtomList);
        Py_DECREF(bubbleVacIndexList);
        return NULL;
    }
    status = PyTuple_SetItem(result, 0, bubbleVacList);
    if (status)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not set item on result");
        Py_DECREF(bubbleVacList);
        Py_DECREF(bubbleAtomList);
        Py_DECREF(bubbleVacIndexList);
        Py_DECREF(result);
        return NULL;
    }
    status = PyTuple_SetItem(result, 1, bubbleAtomList);
    if (status)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not set item on result");
        Py_DECREF(bubbleAtomList);
        Py_DECREF(bubbleVacIndexList);
        Py_DECREF(result);
        return NULL;
    }
    status = PyTuple_SetItem(result, 2, bubbleVacIndexList);
    if (status)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not set item on result");
        Py_DECREF(bubbleVacIndexList);
        Py_DECREF(result);
        return NULL;
    }
    
    /* create numpy array for vacancies too */
    {
        npy_intp dims[1];
        PyArrayObject *vacsnp=NULL;
        
        /* allocate numpy array */
        dims[0] = NVacancies;
        vacsnp = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
        if (vacsnp == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate vacsnp");
            Py_DECREF(result);
            return NULL;
        }
        
        /* populate array with vacancies */
        for (i = 0; i < NVacancies; i++) IIND1(vacsnp, i) = vacancies[i];
        
        /* add to result */
        status = PyTuple_SetItem(result, 3, PyArray_Return(vacsnp));
        if (status)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not set item on result");
            Py_DECREF(vacsnp);
            Py_DECREF(result);
            return NULL;
        }
    }
    
    return result;
}

/*******************************************************************************
 * Determine the cluster index for the bubble atoms
 *******************************************************************************/
static int
getClusterIndexForBubbleAtoms(int NBubbleAtoms, int *bubbleAtomIndices, double *pos, int NVacancies, int *vacancies, double *refPos,
        int *bubbleAtomCluster, int *NBubbleAtomsCluster, double *cellDims, int *pbc, double vacBubbleRad, int NClusters,
        int *vacancyCluster, int *counters)
{
    int i, numBubbles;
    double *vacPos, *bubbleAtomPos;
    struct NeighbourList2 *nebList;
    
    
#ifdef DEBUG
    printf("BUBBLESC:   Finding cluster indices for bubble atoms...\n");
#endif
    
    /* vacancy positions */
    vacPos = malloc(3 * NVacancies * sizeof(double));
    if (vacPos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate vacPos");
        return 1;
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
        PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleAtomPos");
        return 2;
    }
    for (i = 0; i < NBubbleAtoms; i++)
    {
        int i3 = 3 * i;
        int indx3 = 3 * bubbleAtomIndices[i];
        bubbleAtomPos[i3    ] = pos[indx3    ];
        bubbleAtomPos[i3 + 1] = pos[indx3 + 1];
        bubbleAtomPos[i3 + 2] = pos[indx3 + 2];
    }
    
    /* we build a neighbour list for each bubble atom, containing neighbouring vacancies */
    nebList = constructNeighbourList2DiffPos(NBubbleAtoms, bubbleAtomPos, NVacancies, vacPos, cellDims, pbc, vacBubbleRad);
    free(vacPos);
    free(bubbleAtomPos);
    if (nebList == NULL) return 3;
    
    /* sort the neighbour list by separation */
    for (i = 0; i < NBubbleAtoms; i++)
        qsort(nebList[i].neighbour, nebList[i].neighbourCount, sizeof(struct Neighbour), compare_two_nebs);
    
    /* initialise array for storing cluster indexes of bubble atoms */
    for (i = 0; i < NBubbleAtoms; i++) bubbleAtomCluster[i] = -1;
    
    /* loop over bubble atoms and put in clusters (note it is possible for a bubble atom not to belong to a cluster!) */
    for (i = 0; i < NBubbleAtoms; i++)
    {
        if (nebList[i].neighbourCount > 0)
        {
            struct Neighbour neb = nebList[i].neighbour[0];
            if (neb.separation < vacBubbleRad)
            {
                int vacIndex = neb.index;
                int clusterIndex = vacancyCluster[vacIndex];
                bubbleAtomCluster[i] = clusterIndex;
                NBubbleAtomsCluster[clusterIndex]++;
            }
        }
    }
    
    /* free */
    freeNeighbourList2(nebList, NBubbleAtoms);
    
    /* number of bubbles */
    numBubbles = 0;
    for (i = 0; i < NClusters; i++) if (NBubbleAtomsCluster[i] > 0) numBubbles++;
#ifdef DEBUG
    printf("BUBBLESC:     Number of bubbles: %d (%d vac clusters)\n", numBubbles, NClusters);
#endif
    counters[0] = numBubbles;
    
    return 0;
}

/*******************************************************************************
 * Find clusters of vacancies
 *******************************************************************************/
static int
findVacancyClusters(int NVacancies, int *vacancies, double *refPos, int *defectCluster, int *NDefectsCluster, double clusterRadius,
        double *cellDims, int *pbc, int *counters)
{
    int i, boxstat, NClusters, maxNumInCluster;
    double *defectPos, approxBoxWidth, maxSep2;
    struct Boxes *boxes;
    

#ifdef DEBUG
    printf("BUBBLESC:   Finding clusters of vacancies...\n");
#endif
    
    /* build positions array of all defects */
    defectPos = malloc(3 * NVacancies * sizeof(double));
    if (defectPos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate defectPos");
        return 1;
    }
    
    /* add defects positions: vac then int then ant */
    for (i = 0; i < NVacancies; i++)
    {
        int index  = vacancies[i];
        int i3 = i * 3;
        int index3 = index * 3;
        defectPos[i3    ] = refPos[index3    ];
        defectPos[i3 + 1] = refPos[index3 + 1];
        defectPos[i3 + 2] = refPos[index3 + 2];
    }
    
    /* box defects */
    approxBoxWidth = clusterRadius;
    boxes = setupBoxes(approxBoxWidth, pbc, cellDims);
    if (boxes == NULL)
    {
        free(defectPos);
        return 2;
    }
    boxstat = putAtomsInBoxes(NVacancies, defectPos, boxes);
    if (boxstat)
    {
        free(defectPos);
        return 3;
    }
    
    /* rad squared */
    maxSep2 = clusterRadius * clusterRadius;
    
    /* initialise cluster array
     * = -1 : not yet allocated
     * > -1 : cluster ID of defect
     */
    for (i = 0; i < NVacancies; i++) defectCluster[i] = -1;
    
    /* loop over defects */
    NClusters = 0;
    maxNumInCluster = -9999;
    for (i = 0; i < NVacancies; i++)
    {
        /* skip if atom already allocated */
        if (defectCluster[i] == -1)
        {
            int numInCluster;
            
            /* allocate cluster ID */
            defectCluster[i] = NClusters;
            NClusters++;
            
            numInCluster = 1;
            
            /* recursive search for cluster atoms */
            numInCluster = findDefectNeighbours(i, defectCluster[i], numInCluster, defectCluster, defectPos, boxes, maxSep2, cellDims, pbc);
            if (numInCluster < 0) return -1;
            
            NDefectsCluster[defectCluster[i]] = numInCluster;
            maxNumInCluster = (numInCluster > maxNumInCluster) ? numInCluster : maxNumInCluster;
        }
    }
    
    freeBoxes(boxes);
    free(defectPos);
    
    if (NClusters < 0)
    {
        free(NDefectsCluster);
        return 5;
    }
    
    counters[0] = NClusters;
    
#ifdef DEBUG
    printf("BUBBLESC:     Found %d vacancy clusters\n", NClusters);
#endif

    return 0;
}

/*******************************************************************************
 * recursive search for neighbouring defects
 *******************************************************************************/
static int findDefectNeighbours(int index, int clusterID, int numInCluster, int* atomCluster, double *pos, struct Boxes *boxes, 
                                double maxSep2, double *cellDims, int *PBC)
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
            if ((index == index2) || (atomCluster[index2] != -1)) continue;
            
            /* calculate separation */
            sep2 = atomicSeparation2( pos[3*index], pos[3*index+1], pos[3*index+2], pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                      cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2] );
            
            /* check if neighbours */
            if (sep2 < maxSep2)
            {
                atomCluster[index2] = clusterID;
                numInCluster++;
                
                /* search for neighbours to this new cluster atom */
                numInCluster = findDefectNeighbours(index2, clusterID, numInCluster, atomCluster, pos, boxes, maxSep2, cellDims, PBC);
                if (numInCluster < 0) return -1;
            }
        }
    }
    
    return numInCluster;
}

/*******************************************************************************
 * Refine the list of vacancies:
 *   - Vacancies that are close to a H will remain as vacancies
 *   - Vacancies that are close to an interstitial are taken out of the list
 *******************************************************************************/
static int
refineVacancies(int NBubbleAtoms, int *bubbleIndices, int NVacanciesOld, int *vacancies, int NInterstitials, int *interstitials,
        int NSplitInterstitials, int *splitInterstitials, double *pos, double *refPos, int *counters, double *cellDims, int *pbc,
        double vacBubbleRad, double vacIntRad)
{
    int i, NVacancies, count, NIntsForPos, *vacMask;
    double *vacPos, *bubbleAtomPos, *intPos;
    struct NeighbourList2 *nebListBubs, *nebListInts;
    
    
#ifdef DEBUG
    printf("BUBBLESC:   Refining list of vacancies...\n");
#endif
    
    /* vacancy positions */
    vacPos = malloc(3 * NVacanciesOld * sizeof(double));
    if (vacPos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate vacPos");
        return 1;
    }
    for (i = 0; i < NVacanciesOld; i++)
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
        free(vacPos);
        PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleAtomPos");
        return 2;
    }
    for (i = 0; i < NBubbleAtoms; i++)
    {
        int i3 = 3 * i;
        int indx3 = 3 * bubbleIndices[i];
        bubbleAtomPos[i3    ] = pos[indx3    ];
        bubbleAtomPos[i3 + 1] = pos[indx3 + 1];
        bubbleAtomPos[i3 + 2] = pos[indx3 + 2];
    }
    
    /* we build a neighbour list for each vacancy, containing neighbouring bubble atoms */
    nebListBubs = constructNeighbourList2DiffPos(NVacanciesOld, vacPos, NBubbleAtoms, bubbleAtomPos, cellDims, pbc, vacBubbleRad);
    free(bubbleAtomPos);
    if (nebListBubs == NULL)
    {
        free(vacPos);
        return 3;
    }
    
    /* sort the neighbour list by separation */
    for (i = 0; i < NVacanciesOld; i++)
        qsort(nebListBubs[i].neighbour, nebListBubs[i].neighbourCount, sizeof(struct Neighbour), compare_two_nebs);
    
    /* interstitial positions */
    NIntsForPos = NInterstitials + 2 * NSplitInterstitials;
    intPos = malloc(3 * NIntsForPos * sizeof(double));
    if (intPos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate intPos");
        freeNeighbourList2(nebListBubs, NVacanciesOld);
        free(vacPos);
    }
    count = 0;
    for (i = 0; i < NInterstitials; i++)
    {
        int c3 = count * 3;
        int indx3 = 3 * interstitials[i];
        intPos[c3    ] = pos[indx3    ];
        intPos[c3 + 1] = pos[indx3 + 1];
        intPos[c3 + 2] = pos[indx3 + 2];
        count++;
    }
    for (i = 0; i < NSplitInterstitials; i++)
    {
        int j;
        
        for (j = 1; j < 3; j++)
        {
            int c3 = count * 3;
            int indx3 = 3 * splitInterstitials[3 * i + j];
            intPos[c3    ] = pos[indx3    ];
            intPos[c3 + 1] = pos[indx3 + 1];
            intPos[c3 + 2] = pos[indx3 + 2];
            count++;
        }
    }
    
    /* we build a neighbour list for each vacancy, containing neighbouring interstitial atoms */
    nebListInts = constructNeighbourList2DiffPos(NVacanciesOld, vacPos, NIntsForPos, intPos, cellDims, pbc, vacIntRad);
    free(intPos);
    free(vacPos);
    if (nebListInts == NULL)
    {
        freeNeighbourList2(nebListBubs, NVacanciesOld);
        return 4;
    }
    
    /* sort the neighbour list by separation */
    for (i = 0; i < NVacanciesOld; i++)
        qsort(nebListInts[i].neighbour, nebListInts[i].neighbourCount, sizeof(struct Neighbour), compare_two_nebs);
    
    /* vacancy mask array */
    vacMask = calloc(NVacanciesOld, sizeof(int));
    if (vacMask == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate vacMask");
        freeNeighbourList2(nebListBubs, NVacanciesOld);
        freeNeighbourList2(nebListInts, NVacanciesOld);
        return 5;
    }
    
    /* mark vacs with neighbour bubble atoms as definitely vacancies */
    for (i = 0; i < NVacanciesOld; i++)
    {
        if (nebListBubs[i].neighbourCount > 0)
        {
            if (nebListBubs[i].neighbour[0].separation < vacBubbleRad)
            {
                vacMask[i] = 1;
            }
        }
    }
    
    /* mark vacancies with neighbour ints as not vacancies */
    for (i = 0; i < NVacanciesOld; i++)
    {
        if (vacMask[i] == 0 && nebListInts[i].neighbourCount > 0)
        {
            if (nebListInts[i].neighbour[0].separation < vacIntRad)
            {
                vacMask[i] = -1;
            }
        }
    }
    
    /* free neighbour lists */
    freeNeighbourList2(nebListBubs, NVacanciesOld);
    freeNeighbourList2(nebListInts, NVacanciesOld);
    
    /* refresh vacancies array */
    NVacancies = 0;
    for (i = 0; i < NVacanciesOld; i++)
    {
        if (vacMask[i] > -1)
        {
            vacancies[NVacancies++] = vacancies[i];
        }
    }
    
    /* free */
    free(vacMask);
    
    /* store new number of vacancies */
    counters[0] = NVacancies;
    
    return 0;
}


/*******************************************************************************
 * identify split interstitials (new)
 *******************************************************************************/
static int
identifySplitInterstitialsNew(int NVacancies, int *vacancies, int NInterstitials, int *interstitials, int *splitInterstitials,
        double *pos, double *refPos, int *PBC, double *cellDims, int *counters, double vacancyRadius)
{
    int i, NSplit;
    double *intPos, *vacPos, maxSep;
    struct NeighbourList2 *nebListInts;
    struct NeighbourList2 *nebListVacs;
    

#ifdef DEBUG
    printf("BUBBLESC:   Identifying split interstitials (new)\n");
#endif
    
    /* build positions array of all vacancies and interstitials */
    vacPos = malloc(3 * NVacancies * sizeof(double));
    if (vacPos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate vacPos");
        return 1;
    }
    intPos = malloc(3 * NInterstitials * sizeof(double));
    if (intPos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate intPos");
        free(vacPos);
        return 2;
    }
    
    /* add vacancy positions */
    for (i = 0; i < NVacancies; i++)
    {
        int index = vacancies[i];
        int index3 = 3 * index;
        int i3 = 3 * i;
        vacPos[i3    ] = refPos[index3    ];
        vacPos[i3 + 1] = refPos[index3 + 1];
        vacPos[i3 + 2] = refPos[index3 + 2];
    }
    
    /* interstitial positions */
    for (i = 0; i < NInterstitials; i++)
    {
        int index = interstitials[i];
        int index3 = 3 * index;
        int i3 = i * 3;
        intPos[i3    ] = pos[index3    ];
        intPos[i3 + 1] = pos[index3 + 1];
        intPos[i3 + 2] = pos[index3 + 2];
    }
    
    /* max separation */
    maxSep = 2.0 * vacancyRadius;
    
    /* construct list of neighbouring interstitials for each vacancy */
    nebListVacs = constructNeighbourList2DiffPos(NVacancies, vacPos, NInterstitials, intPos, cellDims, PBC, maxSep);
    
    if (nebListVacs == NULL)
    {
        free(vacPos);
        free(intPos);
        return 3;
    }
    
    /* construct list of neighbouring vacancies for each interstitial */
    nebListInts = constructNeighbourList2DiffPos(NInterstitials, intPos, NVacancies, vacPos, cellDims, PBC, maxSep);
    
    /* free position arrays (only required for constructing neighbour lists) */
    free(intPos);
    free(vacPos);
    
    if (nebListInts == NULL)
    {
        freeNeighbourList2(nebListVacs, NVacancies);
        return 4;
    }
    
    /* sort the neighbour lists by separation */
    for (i = 0; i < NVacancies; i++)
        qsort(nebListVacs[i].neighbour, nebListVacs[i].neighbourCount, sizeof(struct Neighbour), compare_two_nebs);
    
    for (i = 0; i < NInterstitials; i++)
        qsort(nebListInts[i].neighbour, nebListInts[i].neighbourCount, sizeof(struct Neighbour), compare_two_nebs);
    
    /* loop over vacancies, checking if they belong to a split interstitial */
    NSplit = 0;
    for (i = 0; i < NVacancies; i++)
    {
        int numVacNebs;
        struct NeighbourList2 vacNebs;
        
        /* list of neighbouring interstitials for this vacancy */
        vacNebs = nebListVacs[i];
        numVacNebs = vacNebs.neighbourCount;
        
        /* proceed only if have at least 2 neighbouring interstitials */
        if (numVacNebs > 1)
        {
            int j, splitCount, splitIndexes[2];
            
            /* loop over neighbouring interstitials until we find 2 interstitials that are closest
             * to this vacancy or we run out of neighbours
             */
            j = 0;
            splitCount = 0;
            while (j < numVacNebs && splitCount != 2)
            {
                struct Neighbour vacNeb = vacNebs.neighbour[j];
                struct NeighbourList2 intNebs = nebListInts[vacNeb.index];
                int numIntNebs = intNebs.neighbourCount;
                
                /* check if closest vacancy is this one */
                if (numIntNebs > 0 && intNebs.neighbour[0].index == i)
                    splitIndexes[splitCount++] = vacNeb.index;
                
                j++;
            }
            
            /* check if we have found a split interstitial */
            if (splitCount == 2)
            {
                int n3 = 3 * NSplit;

                /* store vacancy in split interstitials array */
                splitInterstitials[n3] = vacancies[i];

                /* remove from vacancies array */
                vacancies[i] = -1;

                /* store interstitials */
                for (j = 0; j < 2; j++)
                {
                    int intIndex = splitIndexes[j];

                    /* store in split interstitials array */
                    splitInterstitials[n3 + j + 1] = interstitials[intIndex];

                    /* remove from interstitials array */
                    interstitials[intIndex] = -1;
                }
                
                NSplit++;
            }
            
        }
    }
    
#ifdef DEBUG
    printf("DEFECTSC:     Found %d split interstitials\n", NSplit);
#endif
    
    /* free memory */
    freeNeighbourList2(nebListVacs, NVacancies);
    freeNeighbourList2(nebListInts, NInterstitials);
    
    if (NSplit)
    {
        int count, numIntIn = NInterstitials;
        
        /* recreate interstitials arrays */
        count = 0;
        for (i = 0; i < NInterstitials; i++)
            if (interstitials[i] != -1) interstitials[count++] = interstitials[i];
        NInterstitials = count;
        
        /* recreate vacancies array */
        count = 0;
        for (i = 0; i < NVacancies; i++)
            if (vacancies[i] != -1) vacancies[count++] = vacancies[i];
        NVacancies = count;

        /* sanity check */
        if (numIntIn != NInterstitials + NSplit * 2)
        {
            char errstring[256];
            sprintf(errstring, "Interstitial atoms lost/gained during split detection: %d + %d * 2 != %d (%d vacancies)", NInterstitials,
                    NSplit, numIntIn, NVacancies);
            PyErr_SetString(PyExc_RuntimeError, errstring);
            return 7;
        }
    }

    /* store counters */
    counters[0] = NVacancies;
    counters[1] = NInterstitials;
    counters[2] = NSplit;
    
    return 0;
}

/*******************************************************************************
 ** Classify vacancies and interstitials
 *******************************************************************************/
static int
classifyVacsAndInts(double vacancyRadius, int NAtoms, double *pos, int refNAtoms, double *refPos,
        int *PBC, double *cellDims, int *counters, int *vacancies, int *interstitials,
        int NBubbleAtoms, int *bubbleAtomIndexes)
{
    int boxstat, i;
    int *possibleVacancy, *possibleInterstitial;
    int NVacancies, NInterstitials;
    int *bubbleAtomMask;
    double approxBoxWidth, vacRad2;
    struct Boxes *boxes;
    
    
    /* approx width, must be at least vacRad
     * should vary depending on size of cell
     * ie. don't want too many boxes
     */
    approxBoxWidth = (vacancyRadius > 3.0) ? vacancyRadius : 3.0;
    
    /* box atoms */
    boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
    if (boxes == NULL) return 1;
    boxstat = putAtomsInBoxes(NAtoms, pos, boxes);
    if (boxstat) return 2;
    
    /* allocate local arrays for checking atoms */
    possibleVacancy = malloc(refNAtoms * sizeof(int));
    if (possibleVacancy == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate possibleVacancy");
        freeBoxes(boxes);
        return 3;
    }
    
    possibleInterstitial = malloc(NAtoms * sizeof(int));
    if (possibleInterstitial == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate possibleInterstitial");
        freeBoxes(boxes);
        free(possibleVacancy);
        return 4;
    }
    
    bubbleAtomMask = calloc(NAtoms, sizeof(int));
    if (bubbleAtomMask == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate bubbleAtomMask");
        freeBoxes(boxes);
        free(possibleVacancy);
        free(possibleInterstitial);
        return 4;
    }
    
    /* initialise arrays */
    for (i = 0; i < NAtoms; i++) possibleInterstitial[i] = 1;
    for (i = 0; i < refNAtoms; i++) possibleVacancy[i] = 1;
    for (i = 0; i < NBubbleAtoms; i++) 
    {
        int index = bubbleAtomIndexes[i];
        bubbleAtomMask[index] = 1;
        possibleInterstitial[index] = 0;
    }
    
    /* constant */
    vacRad2 = vacancyRadius * vacancyRadius;
    
    /* loop over reference sites */
    for (i = 0; i < refNAtoms; i++)
    {
        int boxNebList[27], boxIndex, j, boxNebListSize;
        int nearestIndex = -1;
        int occupancyCount = 0;
        int i3 = 3 * i;
        double refxpos, refypos, refzpos;
        double nearestSep2 = 9999.0;

        refxpos = refPos[i3    ];
        refypos = refPos[i3 + 1];
        refzpos = refPos[i3 + 2];

        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(refxpos, refypos, refzpos, boxes);
        if (boxIndex < 0)
        {
            freeBoxes(boxes);
            free(possibleInterstitial);
            free(possibleVacancy);
            free(bubbleAtomMask);
            return 7;
        }

        /* find neighbouring boxes */
        boxNebListSize = getBoxNeighbourhood(boxIndex, boxNebList, boxes);

//        printf("Checking site %d for occupancy (%lf, %lf, %lf)\n", i, refxpos, refypos, refzpos);
        
        /* loop over neighbouring boxes */
        for (j = 0; j < boxNebListSize; j++)
        {
            int checkBox, k;

            checkBox = boxNebList[j];

            /* loop over all input atoms in the box */
            for (k = 0; k < boxes->boxNAtoms[checkBox]; k++)
            {
                int index, index3;
                double xpos, ypos, zpos, sep2;

                /* index of this input atom */
                index = boxes->boxAtoms[checkBox][k];

                /* skip if a bubble atom */
                if (bubbleAtomMask[index]) continue;
                
                /* atom position */
                index3 = 3 * index;
                xpos = pos[index3    ];
                ypos = pos[index3 + 1];
                zpos = pos[index3 + 2];

                /* atomic separation of possible vacancy and possible interstitial */
                sep2 = atomicSeparation2(xpos, ypos, zpos, refxpos, refypos, refzpos,
                                         cellDims[0], cellDims[1], cellDims[2],
                                         PBC[0], PBC[1], PBC[2]);

                /* if within vacancy radius, is it an antisite or normal lattice point */
                if (sep2 < vacRad2)
                {
//                    printf("  Input atom %d within vac rad: sep = %lf (%lf, %lf, %lf)\n", index, sqrt(sep2), xpos, ypos, zpos);
                    
                    /* check whether this is the closest atom to this vacancy */
                    if (sep2 < nearestSep2)
                    {
                        if (possibleInterstitial[index])
                        {
                            nearestSep2 = sep2;
                            nearestIndex = index;
                        }
                    }
                    
                    occupancyCount++;
                }
            }
        }

        /* classify - check the atom that was closest to this site (within the vacancy radius) */
        if (nearestIndex != -1)
        {
//            printf("Classifying site %d: nearest index = %d (sep = %lf)\n", i, nearestIndex, sqrt(nearestSep2));

            /* not an interstitial or vacancy */
            possibleInterstitial[nearestIndex] = 0;
            possibleVacancy[i] = 0;

//            if (occupancyCount > 1)
//                printf("INFO: Occupancy for site %d = %d\n", i, occupancyCount);
        }
    }
    
    /* free box arrays */
    freeBoxes(boxes);

    /* now classify defects */
    /* vacancies */
    NVacancies = 0;
    for (i = 0; i < refNAtoms; i++ )
        if (possibleVacancy[i] == 1) vacancies[NVacancies++] = i;
    
    /* interstitials */
    NInterstitials = 0;
    for (i = 0; i < NAtoms; i++ )
        if (possibleInterstitial[i] == 1) interstitials[NInterstitials++] = i;
    
    /* store counters */
    counters[0] = NVacancies;
    counters[1] = NInterstitials;
    
    /* free arrays */
    free(possibleVacancy);
    free(possibleInterstitial);
    free(bubbleAtomMask);
    
    return 0;
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
