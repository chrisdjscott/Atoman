/*******************************************************************************
 ** C extension to calculate the radial distribution function
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "boxeslib.h"
#include "utilities.h"
#include "array_utils.h"
#include "constants.h"


static PyObject* calculateRDF(PyObject*, PyObject*);
static int computeHistogram(int, int, int*, double*, int*, double*, int*, int*, double,
        double, double, int, double*);
static void normaliseRDF(int, int, int, int, double, double, double*, double*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"calculateRDF", calculateRDF, METH_VARARGS, "Calculate the RDF for the selected atoms"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_rdf(void)
{
    (void)Py_InitModule("_rdf", methods);
    import_array();
}

/*******************************************************************************
 ** Calculate the radial distribution function for the given selections of
 ** visible atoms.
 ** 
 ** Inputs are:
 **     - visibleAtoms: indices of atoms that are to be used for the calculation
 **     - specie: array containing the species index for each atom
 **     - pos: array containing the positions of the atoms
 **     - specieID1: the species of the first selection of atoms
 **     - specieID2: the species of the second selection of atoms
 **     - cellDims: the size of the simulation cell
 **     - pbc: periodic boundary conditions
 **     - start: minimum separation to use when constructing the histogram
 **     - finish: maximum separation to use when constructing the histogram
 **     - interval: the interval between histogram bins
 **     - numBins: the number of histogram bins
 **     - rdf: the result is returned in this array
 *******************************************************************************/
static PyObject*
calculateRDF(PyObject *self, PyObject *args)
{
    int numVisible, *visibleAtoms, *specie, specieID1, specieID2, *pbc, numBins;
    int numThreads, numAtoms;
    double *pos, *cellDims, start, finish, *rdf;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *pbcIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *rdfIn=NULL;
    int i, status, *sel1, *sel2, sel1cnt, sel2cnt, duplicates;
    double interval;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!iiO!O!dddiO!i", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &specieIn,
            &PyArray_Type, &posIn, &specieID1, &specieID2, &PyArray_Type, &cellDimsIn, &PyArray_Type, &pbcIn, &start,
            &finish, &interval, &numBins, &PyArray_Type, &rdfIn, &numThreads))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    numVisible = (int) visibleAtomsIn->dimensions[0];
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    numAtoms = (int) specieIn->dimensions[0];
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(rdfIn)) return NULL;
    rdf = pyvector_to_Cptr_double(rdfIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(pbcIn)) return NULL;
    pbc = pyvector_to_Cptr_int(pbcIn);
    
    /* initialise result array to zero */
    for (i = 0; i < numBins; i++) rdf[i] = 0.0;
    
    /* create the selections of atoms and check for number of duplicates */
    sel1 = malloc(numVisible * sizeof(int));
    if (sel1 == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate sel1");
        return NULL;
    }
    sel2 = malloc(numVisible * sizeof(int));
    if (sel2 == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate sel2");
        free(sel1);
        return NULL;
    }
    sel1cnt = 0;
    sel2cnt = 0;
    duplicates = 0;
    for (i = 0; i < numVisible; i++)
    {
        int index = visibleAtoms[i];
        
        /* check if this atom is in the first selection (negative means all species) */
        if (specieID1 < 0 || specie[index] == specieID1)
        {
            sel1[i] = 1;
            sel1cnt++;
        }
        else sel1[i] = 0;
        
        /* check if this atom is in the second selection (negative means all species) */
        if (specieID2 < 0 || specie[index] == specieID2)
        {
            sel2[i] = 1;
            sel2cnt++;
        }
        else sel2[i] = 0;
        
        /* count the number of atoms that are in both selections */
        if (sel1[i] && sel2[i]) duplicates++;
    }
    
    /* compute the histogram for the RDF */
    status = computeHistogram(numAtoms, numVisible, visibleAtoms, pos, pbc, cellDims, 
            sel1, sel2, start, finish, interval, numThreads, rdf);
    
    /* free memory used for selections */
    free(sel1);
    free(sel2);
    
    /* return if there was an error */
    if (status) return NULL;
    
    /* normalise the rdf */
    normaliseRDF(numBins, sel1cnt, sel2cnt, duplicates, start, interval, cellDims, rdf);
    
    /* return None */
    Py_INCREF(Py_None);
    return Py_None;
}

/*******************************************************************************
 ** Compute the histogram for the RDF
 *******************************************************************************/
static int
computeHistogram(int NAtoms, int NVisible, int *visibleAtoms, double *pos, int *PBC, double *cellDims,
        int *sel1, int *sel2, double start, double finish, double interval, int numThreads, double *hist)
{
    int i, errorCount, boxstat;
    double *visiblePos, approxBoxWidth;
    const double start2 = start * start;
    const double finish2 = finish * finish;
    struct Boxes *boxes;
    
    
    /* positions of visible atoms */
    if (NAtoms == NVisible) visiblePos = pos;
    else
    {
        visiblePos = malloc(3 * NVisible * sizeof(double));
        if (visiblePos == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate visiblePos");
            return 1;
        }
        for (i = 0; i < NVisible; i++)
        {
            int index = visibleAtoms[i];
            int i3 = 3 * i;
            int ind3 = 3 * index;
            visiblePos[i3    ] = pos[ind3    ];
            visiblePos[i3 + 1] = pos[ind3 + 1];
            visiblePos[i3 + 2] = pos[ind3 + 2];
        }
    }
    
    /* spatial decomposition - box width must be at least `finish` */
    approxBoxWidth = finish;
    boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
    if (boxes == NULL)
    {
        if (NAtoms != NVisible) free(visiblePos);
        return 2;
    }
    boxstat = putAtomsInBoxes(NVisible, visiblePos, boxes);
    if (NAtoms != NVisible) free(visiblePos);
    if (boxstat) return 3;
    
    /* loop over visible atoms */
    errorCount = 0;
    #pragma omp parallel for reduction(+: errorCount) num_threads(numThreads)
    for (i = 0; i < NVisible; i++)
    {
        int j, index, ind3, boxIndex, boxNebList[27], boxNebListSize;
        double rxa, rya, rza;
        
        /* skip if this atom is not in the first selection */
        if (!sel1[i]) continue;
        
        /* the index of this atom in the pos array */
        index = visibleAtoms[i];
        
        /* position of this atom and its box index */
        ind3 = index * 3;
        rxa = pos[ind3    ];
        rya = pos[ind3 + 1];
        rza = pos[ind3 + 2];
        boxIndex = boxIndexOfAtom(rxa, rya, rza, boxes);
        if (boxIndex < 0) errorCount++;
        
        if (!errorCount)
        {
            /* find neighbouring boxes */
            boxNebListSize = getBoxNeighbourhood(boxIndex, boxNebList, boxes);
            
            /* loop over the box neighbourhood */
            for (j = 0; j < boxNebListSize; j++)
            {
                int k;
                int checkBox = boxNebList[j];
                
                for (k = 0; k < boxes->boxNAtoms[checkBox]; k++)
                {
                    int visIndex, index2, ind23;
                    double sep2;
                    
                    /* the index of this atom in the visibleAtoms array */
                    visIndex = boxes->boxAtoms[checkBox][k];
                    
                    /* skip if this atom is not in the second selection */
                    if (!sel2[visIndex]) continue;
                    
                    /* atom index */
                    index2 = visibleAtoms[visIndex];
                    
                    /* skip if same atom */
                    if (index == index2) continue;
                    
                    /* atomic separation */
                    ind23 = index2 * 3;
                    sep2 = atomicSeparation2(rxa, rya, rza,
                                             pos[ind23], pos[ind23 + 1], pos[ind23 + 2],
                                             cellDims[0], cellDims[1], cellDims[2], 
                                             PBC[0], PBC[1], PBC[2]);
                    
                    /* put in bin */
                    if (sep2 >= start2 && sep2 < finish2)
                    {
                        int binIndex;
                        double sep;
                        
                        sep = sqrt(sep2);
                        binIndex = (int) ((sep - start) / interval);
                        #pragma omp atomic
                        hist[binIndex]++;
                    }
                }
            }
        }
    }
    
    /* free memory */
    freeBoxes(boxes);
    
    /* raise an exception if there were any errors */
    if (errorCount)
    {
        PyErr_SetString(PyExc_RuntimeError, 
                "computeHistogram failed; probably box index error (check stderr)");
        return 4;
    }
    
    return 0;
}

/*******************************************************************************
 ** Normalise the RDF
 *******************************************************************************/
static void
normaliseRDF(int numBins, int sel1cnt, int sel2cnt, int duplicates, double start, double interval,
        double *cellDims, double *rdf)
{
    int i;
    double pair_dens;
    const double fourThirdsPi = 4.0 / 3.0 * CONST_PI;

    /* compute inverse of pair density (volume / number of pairs) */
    pair_dens = cellDims[0] * cellDims[1] * cellDims[2];
    pair_dens /= ((double)sel1cnt * (double)sel2cnt - (double)duplicates);

    /* loop over histogram bins */
    for (i = 0; i < numBins; i++)
    {
        double rInner, rOuter, norm, shellVolume;

        if (rdf[i] != 0.0)
        {
            /* calculate the volume of this shell */
            rInner = interval * i + start;
            rOuter = interval * (i + 1) + start;
            shellVolume = fourThirdsPi * (pow(rOuter, 3.0) - pow(rInner, 3.0));
            
            /* normalisation factor is 1 / (pair_density * shellVolume) */
            norm = pair_dens / shellVolume;
            rdf[i] = rdf[i] * norm;
        }
    }
}
