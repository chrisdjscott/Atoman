/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** Calculate RDF
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <omp.h>
#include "boxeslib.h"
#include "utilities.h"
#include "array_utils.h"


static PyObject* calculateRDF(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"calculateRDF", calculateRDF, METH_VARARGS, "Calculate RDF for visible atoms"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
initrdf(void)
{
    (void)Py_InitModule("rdf", methods);
    import_array();
}

/*******************************************************************************
 ** Calculate RDF function
 *******************************************************************************/
static PyObject*
calculateRDF(PyObject *self, PyObject *args)
{
    int NVisible, *visibleAtoms, *specie, specieID1, specieID2, *PBC, num;
    int numThreads, NAtoms;
    double *pos, *cellDims, start, finish, *rdf;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *rdfIn=NULL;
    
    int i, boxstat, errorCount;
    int *sel1, *sel2, sel1cnt, sel2cnt, duplicates;
    double approxBoxWidth;
    double interval;
    double start2, finish2;
    double *visiblePos;
    struct Boxes *boxes;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!iiO!O!dddiO!i", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &specieIn,
            &PyArray_Type, &posIn, &specieID1, &specieID2, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &start,
            &finish, &interval, &num, &PyArray_Type, &rdfIn, &numThreads))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisible = (int) visibleAtomsIn->dimensions[0];
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    NAtoms = (int) specieIn->dimensions[0];
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(rdfIn)) return NULL;
    rdf = pyvector_to_Cptr_double(rdfIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    /* box width for spatial decomposition must be at least `finish` */
    approxBoxWidth = finish;
    
    /* create the selections of atoms and check for number of duplicates */
    sel1 = malloc(NVisible * sizeof(int));
    if (sel1 == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate sel1");
        return NULL;
    }
    sel2 = malloc(NVisible * sizeof(int));
    if (sel2 == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate sel2");
        free(sel1);
        return NULL;
    }
    sel1cnt = 0;
    sel2cnt = 0;
    duplicates = 0;
    for (i = 0; i < NVisible; i++)
    {
        int index = visibleAtoms[i];
        
        /* check if this atom is in the first selection */
        if (specieID1 < 0 || (specieID1 >= 0 && specie[index] == specieID1))
        {
            sel1[i] = 1;
            sel1cnt++;
        }
        else sel1[i] = 0;
        
        /* check if this atom is in the second selection */
        if (specieID2 < 0 || (specieID2 >= 0 && specie[index] == specieID2))
        {
            sel2[i] = 1;
            sel2cnt++;
        }
        else sel2[i] = 0;
        
        /* count the number of atoms that are in both selections */
        if (sel1[i] && sel2[i]) duplicates++;
    }
    
    /* positions of visible atoms */
    if (NAtoms == NVisible) visiblePos = pos;
    else
    {
        visiblePos = malloc(3 * NVisible * sizeof(double));
        if (visiblePos == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate visiblePos");
            free(sel1);
            free(sel2);
            return NULL;
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
    
    /* box atoms */
    boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
    if (boxes == NULL)
    {
        if (NAtoms != NVisible) free(visiblePos);
        free(sel1);
        free(sel2);
        return NULL;
    }
    boxstat = putAtomsInBoxes(NVisible, visiblePos, boxes);
    if (NAtoms != NVisible) free(visiblePos);
    else visiblePos = NULL;
    if (boxstat)
    {
        free(sel1);
        free(sel2);
        return NULL;
    }
    
    start2 = start * start;
    finish2 = finish * finish;
    
    /* loop over atoms */
    errorCount = 0;
    #pragma omp parallel for reduction(+: errorCount) num_threads(numThreads)
    for (i = 0; i < NVisible; i++)
    {
        int j, index, ind3, boxIndex, boxNebList[27], boxNebListSize;
        double rxa, rya, rza;
        
        /* skip if not in first selection */
        if (!sel1[i]) continue;
        
        /* atom index */
        index = visibleAtoms[i];
        
        /* get box index of this atom */
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
            
            /* loop over box neighbourhood */
            for (j = 0; j < boxNebListSize; j++)
            {
                int k;
                
                boxIndex = boxNebList[j];
                
                for (k = 0; k < boxes->boxNAtoms[boxIndex]; k++)
                {
                    int visIndex, index2, ind23;
                    double sep2;
                    
                    /* visible index */
                    visIndex = boxes->boxAtoms[boxIndex][k];
                    
                    /* skip if not in second selection */
                    if (!sel2[visIndex]) continue;
                    
                    /* atom index */
                    index2 = visibleAtoms[visIndex];
                    
                    /* skip if same atom */
                    if (index == index2) continue;
                    
                    /* atomic separation */
                    ind23 = index2 * 3;
                    sep2 = atomicSeparation2(rxa, rya, rza,
                                             pos[ind23    ], pos[ind23 + 1], pos[ind23 + 2],
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
                        rdf[binIndex]++;
                    }
                }
            }
        }
    }
    
    /* free */
    freeBoxes(boxes);
    free(sel1);
    free(sel2);
    
    if (errorCount)
    {
        PyErr_SetString(PyExc_RuntimeError, "RDF loop failed; probably box index error (check stderr)");
        return NULL;
    }
    
    /* normalise rdf */
    {
        double pair_dens;
        double fourThirdsPi = 4.0 / 3.0 * M_PI;

        /* compute inverse of pair density (volume / number of pairs) */
        pair_dens = cellDims[0] * cellDims[1] * cellDims[2];
        pair_dens /= ((double)sel1cnt * (double)sel2cnt - (double)duplicates);

        /* loop over histogram bins */
        for (i = 0; i < num; i++)
        {
            double r_inner, r_outer, norm_f, shellVolume;

            if (rdf[i] != 0.0)
            {
                /* calculate shell volume */
                r_inner = interval * i + start;
                r_outer = interval * (i + 1) + start;
                shellVolume = fourThirdsPi * (pow(r_outer, 3.0) - pow(r_inner, 3.0));
                
                /* normalisation factor is 1 / (pair_density * shellVolume) */
                norm_f = pair_dens / shellVolume;
                rdf[i] = rdf[i] * norm_f;
            }
        }
    }
    
    return Py_BuildValue("i", 0);
}
