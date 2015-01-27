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
	int NVisible, *visibleAtoms, NAtoms, *specie, specieID1, specieID2, *PBC, num;
	int OMP_NUM_THREADS;
	double *pos, *minPos, *maxPos, *cellDims, start, finish, *rdf;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *specieIn=NULL;
	PyArrayObject *PBCIn=NULL;
	PyArrayObject *posIn=NULL;
	PyArrayObject *minPosIn=NULL;
	PyArrayObject *maxPosIn=NULL;
	PyArrayObject *cellDimsIn=NULL;
	PyArrayObject *rdfIn=NULL;
	
    int i, norm_n, norm_nref;
    int spec1cnt, spec2cnt, fullShellCount;
    int *sel1, *sel2, sel1cnt, sel2cnt, duplicates;
    double approxBoxWidth, avgAtomDensity;
    double volume, normFactor, rho;
    double interval;
    double start2, finish2;
    double *visiblePos;
    struct Boxes *boxes;
    
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!O!iiO!O!O!O!dddiO!i", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &specieIn,
			&PyArray_Type, &posIn, &specieID1, &specieID2, &PyArray_Type, &minPosIn, &PyArray_Type, &maxPosIn,
			&PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &start, &finish, &interval, &num, &PyArray_Type,
			&rdfIn, &OMP_NUM_THREADS))
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
	
	if (not_doubleVector(minPosIn)) return NULL;
	minPos = pyvector_to_Cptr_double(minPosIn);
	
	if (not_doubleVector(maxPosIn)) return NULL;
	maxPos = pyvector_to_Cptr_double(maxPosIn);
	
	if (not_doubleVector(cellDimsIn)) return NULL;
	cellDims = pyvector_to_Cptr_double(cellDimsIn);
	
	if (not_intVector(PBCIn)) return NULL;
	PBC = pyvector_to_Cptr_int(PBCIn);

	/* set number of openmp threads to use */
    omp_set_num_threads(OMP_NUM_THREADS);
	
    /* approx box width */
	/* may not be any point boxing... */
    approxBoxWidth = finish;
    
    printf("DBGRDF: NVis %d; NAtom %d; SPEC1 %d; SPEC2 %d\n", NVisible, NAtoms, specieID1, specieID2);
    printf("DBGRDF: NUMBOX %d; INTERVAL %lf; START %lf; FINISH %lf\n", num, interval, start, finish);

    /* handle duplicates */
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
        return NULL;
    }
    sel1cnt = 0;
    for (i = 0; i < NVisible; i++)
    {
        int index = visibleAtoms[i];
        if (specieID1 < 0 || (specieID1 >= 0 && specie[index] == specieID1))
        {
            sel1[i] = 1;
            sel1cnt++;
        }
        else
            sel1[i] = 0;
    }
    sel2cnt = 0;
    for (i = 0; i < NVisible; i++)
    {
        int index = visibleAtoms[i];
        if (specieID2 < 0 || (specieID2 >= 0 && specie[index] == specieID2))
        {
            sel2[i] = 1;
            sel2cnt++;
        }
        else
            sel2[i] = 0;
    }
    duplicates = 0;
    for (i = 0; i < NVisible; i++) if (sel1[i] && sel2[i]) duplicates++;
    
    free(sel1);
    free(sel2);
    
    printf("DBGRDF: SEL1 %d; SEL2 %d; DUP %d\n", sel1cnt, sel2cnt, duplicates);

    /* position of visible atoms */
    visiblePos = malloc(3 * NVisible * sizeof(double));
    if (visiblePos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate visiblePos");
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
    
    /* box atoms */
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(NVisible, visiblePos, boxes);
    free(visiblePos);
    
    start2 = start * start;
    finish2 = finish * finish;
    
    /* loop over atoms */
    #pragma omp parallel for
    for (i = 0; i < NVisible; i++)
    {
        int j, index, ind3, boxIndex, boxNebList[27];
        double rxa, rya, rza;
        
        index = visibleAtoms[i];
        
        /* skip if not selected specie */
        if (specieID1 >= 0 && specie[index] != specieID1) continue;
        
        /* get box index of this atom */
        ind3 = index * 3;
        rxa = pos[ind3    ];
        rya = pos[ind3 + 1];
        rza = pos[ind3 + 2];
        boxIndex = boxIndexOfAtom(rxa, rya, rza, boxes);
        
        /* find neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over box neighbourhood */
        for (j = 0; j < 27; j++)
        {
            int k;
            
            boxIndex = boxNebList[j];
            
            for (k = 0; k < boxes->boxNAtoms[boxIndex]; k++)
            {
                int visIndex, index2, ind23;
                double sep2;
                
                visIndex = boxes->boxAtoms[boxIndex][k];
                index2 = visibleAtoms[visIndex];
                
//                if (index2 <= index) continue; // cannot do this because spec1 and spec2 may differ...
                if (index == index2) continue;
                
                /* skip if not selected specie */
                if (specieID2 >= 0 && specie[index2] != specieID2) continue;
                
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
    
    /* normalise rdf */
    {
        double pair_dens;

        pair_dens = cellDims[0] * cellDims[1] * cellDims[2];
        pair_dens /= ((double)sel1cnt * (double)sel2cnt - (double)duplicates);
        printf("PAIR DENS %d\n", pair_dens);
        printf("npair %d\n", sel1cnt * sel2cnt - duplicates);

        for (i = 0; i < num; i++)
        {
            double r_inner, r_outer, norm_f, shellVolume;
            double histv;

            histv = rdf[i];
            r_inner = interval * i + start;
            r_outer = interval * (i + 1) + start;
            shellVolume = 4.0 / 3.0 * M_PI * (pow(r_outer, 3.0) - pow(r_inner, 3.0));
            norm_f = pair_dens / shellVolume;
            rdf[i] = rdf[i] * norm_f;
            printf("HIST %d (%lf -> %lf): histv %lf; vol %lf; norm_f = %lf; rdf %lf\n", i, r_inner, r_outer, histv, shellVolume, norm_f, rdf[i]);
        }
    }

    /* calculate shell volumes and average atom density */
//    avgAtomDensity = 0.0;
//    fullShellCount = 0;
//    for (i=0; i<num; i++)
//    {
//        double ini, fin, shellVolume;
//
//        ini = i * interval + start;
//        fin = (i + 1.0) * interval + start;
//
//        shellVolume = (4.0 / 3.0) * M_PI * (pow(fin, 3.0) - pow(ini, 3));
//        rdf[i] = rdf[i] / shellVolume;
//
//        if (rdf[i] > 0)
//        {
//            avgAtomDensity += rdf[i];
//            fullShellCount++;
//        }
//    }
//    avgAtomDensity = avgAtomDensity / fullShellCount;
//
//    /* divide by average atom density */
//    for (i=0; i<num; i++)
//        rdf[i] = rdf[i] / avgAtomDensity;
    
    /* free */
    freeBoxes(boxes);
    
    return Py_BuildValue("i", 0);
}
