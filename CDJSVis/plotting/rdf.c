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
	
    int i;
    int fullShellCount;
    double approxBoxWidth;
    double avgAtomDensity, shellVolume;
    double ini, fin, interval;
    double start2, finish2;
    struct Boxes *boxes;
    
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!O!iiO!O!O!O!ddiO!i", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &specieIn,
			&PyArray_Type, &posIn, &specieID1, &specieID2, &PyArray_Type, &minPosIn, &PyArray_Type, &maxPosIn, 
			&PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &start, &finish, &num, &PyArray_Type, &rdfIn, 
			&OMP_NUM_THREADS))
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
    
    /* box atoms */
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(NAtoms, pos, boxes);
    
    interval = (finish - start) / ((double) num);
    
    start2 = start * start;
    finish2 = finish * finish;
    
    /* loop over atoms */
    #pragma omp parallel for
    for (i=0; i<NVisible; i++)
    {
        int j, index, boxIndex, boxNebList[27];
        
        index = visibleAtoms[i];
        
        /* skip if not selected specie */
        if (specieID1 >= 0 && specie[index] != specieID1) continue;
        
        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(pos[3*index], pos[3*index+1], pos[3*index+2], boxes);
        
        /* find neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over box neighbourhood */
        for (j=0; j<27; j++)
        {
            int k;
            
            boxIndex = boxNebList[j];
            
            for (k=0; k<boxes->boxNAtoms[boxIndex]; k++)
            {
                int index2;
                double sep2;
                
                index2 = boxes->boxAtoms[boxIndex][k];
                
                if (index2 <= index) continue;
                
                /* skip if not selected specie */
                if (specieID2 >= 0 && specie[index2] != specieID2) continue;
                
                /* atomic separation */
                sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                         pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
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
                    rdf[binIndex] += 2;
                }
            }
        }
    }
    
    /* calculate shell volumes and average atom density */
    avgAtomDensity = 0.0;
    fullShellCount = 0;
    for (i=0; i<num; i++)
    {
        ini = i * interval + start;
        fin = (i + 1.0) * interval + start;
        
        shellVolume = (4.0 / 3.0) * M_PI * (pow(fin, 3.0) - pow(ini, 3));
        
        rdf[i] = rdf[i] / shellVolume;
        
        if (rdf[i] > 0)
        {
            avgAtomDensity += rdf[i];
            fullShellCount++;
        }
    }
    
    avgAtomDensity = avgAtomDensity / fullShellCount;
    
    /* divide by average atom density */
    for (i=0; i<num; i++)
        rdf[i] = rdf[i] / avgAtomDensity;
    
    /* free */
    freeBoxes(boxes);
    
    return Py_BuildValue("i", 0);
}
