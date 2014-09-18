/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** Picker routines
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "boxeslib.h"
#include "utilities.h"
#include "array_utils.h"


static PyObject* pickObject(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"pickObject", pickObject, METH_VARARGS, "Check if an object has been pickeds"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
initpicker(void)
{
    (void)Py_InitModule("picker", methods);
    import_array();
}

/*******************************************************************************
 ** Check if an object has been picked
 *******************************************************************************/
static PyObject*
pickObject(PyObject *self, PyObject *args)
{
    int visibleAtomsDim, *visibleAtoms, vacsDim, *vacs, intsDim, *ints, onAntsDim, *onAnts, splitsDim, *splits;
    int *PBC, *specie, *refSpecie;
    double *pickPos, *pos, *refPos, *cellDims, *minPos, *maxPos, *specieCovRad, *refSpecieCovRad, *result;  
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *vacsIn=NULL;
    PyArrayObject *intsIn=NULL;
    PyArrayObject *onAntsIn=NULL;
    PyArrayObject *splitsIn=NULL;
    PyArrayObject *pickPosIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *refPosIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *minPosIn=NULL;
    PyArrayObject *maxPosIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *refSpecieIn=NULL;
    PyArrayObject *specieCovRadIn=NULL;
    PyArrayObject *refSpecieCovRadIn=NULL;
    PyArrayObject *resultIn=NULL;
    
    int i, k, index, boxIndex;
    int boxNebList[27], realIndex;
    int minSepIndex, NVis, count;
    int minSepType;
    double approxBoxWidth, *visPos;
    double sep2, minSep, rad, sep;
    double *visCovRad;
    struct Boxes *boxes;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &vacsIn, 
            &PyArray_Type, &intsIn, &PyArray_Type, &onAntsIn, &PyArray_Type, &splitsIn, &PyArray_Type, &pickPosIn, &PyArray_Type, 
            &posIn, &PyArray_Type, &refPosIn, &PyArray_Type, &PBCIn, &PyArray_Type, &cellDimsIn, &PyArray_Type, &minPosIn, 
            &PyArray_Type, &maxPosIn, &PyArray_Type, &specieIn, &PyArray_Type, &refSpecieIn, &PyArray_Type, &specieCovRadIn,
            &PyArray_Type, &refSpecieCovRadIn, &PyArray_Type, &resultIn))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    visibleAtomsDim = (int) visibleAtomsIn->dimensions[0];
    
    if (not_intVector(vacsIn)) return NULL;
    vacs = pyvector_to_Cptr_int(vacsIn);
    vacsDim = (int) vacsIn->dimensions[0];
    
    if (not_intVector(intsIn)) return NULL;
    ints = pyvector_to_Cptr_int(intsIn);
    intsDim = (int) intsIn->dimensions[0];
    
    if (not_intVector(onAntsIn)) return NULL;
    onAnts = pyvector_to_Cptr_int(onAntsIn);
    onAntsDim = (int) onAntsIn->dimensions[0];
    
    if (not_intVector(splitsIn)) return NULL;
    splits = pyvector_to_Cptr_int(splitsIn);
    splitsDim = (int) splitsIn->dimensions[0];
    
    if (not_doubleVector(pickPosIn)) return NULL;
    pickPos = pyvector_to_Cptr_double(pickPosIn);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(refPosIn)) return NULL;
    refPos = pyvector_to_Cptr_double(refPosIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(minPosIn)) return NULL;
    minPos = pyvector_to_Cptr_double(minPosIn);
    
    if (not_doubleVector(maxPosIn)) return NULL;
    maxPos = pyvector_to_Cptr_double(maxPosIn);
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    
    if (not_intVector(refSpecieIn)) return NULL;
    refSpecie = pyvector_to_Cptr_int(refSpecieIn);
    
    if (not_doubleVector(specieCovRadIn)) return NULL;
    specieCovRad = pyvector_to_Cptr_double(specieCovRadIn);
    
    if (not_doubleVector(refSpecieCovRadIn)) return NULL;
    refSpecieCovRad = pyvector_to_Cptr_double(refSpecieCovRadIn);
    
    if (not_doubleVector(resultIn)) return NULL;
    result = pyvector_to_Cptr_double(resultIn);
    
    
    // this should be detected automatically depending on cell size...
    approxBoxWidth = 4.0;
    
    
    if (visibleAtomsDim > 0)
    {
        /* vis atoms pos */
        visPos = malloc(3 * visibleAtomsDim * sizeof(double));
        if (visPos == NULL)
        {
            printf("ERROR: could not allocate visPos\n");
            exit(1);
        }
        
        for (i=0; i<visibleAtomsDim; i++)
        {
            index = visibleAtoms[i];
            visPos[3*i] = pos[3*index];
            visPos[3*i+1] = pos[3*index+1];
            visPos[3*i+2] = pos[3*index+2];
        }
        
        /* box vis atoms */
        boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
        putAtomsInBoxes(visibleAtomsDim, visPos, boxes);
        
        /* box index of picked pos */
        boxIndex = boxIndexOfAtom(pickPos[0], pickPos[1], pickPos[2], boxes);
        
        /* neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over neighbouring boxes, looking for nearest atom */
        minSep = 9999999.0;
        minSepIndex = -1;
        for (i=0; i<27; i++)
        {
            boxIndex = boxNebList[i];
            
            for (k=0; k<boxes->boxNAtoms[boxIndex]; k++)
            {
                index = boxes->boxAtoms[boxIndex][k];
                
                /* atomic separation */
                sep2 = atomicSeparation2(pickPos[0], pickPos[1], pickPos[2], 
                                         visPos[3*index], visPos[3*index+1], visPos[3*index+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                
                /* need radius too */
                realIndex = visibleAtoms[index];
                
                rad = specieCovRad[specie[realIndex]];
                
                sep = sqrt(sep2);
                
                /* if separation is greater than radius, subtract radius, 
                 * otherwise set to zero (within radius is good enough for me)
                 */
                sep = (sep > rad) ? sep - rad : 0.0;
                
                if (sep < minSep)
                {
                    minSep = sep;
                    minSepIndex = index;
                }
            }
        }
        
        /* store result */
        result[0] = 0;
        result[1] = minSepIndex;
        result[2] = minSep;
        
        /* tidy up */
        free(visPos);
        freeBoxes(boxes);
    }
    else
    {
        /* build positions array of all defects */
        NVis = vacsDim + intsDim + onAntsDim + splitsDim;
        visPos = malloc(3 * NVis * sizeof(double));
        if (visPos == NULL)
        {
            printf("ERROR: could not allocate visPos\n");
            exit(50);
        }
        
        visCovRad = malloc(NVis * sizeof(double));
        if (visCovRad == NULL)
        {
            printf("ERROR: could not allocate visCovRad\n");
            exit(50);
        }
        
        /* add defects positions: vac then int then ant */
        count = 0;
        for (i=0; i<vacsDim; i++)
        {
            index = vacs[i];
            
            visPos[3*count] = refPos[3*index];
            visPos[3*count+1] = refPos[3*index+1];
            visPos[3*count+2] = refPos[3*index+2];
            
            visCovRad[count] = refSpecieCovRad[refSpecie[index]] * 1.2;// * 0.75; // multiply by 0.75 because vacs aren't full size (rendering.py)
            
            count++;
        }
        
        for (i=0; i<intsDim; i++)
        {
            index = ints[i];
            
            visPos[3*count] = pos[3*index];
            visPos[3*count+1] = pos[3*index+1];
            visPos[3*count+2] = pos[3*index+2];
            
            visCovRad[count] = specieCovRad[specie[index]];
            
            count++;
        }
        
        for (i=0; i<onAntsDim; i++)
        {
            index = onAnts[i];
            
            visPos[3*count] = pos[3*index];
            visPos[3*count+1] = pos[3*index+1];
            visPos[3*count+2] = pos[3*index+2];
            
            visCovRad[count] = specieCovRad[specie[index]];
            
            count++;
        }
        
        for (i=0; i<splitsDim/3; i++)
        {
            index = splits[3*i];
            visPos[3*count] = refPos[3*index];
            visPos[3*count+1] = refPos[3*index+1];
            visPos[3*count+2] = refPos[3*index+2];
            visCovRad[count] = refSpecieCovRad[refSpecie[index]];
            count++;
            
            index = splits[3*i+1];
            visPos[3*count] = pos[3*index];
            visPos[3*count+1] = pos[3*index+1];
            visPos[3*count+2] = pos[3*index+2];
            visCovRad[count] = specieCovRad[specie[index]];
            count++;
            
            index = splits[3*i+2];
            visPos[3*count] = pos[3*index];
            visPos[3*count+1] = pos[3*index+1];
            visPos[3*count+2] = pos[3*index+2];
            visCovRad[count] = specieCovRad[specie[index]];
            count++;
        }
        
        /* box vis atoms */
        boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
        putAtomsInBoxes(NVis, visPos, boxes);
        
        /* box index of picked pos */
        boxIndex = boxIndexOfAtom(pickPos[0], pickPos[1], pickPos[2], boxes);
        
        /* neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over neighbouring boxes, looking for nearest atom */
        minSep = 9999999.0;
        minSepIndex = -1;
        for (i=0; i<27; i++)
        {
            boxIndex = boxNebList[i];
            
            for (k=0; k<boxes->boxNAtoms[boxIndex]; k++)
            {
                index = boxes->boxAtoms[boxIndex][k];
                
                /* atomic separation */
                sep2 = atomicSeparation2(pickPos[0], pickPos[1], pickPos[2], 
                                         visPos[3*index], visPos[3*index+1], visPos[3*index+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                
                /* need radius too */
                rad = visCovRad[index];
                                
                sep = sqrt(sep2);
                
                /* if separation is greater than radius, subtract radius, 
                 * otherwise set to zero (within radius is good enough for me)
                 */
                sep = (sep > rad) ? sep - rad : 0.0;
                
                if (sep < minSep)
                {
                    minSep = sep;
                    minSepIndex = index;
                }
                
            }
        }
        
        if (minSepIndex < vacsDim)
        {
            minSepType = 1;
        }
        else if (minSepIndex < vacsDim + intsDim)
        {
            minSepIndex -= vacsDim;
            minSepType = 2;
        }
        else if (minSepIndex < vacsDim + intsDim + onAntsDim)
        {
            minSepIndex -= vacsDim + intsDim;
            minSepType = 3;
        }
        else
        {
            minSepIndex -= vacsDim + intsDim + onAntsDim;
            minSepIndex = (int) (minSepIndex / 3);
            minSepType = 4;
        }
        
        /* store result */
        result[0] = minSepType;
        result[1] = minSepIndex;
        result[2] = minSep;
        
        /* tidy up */
        freeBoxes(boxes);
        free(visPos);
        free(visCovRad);
    }
    
    return Py_BuildValue("i", 0);
}




