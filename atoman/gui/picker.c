/*******************************************************************************
 ** Picker routines
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "visclibs/boxeslib.h"
#include "visclibs/utilities.h"
#include "visclibs/array_utils.h"

static PyObject* pickObject(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"pickObject", pickObject, METH_VARARGS, "Check if an object has been picked"},
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
    double *pickPos, *pos, *refPos, *cellDims, *specieCovRad, *refSpecieCovRad, *result;  
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
    PyArrayObject *specieIn=NULL;
    PyArrayObject *refSpecieIn=NULL;
    PyArrayObject *specieCovRadIn=NULL;
    PyArrayObject *refSpecieCovRadIn=NULL;
    PyArrayObject *resultIn=NULL;
    
    int minSepIndex, minSepType;
    double approxBoxWidth, minSepRad, minSep2, minSep;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!O!O!O!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &vacsIn, 
            &PyArray_Type, &intsIn, &PyArray_Type, &onAntsIn, &PyArray_Type, &splitsIn, &PyArray_Type, &pickPosIn, &PyArray_Type, 
            &posIn, &PyArray_Type, &refPosIn, &PyArray_Type, &PBCIn, &PyArray_Type, &cellDimsIn, &PyArray_Type, &specieIn,
            &PyArray_Type, &refSpecieIn, &PyArray_Type, &specieCovRadIn, &PyArray_Type, &refSpecieCovRadIn, &PyArray_Type,
            &resultIn))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    visibleAtomsDim = (int) PyArray_DIM(visibleAtomsIn, 0);
    
    if (not_intVector(vacsIn)) return NULL;
    vacs = pyvector_to_Cptr_int(vacsIn);
    vacsDim = (int) PyArray_DIM(vacsIn, 0);
    
    if (not_intVector(intsIn)) return NULL;
    ints = pyvector_to_Cptr_int(intsIn);
    intsDim = (int) PyArray_DIM(intsIn, 0);
    
    if (not_intVector(onAntsIn)) return NULL;
    onAnts = pyvector_to_Cptr_int(onAntsIn);
    onAntsDim = (int) PyArray_DIM(onAntsIn, 0);
    
    if (not_intVector(splitsIn)) return NULL;
    splits = pyvector_to_Cptr_int(splitsIn);
    splitsDim = (int) PyArray_DIM(splitsIn, 0);
    
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
    
    /* initialise */
    minSep2 = 9999999.0;
    minSepIndex = -1;
    minSepRad = 0.0;
    minSepType = 0;
    
    /* pick atoms (if there are any) */
    if (visibleAtomsDim > 0)
    {
        int i, boxIndex, boxNebList[27], boxstat;
        int boxNebListSize;
        double *visPos;
        struct Boxes *boxes;
        
#ifdef DEBUG
        printf("PICKC: Picking atoms\n");
#endif
        
        /* vis atoms pos */
        visPos = malloc(3 * visibleAtomsDim * sizeof(double));
        if (visPos == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not allocate visPos");
            return NULL;
        }
        
        for (i = 0; i < visibleAtomsDim; i++)
        {
            int i3 = 3 * i;
            int index = visibleAtoms[i];
            int index3 = 3 * index;
            visPos[i3    ] = pos[index3    ];
            visPos[i3 + 1] = pos[index3 + 1];
            visPos[i3 + 2] = pos[index3 + 2];
        }
        
        /* box vis atoms */
        boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
        if (boxes == NULL)
        {
            free(visPos);
            return NULL;
        }
        boxstat = putAtomsInBoxes(visibleAtomsDim, visPos, boxes);
        if (boxstat)
        {
            free(visPos);
            return NULL;
        }
        
        /* box index of picked pos */
        boxIndex = boxIndexOfAtom(pickPos[0], pickPos[1], pickPos[2], boxes);
        if (boxIndex < 0)
        {
            free(visPos);
            freeBoxes(boxes);
            return NULL;
        }
        
        /* neighbouring boxes */
        boxNebListSize = getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over neighbouring boxes, looking for nearest atom */
        for (i = 0; i < boxNebListSize; i++)
        {
            int k;
            int checkBox = boxNebList[i];
            
            for (k = 0; k < boxes->boxNAtoms[checkBox]; k++)
            {
                int index, realIndex;
                double sep2, rad;
                
                index = boxes->boxAtoms[checkBox][k];
                
                /* atomic separation */
                sep2 = atomicSeparation2(pickPos[0], pickPos[1], pickPos[2], 
                                         visPos[3*index], visPos[3*index+1], visPos[3*index+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                
                /* need radius too */
                realIndex = visibleAtoms[index];
                rad = specieCovRad[specie[realIndex]];
                
                if (sep2 < minSep2)
                {
                    minSep2 = sep2;
                    minSepIndex = index;
                    minSepRad = rad;
                }
            }
        }
        
        /* free memory */
        free(visPos);
        freeBoxes(boxes);
    }
    
    /* now check defects (if there are any) */
    if (vacsDim + intsDim + onAntsDim + splitsDim > 0)
    {
        int i, NVis, count, boxNebListSize, minSepIsDefect;
        int boxIndex, boxNebList[27], boxstat;
        double *visPos, *visCovRad;
        struct Boxes *boxes;
        
#ifdef DEBUG
        printf("PICKC: Picking defect\n");
#endif
        
        /* build positions array of all defects */
        NVis = vacsDim + intsDim + onAntsDim + splitsDim;
        visPos = malloc(3 * NVis * sizeof(double));
        if (visPos == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not allocate visPos");
            return NULL;
        }
        
        visCovRad = malloc(NVis * sizeof(double));
        if (visCovRad == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not allocate visCovRad");
            free(visPos);
            return NULL;
        }
        
        /* add defects positions: vac then int then ant */
        count = 0;
        for (i = 0; i < vacsDim; i++)
        {
            int c3 = 3 * count;
            int index = vacs[i];
            int index3 = 3 * index;
            
            visPos[c3    ] = refPos[index3    ];
            visPos[c3 + 1] = refPos[index3 + 1];
            visPos[c3 + 2] = refPos[index3 + 2];
            
            visCovRad[count++] = refSpecieCovRad[refSpecie[index]] * 1.2; // * 0.75; // multiply by 0.75 because vacs aren't full size (rendering.py)
        }
        
        for (i = 0; i < intsDim; i++)
        {
            int c3 = 3 * count;
            int index = ints[i];
            int index3 = 3 * index;
            
            visPos[c3    ] = pos[index3    ];
            visPos[c3 + 1] = pos[index3 + 1];
            visPos[c3 + 2] = pos[index3 + 2];
            
            visCovRad[count++] = specieCovRad[specie[index]];
        }
        
        for (i = 0; i < onAntsDim; i++)
        {
            int c3 = 3 * count;
            int index = onAnts[i];
            int index3 = 3 * index;
            
            visPos[c3    ] = pos[index3    ];
            visPos[c3 + 1] = pos[index3 + 1];
            visPos[c3 + 2] = pos[index3 + 2];
            
            visCovRad[count++] = specieCovRad[specie[index]];
        }
        
        for (i = 0; i < splitsDim / 3; i++)
        {
            int i3 = 3 * i;
            int index, c3, index3;
            
            index = splits[i3    ];
            index3 = index * 3;
            c3 = count * 3;
            visPos[c3    ] = refPos[index3    ];
            visPos[c3 + 1] = refPos[index3 + 1];
            visPos[c3 + 2] = refPos[index3 + 2];
            visCovRad[count++] = refSpecieCovRad[refSpecie[index]];
            
            index = splits[i3 + 1];
            index3 = index * 3;
            c3 = count * 3;
            visPos[c3    ] = pos[index3    ];
            visPos[c3 + 1] = pos[index3 + 1];
            visPos[c3 + 2] = pos[index3 + 2];
            visCovRad[count++] = specieCovRad[specie[index]];
            
            index = splits[i3 + 2];
            index3 = index * 3;
            c3 = count * 3;
            visPos[c3    ] = pos[index3    ];
            visPos[c3 + 1] = pos[index3 + 1];
            visPos[c3 + 2] = pos[index3 + 2];
            visCovRad[count++] = specieCovRad[specie[index]];
        }
        
        /* box vis atoms */
        boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
        if (boxes == NULL)
        {
            free(visPos);
            free(visCovRad);
            return NULL;
        }
        boxstat = putAtomsInBoxes(NVis, visPos, boxes);
        if (boxstat)
        {
            free(visPos);
            free(visCovRad);
            return NULL;
        }
        
        /* box index of picked pos */
        boxIndex = boxIndexOfAtom(pickPos[0], pickPos[1], pickPos[2], boxes);
        if (boxIndex < 0)
        {
            free(visPos);
            free(visCovRad);
            freeBoxes(boxes);
            return NULL;
        }
        
        /* neighbouring boxes */
        boxNebListSize = getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over neighbouring boxes, looking for nearest atom */
        minSepIsDefect = 0;
        for (i = 0; i < boxNebListSize; i++)
        {
            int k;
            int checkBox = boxNebList[i];
            
            for (k = 0; k < boxes->boxNAtoms[checkBox]; k++)
            {
                int index;
                double sep2, rad;
                
                index = boxes->boxAtoms[checkBox][k];
                
                /* atomic separation */
                sep2 = atomicSeparation2(pickPos[0], pickPos[1], pickPos[2], 
                                         visPos[3*index], visPos[3*index+1], visPos[3*index+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                
                /* need radius too */
                rad = visCovRad[index];
                
                if (sep2 < minSep2)
                {
                    minSep2 = sep2;
                    minSepIndex = index;
                    minSepRad = rad;
                    minSepIsDefect = 1;
                }
            }
        }
        
        if (minSepIsDefect)
        {
            /* picked type */
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
        }
        
        /* free memory */
        freeBoxes(boxes);
        free(visPos);
        free(visCovRad);
    }
    
    /* min separation */
    minSep = sqrt(minSep2);
    /* if separation is greater than radius, subtract radius, 
     * otherwise set to zero (within radius is good enough for me)
     */
    minSep = (minSep > minSepRad) ? minSep - minSepRad : 0.0;
    
    /* store result */
    result[0] = minSepType;
    result[1] = minSepIndex;
    result[2] = minSep;
    
#ifdef DEBUG
    printf("PICKC: End\n");
#endif
    
    return Py_BuildValue("i", 0);
}
