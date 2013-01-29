/* picker_c.i */

/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** SWIG interface file to picker_c.c
 *******************************************************************************/

/* header */
%module picker_c

%include carrays.i
%include typemaps.i
%include cstring.i

%{
#define SWIG_FILE_WITH_INIT
%}

/* required to pass numpy arrays */
%include "numpy.i"
%init %{
import_array();
%}

/* required to pass arrays of chars */
%numpy_typemaps(char, NPY_STRING, int)

/* define numpy arrays here */
/* 1d arrays of doubles */
%apply (int DIM1, double* INPLACE_ARRAY1) {
    (int posDim, double* pos),
    (int specieCovRadDim, double* specieCovRad),
    (int refPosDim, double *refPos),
    (int refSpecieCovRadDim, double* refSpecieCovRad),
    (int chargeDim, double* charge),
    (int pickPosDim, double *pickPos),
    (int cellDimsDim, double *cellDims),
    (int minPosDim, double *minPos),
    (int maxPosDim, double *maxPos),
    (int resultDim, double *result)
}

/* 2d array of doubles */
%apply (int DIM1, int DIM2, double *INPLACE_ARRAY2) {
    (int specieRGBDim1, int specieRGBDim2, double *specieRGB),
    (int refSpecieRGBDim1, int refSpecieRGBDim2, double *refSpecieRGB)
}

/* 1d arrays of chars */
%apply (int DIM1, char* INPLACE_ARRAY1) {
    (int dim6, char* sym),
    (int speclistDim, char* specieList_c)
}

/* 1d arrays of ints */
%apply (int DIM1, int* INPLACE_ARRAY1) {
    (int specieDim, int* specie),
    (int visibleAtomsDim, int *visibleAtoms),
    (int vacsDim, int *vacs),
    (int intsDim, int *ints),
    (int antsDim, int *ants),
    (int onAntsDim, int *onAnts),
    (int refSpecieDim, int *refSpecie),
    (int splitsDim, int *splits),
    (int PBCDim, int *PBC)
}

/* define functions here */
%{
extern int pickObject(int visibleAtomsDim, int *visibleAtoms, int vacsDim, int *vacs, int intsDim, int *ints, 
               int onAntsDim, int *onAnts, int splitsDim, int *splits, int pickPosDim, double *pickPos,
               int posDim, double *pos, int refPosDim, double *refPos, int PBCDim, int *PBC, 
               int cellDimsDim, double *cellDims, int minPosDim, double *minPos, int maxPosDim, double *maxPos,
               int specieDim, int* specie, int refSpecieDim, int *refSpecie, int specieCovRadDim, double* specieCovRad,
               int refSpecieCovRadDim, double* refSpecieCovRad, int resultDim, double *result);
%}

extern int pickObject(int visibleAtomsDim, int *visibleAtoms, int vacsDim, int *vacs, int intsDim, int *ints, 
               int onAntsDim, int *onAnts, int splitsDim, int *splits, int pickPosDim, double *pickPos,
               int posDim, double *pos, int refPosDim, double *refPos, int PBCDim, int *PBC, 
               int cellDimsDim, double *cellDims, int minPosDim, double *minPos, int maxPosDim, double *maxPos,
               int specieDim, int* specie, int refSpecieDim, int *refSpecie, int specieCovRadDim, double* specieCovRad,
               int refSpecieCovRadDim, double* refSpecieCovRad, int resultDim, double *result);


