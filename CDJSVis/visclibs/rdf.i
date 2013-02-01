/* rdf.i */

/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** SWIG interface file to rdf.c
 *******************************************************************************/

/* header */
%module rdf

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
%apply ( int DIM1, double* INPLACE_ARRAY1 ) {
    (int posDim, double* pos),
    (int minPosDim, double *minPos),
    (int maxPosDim, double *maxPos),
    (int cellDimsDim, double *cellDims),
    (int rdfDim, double *rdf),
    (int resultDim, double *result)
}

%apply ( int DIM1, char* INPLACE_ARRAY1 ) {
    (int specieListDim, char* specieList),
    (int specieListRefDim, char* specieListRef)
}

%apply ( int DIM1, int* INPLACE_ARRAY1 ) {
    (int specieDim, int* specie),
    (int PBCDim, int *PBC),
    (int NVisible, int *visibleAtoms)
}

%apply ( int DIM1, int DIM2, int *INPLACE_ARRAY2 ) {
    (int onAntSpecCntDim1, int onAntSpecCntDim2, int *onAntSpecCount),
    (int splitIntSpecCntDim1, int splitIntSpecCntDim2, int* splitIntSpecCount)
}

/* define functions here */
%{
extern int calculateRDF(int NVisible, int *visibleAtoms, int specieDim, int *specie, int posDim, double *pos, int specieID1, int specieID2, 
                        int minPosDim, double *minPos, int maxPosDim, double *maxPos, int cellDimsDim, double *cellDims, int PBCDim, int *PBC,
                        double start, double finish, int num, int rdfDim, double *rdf);
%}


extern int calculateRDF(int NVisible, int *visibleAtoms, int specieDim, int *specie, int posDim, double *pos, int specieID1, int specieID2, 
                        int minPosDim, double *minPos, int maxPosDim, double *maxPos, int cellDimsDim, double *cellDims, int PBCDim, int *PBC,
                        double start, double finish, int num, int rdfDim, double *rdf);
