/* defects_c.i */

/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** SWIG interface file to defects.c
 *******************************************************************************/

/* header */
%module defects_c

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
%apply ( int DIM1, double* IN_ARRAY1 ) {(int posDim, double* pos),
                                        (int refPosDim, double* refPos)}

%apply ( int DIM1, char* INPLACE_ARRAY1 ) {(int specieListDim, char* specieList),
                                           (int specieListRefDim, char* specieListRef)}

%apply ( int DIM1, int* INPLACE_ARRAY1 ) {(int specieDim, int* specie),
                                          (int specieRefDim, int* specieRef),
                                          (int vacDim, int* vacancies),
                                          (int intDim, int* interstitials),
                                          (int antDim, int* antisites),
                                          (int onAntDim, int* onAntisites),
                                          (int NDefectsTypeDim, int* NDefectsType),
                                          (int exclSpecInputDim, int* exclSpecInput),
                                          (int exclSpecRefDim, int* exclSpecRef)}

/* define functions here */
%{
extern int findDefects( int includeVacs, int includeInts, int includeAnts, int NDefectsTypeDim, int* NDefectsType, int vacDim, int* vacancies, 
                        int intDim, int* interstitials, int antDim, int* antisites, int onAntDim, int* onAntisites, int exclSpecInputDim, 
                        int* exclSpecInput, int exclSpecRefDim, int* exclSpecRef, int NAtoms, int specieListDim, char* specieList, 
                        int specieDim, int* specie, int posDim, double* pos, int refNAtoms, int specieListRefDim, char* specieListRef, 
                        int specieRefDim, int* specieRef, int refPosDim, double* refPos, double xdim, double ydim, double zdim, int pbcx, 
                        int pbcy, int pbcz, double vacancyRadius, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax );
%}

extern int findDefects( int includeVacs, int includeInts, int includeAnts, int NDefectsTypeDim, int* NDefectsType, int vacDim, int* vacancies, 
                        int intDim, int* interstitials, int antDim, int* antisites, int onAntDim, int* onAntisites, int exclSpecInputDim, 
                        int* exclSpecInput, int exclSpecRefDim, int* exclSpecRef, int NAtoms, int specieListDim, char* specieList, 
                        int specieDim, int* specie, int posDim, double* pos, int refNAtoms, int specieListRefDim, char* specieListRef, 
                        int specieRefDim, int* specieRef, int refPosDim, double* refPos, double xdim, double ydim, double zdim, int pbcx, 
                        int pbcy, int pbcz, double vacancyRadius, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax );
