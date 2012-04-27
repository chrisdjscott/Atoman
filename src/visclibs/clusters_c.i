/* clusters_c.i */

/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** SWIG interface file to clusters_c.c
 *******************************************************************************/

/* header */
%module clusters_c

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
%apply ( int DIM1, double* INPLACE_ARRAY1 ) {(int posDim, double* pos),
                                             (int refPosDim, double* refPos),
                                             (int cellDimsDim, double *cellDims),
                                             (int minPosDim, double *minPos),
                                             (int maxPosDim, double *maxPos)}

%apply ( int DIM1, char* INPLACE_ARRAY1 ) {(int specieListDim, char* specieList),
                                           (int specieListRefDim, char* specieListRef)}

%apply ( int DIM1, int* INPLACE_ARRAY1 ) {(int specieDim, int* specie),
                                          (int specieRefDim, int* specieRef),
                                          (int visibleAtomsDim, int *visibleAtoms),
                                          (int clusterArrayDim, int *clusterArray),
                                          (int PBCDim, int *PBC),
                                          (int resultsDim, int *results)}

/* define functions here */
%{
extern void findClusters(int visibleAtomsDim, int *visibleAtoms, int posDim, double *pos, int clusterArrayDim, int *clusterArray, double neighbourRad, 
                         int cellDimsDim, double *cellDims, int PBCDim, int *PBC, int minPosDim, double *minPos, int maxPosDim, double *maxPos, 
                         int minClusterSize, int resultsDim, int *results);
%}

extern void findClusters(int visibleAtomsDim, int *visibleAtoms, int posDim, double *pos, int clusterArrayDim, int *clusterArray, double neighbourRad, 
                         int cellDimsDim, double *cellDims, int PBCDim, int *PBC, int minPosDim, double *minPos, int maxPosDim, double *maxPos, 
                         int minClusterSize, int resultsDim, int *results);
