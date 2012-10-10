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
%apply ( int DIM1, double* IN_ARRAY1 ) {
    (int posDim, double* pos),
    (int refPosDim, double* refPos),
    (int minPosDim, double *minPos),
    (int maxPosDim, double *maxPos),
    (int cellDimsDim, double *cellDims)
}

%apply ( int DIM1, char* INPLACE_ARRAY1 ) {
    (int specieListDim, char* specieList),
    (int specieListRefDim, char* specieListRef)
}

%apply ( int DIM1, int* INPLACE_ARRAY1 ) {
    (int specieDim, int* specie),
    (int specieRefDim, int* specieRef),
    (int vacDim, int* vacancies),
    (int intDim, int* interstitials),
    (int antDim, int* antisites),
    (int onAntDim, int* onAntisites),
    (int NDefectsTypeDim, int* NDefectsType),
    (int exclSpecInputDim, int* exclSpecInput),
    (int exclSpecRefDim, int* exclSpecRef),
    (int PBCDim, int *PBC),
    (int defectClusterDim, int *defectCluster),
    (int vacSpecCountDim, int *vacSpecCount),
    (int intSpecCountDim, int *intSpecCount),
    (int antSpecCountDim, int *antSpecCount),
    (int splitIntDim, int *splitInterstitials)
}

%apply ( int DIM1, int DIM2, int *INPLACE_ARRAY2 ) {
    (int onAntSpecCntDim1, int onAntSpecCntDim2, int *onAntSpecCount),
    (int splitIntSpecCntDim1, int splitIntSpecCntDim2, int* splitIntSpecCount)
}

/* define functions here */
%{
extern int findDefects( int includeVacs, int includeInts, int includeAnts, int NDefectsTypeDim, int* NDefectsType, int vacDim, int* vacancies, 
                        int intDim, int* interstitials, int antDim, int* antisites, int onAntDim, int* onAntisites, int exclSpecInputDim, 
                        int* exclSpecInput, int exclSpecRefDim, int* exclSpecRef, int NAtoms, int specieListDim, char* specieList, 
                        int specieDim, int* specie, int posDim, double* pos, int refNAtoms, int specieListRefDim, char* specieListRef, 
                        int specieRefDim, int* specieRef, int refPosDim, double* refPos, int cellDimsDim, double *cellDims, int PBCDim, 
                        int *PBC, double vacancyRadius, int minPosDim, double *minPos, int maxPosDim, double *maxPos, int findClusters,
                        double clusterRadius, int defectClusterDim, int *defectCluster, int vacSpecCountDim, int *vacSpecCount, 
                        int intSpecCountDim, int *intSpecCount, int antSpecCountDim, int *antSpecCount, int onAntSpecCntDim1, 
                        int onAntSpecCntDim2, int *onAntSpecCount, int splitIntSpecCntDim1, int splitIntSpecCntDim2, int* splitIntSpecCount,
                        int minClusterSize, int maxClusterSize, int splitIntDim, int *splitInterstitials, int identifySplits);
%}

extern int findDefects( int includeVacs, int includeInts, int includeAnts, int NDefectsTypeDim, int* NDefectsType, int vacDim, int* vacancies, 
                        int intDim, int* interstitials, int antDim, int* antisites, int onAntDim, int* onAntisites, int exclSpecInputDim, 
                        int* exclSpecInput, int exclSpecRefDim, int* exclSpecRef, int NAtoms, int specieListDim, char* specieList, 
                        int specieDim, int* specie, int posDim, double* pos, int refNAtoms, int specieListRefDim, char* specieListRef, 
                        int specieRefDim, int* specieRef, int refPosDim, double* refPos, int cellDimsDim, double *cellDims, int PBCDim, 
                        int *PBC, double vacancyRadius, int minPosDim, double *minPos, int maxPosDim, double *maxPos, int findClusters,
                        double clusterRadius, int defectClusterDim, int *defectCluster, int vacSpecCountDim, int *vacSpecCount, 
                        int intSpecCountDim, int *intSpecCount, int antSpecCountDim, int *antSpecCount, int onAntSpecCntDim1, 
                        int onAntSpecCntDim2, int *onAntSpecCount, int splitIntSpecCntDim1, int splitIntSpecCntDim2, int* splitIntSpecCount,
                        int minClusterSize, int maxClusterSize, int splitIntDim, int *splitInterstitials, int identifySplits);
