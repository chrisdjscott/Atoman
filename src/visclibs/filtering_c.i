/* filtering_c.i */

/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** SWIG interface file to filtering_c.c
 *******************************************************************************/

/* header */
%module filtering_c

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
    (int refPosDim, double *refPos),
    (int cellDimsDim, double *cellDims),
    (int KEDim, double *KE),
    (int PEDim, double *PE),
    (int chargeDim, double *charge),
    (int scalarsDim, double *scalars)
}

/* 1d arrays of chars */
%apply (int DIM1, char* INPLACE_ARRAY1) {
    (int speclistDim, char* specieList_c)
}

/* 1d arrays of ints */
%apply (int DIM1, int* INPLACE_ARRAY1) {
    (int specieDim, int* specie),
    (int NVisibleIn, int* visibleAtoms),
    (int visSpecDim, int* visSpec),
    (int PBCDim, int *PBC)
}

/* define functions here */
%{
extern int specieFilter(int NVisibleIn, int *visibleAtoms, int visSpecDim, int* visSpec, int specieDim, int *specie);
    
extern int cropFilter(int NVisibleIn, int* visibleAtoms, int posDim, double* pos, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, int xEnabled, int yEnabled, int zEnabled);

extern int displacementFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int posDim, double *pos, int refPosDim, double *refPos, 
                              int cellDimsDim, double *cellDims, int PBCDim, int *PBC, double minDisp, double maxDisp);

extern int KEFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int KEDim, double *KE, double minKE, double maxKE);

extern int PEFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int PEDim, double *PE, double minPE, double maxPE);

extern int chargeFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int chargeDim, double *charge, double minCharge, double maxCharge);

extern int cropSphereFilter(int NVisibleIn, int *visibleAtoms, int posDim, double *pos, double xCentre, 
                            double yCentre, double zCentre, double radius, int cellDimsDim, double *cellDims, 
                            int PBCDim, int *PBC, int invertSelection);
%}

extern int specieFilter(int NVisibleIn, int *visibleAtoms, int visSpecDim, int* visSpec, int specieDim, int *specie);

extern int cropFilter(int NVisibleIn, int* visibleAtoms, int posDim, double* pos, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, int xEnabled, int yEnabled, int zEnabled);

extern int displacementFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int posDim, double *pos, int refPosDim, double *refPos, 
                              int cellDimsDim, double *cellDims, int PBCDim, int *PBC, double minDisp, double maxDisp);

extern int KEFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int KEDim, double *KE, double minKE, double maxKE);

extern int PEFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int PEDim, double *PE, double minPE, double maxPE);

extern int chargeFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int chargeDim, double *charge, double minCharge, double maxCharge);

extern int cropSphereFilter(int NVisibleIn, int *visibleAtoms, int posDim, double *pos, double xCentre, 
                            double yCentre, double zCentre, double radius, int cellDimsDim, double *cellDims, 
                            int PBCDim, int *PBC, int invertSelection);
