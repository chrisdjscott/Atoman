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
%apply (int DIM1, double* INPLACE_ARRAY1) {(int posDim, double* pos)}

/* 1d arrays of chars */
%apply (int DIM1, char* INPLACE_ARRAY1) {(int dim6, char* sym),
        (int speclistDim, char* specieList_c)}

/* 1d arrays of ints */
%apply (int DIM1, int* INPLACE_ARRAY1) {(int specieDim, int* specie),
                                        (int NVisibleIn, int* visibleAtoms),
                                        (int visSpecDim, int* visSpec)}

/* define functions here */
%{
    extern int specieFilter(int NVisibleIn, int *visibleAtoms, int visSpecDim, int* visSpec, int specieDim, int *specie);
%}

extern int specieFilter(int NVisibleIn, int *visibleAtoms, int visSpecDim, int* visSpec, int specieDim, int *specie);
