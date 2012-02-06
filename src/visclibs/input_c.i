/* io_module.i */

/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** SWIG interface file to io_module.c
 *******************************************************************************/

/* header */
%module input_c

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
%apply (int DIM1, double* INPLACE_ARRAY1) {(int dim1, double* pos),
        (int dim2, double* charge),
        (int dim3, double* KE),
        (int dim4, double* PE),
        (int dim5, double* force),
        (int dim15, double* maxPos),
        (int dim16, double* minPos)}

/* 1d arrays of chars */
%apply (int DIM1, char* INPLACE_ARRAY1) {(int dim6, char* sym),
        (int dim7, char* specieList_c)}

/* 1d arrays of ints */
%apply (int DIM1, int* INPLACE_ARRAY1) {(int dim8, int* specieCount_c)}

/* define functions here */
%{
extern void readLatticeLBOMD( char* file, int dim6, char* sym, int dim1, double* pos, int dim2, double* charge, int dim7, char* specieList_c, int dim8, int* specieCount_c, int dim15, double* maxPos, int dim16, double* minPos, int verboseLevel );

extern void writeLatticeLBOMD( char* file, int NAtoms, double xdim, double ydim, double zdim, int dim6, char* sym, int dim1, double* pos, int dim2, double* charge );

extern void readRef( char* file, int dim6, char* sym, int dim1, double* pos, int dim2, double* charge, int dim3, double* KE, int dim4, double* PE, int dim5, double* force, int dim7, char* specieList_c, int dim8, int* specieCount_c, int dim15, double* maxPos, int dim16, double* minPos );
%}

extern void readLatticeLBOMD( char* file, int dim6, char* sym, int dim1, double* pos, int dim2, double* charge, int dim7, char* specieList_c, int dim8, int* specieCount_c, int dim15, double* maxPos, int dim16, double* minPos, int verboseLevel );

extern void writeLatticeLBOMD( char* file, int NAtoms, double xdim, double ydim, double zdim, int dim6, char* sym, int dim1, double* pos, int dim2, double* charge );

extern void readRef( char* file, int dim6, char* sym, int dim1, double* pos, int dim2, double* charge, int dim3, double* KE, int dim4, double* PE, int dim5, double* force, int dim7, char* specieList_c, int dim8, int* specieCount_c, int dim15, double* maxPos, int dim16, double* minPos );
