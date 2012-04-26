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
%apply (int DIM1, double* INPLACE_ARRAY1) {(int posDim, double* pos),
        (int chargeDim, double* charge),
        (int KEDim, double* KE),
        (int PEDim, double* PE),
        (int forceDim, double* force),
        (int maxPosDim, double* maxPos),
        (int minPosDim, double* minPos)}

/* 1d arrays of chars */
%apply (int DIM1, char* INPLACE_ARRAY1) {(int dim6, char* sym),
        (int speclistDim, char* specieList_c)}

/* 1d arrays of ints */
%apply (int DIM1, int* INPLACE_ARRAY1) {(int specCountDim, int* specieCount_c),
                                        (int specieDim, int* specie)}

/* define functions here */
%{
extern void readLatticeLBOMD( char* file, int specieDim, int* specie, int posDim, double* pos, int chargeDim, double* charge, int speclistDim, char* specieList_c, int specCountDim, int* specieCount_c, int maxPosDim, double* maxPos, int minPosDim, double* minPos );

extern void writeLatticeLBOMD( char* file, int NAtoms, double xdim, double ydim, double zdim, int speclistDim, char* specieList_c, int specieDim, int* specie, int posDim, double* pos, int chargeDim, double* charge );

extern void readRef( char* file, int specieDim, int* specie, int posDim, double* pos, int chargeDim, double* charge, int KEDim, double* KE, int PEDim, double* PE, int forceDim, double* force, int speclistDim, char* specieList_c, int specCountDim, int* specieCount_c, int maxPosDim, double* maxPos, int minPosDim, double* minPos );

extern void readLBOMDXYZ( char* file, int posDim, double* pos, int chargeDim, double* charge, int KEDim, double* KE, int PEDim, double* PE, int forceDim, double* force, int maxPosDim, double* maxPos, int minPosDim, double* minPos, int xyzformat );
%}

extern void readLatticeLBOMD( char* file, int specieDim, int* specie, int posDim, double* pos, int chargeDim, double* charge, int speclistDim, char* specieList_c, int specCountDim, int* specieCount_c, int maxPosDim, double* maxPos, int minPosDim, double* minPos );

extern void writeLatticeLBOMD( char* file, int NAtoms, double xdim, double ydim, double zdim, int speclistDim, char* specieList_c, int specieDim, int* specie, int posDim, double* pos, int chargeDim, double* charge );

extern void readRef( char* file, int specieDim, int* specie, int posDim, double* pos, int chargeDim, double* charge, int KEDim, double* KE, int PEDim, double* PE, int forceDim, double* force, int speclistDim, char* specieList_c, int specCountDim, int* specieCount_c, int maxPosDim, double* maxPos, int minPosDim, double* minPos );

extern void readLBOMDXYZ( char* file, int posDim, double* pos, int chargeDim, double* charge, int KEDim, double* KE, int PEDim, double* PE, int forceDim, double* force, int maxPosDim, double* maxPos, int minPosDim, double* minPos, int xyzformat );
