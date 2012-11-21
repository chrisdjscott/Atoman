/* vectors.i */

/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** SWIG interface file to vectors.c
 *******************************************************************************/

/* header */
%module vectors

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
%apply ( int DIM1, double* INPLACE_ARRAY1 ) {(int dim4, double* pos1),
                (int dim12, double* pos2),
                (int dim1, double* returnVector),
                (int dim2, double* cellDims)}

%apply (int DIM1, int* INPLACE_ARRAY1) {
	(int displLen, int* displAtoms)
}

/* define functions here */
%{
extern void separationVector( int dim1, double* returnVector, int length, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern void imageSeparationVector( int dim1, double *returnVector, int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern void imageSeparationVector_fixed( int dim1, double *returnVector, int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern double separationMagnitude( int length, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern double imageSeparationMagnitude( int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern double imageSeparationMagnitude_fixed( int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern double magnitude( int dim4, double* pos1 );

extern double dotProduct( int dim4, double* pos1, int dim12, double* pos2 );

extern void addVectorsInplace( int dim4, double* pos1, int dim12, double* pos2 );

extern void subtractVectors( int dim1, double* returnVector, int dim4, double* pos1, int dim12, double* pos2 );

extern void addVectors( int dim1, double* returnVector, int dim4, double* pos1, int dim12, double* pos2 );

extern void maxMovement( int dim1, double* returnVector, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern void scaleVector( int dim1, double* returnVector, int dim4, double* pos1, double factor );

extern void calcForceInfNorm(int dim1, double* returnVector, int dim4, double* pos1);

extern void applyPartialDisplacement( int dim4, double* pos1, int dim12, double* pos2, int displLen, int* displAtoms);

extern void applyPartialDisplacementReverse( int dim4, double* pos1, int dim12, double* pos2, int displLen, int* displAtoms);

extern double partialDotProduct( int dim4, double* pos1, int dim12, double* pos2, int displLen, int* displAtoms);
%}

extern void separationVector( int dim1, double* returnVector, int length, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern void imageSeparationVector( int dim1, double *returnVector, int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern void imageSeparationVector_fixed( int dim1, double *returnVector, int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern double separationMagnitude( int length, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern double imageSeparationMagnitude( int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern double magnitude( int dim4, double* pos1 );

extern double imageSeparationMagnitude_fixed( int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern double dotProduct( int dim4, double* pos1, int dim12, double* pos2 );

extern void addVectorsInplace( int dim4, double* pos1, int dim12, double* pos2 );

extern void subtractVectors( int dim1, double* returnVector, int dim4, double* pos1, int dim12, double* pos2 );

extern void maxMovement( int dim1, double* returnVector, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz );

extern void addVectors( int dim1, double* returnVector, int dim4, double* pos1, int dim12, double* pos2 );

extern void scaleVector( int dim1, double* returnVector, int dim4, double* pos1, double factor );

extern void calcForceInfNorm(int dim1, double* returnVector, int dim4, double* pos1);

extern void applyPartialDisplacement( int dim4, double* pos1, int dim12, double* pos2, int displLen, int* displAtoms);

extern void applyPartialDisplacementReverse( int dim4, double* pos1, int dim12, double* pos2, int displLen, int* displAtoms);

extern double partialDotProduct( int dim4, double* pos1, int dim12, double* pos2, int displLen, int* displAtoms);
