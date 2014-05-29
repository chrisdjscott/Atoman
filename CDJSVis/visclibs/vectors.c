
/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Find defects and return the sub-system surrounding them
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "vectors.h"


/*******************************************************************************
 * eliminate pbc flicker
 *******************************************************************************/
int eliminatePBCFlicker(int NAtoms, double *pos, double *previousPos, double *cellDims, int *pbc)
{
    int i, j, count;
    double sep, absSep, halfDims[3];
    
    
    /* half of cellDims */
    halfDims[0] = cellDims[0] * 0.5;
    halfDims[1] = cellDims[1] * 0.5;
    halfDims[2] = cellDims[2] * 0.5;
    
    /* loop over atoms */
    count = 0;
    for (i = 0; i < NAtoms; i++)
    {
        sep = atomicSeparation2(pos[3*i], pos[3*i+1], pos[3*i+2], previousPos[3*i], previousPos[3*i+1], previousPos[3*i+2], 
                                cellDims[0], cellDims[1], cellDims[2], pbc[0], pbc[1], pbc[2]);
        
        for (j = 0; j < 3; j++)
        {
            if (sep < 1.0 && pos[3*i+j] < 1.0)
            {
                absSep = fabs(pos[3*i+j] - previousPos[3*i+j]);
                if (absSep > halfDims[j])
                {
                    pos[3*i+j] += cellDims[j];
                    count++;
                }
            }
            
            else if (sep < 1.0 && fabs(cellDims[j] - pos[3*i+j]) < 1.0)
            {
                absSep = fabs(pos[3*i+j] - previousPos[3*i+j]);
                if (absSep > halfDims[j])
                {
                    pos[3*i+j] -= cellDims[j];
                    count++;
                }
            }
        }
    }
    
    return count;
}

/*******************************************************************************
 * Calc infinity norm of force array
 *******************************************************************************/
void calcForceInfNorm(int dim1, double* returnVector, int dim4, double* pos1)
{
    int i, maxIndex, NAtoms;
    double max, mag;
    
    NAtoms = dim4 / 3;
    
    max = -1.0;
    maxIndex = -1;
    for (i=0; i<NAtoms; i++)
    {
        mag = pos1[3*i] * pos1[3*i] + pos1[3*i+1] * pos1[3*i+1] + pos1[3*i+2] * pos1[3*i+2];
        
        if (mag > max)
        {
            max = mag;
            maxIndex = i;
        }
    }
    
    max = sqrt(max);
    
    returnVector[0] = max;
    returnVector[1] = maxIndex;
}


/*******************************************************************************
 * Find max moved atom and average movement
 *******************************************************************************/
void maxMovement( int dim1, double* returnVector, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz )
{
    int i, N;
    double r2, r;
    double sum, avgMove, sumSep;
    double maxMove, maxMoveIndex;
    
    N = dim4 / 3;
    
    /* loop over atoms and calculate displacement from original */
    maxMove = -1;
    maxMoveIndex = -1;
    sum = 0.0;
    sumSep = 0.0;
    for (i=0; i<N; i++)
    {
        /* separation between current and initial positions */
        r2 = atomicSeparation2( pos1[3*i], pos1[3*i+1], pos1[3*i+2], pos2[3*i], pos2[3*i+1], pos2[3*i+2], cellDims[0], cellDims[4], cellDims[8], pbcx, pbcy, pbcz );
        r = sqrt(r2);
        
        /* is this the max movement */
        if (r > maxMove)
        {
            maxMove = r;
            maxMoveIndex = i;
        }
        
        /* sum movements */
        sum += r;
        
        /* sum square of movements to get separation */
        sumSep += r2;
    }
    
    /* average movement */
    avgMove = sum / N;
    
    returnVector[0] = maxMoveIndex;
    returnVector[1] = maxMove;
    returnVector[2] = avgMove;
    returnVector[3] = sqrt(sumSep);
}


/*******************************************************************************
 * Find separation vector between two pos vectors
 *******************************************************************************/
int separationVector(int length, double* returnVector, double* pos1, double* pos2, double* cellDims, int *PBC)
{
    int i;
    double atomSepVec[3];
    
    
    for ( i=0; i<length; i++ )
    {
        atomSeparationVector(atomSepVec, pos1[3*i], pos1[3*i+1], pos1[3*i+2], pos2[3*i], pos2[3*i+1], pos2[3*i+2], 
                             cellDims[0], cellDims[4], cellDims[8], PBC[0], PBC[1], PBC[2]);
        
        returnVector[3*i] = atomSepVec[0];
        returnVector[3*i+1] = atomSepVec[1];
        returnVector[3*i+2] = atomSepVec[2];
    }
    
    return 0;
}


/*******************************************************************************
 * return magnitude of separation vector between two pos vectors
 *******************************************************************************/
double separationMagnitude(int length, double* pos1, double* pos2, double* cellDims, int *PBC)
{
    int i;
    double sum, r2;
    
    
    sum = 0;
    for ( i=0; i<length; i++ )
    {
        r2 = atomicSeparation2( pos1[3*i], pos1[3*i+1], pos1[3*i+2], pos2[3*i], pos2[3*i+1], pos2[3*i+2], cellDims[0], cellDims[4], cellDims[8], PBC[0], PBC[1], PBC[2]);
        
        sum += r2;
    }
    
    return sqrt(sum);
}


/*******************************************************************************
 * image separation (both moving)
 *******************************************************************************/
void imageSeparationVector( int dim1, double *returnVector, int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz )
{
    int i, offset1, offset2;
    double atomSepVec[3];
    
    
    /* all positions are stored in one array, so calculate offsets */
    offset1 = 3 * (image1 - 1) * imageNAtoms;
    offset2 = 3 * (image2 - 1) * imageNAtoms;
    
    for ( i=0; i<imageNAtoms; i++ )
    {
        atomSeparationVector( atomSepVec, pos1[offset1+3*i], pos1[offset1+3*i+1], pos1[offset1+3*i+2], pos1[offset2+3*i], pos1[offset2+3*i+1], pos1[offset2+3*i+2], cellDims[0], cellDims[4], cellDims[8], pbcx, pbcy, pbcz );
        
        returnVector[3*i] = atomSepVec[0];
        returnVector[3*i+1] = atomSepVec[1];
        returnVector[3*i+2] = atomSepVec[2];
    }
}


/*******************************************************************************
 * image separation (fixed end point)
 *******************************************************************************/
void imageSeparationVector_fixed( int dim1, double *returnVector, int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz )
{
    int i, offset;
    double atomSepVec[3];
    
    
    /* image1 is initial state */
    if ( image1 == 0 )
    {
        offset = 3 * (image2 - 1) * imageNAtoms;
        
        for ( i=0; i<imageNAtoms; i++ )
        {
            atomSeparationVector( atomSepVec, pos1[3*i], pos1[3*i+1], pos1[3*i+2], pos2[offset+3*i], pos2[offset+3*i+1], pos2[offset+3*i+2], cellDims[0], cellDims[4], cellDims[8], pbcx, pbcy, pbcz );
            
            returnVector[3*i] = atomSepVec[0];
            returnVector[3*i+1] = atomSepVec[1];
            returnVector[3*i+2] = atomSepVec[2];
        }
    }
    /* otherwise assume image2 is finalState image */
    else
    {
        offset = 3 * (image1 - 1) * imageNAtoms;
        
        for ( i=0; i<imageNAtoms; i++ )
        {
            atomSeparationVector( atomSepVec, pos1[offset+3*i], pos1[offset+3*i+1], pos1[offset+3*i+2], pos2[3*i], pos2[3*i+1], pos2[3*i+2], cellDims[0], cellDims[4], cellDims[8], pbcx, pbcy, pbcz );
            
            returnVector[3*i] = atomSepVec[0];
            returnVector[3*i+1] = atomSepVec[1];
            returnVector[3*i+2] = atomSepVec[2];
        }
    }
}


/*******************************************************************************
 * magnitude image separation (both moving)
 *******************************************************************************/
double imageSeparationMagnitude( int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz )
{
    int i, offset1, offset2;
    double r2, sum;
    
    
    /* all positions are stored in one array, so calculate offsets */
    offset1 = 3 * (image1 - 1) * imageNAtoms;
    offset2 = 3 * (image2 - 1) * imageNAtoms;
    
    sum = 0;
    for ( i=0; i<imageNAtoms; i++ )
    {
        r2 = atomicSeparation2( pos1[offset1+3*i], pos1[offset1+3*i+1], pos1[offset1+3*i+2], pos1[offset2+3*i], pos1[offset2+3*i+1], pos1[offset2+3*i+2], cellDims[0], cellDims[4], cellDims[8], pbcx, pbcy, pbcz );
        
        sum += r2;
    }
    
    return sqrt(sum);
}


/*******************************************************************************
 * magnitude of image separation (fixed end point)
 *******************************************************************************/
double imageSeparationMagnitude_fixed( int image1, int image2, int imageNAtoms, int dim4, double* pos1, int dim12, double* pos2, int dim2, double* cellDims, int pbcx, int pbcy, int pbcz )
{
    int i, offset;
    double r2, sum;
    
    
    sum = 0;
    
    /* image1 is initial state */
    if ( image1 == 0 )
    {
        offset = 3 * (image2 - 1) * imageNAtoms;
        
        for ( i=0; i<imageNAtoms; i++ )
        {
            r2 = atomicSeparation2( pos1[3*i], pos1[3*i+1], pos1[3*i+2], pos2[offset+3*i], pos2[offset+3*i+1], pos2[offset+3*i+2], cellDims[0], cellDims[4], cellDims[8], pbcx, pbcy, pbcz );
            
            sum += r2;
        }
    }
    /* otherwise assume image2 is finalState image */
    else
    {
        offset = 3 * (image1 - 1) * imageNAtoms;
        
        for ( i=0; i<imageNAtoms; i++ )
        {
            r2 = atomicSeparation2( pos1[offset+3*i], pos1[offset+3*i+1], pos1[offset+3*i+2], pos2[3*i], pos2[3*i+1], pos2[3*i+2], cellDims[0], cellDims[4], cellDims[8], pbcx, pbcy, pbcz );
            
            sum += r2;
        }
    }
    
    return sqrt(sum);
}


/*******************************************************************************
 * Return the magnitude of given vector
 *******************************************************************************/
double magnitude(int length, double* pos1)
{
    int i;
    double sum;
    
    
    sum = 0.0;
    for ( i=0; i<length; i++ )
    {
        sum += pos1[i] * pos1[i];
    }
    
    return sqrt(sum);
}


/*******************************************************************************
 * Add two vectors. Result is saved in pos1.
 *******************************************************************************/
void addVectorsInplace( int dim4, double* pos1, int dim12, double* pos2 )
{
    int i;
    
    
    for ( i=0; i<dim4; i++ )
    {
        pos1[i] += pos2[i];
    }
}


/*******************************************************************************
 * Scale vector 1, modify in place
 *******************************************************************************/
void scaleVector( int dim1, double* returnVector, int dim4, double* pos1, double factor )
{
    int i;
    
    
    for ( i=0; i<dim4; i++ )
    {
        returnVector[i] = pos1[i] * factor;
    }
}


/*******************************************************************************
 * Subtract two vectors. Result is saved in returnVector.
 *******************************************************************************/
void subtractVectors( int dim1, double* returnVector, int dim4, double* pos1, int dim12, double* pos2 )
{
    int i;
    

    for ( i=0; i<dim4; i++ )
    {
        returnVector[i] = pos1[i] - pos2[i];
    }
}


/*******************************************************************************
 * Add two vectors. Result is saved in returnVector.
 *******************************************************************************/
void addVectors( int dim1, double* returnVector, int dim4, double* pos1, int dim12, double* pos2 )
{
    int i;
    

    for ( i=0; i<dim4; i++ )
    {
        returnVector[i] = pos1[i] + pos2[i];
    }
}


/*******************************************************************************
 * Return dot product of given vectors
 *******************************************************************************/
double dotProduct( int dim4, double* pos1, int dim12, double* pos2 )
{
    int i;
    double sum;


    sum = 0.0;
    for ( i=0; i<dim4; i++ )
    {
        sum += pos1[i] * pos2[i];
    }

    return sum;
}

/*******************************************************************************
 *
 *******************************************************************************/
void applyPartialDisplacement( int dim4, double* pos1, int dim12, double* pos2,
        int displLen, int* displAtoms)
{
    int i, j, atom;

    if ((dim12 == (displLen*3)) && (displLen > 0)) {

        for ( i = 0; i < displLen; i++ ) {
            atom = displAtoms[i];

            for ( j = 0; j < 3; j++ ) {
                pos1[3*atom+j] += pos2[3*i+j];
            }
        }
    }
}

/*******************************************************************************
 *
 *******************************************************************************/
void applyPartialDisplacementReverse( int dim4, double* pos1, int dim12, double* pos2,
        int displLen, int* displAtoms)
{
    int i, j, atom;

    if ((dim12 == (displLen*3)) && (displLen > 0)) {

        for ( i = 0; i < displLen; i++ ) {
            atom = displAtoms[i];

            for ( j = 0; j < 3; j++ ) {
                pos2[3*i+j] += pos1[3*atom+j];
            }
        }
    }
}

/*******************************************************************************
 *
 *******************************************************************************/
double partialDotProduct( int dim4, double* pos1, int dim12, double* pos2,
        int displLen, int* displAtoms)
{
    double sum = 0.0;
    int i, j, atom;

    if ((dim12 == (displLen*3)) && (displLen > 0)) {

        for ( i = 0; i < displLen; i++ ) {
            atom = displAtoms[i];

            for ( j = 0; j < 3; j++ ) {
                sum += pos1[3*atom+j] * pos2[3*i+j];
            }
        }
    }

    return sum;
}

