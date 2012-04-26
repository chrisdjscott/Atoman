
/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Functions associated with spatially decomposing lattice into boxes
 *******************************************************************************/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "boxes.h"


void boxIJKIndices( int, int*, int );
int boxIndexFromIJK( int, int, int );


/*******************************************************************************
 ** free box arrays
 *******************************************************************************/
void cleanupBoxes()
{
    free(boxNAtoms);
    free(boxAtoms);
}


/*******************************************************************************
 ** estup boxes
 *******************************************************************************/
void setupBoxes( double *minPos, double *maxPos, double approxBoxWidth )
{
    int i;
    double cellLength;
    
    
    for (i=0; i<3; i++)
    {
        /* size of the region in this direction */
        cellLength = maxPos[i] - minPos[i];
        cellLength = (cellLength < 1.0) ? 1.0 : cellLength;
        
        /* number of boxes in this direction */
        NBoxes[i] = (int) (cellLength / approxBoxWidth);
        NBoxes[i] = (NBoxes[i] == 0) ? 1 : NBoxes[i];
        
        /* length of box side */
        boxWidth[i] = cellLength / NBoxes[i];
        
        boxMinPos[i] = minPos[i];
        boxMaxPos[i] = maxPos[i];
    }
    
    totNBoxes = NBoxes[0] * NBoxes[1] * NBoxes[2];
    
    /* allocate arrays */
    boxNAtoms = calloc(totNBoxes, sizeof(int));
    if (boxNAtoms == NULL)
    {
        printf("ERROR: Boxes: could not allocate boxNAtoms\n");
        exit(1);
    }
    boxAtoms = malloc( totNBoxes * maxAtomsPerBox * sizeof(int));
    if (boxAtoms == NULL)
    {
        printf("ERROR: Boxes: could not allocate boxAtoms\n");
        exit(1);
    }
}


/*******************************************************************************
 ** put atoms into boxes
 *******************************************************************************/
void putAtomsInBoxes( int NAtoms, double* pos )
{
    long i;
    long index, boxIndex, boxAtomsIndex;
    
            
    for ( i=0; i<NAtoms; i++ )
    {
        /* find which box this atom should be in */
        boxIndex = boxIndexOfAtom( pos[3*i], pos[3*i+1], pos[3*i+2] );
        
        /* add atom to the box */
        boxAtomsIndex = boxIndex*maxAtomsPerBox+boxNAtoms[boxIndex];
        boxAtoms[boxAtomsIndex] = i;
        boxNAtoms[boxIndex]++;
    }
}


/*******************************************************************************
 ** returns box index of given atom
 *******************************************************************************/
int boxIndexOfAtom( double xpos, double ypos, double zpos )
{
    int posintx, posinty, posintz;
    int boxIndex;
    
    posintx = (int) ( (xpos - boxMinPos[0]) / boxWidth[0] );
    if ( posintx >= NBoxes[0])
    {
        posintx = NBoxes[0] - 1;
    }
    
    posinty = (int) ( (ypos - boxMinPos[1]) / boxWidth[1] );
    if ( posinty >= NBoxes[1])
    {
        posinty = NBoxes[1] - 1;
    }
    
    posintz = (int) ( (zpos - boxMinPos[2]) / boxWidth[2] );
    if ( posintz >= NBoxes[2])
    {
        posintz = NBoxes[2] - 1;
    }
    
    /* think this numbers by x then z then y (can't remember why I wanted it like this) */
    boxIndex = (int) (posintx + posintz * NBoxes[0] + posinty * NBoxes[0] * NBoxes[2]);
    
    if (boxIndex < 0)
    {
        printf("WARNING: boxIndexOfAtom (CLIB): boxIndex < 0: %d, %d %d %d\n", boxIndex, posintx, posinty, posintz);
        printf("         pos = %f %f %f, box widths = %f %f %f, NBoxes = %d %d %d\n", xpos, ypos, zpos, boxWidth[0], boxWidth[1], boxWidth[2], NBoxes[0], NBoxes[1], NBoxes[2]);
        printf("         min box pos = %f %f %f\n", boxMinPos[0], boxMinPos[1], boxMinPos[2]);
    }
    
    return boxIndex;
}
    

/*******************************************************************************
 ** return i, j, k indices of box
 *******************************************************************************/
void boxIJKIndices( int dim1, int* ijkIndices, int boxIndex )
{
    int xint, yint, zint, tmp;
    
    
    // maybe should check here that dim1 == 3
    
    yint = (int) ( boxIndex / (NBoxes[0] * NBoxes[2]) );
    
    tmp = boxIndex - yint * NBoxes[0] * NBoxes[2];
    zint = (int) ( tmp / NBoxes[0] );
    
    xint = (int) (tmp - zint * NBoxes[0]);
    
    ijkIndices[0] = xint;
    ijkIndices[1] = yint;
    ijkIndices[2] = zint;
}


/*******************************************************************************
 ** returns the box index of box with given i,j,k indices
 *******************************************************************************/
int boxIndexFromIJK( int xindex, int yindex, int zindex )
{
    int xint, yint, zint, box;
    
    
    xint = xindex;
    zint = NBoxes[0] * zindex;
    yint = NBoxes[0] * NBoxes[2] * yindex;
    
    box = (int) (xint + yint + zint);
    
    return box;
}


/*******************************************************************************
 ** returns neighbourhood of given box
 *******************************************************************************/
void getBoxNeighbourhood( int mainBox, int* boxNeighbourList )
{
    int mainBoxIJK[3];
    int i, j, k;
    int posintx, posinty, posintz;
    int index, count;
    int xint,yint,zint,tmp;
    
    
    /* first get i,j,k indices of the main box */
    boxIJKIndices( 3, mainBoxIJK, mainBox );
            
    /* loop over each direction */
    count = 0;
    for ( i=0; i<3; i++ )
    {
        posintx = mainBoxIJK[0] - 1 + i;
        /* wrap */
        if ( posintx < 0 )
        {
            posintx += NBoxes[0];
        }
        else if ( posintx >= NBoxes[0] )
        {
            posintx -= NBoxes[0];
        }
        
        for ( j=0; j<3; j++ )
        {
            posinty = mainBoxIJK[1] - 1 + j;
            /* wrap */
            if ( posinty < 0 )
            {
                posinty += NBoxes[1];
            }
            else if ( posinty >= NBoxes[1] )
            {
                posinty -= NBoxes[1];
            }
            
            for ( k=0; k<3; k++ )
            {
                posintz = mainBoxIJK[2] - 1 + k;
                /* wrap */
                if ( posintz < 0 )
                {
                    posintz += NBoxes[2];
                }
                else if ( posintz >= NBoxes[2] )
                {
                    posintz -= NBoxes[2];
                }
                
                /* get index of box from this i,j,k */
                index = boxIndexFromIJK( posintx, posinty, posintz );
                boxNeighbourList[count] = index;
                count++;
            }
        }
    }
}
