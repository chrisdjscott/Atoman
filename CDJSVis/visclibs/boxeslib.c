
/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Functions associated with spatially decomposing a system of atoms into boxes
 ** 
 ** 
 ** Include boxeslib.h to use this library
 ** 
 ** Call setupBoxes() to return the Boxes structure
 ** 
 ** Call putAtomInBoxes() to add atoms to the boxes
 ** 
 ** The Boxes structure must be freed by calling freeBoxes()
 ** 
 *******************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "boxeslib.h"



/*******************************************************************************
 ** create and return pointer to Boxes structure
 ** #TODO: if PBCs are set min/max pos should be equal to cell dims
 *******************************************************************************/
struct Boxes * setupBoxes(double approxBoxWidth, double *minPos, double *maxPos, int *PBC, double *cellDims)
{
    int i;
    double cellLength;
    struct Boxes *boxes;
    
    
    /* allocate space for boxes struct */
    boxes = malloc(1 * sizeof(struct Boxes));
    if (boxes == NULL)
    {
        printf("ERROR: could not allocate boxes\n");
        exit(50);
    }
    
    /* setup boxes */
    for (i=0; i<3; i++)
    {
        /* store some parameters */
        boxes->PBC[i] = PBC[i];
        boxes->cellDims[i] = cellDims[i];
        
        /* if PBC box cell, otherwise box min-max pos */
        if (boxes->PBC[i] == 0)
        {
            boxes->minPos[i] = minPos[i];
            boxes->maxPos[i] = maxPos[i];
        }
        else
        {
            boxes->minPos[i] = 0.0;
            boxes->maxPos[i] = boxes->cellDims[i];
        }
        
        /* size of the region in this direction */
        cellLength = boxes->maxPos[i] - boxes->minPos[i];
        cellLength = (cellLength < 1.0) ? 1.0 : cellLength;
        
        /* number of boxes in this direction */
        boxes->NBoxes[i] = (int) (cellLength / approxBoxWidth);
        boxes->NBoxes[i] = (boxes->NBoxes[i] == 0) ? 1 : boxes->NBoxes[i];
        
        /* length of box side */
        boxes->boxWidth[i] = cellLength / boxes->NBoxes[i];
    }
    boxes->totNBoxes = boxes->NBoxes[0] * boxes->NBoxes[1] * boxes->NBoxes[2];
    
    boxes->allocChunk = 50;
    
    /* allocate arrays */
    boxes->boxNAtoms = calloc(boxes->totNBoxes, sizeof(int));
    if (boxes->boxNAtoms == NULL)
    {
        printf("ERROR: could not allocate boxNAtoms\n");
        exit(50);
    }
    
    boxes->boxAtoms = malloc(boxes->totNBoxes * sizeof(int *));
    if (boxes->boxAtoms == NULL)
    {
        printf("ERROR: could not allocate boxAtoms\n");
        exit(50);
    }
        
    return boxes;
}


/*******************************************************************************
 ** put atoms into boxes
 *******************************************************************************/
void putAtomsInBoxes(int NAtoms, double *pos, struct Boxes *boxes)
{
    int i, boxIndex, newsize;
    
    
    for (i=0; i<NAtoms; i++)
    {
        boxIndex = boxIndexOfAtom(pos[3*i], pos[3*i+1], pos[3*i+2], boxes);
        
        /* check if this box is empty or full */
        if (boxes->boxNAtoms[boxIndex] == 0)
        {
            /* allocate this box */
            boxes->boxAtoms[boxIndex] = malloc(boxes->allocChunk * sizeof(int));
            if (boxes->boxAtoms[boxIndex] == NULL)
            {
                printf("ERROR: could not allocate boxAtoms[%d]\n", boxIndex);
                exit(50);
            }
        }
        
        else if (boxes->boxNAtoms[boxIndex] % boxes->allocChunk == 0)
        {
            /* realloc more space */
            newsize = boxes->boxNAtoms[boxIndex] + boxes->allocChunk;
            boxes->boxAtoms[boxIndex] = realloc(boxes->boxAtoms[boxIndex], newsize * sizeof(int));
            if (boxes->boxAtoms[boxIndex] == NULL)
            {
                printf("ERROR: could not reallocate boxAtoms[%d]: %d\n", boxIndex, newsize);
                exit(50);
            }
        }
        
        /* add atom to box */
        boxes->boxAtoms[boxIndex][boxes->boxNAtoms[boxIndex]] = i;
        boxes->boxNAtoms[boxIndex]++;
    }
}


/*******************************************************************************
 ** returns box index of given atom
 *******************************************************************************/
int boxIndexOfAtom(double xpos, double ypos, double zpos, struct Boxes *boxes)
{
    int posintx, posinty, posintz;
    int boxIndex;
    
    
    /* if atom is outside boxes min/max pos we wrap or translate it back depending on PBCs */
    if (xpos > boxes->maxPos[0] || xpos < boxes->minPos[0])
    {
        if (boxes->PBC[0] == 1)
        {
            /* wrap position */
            xpos = xpos - floor( xpos / boxes->cellDims[0] ) * boxes->cellDims[0];
        }
        else
        {
            if (xpos > boxes->maxPos[0])
            {
                /* put it in the end box */
                xpos = boxes->maxPos[0] - 0.5 * boxes->boxWidth[0];
            }
            else
            {
                /* put it in the end box */
                xpos = boxes->minPos[0] + 0.5 * boxes->boxWidth[0];
            }
        }
    }
    
    if (ypos > boxes->maxPos[1] || ypos < boxes->minPos[1])
    {
        if (boxes->PBC[1] == 1)
        {
            /* wrap position */
            ypos = ypos - floor( ypos / boxes->cellDims[1] ) * boxes->cellDims[1];
        }
        else
        {
            if (ypos > boxes->maxPos[1])
            {
                /* put it in the end box */
                ypos = boxes->maxPos[1] - 0.5 * boxes->boxWidth[1];
            }
            else
            {
                /* put it in the end box */
                ypos = boxes->minPos[1] + 0.5 * boxes->boxWidth[1];
            }
        }
    }
    
    if (zpos > boxes->maxPos[2] || zpos < boxes->minPos[2])
    {
        if (boxes->PBC[2] == 1)
        {
            /* wrap position */
            zpos = zpos - floor( zpos / boxes->cellDims[2] ) * boxes->cellDims[2];
        }
        else
        {
            if (zpos > boxes->maxPos[2])
            {
                /* put it in the end box */
                zpos = boxes->maxPos[2] - 0.5 * boxes->boxWidth[2];
            }
            else
            {
                /* put it in the end box */
                zpos = boxes->minPos[2] + 0.5 * boxes->boxWidth[2];
            }
        }
    }
    
    /* find box for atom */
    posintx = (int) ( (xpos - boxes->minPos[0]) / boxes->boxWidth[0] );
    if ( posintx >= boxes->NBoxes[0])
    {
        posintx = boxes->NBoxes[0] - 1;
    }
    
    posinty = (int) ( (ypos - boxes->minPos[1]) / boxes->boxWidth[1] );
    if ( posinty >= boxes->NBoxes[1])
    {
        posinty = boxes->NBoxes[1] - 1;
    }
    
    posintz = (int) ( (zpos - boxes->minPos[2]) / boxes->boxWidth[2] );
    if ( posintz >= boxes->NBoxes[2])
    {
        posintz = boxes->NBoxes[2] - 1;
    }
    
    /* this numbers by x then z then y (can't remember why I wanted it like this) */
    boxIndex = (int) (posintx + posintz * boxes->NBoxes[0] + posinty * boxes->NBoxes[0] * boxes->NBoxes[2]);
    
    if (boxIndex < 0)
    {
        printf("WARNING: boxIndexOfAtom (CLIB): boxIndex < 0: %d, %d %d %d\n", boxIndex, posintx, posinty, posintz);
        printf("         pos = %f %f %f, box widths = %f %f %f, NBoxes = %d %d %d\n", xpos, ypos, zpos, boxes->boxWidth[0], boxes->boxWidth[1], 
                boxes->boxWidth[2], boxes->NBoxes[0], boxes->NBoxes[1], boxes->NBoxes[2]);
        printf("         min box pos = %f %f %f\n", boxes->minPos[0], boxes->minPos[1], boxes->minPos[2]);
    }
    
    return boxIndex;
}


/*******************************************************************************
 ** return i, j, k indices of box
 *******************************************************************************/
void boxIJKIndices(int dim1, int* ijkIndices, int boxIndex, struct Boxes *boxes)
{
    int xint, yint, zint, tmp;
    
    
    // maybe should check here that dim1 == 3
    
    yint = (int) ( boxIndex / (boxes->NBoxes[0] * boxes->NBoxes[2]) );
    
    tmp = boxIndex - yint * boxes->NBoxes[0] * boxes->NBoxes[2];
    zint = (int) ( tmp / boxes->NBoxes[0] );
    
    xint = (int) (tmp - zint * boxes->NBoxes[0]);
    
    ijkIndices[0] = xint;
    ijkIndices[1] = yint;
    ijkIndices[2] = zint;
}


/*******************************************************************************
 ** returns the box index of box with given i,j,k indices
 *******************************************************************************/
int boxIndexFromIJK(int xindex, int yindex, int zindex, struct Boxes *boxes)
{
    int xint, yint, zint, box;
    
    
    xint = xindex;
    zint = boxes->NBoxes[0] * zindex;
    yint = boxes->NBoxes[0] * boxes->NBoxes[2] * yindex;
    
    box = (int) (xint + yint + zint);
    
    return box;
}


/*******************************************************************************
 ** returns neighbourhood of given box
 *******************************************************************************/
void getBoxNeighbourhood(int mainBox, int* boxNeighbourList, struct Boxes *boxes)
{
    int mainBoxIJK[3];
    int i, j, k;
    int posintx, posinty, posintz;
    int index, count;
    
    
    /* first get i,j,k indices of the main box */
    boxIJKIndices( 3, mainBoxIJK, mainBox, boxes );
            
    /* loop over each direction */
    count = 0;
    for ( i=0; i<3; i++ )
    {
        posintx = mainBoxIJK[0] - 1 + i;
        /* wrap */
        if ( posintx < 0 )
        {
            posintx += boxes->NBoxes[0];
        }
        else if ( posintx >= boxes->NBoxes[0] )
        {
            posintx -= boxes->NBoxes[0];
        }
        
        for ( j=0; j<3; j++ )
        {
            posinty = mainBoxIJK[1] - 1 + j;
            /* wrap */
            if ( posinty < 0 )
            {
                posinty += boxes->NBoxes[1];
            }
            else if ( posinty >= boxes->NBoxes[1] )
            {
                posinty -= boxes->NBoxes[1];
            }
            
            for ( k=0; k<3; k++ )
            {
                posintz = mainBoxIJK[2] - 1 + k;
                /* wrap */
                if ( posintz < 0 )
                {
                    posintz += boxes->NBoxes[2];
                }
                else if ( posintz >= boxes->NBoxes[2] )
                {
                    posintz -= boxes->NBoxes[2];
                }
                
                /* get index of box from this i,j,k */
                index = boxIndexFromIJK( posintx, posinty, posintz, boxes );
                boxNeighbourList[count] = index;
                count++;
            }
        }
    }
}


/*******************************************************************************
 ** free boxes memory
 *******************************************************************************/
void freeBoxes(struct Boxes *boxes)
{
    int i;
    
    
    for (i=0; i<boxes->totNBoxes; i++)
    {
        if (boxes->boxNAtoms[i] > 0)
        {
            free(boxes->boxAtoms[i]);
        }
    }
    free(boxes->boxAtoms);
    free(boxes->boxNAtoms);
    free(boxes);
}
