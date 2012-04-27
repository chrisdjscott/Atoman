
/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Functions associated with spatially decomposing a system of atoms into boxes
 ** 
 ** 
 ** Include boxes.h to use this library
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
#include "boxes.h"



/*******************************************************************************
 ** create and return pointer to Boxes structure
 *******************************************************************************/
struct Boxes * setupBoxes(double approxBoxWidth, double *minPos, double *maxPos, int *PBC, int maxAtomsPerBox)
{
    int i;
    double cellLength;
    struct Boxes *boxes;
    
    
    /* allocate space for boxes struct */
    boxes = malloc( 1 * sizeof(struct Boxes));
    if (boxes == NULL)
    {
        printf("ERROR: could not allocate boxes\n");
        exit(50);
    }
    
    /* setup boxes */
    for (i=0; i<3; i++)
    {
        boxes->minPos[i] = minPos[i];
        boxes->maxPos[i] = maxPos[i];
        boxes->PBC[i] = PBC[i];
        
        /* size of the region in this direction */
        cellLength = maxPos[i] - minPos[i];
        cellLength = (cellLength < 1.0) ? 1.0 : cellLength;
        
        /* number of boxes in this direction */
        boxes->NBoxes[i] = (int) (cellLength / approxBoxWidth);
        boxes->NBoxes[i] = (boxes->NBoxes[i] == 0) ? 1 : boxes->NBoxes[i];
        
        /* length of box side */
        boxes->boxWidth[i] = cellLength / boxes->NBoxes[i];
    }
    boxes->totNBoxes = boxes->NBoxes[0] * boxes->NBoxes[1] * boxes->NBoxes[2];
    boxes->maxAtomsPerBox = maxAtomsPerBox;
    
//    printf("DEBUG setupBoxes:\n");
//    printf("  minPos = %f, %f, %f\n", boxes->minPos[0], boxes->minPos[1], boxes->minPos[2]);
//    printf("  minPosIN = %f, %f, %f\n", minPos[0], minPos[1], minPos[2]);
//    printf("  maxPos = %f, %f, %f\n", boxes->maxPos[0], boxes->maxPos[1], boxes->maxPos[2]);
//    printf("  maxPosIN = %f, %f, %f\n", maxPos[0], maxPos[1], maxPos[2]);
//    printf("  width = %f, %f, %f\n", boxes->boxWidth[0], boxes->boxWidth[1], boxes->boxWidth[2]);
//    printf("  NBoxes = %d, %d, %d\n", boxes->NBoxes[0], boxes->NBoxes[1], boxes->NBoxes[2]);
//    printf("END DEBUG setupBoxes:\n");
    
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
    int i, boxIndex;
    
    
    for (i=0; i<NAtoms; i++)
    {
        boxIndex = boxIndexOfAtom(pos[3*i], pos[3*i+1], pos[3*i+2], boxes);
        
        /* check if this box is full/already allocated */
        if (boxes->boxNAtoms[boxIndex] + 1 == boxes->maxAtomsPerBox)
        {
            printf("ERROR: box limit reached\n");
            exit(140);
        }
        
        else if (boxes->boxNAtoms[boxIndex] == 0)
        {
            /* allocate this box */
            boxes->boxAtoms[boxIndex] = malloc(boxes->maxAtomsPerBox * sizeof(int));
            if (boxes->boxAtoms[boxIndex] == NULL)
            {
                printf("ERROR: could not allocate boxAtoms[%d]\n", boxIndex);
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
 ** TODO: handle if atom outside min and max pos
 *******************************************************************************/
int boxIndexOfAtom(double xpos, double ypos, double zpos, struct Boxes *boxes)
{
    int posintx, posinty, posintz;
    int boxIndex;
    
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
    
    /* think this numbers by x then z then y (can't remember why I wanted it like this) */
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
    int xint, yint, zint, tmp;
    
    
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
