
/*******************************************************************************
 ** Copyright Chris Scott 2011
 ** Filtering routines written in C to improve performance
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


/*******************************************************************************
 ** Specie filter
 *******************************************************************************/
int specieFilter(int NVisibleIn, int *visibleAtoms, int visSpecDim, int* visSpec, int specieDim, int *specie)
{
    int i, j, index, match, NVisible;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        match = 0;
        for (j=0; j<visSpecDim; j++)
        {
            if (specie[index] == visSpec[j])
            {
                match = 1;
                break;
            }
        }
        
        if (match)
        {
            visibleAtoms[NVisible] = index;
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Crop filter
 *******************************************************************************/
int cropFilter(int NVisibleIn, int* visibleAtoms, int posDim, double* pos, double xmin, double xmax,
               double ymin, double ymax, double zmin, double zmax, int xEnabled, int yEnabled, int zEnabled)
{
    int i, index, NVisible;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (xEnabled == 1)
        {
            if (pos[3*index] < xmin || pos[3*index] > xmax)
            {
                continue;
            }
        }
        
        if (yEnabled == 1)
        {
            if (pos[3*index+1] < ymin || pos[3*index+1] > ymax)
            {
                continue;
            }
        }
        
        if (zEnabled == 1)
        {
            if (pos[3*index+2] < zmin || pos[3*index+2] > zmax)
            {
                continue;
            }
        }
        
        visibleAtoms[NVisible] = index;
        NVisible++;
    }
    
    return NVisible;
}
