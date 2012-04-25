
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
