/*******************************************************************************
 ** Copyright Chris Scott 2013
 ** Calculate bonds
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../visclibs/boxeslib.h"
#include "../visclibs/utilities.h"
#include "bonds.h"



/*******************************************************************************
 * Calculate bonds
 *******************************************************************************/
int calculateBonds(int NVisible, int *visibleAtoms, double *pos, int *specie, int NSpecies, double *bondMinArray, double *bondMaxArray, 
                   double approxBoxWidth, int maxBondsPerAtom, double *cellDims, int *PBC, double *minPos, double *maxPos,
                   int *bondArray, int *NBondsArray, double *bondVectorArray)
{
    int i, j, k, index, index2, visIndex;
    int speca, specb, count;
    int boxIndex, boxNebList[27];
    double *visiblePos, sep2, sep;
    double sepVec[3];
    struct Boxes *boxes;
    
    
    printf("BONDS CLIB\n");
    printf("N VIS: %d\n", NVisible);
    
    for (i=0; i<NSpecies; i++)
    {
        for (j=i; j<NSpecies; j++)
        {
            printf("%d - %d: %lf -> %lf\n", i, j, bondMinArray[i*NSpecies+j], bondMaxArray[i*NSpecies+j]);
        }
    }
    
    /* construct visible pos array */
    visiblePos = malloc(3 * NVisible * sizeof(double));
    if (visiblePos == NULL)
    {
        printf("ERROR: could not allocate visiblePos\n");
        exit(50);
    }
    
    for (i=0; i<NVisible; i++)
    {
        index = visibleAtoms[i];
        
        visiblePos[3*i] = pos[3*index];
        visiblePos[3*i+1] = pos[3*index+1];
        visiblePos[3*i+2] = pos[3*index+2];
    }
    
    /* box visible atoms */
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(NVisible, visiblePos, boxes);
    
    /* loop over visible atoms */
    count = 0;
    for (i=0; i<NVisible; i++)
    {
        index = visibleAtoms[i];
        
        speca = specie[index];
        
        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(pos[3*index], pos[3*index+1], pos[3*index+2], boxes);
        
        /* find neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over box neighbourhood */
        for (j=0; j<27; j++)
        {
            boxIndex = boxNebList[j];
            
            for (k=0; k<boxes->boxNAtoms[boxIndex]; k++)
            {
                visIndex = boxes->boxAtoms[boxIndex][k];
                index2 = visibleAtoms[visIndex];
                
                if (index >= index2)
                {
                    continue;
                }
                
                specb = specie[index2];
                
                if (bondMinArray[speca*NSpecies+specb] == 0.0 && bondMaxArray[speca*NSpecies+specb] == 0.0)
                {
                    continue;
                }
                
                /* atomic separation */
                sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                         pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                
                sep = sqrt(sep2);
                
                /* check if these atoms are bonded */
                if (sep >= bondMinArray[speca*NSpecies+specb] && sep <= bondMaxArray[speca*NSpecies+specb])
                {
                    if (NBondsArray[i] + 1 == maxBondsPerAtom)
                    {
                        printf("ERROR: maxBondsPerAtom exceeded\n");
                        return 1;
                    }
                    
                    bondArray[count] = visIndex;
                    NBondsArray[i]++;
                    
                    /* separation vector */
                    atomSeparationVector(sepVec, pos[3*index], pos[3*index+1], pos[3*index+2], 
                                         pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                    
                    bondVectorArray[3*count] = sepVec[0] / 2.0;
                    bondVectorArray[3*count+1] = sepVec[1] / 2.0;
                    bondVectorArray[3*count+2] = sepVec[2] / 2.0;
                    
                    count++;
                }
            }
        }
    }
    
    printf("  N BONDS TOT: %d\n", count);
    
    /* free */
    free(visiblePos);
    freeBoxes(boxes);
    
    return 0;
}

