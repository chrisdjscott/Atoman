/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Picker routines
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "boxeslib.h"
#include "utilities.h"


int pickObject(int visibleAtomsDim, int *visibleAtoms, int vacsDim, int *vacs, int intsDim, int *ints, 
               int onAntsDim, int *onAnts, int splitsDim, int *splits, int pickPosDim, double *pickPos,
               int posDim, double *pos, int refPosDim, double *refPos, int PBCDim, int *PBC, 
               int cellDimsDim, double *cellDims, int minPosDim, double *minPos, int maxPosDim, double *maxPos,
               int specieDim, int* specie, int refSpecieDim, int *refSpecie, int specieCovRadDim, double* specieCovRad,
               int refSpecieCovRadDim, double* refSpecieCovRad, int resultDim, double *result)
{
    int i, k, index, boxIndex;
    int boxNebList[27], realIndex;
    int minSepIndex;
    double approxBoxWidth, *visPos;
    double sep2, minSep, rad, sep;
    struct Boxes *boxes;
    
    
    approxBoxWidth = 4.0;
    
    
    if (visibleAtomsDim > 0)
    {
        /* vis atoms pos */
        visPos = malloc(3 * visibleAtomsDim * sizeof(double));
        if (visPos == NULL)
        {
            printf("ERROR: could not allocate visPos\n");
            exit(1);
        }
        
        for (i=0; i<visibleAtomsDim; i++)
        {
            index = visibleAtoms[i];
            visPos[3*i] = pos[3*index];
            visPos[3*i+1] = pos[3*index+1];
            visPos[3*i+2] = pos[3*index+2];
        }
        
        /* box vis atoms */
        boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
        putAtomsInBoxes(visibleAtomsDim, visPos, boxes);
        
        /* box index of picked pos */
        boxIndex = boxIndexOfAtom(pickPos[0], pickPos[1], pickPos[2], boxes);
        
        /* neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over neighbouring boxes, looking for nearest atom */
        minSep = 9999999.0;
        minSepIndex = -1;
        for (i=0; i<27; i++)
        {
            boxIndex = boxNebList[i];
            
            for (k=0; k<boxes->boxNAtoms[boxIndex]; k++)
            {
                index = boxes->boxAtoms[boxIndex][k];
                
                /* atomic separation */
                sep2 = atomicSeparation2(pickPos[0], pickPos[1], pickPos[2], 
                                         visPos[3*index], visPos[3*index+1], visPos[3*index+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                
                /* need radius too */
                realIndex = visibleAtoms[index];
                
                rad = specieCovRad[specie[realIndex]];
                
                sep = fabs(sqrt(sep2) - rad);
                
                if (sep < minSep)
                {
                    minSep = sep;
                    minSepIndex = index;
                }
                
            }
        }
        
        /* store result */
        result[0] = 0;
        result[1] = minSepIndex;
        result[2] = minSep;
        
        /* tidy up */
        free(visPos);
        freeBoxes(boxes);
    }
    else
    {
        printf("Picking defect\n");
        
        
        
        
        
    }
    
    return 0;
}




