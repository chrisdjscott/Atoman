/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Find clusters of atoms
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "boxes.h"
#include "utilities.h"


int findNeighbours(int, int, int, int *, double *, double, struct Boxes *, double *, int *);


/*******************************************************************************
 * Find clusters
 *******************************************************************************/
void findClusters(int visibleAtomsDim, int *visibleAtoms, int posDim, double *pos, int clusterArrayDim, int *clusterArray, double neighbourRad, 
                 int cellDimsDim, double *cellDims, int PBCDim, int *PBC, int minPosDim, double *minPos, int maxPosDim, double *maxPos, 
                 int minClusterSize, int resultsDim, int *results)
{
    int i, index, NClusters, numInCluster;
    int maxNumInCluster;
    double nebRad2, approxBoxWidth;
    double *visiblePos;
    struct Boxes *boxes;
    int *NAtomsCluster, clusterIndex;
    int *NAtomsClusterNew;
    int NVisible, count;
    
    
    visiblePos = malloc(3 * visibleAtomsDim * sizeof(double));
    if (visiblePos == NULL)
    {
        printf("ERROR: could not allocate visiblePos\n");
        exit(50);
    }
    
    for (i=0; i<visibleAtomsDim; i++)
    {
        index = visibleAtoms[i];
        
        visiblePos[3*i] = pos[3*index];
        visiblePos[3*i+1] = pos[3*index+1];
        visiblePos[3*i+2] = pos[3*index+2];
    }
    
    NAtomsCluster = calloc(visibleAtomsDim, sizeof(int));
    if (NAtomsCluster == NULL)
    {
        printf("ERROR: could not allocate NAtomsCluster\n");
        exit(50);
    }
    
    /* box visible atoms */
    approxBoxWidth = neighbourRad;
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(visibleAtomsDim, visiblePos, boxes);
    
    nebRad2 = neighbourRad * neighbourRad;
    
    /* initialise clusters array */
    for (i=0; i<visibleAtomsDim; i++)
    {
        clusterArray[i] = -1;
    }
    
    /* loop over atoms */
    maxNumInCluster = -9999;
    NClusters = 0;
    for (i=0; i<visibleAtomsDim; i++)
    {
        /* skip atom if already allocated */
        if (clusterArray[i] == -1)
        {
            clusterArray[i] = NClusters;
            
            numInCluster = 1;
            
            /* recursive search for cluster atoms */
            numInCluster = findNeighbours(i, clusterArray[i], numInCluster, clusterArray, visiblePos, nebRad2, boxes, cellDims, PBC);
            
            maxNumInCluster = (numInCluster > maxNumInCluster) ? numInCluster : maxNumInCluster;
            
            NAtomsCluster[NClusters] = numInCluster;
            
            NClusters++;
        }
    }
    
    NAtomsClusterNew = calloc(NClusters, sizeof(int));
    if (NAtomsClusterNew == NULL)
    {
        printf("ERROR: could not allocate NAtomsClusterNew\n");
        exit(50);
    }
    
    /* loop over visible atoms to see if their clusters has more than the min number */
    NVisible = 0;
    for (i=0; i<visibleAtomsDim; i++)
    {
        clusterIndex = clusterArray[i];
        index = visibleAtoms[i];
        
        numInCluster = NAtomsCluster[clusterIndex];
        
        if (numInCluster < minClusterSize)
        {
            continue;
        }
        
        visibleAtoms[NVisible] = index;
        clusterArray[NVisible] = clusterIndex;
        NAtomsClusterNew[clusterIndex]++;
        
        NVisible++;
    }
    
    /* how many clusters now */
    count = 0;
    for (i=0; i<NClusters; i++)
    {
        if (NAtomsClusterNew[i] > 0)
        {
            count += 1;
        }
    }
    
    /* store results */
    results[0] = NVisible;
    results[1] = count;
    
    free(NAtomsClusterNew);
    free(NAtomsCluster);
    freeBoxes(boxes);
    free(visiblePos);
}


/*******************************************************************************
 * recursive search for neighbouring defects
 *******************************************************************************/
int findNeighbours(int index, int clusterID, int numInCluster, int* atomCluster, double *pos, double maxSep2, 
                   struct Boxes *boxes, double *cellDims, int *PBC)
{
    int i, j, index2;
    int boxIndex, boxNebList[27];
    double sep2;
    
    
    /* box of primary atom */
    boxIndex = boxIndexOfAtom(pos[3*index], pos[3*index+1], pos[3*index+2], boxes);
    
    /* find neighbouring boxes */
    getBoxNeighbourhood(boxIndex, boxNebList, boxes);
    
    /* loop over neighbouring boxes */
    for (i=0; i<27; i++)
    {
        boxIndex = boxNebList[i];
        
        for (j=0; j<boxes->boxNAtoms[boxIndex]; j++)
        {
            index2 = boxes->boxAtoms[boxIndex][j];
            
            /* skip itself or if already searched */
            if ((index == index2) || (atomCluster[index2] != -1))
                continue;
            
            /* calculate separation */
            sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                    cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
            
            /* check if neighbours */
            if (sep2 < maxSep2)
            {
                atomCluster[index2] = clusterID;
                numInCluster++;
                
                /* search for neighbours to this new cluster atom */
                numInCluster = findNeighbours(index2, clusterID, numInCluster, atomCluster, pos, maxSep2, boxes, cellDims, PBC);
            }
        }
    }
    
    return numInCluster;
}
