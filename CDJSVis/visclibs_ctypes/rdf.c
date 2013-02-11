/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Calculate RDF
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../visclibs/boxeslib.h"
#include "../visclibs/utilities.h"
#include "rdf.h"


int calculateRDF(int NVisible, int *visibleAtoms, int NAtoms, int *specie, double *pos, int specieID1, int specieID2, 
                 double *minPos, double *maxPos, double *cellDims, int *PBC, double start, double finish, int num, double *rdf)
{
    int i, j, k,index, index2, boxIndex;
    int boxNebList[27], fullShellCount, binIndex;
    double approxBoxWidth, sep2, sep;
    double avgAtomDensity, shellVolume;
    double ini, fin, interval;
    struct Boxes *boxes;
    
    
    /* approx box width??? */
    approxBoxWidth = 5.0; // must be at least interval I guess?
    
    /* box reference atoms */
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(NAtoms, pos, boxes);
    
    interval = (finish - start) / ((double) num);
    
    /* loop over atoms */
    for (i=0; i<NVisible; i++)
    {
        index = visibleAtoms[i];
        
        /* skip if not selected specie */
        if (specieID1 >= 0 && specie[index] != specieID1)
        {
            continue;
        }
        
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
                index2 = boxes->boxAtoms[boxIndex][k];
                
                /* skip if not selected specie */
                if (specieID2 >= 0 && specie[index2] != specieID2)
                {
                    continue;
                }
                
                /* atomic separation */
                sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                         pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2] );
                
                sep = sqrt(sep2);
                
                /* put in bin */
                if (sep >= start && sep < finish)
                {
                    binIndex = (int) ((sep - start) / interval);
                    
                    rdf[binIndex]++;
                }
            }
        }
    }
    
    /* calculate shell volumes and average atom density */
    avgAtomDensity = 0.0;
    fullShellCount = 0;
    for (i=0; i<num; i++)
    {
        ini = i * interval + start;
        fin = (i + 1.0) * interval + start;
        
        shellVolume = (4.0 / 3.0) * 3.1415926536 * (pow(fin, 3.0) - pow(ini, 3));
        
        rdf[i] = rdf[i] / shellVolume;
        
        if (rdf[i] > 0)
        {
            avgAtomDensity = avgAtomDensity + rdf[i];
            fullShellCount++;
        }
    }
    
    avgAtomDensity = avgAtomDensity / fullShellCount;
    
    /* divide by average atom density */
    for (i=0; i<num; i++)
    {
        rdf[i] = rdf[i] / avgAtomDensity;
    }
    
    /* free */
    freeBoxes(boxes);
    
    return 0;
}


