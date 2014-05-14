/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Find defects
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "boxeslib.h"
#include "utilities.h"
#include "defects.h"


int findDefectClusters(int, double *, int *, int *, struct Boxes *, double, double *, int *);
int findDefectNeighbours(int, int, int, int *, double *, struct Boxes *, double, double *, int *);



/*******************************************************************************
 * Search for defects and return the sub-system surrounding them
 *******************************************************************************/
int findDefects(int includeVacs, int includeInts, int includeAnts, int* NDefectsType, int* vacancies, int* interstitials, int* antisites, 
                int* onAntisites, int exclSpecInputDim, int* exclSpecInput, int exclSpecRefDim, int* exclSpecRef, int NAtoms, char* specieList, 
                int* specie, double* pos, int refNAtoms, char* specieListRef, int* specieRef, double* refPosIn, double *cellDims, int *PBC, 
                double vacancyRadius, double *minPos, double *maxPos, int findClustersFlag, double clusterRadius, int *defectCluster, int NSpecies, 
                int *vacSpecCount, int *intSpecCount, int *antSpecCount, int *onAntSpecCount, int* splitIntSpecCount, int minClusterSize, 
                int maxClusterSize, int *splitInterstitials, int identifySplits, int driftCompensation, double *driftVector)
{
    int i, exitLoop, k, j, index;
    double vacRad2;
    int boxNebList[27], index2;
    char symtemp[3], symtemp2[3];
    double xpos, ypos, zpos;
    int checkBox, refIndex, comp, boxIndex;
    double refxpos, refypos, refzpos;
    double sep2;
    int NDefects, NAntisites, NInterstitials, NVacancies;
    int *NDefectsCluster, *NDefectsClusterNew;
    int *possibleVacancy, *possibleInterstitial;
    int *possibleAntisite, *possibleOnAntisite;
    int skip, count, NClusters, clusterIndex, vacCount;
    double approxBoxWidth, splitIntRad;
    struct Boxes *boxes;
    double *defectPos;
    int NVacNew, NIntNew, NAntNew, NSplitNew, numInCluster;
    int NSplitInterstitials, *defectClusterSplit, splitIndexes[3];
    double *refPos;
    
    
    /* drift compensation? */
    if (driftCompensation)
    {
        refPos = malloc(3 * refNAtoms * sizeof(double));
        if (refPos == NULL)
        {
            printf("ERROR: could not allocate refPos\n");
            exit(34);
        }
        
        for (i = 0; i < refNAtoms; i++)
        {
            refPos[3*i] = refPosIn[3*i] + driftVector[0];
            refPos[3*i+1] = refPosIn[3*i+1] + driftVector[1];
            refPos[3*i+2] = refPosIn[3*i+2] + driftVector[2];
        }
    }
    else
    {
        refPos = refPosIn;
    }
    
    /* approx width, must be at least vacRad
     * should vary depending on size of cell
     * ie. don't want too many boxes
     */
    approxBoxWidth = 1.1 * vacancyRadius;
    
    /* box reference atoms */
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(refNAtoms, refPos, boxes);
    
    /* allocate local arrays for checking atoms */
    possibleVacancy = malloc( refNAtoms * sizeof(int) );
    if (possibleVacancy == NULL)
    {
        printf("ERROR: Boxes: could not allocate possibleVacancy\n");
        exit(1);
    }
    
    possibleInterstitial = malloc( NAtoms * sizeof(int) );
    if (possibleInterstitial == NULL)
    {
        printf("ERROR: Boxes: could not allocate possibleInterstitial\n");
        exit(1);
    }
    
    possibleAntisite = malloc( refNAtoms * sizeof(int) );
    if (possibleAntisite == NULL)
    {
        printf("ERROR: Boxes: could not allocate possibleAntisite\n");
        exit(1);
    }
    
    possibleOnAntisite = malloc( refNAtoms * sizeof(int) );
    if (possibleOnAntisite == NULL)
    {
        printf("ERROR: Boxes: could not allocate possibleOnAntisite\n");
        exit(1);
    }
    
    /* initialise arrays */
    for ( i=0; i<NAtoms; i++ )
    {
        possibleInterstitial[i] = 1;
    }
    for ( i=0; i<refNAtoms; i++ )
    {
        possibleVacancy[i] = 1;
        possibleAntisite[i] = 1;
    }
    
    vacRad2 = vacancyRadius * vacancyRadius;
    
    /* loop over all input atoms */
    for ( i=0; i<NAtoms; i++ )
    {        
        xpos = pos[3*i];
        ypos = pos[3*i+1];
        zpos = pos[3*i+2];
        
        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(xpos, ypos, zpos, boxes);
        
        /* find neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
                
        /* loop over neighbouring boxes */
        exitLoop = 0;
        for ( j=0; j<27; j++ )
        {
            if (exitLoop)
            {
                break;
            }
            
            checkBox = boxNebList[j];
            
            /* now loop over all reference atoms in the box */
            for ( k=0; k<boxes->boxNAtoms[checkBox]; k++ )
            {
                refIndex = boxes->boxAtoms[checkBox][k];
                
                /* if this vacancy has already been filled then skip to the next one */
                if ( possibleVacancy[refIndex] == 0 )
                {
                    continue;
                }
                
                refxpos = refPos[3*refIndex];
                refypos = refPos[3*refIndex+1];
                refzpos = refPos[3*refIndex+2];
                
                /* atomic separation of possible vacancy and possible interstitial */
                sep2 = atomicSeparation2( xpos, ypos, zpos, refxpos, refypos, refzpos, 
                                          cellDims[0], cellDims[1], cellDims[2], 
                                          PBC[0], PBC[1], PBC[2] );
                
                /* if within vacancy radius, is it an antisite or normal lattice point */
                if ( sep2 < vacRad2 )
                {
                    /* compare symbols */
                    symtemp[0] = specieList[2*specie[i]];
                    symtemp[1] = specieList[2*specie[i]+1];
                    symtemp[2] = '\0';
                    
                    symtemp2[0] = specieListRef[2*specieRef[refIndex]];
                    symtemp2[1] = specieListRef[2*specieRef[refIndex]+1];
                    symtemp2[2] = '\0';
                    
                    comp = strcmp( symtemp, symtemp2 );
                    if ( comp == 0 )
                    {
                        /* match, so not antisite */
                        possibleAntisite[refIndex] = 0;
                    }
                    else
                    {
                        possibleOnAntisite[refIndex] = i;
                    }
                    
                    /* not an interstitial or vacancy */
                    possibleInterstitial[i] = 0;
                    possibleVacancy[refIndex] = 0;
                    
                    /* no need to check further for this (no longer) possible interstitial */
                    exitLoop = 1;
                    break;
                }
            }
        }
    }
    
    /* free box arrays */
    freeBoxes(boxes);
        
    /* now classify defects */
    NVacancies = 0;
    NInterstitials = 0;
    NAntisites = 0;
    NSplitInterstitials = 0;
    for ( i=0; i<refNAtoms; i++ )
    {
//        skip = 0;
//        for (j=0; j<exclSpecRefDim; j++)
//        {
//            if (specieRef[i] == exclSpecRef[j])
//            {
//                skip = 1;
//            }
//        }
//        
//        if (skip == 1)
//        {
//            continue;
//        }
        
        /* vacancies */
        if (possibleVacancy[i] == 1)
        {
//            if (includeVacs == 1)
//            {
                vacancies[NVacancies] = i;
                NVacancies++;
//            }
        }
        
        /* antisites */
        else if (possibleAntisite[i] == 1) //&& (includeAnts == 1) )
        {
            antisites[NAntisites] = i;
            onAntisites[NAntisites] = possibleOnAntisite[i];
            NAntisites++;
        }
    }
    
//    if (includeInts == 1)
//    {
        for ( i=0; i<NAtoms; i++ )
        {
//            skip = 0;
//            for (j=0; j<exclSpecInputDim; j++)
//            {
//                if (specie[i] == exclSpecInput[j])
//                {
//                    skip = 1;
//                    break;
//                }
//            }
//            
//            if (skip == 1)
//            {
//                continue;
//            }
                        
            /* interstitials */
            if (possibleInterstitial[i] == 1)
            {
                interstitials[NInterstitials] = i;
                NInterstitials++;
            }
        }
//    }
    
    /* free arrays */
    free(possibleVacancy);
    free(possibleInterstitial);
    free(possibleAntisite);
    free(possibleOnAntisite);
    
    /* look for split interstitials */
    if (identifySplits && NVacancies > 0 && NInterstitials > 1)
    {
//        printf("IDENTIFYING SPLIT INTS\n");
        
        /* build positions array of all defects */
        NDefects = NVacancies + NInterstitials;
        defectPos = malloc(3 * NDefects * sizeof(double));
        if (defectPos == NULL)
        {
            printf("ERROR: could not allocate defectPos\n");
            exit(50);
        }
        
        /* add defects positions: vac then int then ant */
        count = 0;
        for (i=0; i<NVacancies; i++)
        {
            index = vacancies[i];
            
            defectPos[3*count] = refPos[3*index];
            defectPos[3*count+1] = refPos[3*index+1];
            defectPos[3*count+2] = refPos[3*index+2];
            
            count++;
        }
        
        for (i=0; i<NInterstitials; i++)
        {
            index = interstitials[i];
            
            defectPos[3*count] = pos[3*index];
            defectPos[3*count+1] = pos[3*index+1];
            defectPos[3*count+2] = pos[3*index+2];
            
            count++;
        }
        
        splitIntRad = 2.0 * vacancyRadius;
        
        /* box defects */
        approxBoxWidth = splitIntRad;
        boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
        putAtomsInBoxes(NDefects, defectPos, boxes);
        
        /* number of defects per cluster */
        NDefectsCluster = malloc(NDefects * sizeof(int));
        if (NDefectsCluster == NULL)
        {
            printf("ERROR: Boxes: could not allocate NDefectsCluster\n");
            exit(1);
        }
        
        /* cluster number */
        defectClusterSplit = malloc(NDefects * sizeof(int));
        if (defectClusterSplit == NULL)
        {
            printf("ERROR: Boxes: could not allocate defectClusterSplit\n");
            exit(1);
        }
        
        /* find clusters */
        NClusters = findDefectClusters(NDefects, defectPos, defectClusterSplit, NDefectsCluster, boxes, splitIntRad, cellDims, PBC);
        
        NDefectsCluster = realloc(NDefectsCluster, NClusters * sizeof(int));
        if (NDefectsCluster == NULL)
        {
            printf("ERROR: could not reallocate NDefectsCluster\n");
            exit(51);
        }
        
        NVacNew = NVacancies;
        NIntNew = NInterstitials;
        NSplitInterstitials = 0;
        
        /* find split ints */
        for (i=0; i<NClusters; i++)
        {
            if (NDefectsCluster[i] == 3)
            {
                /* check if 2 interstitials and 1 vacancy */
//                printf("  POSSIBLE SPLIT INTERSTITIAL\n");
                
                count = 0;
                vacCount = 0;
                for (j=0; j<NDefects; j++)
                {
                    if (defectClusterSplit[j] == i)
                    {
                        if (j < NVacancies)
                        {
                            vacCount += 1;
                        }
                        
                        splitIndexes[count] = j;
                        count++;
                    }
                }
                
                if (vacCount == 1)
                {
//                    printf("    FOUND SPLIT INTERSTITIAL\n");
                    
                    /* indexes */
                    count = 1;
                    for (j=0; j<3; j++)
                    {
                        index = splitIndexes[j];
                        
                        if (index < NVacancies)
                        {
                            index2 = vacancies[index];
                            vacancies[index] = -1;
                            splitInterstitials[3*NSplitInterstitials] = index2;
                            NVacNew--;
                        }
                        else
                        {
                            index2 = interstitials[index - NVacancies];
                            interstitials[index - NVacancies] = -1;
                            splitInterstitials[3*NSplitInterstitials+count] = index2;
                            NIntNew--;
                            count++;
                        }
                    }
                    NSplitInterstitials++;
                }
            }
        }
        
        /* free memory */
        free(defectClusterSplit);
        free(NDefectsCluster);
        free(defectPos);
        freeBoxes(boxes);
        
        /* recreate interstitials array */
        count = 0;
        for (i=0; i<NInterstitials; i++)
        {
            if (interstitials[i] != -1)
            {
                interstitials[count] = interstitials[i];
                count++;
            }
        }
        NInterstitials = count;
        
        /* recreate vacancies array */
        count = 0;
        for (i=0; i<NVacancies; i++)
        {
            if (vacancies[i] != -1)
            {
                vacancies[count] = vacancies[i];
                count++;
            }
        }
        NVacancies = count;
    }
    
//    printf("NVACS %d; NINTS %d; NSPLITINTS %d\n", NVacancies, NInterstitials, NSplitInterstitials);
    
    /* exclude defect types and species here... */
    if (!includeInts)
    {
        NInterstitials = 0;
        NSplitInterstitials = 0;
    }
    else
    {
        NIntNew = 0;
        for (i=0; i<NInterstitials; i++)
        {
            index = interstitials[i];
            
            skip = 0;
            for (j=0; j<exclSpecInputDim; j++)
            {
                if (specie[index] == exclSpecInput[j])
                {
                    skip = 1;
                    break;
                }
            }
            
            if (!skip)
            {
                interstitials[NIntNew] = index;
                NIntNew++;
            }
        }
        NInterstitials = NIntNew;
    }
    
    if (!includeVacs)
    {
        NVacancies = 0;
    }
    else
    {
        NVacNew = 0;
        for (i=0; i<NVacancies; i++)
        {
            index = vacancies[i];
            
            skip = 0;
            for (j=0; j<exclSpecRefDim; j++)
            {
                if (specie[index] == exclSpecRef[j])
                {
                    skip = 1;
                    break;
                }
            }
            
            if (!skip)
            {
                vacancies[NVacNew] = index;
                NVacNew++;
            }
        }
        NVacancies = NVacNew;
    }
    
    if (!includeAnts)
    {
        NAntisites = 0;
    }
    
//    printf("NVACS %d; NINTS %d; NSPLITINTS %d\n", NVacancies, NInterstitials, NSplitInterstitials);
    
    /* find clusters of defects */
    if (findClustersFlag)
    {
        /* build positions array of all defects */
        NDefects = NVacancies + NInterstitials + NAntisites + 3 * NSplitInterstitials;
        defectPos = malloc(3 * NDefects * sizeof(double));
        if (defectPos == NULL)
        {
            printf("ERROR: could not allocate defectPos\n");
            exit(50);
        }
        
        /* add defects positions: vac then int then ant */
        count = 0;
        for (i=0; i<NVacancies; i++)
        {
            index = vacancies[i];
            
            defectPos[3*count] = refPos[3*index];
            defectPos[3*count+1] = refPos[3*index+1];
            defectPos[3*count+2] = refPos[3*index+2];
            
            count++;
        }
        
        for (i=0; i<NInterstitials; i++)
        {
            index = interstitials[i];
            
            defectPos[3*count] = pos[3*index];
            defectPos[3*count+1] = pos[3*index+1];
            defectPos[3*count+2] = pos[3*index+2];
            
            count++;
        }
        
        for (i=0; i<NAntisites; i++)
        {
            index = antisites[i];
            
            defectPos[3*count] = refPos[3*index];
            defectPos[3*count+1] = refPos[3*index+1];
            defectPos[3*count+2] = refPos[3*index+2];
            
            count++;
        }
        
        for (i=0; i<NSplitInterstitials; i++)
        {
            index = splitInterstitials[3*i];
            defectPos[3*count] = refPos[3*index];
            defectPos[3*count+1] = refPos[3*index+1];
            defectPos[3*count+2] = refPos[3*index+2];
            count++;
            
            index = splitInterstitials[3*i+1];
            defectPos[3*count] = pos[3*index];
            defectPos[3*count+1] = pos[3*index+1];
            defectPos[3*count+2] = pos[3*index+2];
            count++;
            
            index = splitInterstitials[3*i+2];
            defectPos[3*count] = pos[3*index];
            defectPos[3*count+1] = pos[3*index+1];
            defectPos[3*count+2] = pos[3*index+2];
            count++;
        }
        
        /* box defects */
        approxBoxWidth = clusterRadius;
        boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
        putAtomsInBoxes(NDefects, defectPos, boxes);
        
        /* number of defects per cluster */
        NDefectsCluster = malloc(NDefects * sizeof(int));
        if (NDefectsCluster == NULL)
        {
            printf("ERROR: Boxes: could not allocate NDefectsCluster\n");
            exit(1);
        }
        
        /* find clusters */
        NClusters = findDefectClusters(NDefects, defectPos, defectCluster, NDefectsCluster, boxes, clusterRadius, cellDims, PBC);
        
        NDefectsCluster = realloc(NDefectsCluster, NClusters * sizeof(int));
        if (NDefectsCluster == NULL)
        {
            printf("ERROR: could not reallocate NDefectsCluster\n");
            exit(51);
        }
        
        /* first we have to adjust the number of atoms in clusters containing split interstitials */
        for (i=0; i<NSplitInterstitials; i++)
        {
            /* assume all lone defects forming split interstitial are in same cluster; maybe should check!? */
            j = NVacancies + NInterstitials + NAntisites;
            clusterIndex = defectCluster[j + 3 * i];
            
            /* subtract 2 because we add 3 instead of 1 */
            NDefectsCluster[clusterIndex] -= 2;
        }
        
        /* now limit by size */
        NDefectsClusterNew = calloc(NClusters, sizeof(int));
        if (NDefectsClusterNew == NULL)
        {
            printf("ERROR: could not allocate NDefectsClusterNew\n");
            exit(50);
        }
        
        /* first vacancies */
        NVacNew = 0;
        for (i=0; i<NVacancies; i++)
        {
            clusterIndex = defectCluster[i];
            index = vacancies[i];
            
            numInCluster = NDefectsCluster[clusterIndex];
            
            if (numInCluster < minClusterSize)
            {
                continue;
            }
            
            if (maxClusterSize >= minClusterSize && numInCluster > maxClusterSize)
            {
                continue;
            }
            
            vacancies[NVacNew] = index;
            defectCluster[NVacNew] = clusterIndex;
            NDefectsClusterNew[clusterIndex]++;
            
            NVacNew++;
        }
        
        /* now interstitials */
        NIntNew = 0;
        for (i=0; i<NInterstitials; i++)
        {
            clusterIndex = defectCluster[NVacancies+i];
            index = interstitials[i];
            
            numInCluster = NDefectsCluster[clusterIndex];
            
            if (numInCluster < minClusterSize)
            {
                continue;
            }
            
            if (maxClusterSize >= minClusterSize && numInCluster > maxClusterSize)
            {
                continue;
            }
            
            interstitials[NIntNew] = index;
            defectCluster[NVacNew+NIntNew] = clusterIndex;
            NDefectsClusterNew[clusterIndex]++;
            
            NIntNew++;
        }
        
        /* antisites */
        NAntNew = 0;
        for (i=0; i<NAntisites; i++)
        {
            clusterIndex = defectCluster[NVacancies+NInterstitials+i];
            index = antisites[i];
            index2 = onAntisites[i];
            
            numInCluster = NDefectsCluster[clusterIndex];
            
            if (numInCluster < minClusterSize)
            {
                continue;
            }
            
            if (maxClusterSize >= minClusterSize && numInCluster > maxClusterSize)
            {
                continue;
            }
            
            antisites[NAntNew] = index;
            onAntisites[NAntNew] = index2;
            defectCluster[NVacNew+NIntNew+NAntNew] = clusterIndex;
            NDefectsClusterNew[clusterIndex]++;
            
            NAntNew++;
        }
        
        /* split interstitials */
        NSplitNew = 0;
        for (i=0; i<NSplitInterstitials; i++)
        {
            /* assume all lone defects forming split interstitial are in same cluster */
            j = NVacancies + NInterstitials + NAntisites;
            clusterIndex = defectCluster[j + 3 * i];
            
            /* note, this number is wrong because we add 3 per split int, not 1 */
            numInCluster = NDefectsCluster[clusterIndex];
            
            if (numInCluster < minClusterSize)
            {
                continue;
            }
            
            if (maxClusterSize >= minClusterSize && numInCluster > maxClusterSize)
            {
                continue;
            }
            
            splitInterstitials[3*NSplitNew] = splitInterstitials[3*i];
            splitInterstitials[3*NSplitNew+1] = splitInterstitials[3*i+1];
            splitInterstitials[3*NSplitNew+2] = splitInterstitials[3*i+2];
            defectCluster[NVacNew+NIntNew+NAntNew+NSplitNew] = clusterIndex;
            NDefectsClusterNew[clusterIndex]++;
            
            NSplitNew++;
        }
        
        /* number of visible defects */
        NVacancies = NVacNew;
        NInterstitials = NIntNew;
        NAntisites = NAntNew;
        NSplitInterstitials = NSplitNew;
        
        /* recalc number of clusters */
        count = 0;
        for (i=0; i<NClusters; i++)
        {
            if (NDefectsClusterNew[i] > 0)
            {
                count++;
            }
        }
        NClusters = count;
        
        /* number of clusters */
        NDefectsType[4] = NClusters;
        
        /* free stuff */
        freeBoxes(boxes);
        free(defectPos);
        free(NDefectsCluster);
        free(NDefectsClusterNew);
    }
    
    /* counters */
    NDefects = NVacancies + NInterstitials + NAntisites + NSplitInterstitials;
    
    NDefectsType[0] = NDefects;
    NDefectsType[1] = NVacancies;
    NDefectsType[2] = NInterstitials;
    NDefectsType[3] = NAntisites;
    NDefectsType[5] = NSplitInterstitials;
    
    /* specie counters */
    for (i=0; i<NVacancies; i++)
    {
        index = vacancies[i];
        vacSpecCount[specieRef[index]]++;
    }
    
    for (i=0; i<NInterstitials; i++)
    {
        index = interstitials[i];
        intSpecCount[specie[index]]++;
    }
    
    for (i=0; i<NAntisites; i++)
    {
        index = antisites[i];
        antSpecCount[specieRef[index]]++;
        
        index2 = onAntisites[i];
        onAntSpecCount[specieRef[index]*NSpecies+specie[index2]]++;
    }
    
    for (i=0; i<NSplitInterstitials; i++)
    {
        index = splitInterstitials[3*i+1];
        index2 = splitInterstitials[3*i+2];
        
        splitIntSpecCount[specie[index]*NSpecies+specie[index2]]++;
    }
    
    if (driftCompensation)
    {
        free(refPos);
    }
    else
    {
        refPos = NULL;
    }
    
    return 0;
}


/*******************************************************************************
 * put defects into clusters
 *******************************************************************************/
int findDefectClusters(int NDefects, double *defectPos, int *defectCluster, int *NDefectsCluster, struct Boxes *boxes, double maxSep, 
                       double *cellDims, int *PBC)
{
    int i, maxNumInCluster;
    int NClusters, numInCluster;
    double maxSep2;
    
    
    maxSep2 = maxSep * maxSep;
    
    /* initialise cluster array
     * = -1 : not yet allocated
     * > -1 : cluster ID of defect
     */
    for (i=0; i<NDefects; i++)
    {
        defectCluster[i] = -1;
    }
    
    /* loop over defects */
    NClusters = 0;
    maxNumInCluster = -9999;
    for (i=0; i<NDefects; i++)
    {
        /* skip if atom already allocated */
        if (defectCluster[i] == -1)
        {
            /* allocate cluster ID */
            defectCluster[i] = NClusters;
            NClusters++;
            
            numInCluster = 1;
            
            /* recursive search for cluster atoms */
            numInCluster = findDefectNeighbours(i, defectCluster[i], numInCluster, defectCluster, defectPos, boxes, maxSep2, cellDims, PBC);
            
            NDefectsCluster[defectCluster[i]] = numInCluster;
            
            maxNumInCluster = (numInCluster > maxNumInCluster) ? numInCluster : maxNumInCluster;
        }
    }
    
    return NClusters;
}


/*******************************************************************************
 * recursive search for neighbouring defects
 *******************************************************************************/
int findDefectNeighbours(int index, int clusterID, int numInCluster, int* atomCluster, double *pos, struct Boxes *boxes, 
                         double maxSep2, double *cellDims, int *PBC)
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
            sep2 = atomicSeparation2( pos[3*index], pos[3*index+1], pos[3*index+2], pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                      cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2] );
            
            /* check if neighbours */
            if (sep2 < maxSep2)
            {
                atomCluster[index2] = clusterID;
                numInCluster++;
                
                /* search for neighbours to this new cluster atom */
                numInCluster = findDefectNeighbours(index2, clusterID, numInCluster, atomCluster, pos, boxes, maxSep2, cellDims, PBC);
            }
        }
    }
    
    return numInCluster;
}



