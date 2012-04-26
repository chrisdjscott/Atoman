/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Find defects
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "boxes.h"
#include "utilities.h"



/*******************************************************************************
 * Search for defects and return the sub-system surrounding them
 *******************************************************************************/
int findDefects( int includeVacs, int includeInts, int includeAnts, int NDefectsTypeDim, int* NDefectsType, int vacDim, int* vacancies, 
                 int intDim, int* interstitials, int antDim, int* antisites, int onAntDim, int* onAntisites, int exclSpecInputDim, 
                 int* exclSpecInput, int exclSpecRefDim, int* exclSpecRef, int NAtoms, int specieListDim, char* specieList, 
                 int specieDim, int* specie, int posDim, double* pos, int refNAtoms, int specieListRefDim, char* specieListRef, 
                 int specieRefDim, int* specieRef, int refPosDim, double* refPos, double xdim, double ydim, double zdim, int pbcx, 
                 int pbcy, int pbcz, double vacancyRadius, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax )
{
    int i, NSpecies, exitLoop, k, j, index;
    double minPos[3], maxPos[3];
    double vacRad2;
    int boxNebList[27], specieIndex;
    char symtemp[3], symtemp2[3];
    double xpos, ypos, zpos;
    int checkBox, refIndex, comp, boxIndex;
    double refxpos, refypos, refzpos;
    double sep2;
    int NDefects, NAntisites, NInterstitials, NVacancies;
    int *possibleVacancy, *possibleInterstitial;
    int *possibleAntisite, *possibleOnAntisite;
    int skip;
    double approxBoxWidth;
    
    printf("CLIB: finding defects\n");
    printf("  vac rad %f\n", vacancyRadius);
    
    /* boxing parameters (100 will always be enough
     * since box width is similar to vacancy radius)
     * SHOULD BE CALCULATED DEPENDING ON APPROXBOXWIDTH!!!!
     */
    maxAtomsPerBox = 100;
    minPos[0] = xmin;
    maxPos[0] = xmax;
    minPos[1] = ymin;
    maxPos[1] = ymax;
    minPos[2] = zmin;
    maxPos[2] = zmax;
    
    /* approx width, must be at least vacRad */
    approxBoxWidth = 1.1 * vacancyRadius;
    
    /* box reference atoms */
    setupBoxes( minPos, maxPos, approxBoxWidth );
    putAtomsInBoxes( refNAtoms, refPos );
    
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
        boxIndex = boxIndexOfAtom(xpos, ypos, zpos);
        
        /* find neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList);
                
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
            for ( k=0; k<boxNAtoms[checkBox]; k++ )
            {
                refIndex = boxAtoms[checkBox*maxAtomsPerBox+k];
                
                /* if this vacancy has already been filled then skip to the next one */
                if ( possibleVacancy[refIndex] == 0 )
                {
                    continue;
                }
                
                refxpos = refPos[3*refIndex];
                refypos = refPos[3*refIndex+1];
                refzpos = refPos[3*refIndex+2];
                
                /* atomic separation of possible vacancy and possible interstitial */
                sep2 = atomicSeparation2( xpos, ypos, zpos, refxpos, refypos, refzpos, xdim, ydim, zdim, pbcx, pbcy, pbcz );
                
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
    cleanupBoxes();
        
    /* now classify defects */
    NVacancies = 0;
    NInterstitials = 0;
    NAntisites = 0;
    for ( i=0; i<refNAtoms; i++ )
    {
        skip = 0;
        for (j=0; j<exclSpecRefDim; j++)
        {
            if (specieRef[i] == exclSpecRef[j])
            {
                skip = 1;
            }
        }
        
        if (skip == 1)
        {
            continue;
        }
        
        /* vacancies */
        if ( (possibleVacancy[i] == 1) && (includeVacs == 1) )
        {
            vacancies[NVacancies] = i;
            NVacancies++;
        }
        
        /* antisites */
        else if ( (possibleAntisite[i] == 1) && (includeAnts == 1) )
        {
            antisites[NAntisites] = i;
            onAntisites[NAntisites] = possibleOnAntisite[i];
            NAntisites++;
        }
    }
    
    if (includeInts == 1)
    {
        for ( i=0; i<NAtoms; i++ )
        {
            skip = 0;
            for (j=0; j<exclSpecInputDim; j++)
            {
                if (specie[i] == exclSpecInput[j])
                {
                    skip = 1;
                    break;
                }
            }
            
            if (skip == 1)
            {
                continue;
            }
                        
            /* interstitials */
            if ( (possibleInterstitial[i] == 1) )
            {
                interstitials[NInterstitials] = i;
                NInterstitials++;
            }
        }
    }
        
    NDefects = NVacancies + NInterstitials + NAntisites;
    
    printf("  found %d defects\n", NDefects);
    printf("    %d vacancies\n", NVacancies);
    printf("    %d interstitials\n", NInterstitials);
    printf("    %d antisites\n", NAntisites);
    
    NDefectsType[0] = NDefects;
    NDefectsType[1] = NVacancies;
    NDefectsType[2] = NInterstitials;
    NDefectsType[3] = NAntisites;
    
    /* free arrays */
    free(possibleVacancy);
    free(possibleInterstitial);
    free(possibleAntisite);
    free(possibleOnAntisite);
    
    return 0;
}
