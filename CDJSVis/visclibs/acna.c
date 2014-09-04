
/*******************************************************************************
 ** Adaptive Common Neighbour Analysis (Stutowski...)
 ** Copyright Chris Scott 2014
 *******************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "boxeslib.h"
#include "neb_list.h"
#include "utilities.h"
#include "acna.h"


/* function prototypes */
int compare_two_nebs(const void *, const void *);
int analyseAtom(int, struct NeighbourList2 *);
int checkForNeighbourBond(int, int, struct NeighbourList2 *, double);
void setNeighbourBond(unsigned int *, int, int, int);
int findCommonNeighbours(unsigned int *, int, unsigned int *);
int findNeighbourBonds(unsigned int *, unsigned int, int, unsigned int *);
int calcMaxChainLength(unsigned int *, int);
int getAdjacentBonds(unsigned int, unsigned int *, int *, unsigned int *, unsigned int *);


/*******************************************************************************
 ** Function that compares two elements in a neighbour list
 *******************************************************************************/
int compare_two_nebs(const void * a, const void * b)
{
    const struct Neighbour *n1 = a;
    const struct Neighbour *n2 = b;
    
    if (n1->separation < n2->separation) return -1;
    else if (n1->separation > n2->separation) return 1;
    else return 0;
}

/*******************************************************************************
 ** perform adaptive common neighbour analysis
 *******************************************************************************/
int adaptiveCommonNeighbourAnalysis(int NVisibleIn, int* visibleAtoms, int posDim, double *pos, int scalarsDim, double *scalars, 
                                    double *minPos, double *maxPos, double *cellDims, int *PBC, int NScalars, double *fullScalars,
                                    double maxBondDistance, int *counters, int filteringEnabled, int *structureVisibility)
{
    int i, NVisible, index;
    int atomStructure;
    double *visiblePos, approxBoxWidth, maxSep2;
    struct Boxes *boxes;
    struct NeighbourList2 *nebList;
    
    
/* first we construct neighbour list for each atom, containing indexes and separations */
    
    /* construct visible pos array */
    visiblePos = malloc(3 * NVisibleIn * sizeof(double));
    if (visiblePos == NULL)
    {
        printf("ERROR: could not allocate visiblePos\n");
        exit(50);
    }
    
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        visiblePos[3*i] = pos[3*index];
        visiblePos[3*i+1] = pos[3*index+1];
        visiblePos[3*i+2] = pos[3*index+2];
    }
    
    /* box visible atoms */
    approxBoxWidth = maxBondDistance;
    maxSep2 = maxBondDistance * maxBondDistance;
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(NVisibleIn, visiblePos, boxes);
    
    /* create neighbour list */
    nebList = constructNeighbourList2(NVisibleIn, visiblePos, boxes, cellDims, PBC, maxSep2);
    
    /* only required for building neb list */
    freeBoxes(boxes);
    free(visiblePos);
    
/* now we order the neighbour lists by separation */
    
    /* if less than min neighbours, mark as disordered!!! */
    
    
    /* sort neighbours by distance */
    for (i = 0; i < NVisibleIn; i++)
        qsort(nebList[i].neighbour, nebList[i].neighbourCount, sizeof(struct Neighbour), compare_two_nebs);
    
/* classify atoms */
    
    for (i = 0; i < NVisibleIn; i++)
    {
    	atomStructure = analyseAtom(i, nebList);
    	scalars[i] = (double) atomStructure;
    	counters[atomStructure]++;
    }
    
/* there should be option to only show atoms of specific structure type */
    
    if (filteringEnabled)
    {
        NVisible = 0;
        for (i = 0; i < NVisibleIn; i++)
        {
            atomStructure = (int) scalars[i];
            if (structureVisibility[atomStructure])
            {
                visibleAtoms[NVisible] = visibleAtoms[i];
                scalars[NVisible++] = scalars[i];
            }
        }
    }
    else NVisible = NVisibleIn;
    
/* tidy up */
    
    freeNeighbourList2(nebList, NVisibleIn);
    
    return NVisible;
}

/*******************************************************************************
 ** classify atom
 *******************************************************************************/
int analyseAtom(int mainIndex, struct NeighbourList2 *nebList)
{
	int i, j, nn, ok, visInd1, visInd2;
	double localScaling, localCutoff;
	double localScalingSum;
	
	
	/* check we have the minimum number of neighbours */
	if (nebList[mainIndex].neighbourCount < MIN_REQUIRED_NEBS)
		return ATOM_STRUCTURE_DISORDERED;
	
/* first we test for FCC, HCP, Icosohedral (12 1NN) */
	
	/* number of neighbours to test for */
	nn = 12;
	
	/* check enough nebs */
	if (nebList[mainIndex].neighbourCount < nn)
		return ATOM_STRUCTURE_DISORDERED;
	
	/* compute local cutoff */
	localScaling = 0.0;
	for (i = 0; i < nn; i++)
	{
		localScaling += nebList[mainIndex].neighbour[i].separation;
	}
	localScaling /= nn;
	localCutoff = localScaling * (1.0 + M_SQRT2) / 2.0;
	
	/* at this point I feel like we should check that the 12 NN are within localCutoff ????? */
	ok = 1;
	for (i = 0; i < nn; i++)
	{
		if (nebList[mainIndex].neighbour[i].separation > localCutoff)
		{
			ok = 0;
			break;
		}
	}
	
	if (ok)
	{
		int n421;
		int n422;
		int n555;
		unsigned int neighbourArray[MAX_REQUIRED_NEBS] = {0};
		
		/* determine bonding between neighbours, based on local cutoff */
		for (i = 0; i < nn; i++)
		{
			visInd1 = nebList[mainIndex].neighbour[i].index;
			setNeighbourBond(neighbourArray, i, i, 0);
			for (j = i + 1; j < nn; j++)
			{
				visInd2 = nebList[mainIndex].neighbour[j].index;
				setNeighbourBond(neighbourArray, i, j, checkForNeighbourBond(visInd1, visInd2, nebList, localCutoff));
			}
		}
		
		n421 = 0;
		n422 = 0;
		n555 = 0;
		for (i = 0; i < nn; i++)
		{
		    int numCommonNeighbours;
		    int numNeighbourBonds;
		    int maxChainLength;
		    unsigned int commonNeighbours;
		    unsigned int neighbourBonds[MAX_REQUIRED_NEBS*MAX_REQUIRED_NEBS] = {0};
		    
		    /* number of common neighbours */
			numCommonNeighbours = findCommonNeighbours(neighbourArray, i, &commonNeighbours);
			if (numCommonNeighbours != 4 && numCommonNeighbours != 5)
				break;
			
			/* number of bonds among common neighbours */
			numNeighbourBonds = findNeighbourBonds(neighbourArray, commonNeighbours, nn, neighbourBonds);
			if (numNeighbourBonds != 2 && numNeighbourBonds != 5)
			    break;
			
			/* number of bonds in the longest continuous chain */
			maxChainLength = calcMaxChainLength(neighbourBonds, numNeighbourBonds);
			if (numCommonNeighbours == 4 && numNeighbourBonds == 2)
			{
			    if (maxChainLength == 1) n421++;
			    else if (maxChainLength == 2) n422++;
			    else break;
			}
			else if (numCommonNeighbours == 5 && numNeighbourBonds == 5 && maxChainLength == 5) n555++;
			else break;
		}
		if (n421 == 12) return ATOM_STRUCTURE_FCC;
		else if (n421 == 6 && n422 == 6) return ATOM_STRUCTURE_HCP;
		else if (n555 == 12) return ATOM_STRUCTURE_ICOSAHEDRAL;
	}
	
/* next we test for BCC (8 1NN + 6 2NN) */
	
	/* number of neighbours to test for */
    nn = 14;
    
    /* check enough nebs */
    if (nebList[mainIndex].neighbourCount < nn)
        return ATOM_STRUCTURE_DISORDERED;
    
    /* compute local cutoff */
    localScaling = 0.0;
    for (i = 0; i < 8; i++)
    {
        localScaling += nebList[mainIndex].neighbour[i].separation;
    }
    localScaling /= 8.0;
    
    localScalingSum = 0.0;
    for (i = 8; i < 14; i++)
    {
        localScalingSum += nebList[mainIndex].neighbour[i].separation;
    }
    localScalingSum /= 6.0;
    
    localCutoff = (1.0 + M_SQRT2) / 4.0 * (2.0 / M_SQRT3 * localScaling + localScalingSum); // divide by 4 not 2 as in the paper
    
    /* at this point I feel like we should check that the 12 NN are within localCutoff ????? */
    ok = 1;
    for (i = 0; i < nn; i++)
    {
        if (nebList[mainIndex].neighbour[i].separation > localCutoff)
        {
            ok = 0;
            break;
        }
    }
    
    if (ok)
    {
        int n444;
        int n666;
        unsigned int neighbourArray[MAX_REQUIRED_NEBS] = {0};
        
        /* determine bonding between neighbours, based on local cutoff */
        for (i = 0; i < nn; i++)
        {
            visInd1 = nebList[mainIndex].neighbour[i].index;
            setNeighbourBond(neighbourArray, i, i, 0);
            for (j = i + 1; j < nn; j++)
            {
                visInd2 = nebList[mainIndex].neighbour[j].index;
                setNeighbourBond(neighbourArray, i, j, checkForNeighbourBond(visInd1, visInd2, nebList, localCutoff));
            }
        }
        
        n444 = 0;
        n666 = 0;
        for (i = 0; i < nn; i++)
        {
            int numCommonNeighbours;
            int numNeighbourBonds;
            int maxChainLength;
            unsigned int commonNeighbours;
            unsigned int neighbourBonds[MAX_REQUIRED_NEBS*MAX_REQUIRED_NEBS] = {0};
            
            /* number of common neighbours */
            numCommonNeighbours = findCommonNeighbours(neighbourArray, i, &commonNeighbours);
            if (numCommonNeighbours != 4 && numCommonNeighbours != 6)
                break;
            
            /* number of bonds among common neighbours */
            numNeighbourBonds = findNeighbourBonds(neighbourArray, commonNeighbours, nn, neighbourBonds);
            if (numNeighbourBonds != 4 && numNeighbourBonds != 6)
                break;
            
            /* number of bonds in the longest continuous chain */
            maxChainLength = calcMaxChainLength(neighbourBonds, numNeighbourBonds);
            if (numCommonNeighbours == 4 && numNeighbourBonds == 4 && maxChainLength == 4) n444++;
            else if (numCommonNeighbours == 6 && numNeighbourBonds == 6 && maxChainLength == 6) n666++;
            else break;
        }
        if (n666 == 8 && n444 == 6) return ATOM_STRUCTURE_BCC;
    }
	
	return ATOM_STRUCTURE_DISORDERED;
}

/*******************************************************************************
 ** find all chains of bonds between common neighbours and determine the length
 ** of the longest continuous chain
 *******************************************************************************/
int calcMaxChainLength(unsigned int *neighbourBonds, int numBonds)
{
    int maxChainLength;
    
    
    maxChainLength = 0;
    while (numBonds)
    {
        int clusterSize;
        unsigned int atomsToProcess, atomsProcessed;
        
        
        /* make a new cluster starting with the first remaining bond to be processed */
        numBonds--;
        
        /* initialise some variables */
        atomsToProcess = neighbourBonds[numBonds];
        atomsProcessed = 0;
        clusterSize = 1;
        
        do 
        {
            unsigned int nextAtom;
            /* determine number of trailing 0-bits in atomsToProcess
             * starting with least significant bit position
             */
#if defined(__GNUC__)
            int nextAtomIndex = __builtin_ctz(atomsToProcess);
#elif defined(_MSC_VER)
            unsigned long nextAtomIndex;
            _BitScanForward(&nextAtomIndex, atomsToProcess);
#else
            #error "Your C compiler is not supported."
#endif
            if (nextAtomIndex < 0 || nextAtomIndex >= 32)
            {
                printf("nextAtomIndex error (%d)\n", nextAtomIndex);
                exit(98);
            }
            
            nextAtom = 1 << nextAtomIndex;
            atomsProcessed |= nextAtom;
            atomsToProcess &= ~nextAtom;
            clusterSize += getAdjacentBonds(nextAtom, neighbourBonds, &numBonds, &atomsToProcess, &atomsProcessed);
        }
        while (atomsToProcess);
        
        if (clusterSize > maxChainLength)
            maxChainLength = clusterSize;
    }
    
    return maxChainLength;
}

/*******************************************************************************
 ** find all chains of bonds
 *******************************************************************************/
int getAdjacentBonds(unsigned int atom, unsigned int *bondsToProcess, int *numBonds, unsigned int *atomsToProcess, unsigned int *atomsProcessed)
{
    int adjacentBonds, b;
    
    
    adjacentBonds = 0;
    for (b = *numBonds - 1; b >= 0; b--)
    {
        if (atom & *bondsToProcess)
        {
            ++adjacentBonds;
//            *atomsToProcess |= *bondsToProcess & (~*atomsProcessed);
            *atomsToProcess = *atomsToProcess | (*bondsToProcess & (~*atomsProcessed));
            memmove(bondsToProcess, bondsToProcess + 1, sizeof(unsigned int) * b);
            *numBonds = *numBonds - 1;
        }
        else ++bondsToProcess;
    }
    
    return adjacentBonds;
}


/*******************************************************************************
 ** find bonds between common nearest neighbours
 *******************************************************************************/
int findNeighbourBonds(unsigned int *neighbourArray, unsigned int commonNeighbours, int numNeighbours, unsigned int *neighbourBonds)
{
    int ni1, n;
    int numBonds;
    int neb_size;
    unsigned int nib[MAX_REQUIRED_NEBS] = {0};
    int nibn;
    unsigned int ni1b;
    unsigned int b;
    
    
    neb_size = MAX_REQUIRED_NEBS * MAX_REQUIRED_NEBS;
    
    numBonds = 0;
    nibn = 0;
    ni1b = 1;
    for (ni1 = 0; ni1 < numNeighbours; ni1++, ni1b <<= 1)
    {
        if (commonNeighbours & ni1b)
        {
            b = commonNeighbours & neighbourArray[ni1];
            for (n = 0; n < nibn; n++)
            {
                if (b & nib[n])
                {
                    if (numBonds > neb_size)
                    {
                        printf("ERROR: num bonds exceeds limit (findNeighbourBonds)\n");
                        exit(58);
                    }
                    neighbourBonds[numBonds++] = ni1b | nib[n];
                }
            }
            
            nib[nibn++] = ni1b;
        }
    }
    
    return numBonds;
}

/*******************************************************************************
 ** find common neighbours
 *******************************************************************************/
int findCommonNeighbours(unsigned int *neighbourArray, int neighbourIndex, unsigned int *commonNeighbours)
{
#ifdef __GNUC__
	*commonNeighbours = neighbourArray[neighbourIndex];
	
	/* Count the number of bits set in neighbor bit field. */
	return __builtin_popcount(*commonNeighbours); // GNU g++ specific
#else
	unsigned int v;
	
	*commonNeighbours = neighbourArray[neighbourIndex];
	
	/* Count the number of bits set in neighbor bit field. */
	v = *commonNeighbours - ((*commonNeighbours >> 1) & 0x55555555);
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
	return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
}

/*******************************************************************************
 ** check if two neighbours are bonded
 *******************************************************************************/
int checkForNeighbourBond(int visInd1, int visInd2, struct NeighbourList2 *nebList, double cutoff)
{
	int i, bonded;
	
	
	bonded = 0;
	for (i = 0; i < nebList[visInd1].neighbourCount; i++)
	{
		if (nebList[visInd1].neighbour[i].index == visInd2 && nebList[visInd1].neighbour[i].separation <= cutoff)
		{
			bonded = 1;
			break;
		}
	}
	
	return bonded;
}

/*******************************************************************************
 ** set neighbour bond
 *******************************************************************************/
void setNeighbourBond(unsigned int *neighbourArray, int index1, int index2, int bonded)
{
	if (bonded)
	{
		neighbourArray[index1] |= (1<<index2);
		neighbourArray[index2] |= (1<<index1);
	}
	else
	{
		neighbourArray[index1] &= ~(1<<index2);
		neighbourArray[index2] &= ~(1<<index1);
	}
}
