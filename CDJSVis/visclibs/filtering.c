
/*******************************************************************************
 ** Copyright Chris Scott 2011
 ** Filtering routines written in C to improve performance
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"
#include "boxeslib.h"
#include "filtering.h"
#include "constants.h"


/*******************************************************************************
 ** Specie filter
 *******************************************************************************/
int specieFilter(int NVisibleIn, int *visibleAtoms, int visSpecDim, int* visSpec, int specieDim, int *specie,
		         int scalarsDim, double *scalars)
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
            
            if (scalarsDim == NVisibleIn)
            	scalars[NVisible] = scalars[i];
            
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Slice filter
 *******************************************************************************/
int sliceFilter(int NVisibleIn, int *visibleAtoms, int posDim, double *pos, double x0,
                double y0, double z0, double xn, double yn, double zn, int invert,
                int scalarsDim, double *scalars)
{
    int i, NVisible, index;
    double mag, xd, yd, zd, dotProd, distanceToPlane;
    
    /* normalise (xn, yn, zn) */
    mag = sqrt(x0 * x0 + y0 * y0 + z0 * z0);
    xn = xn / mag;
    yn = yn / mag;
    zn = zn / mag;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        xd = pos[3*index] - x0;
        yd = pos[3*index+1] - y0;
        zd = pos[3*index+2] - z0;
        
        dotProd = xd * xn + yd * yn + zd * zn;
        distanceToPlane = dotProd / mag;
        
        if ((invert && distanceToPlane > 0) || (!invert && distanceToPlane < 0))
        {
            visibleAtoms[NVisible] = index;
            
            if (scalarsDim == NVisibleIn)
				scalars[NVisible] = scalars[i];
            
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Crop sphere filter
 *******************************************************************************/
int cropSphereFilter(int NVisibleIn, int *visibleAtoms, int posDim, double *pos, double xCentre, 
                     double yCentre, double zCentre, double radius, double *cellDims, 
                     int *PBC, int invertSelection, int scalarsDim, double *scalars)
{
    int i, NVisible, index;
    double radius2, sep2;
    
    
    radius2 = radius * radius;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                 xCentre, yCentre, zCentre, 
                                 cellDims[0], cellDims[1], cellDims[2], 
                                 PBC[0], PBC[1], PBC[2]);
        
        if (sep2 < radius2)
        {
            if (invertSelection)
            {
                visibleAtoms[NVisible] = index;
                
                if (scalarsDim == NVisibleIn)
					scalars[NVisible] = scalars[i];
                
                NVisible++;
            }
        }
        else
        {
            if (!invertSelection)
            {
                visibleAtoms[NVisible] = index;
                
                if (scalarsDim == NVisibleIn)
					scalars[NVisible] = scalars[i];
                
                NVisible++;
            }
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Crop filter
 *******************************************************************************/
int cropFilter(int NVisibleIn, int* visibleAtoms, int posDim, double* pos, double xmin, double xmax,
               double ymin, double ymax, double zmin, double zmax, int xEnabled, int yEnabled, int zEnabled,
               int invertSelection, int scalarsDim, double *scalars)
{
    int i, index, NVisible, add;
    double rx, ry, rz;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        rx = pos[3*index];
        ry = pos[3*index+1];
        rz = pos[3*index+2];
        
        add = 1;
        if (xEnabled == 1)
        {
            if (rx < xmin || rx > xmax)
            {
				add = 0;
            }
        }
        
        if (add && yEnabled == 1)
        {
            if (ry < ymin || ry > ymax)
            {
				add = 0;
            }
        }
        
        if (add && zEnabled == 1)
        {
            if (rz < zmin || rz > zmax)
            {
				add = 0;
            }
        }
        
        if ((add && !invertSelection) || (!add && invertSelection))
        {
			visibleAtoms[NVisible] = index;
			
			if (scalarsDim == NVisibleIn)
				scalars[NVisible] = scalars[i];
			
			NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Displacement filter
 *******************************************************************************/
int displacementFilter(int NVisibleIn, int* visibleAtoms, int scalarsDim, double *scalars, int posDim, double *pos, int refPosDim, double *refPos, 
                       double *cellDims, int *PBC, double minDisp, double maxDisp)
{
    int i, NVisible, index;
    double sep2, maxDisp2, minDisp2;
    
    
    minDisp2 = minDisp * minDisp;
    maxDisp2 = maxDisp * maxDisp;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                 refPos[3*index], refPos[3*index+1], refPos[3*index+2], 
                                 cellDims[0], cellDims[1], cellDims[2], 
                                 PBC[0], PBC[1], PBC[2]);
        
        if (sep2 <= maxDisp2 && sep2 >= minDisp2)
        {
            visibleAtoms[NVisible] = index;
            scalars[NVisible] = sqrt(sep2);
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Kinetic energy filter
 *******************************************************************************/
int KEFilter(int NVisibleIn, int* visibleAtoms, int KEDim, double *KE, double minKE, double maxKE,
		     int scalarsDim, double *scalars)
{
    int i, NVisible, index;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (KE[index] < minKE || KE[index] > maxKE)
        {
            continue;
        }
        else
        {
            visibleAtoms[NVisible] = index;
            
            if (scalarsDim == NVisibleIn)
				scalars[NVisible] = scalars[i];
            
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Potential energy filter
 *******************************************************************************/
int PEFilter(int NVisibleIn, int* visibleAtoms, int PEDim, double *PE, double minPE, double maxPE,
		     int scalarsDim, double *scalars)
{
    int i, NVisible, index;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (PE[index] < minPE || PE[index] > maxPE)
        {
            continue;
        }
        else
        {
            visibleAtoms[NVisible] = index;
            
            if (scalarsDim == NVisibleIn)
				scalars[NVisible] = scalars[i];
            
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 ** Charge energy filter
 *******************************************************************************/
int chargeFilter(int NVisibleIn, int* visibleAtoms, int chargeDim, double *charge, double minCharge, double maxCharge,
		  	     int scalarsDim, double *scalars)
{
    int i, NVisible, index;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (charge[index] < minCharge || charge[index] > maxCharge)
        {
            continue;
        }
        else
        {
            visibleAtoms[NVisible] = index;
            
            if (scalarsDim == NVisibleIn)
				scalars[NVisible] = scalars[i];
            
            NVisible++;
        }
    }
    
    return NVisible;
}


/*******************************************************************************
 * Calculate coordination number
 *******************************************************************************/
int coordNumFilter(int NVisible, int *visibleAtoms, double *pos, int *specie, int NSpecies, double *bondMinArray, double *bondMaxArray, 
                   double approxBoxWidth, double *cellDims, int *PBC, double *minPos, double *maxPos,
                   double *coordArray, int minCoordNum, int maxCoordNum)
{
    int i, j, k, index, index2, visIndex;
    int speca, specb, count, NVisibleNew;
    int boxIndex, boxNebList[27];
    double *visiblePos, sep2, sep;
    struct Boxes *boxes;
    
    
//    printf("BONDS CLIB\n");
//    printf("N VIS: %d\n", NVisible);
//    
//    for (i=0; i<NSpecies; i++)
//    {
//        for (j=i; j<NSpecies; j++)
//        {
//            printf("%d - %d: %lf -> %lf\n", i, j, bondMinArray[i*NSpecies+j], bondMaxArray[i*NSpecies+j]);
//        }
//    }
    
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
    
    /* free visible pos */
    free(visiblePos);
    
    /* zero coord array */
    for (i=0; i<NVisible; i++)
    {
        coordArray[i] = 0;
    }
    
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
                    coordArray[i]++;
                    coordArray[visIndex]++;
                    
                    count++;
                }
            }
        }
    }
    
    /* filter */
    NVisibleNew = 0;
    for (i=0; i<NVisible; i++)
    {
        if (coordArray[i] >= minCoordNum && coordArray[i] <= maxCoordNum)
        {
            visibleAtoms[NVisibleNew] = visibleAtoms[i];
            coordArray[NVisibleNew] = coordArray[i];
            
            NVisibleNew++;
        }
    }
    
    /* free */
    freeBoxes(boxes);
    
    return NVisibleNew;
}

/*******************************************************************************
 ** Voronoi volume filter
 *******************************************************************************/
int voronoiVolumeFilter(int NVisibleIn, int* visibleAtoms, int volumeDim, double *volume, double minVolume, double maxVolume,
		  	            int scalarsDim, double *scalars)
{
    int i, NVisible, index;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (volume[index] < minVolume || volume[index] > maxVolume)
        {
            continue;
        }
        else
        {
            visibleAtoms[NVisible] = index;
            scalars[NVisible] = volume[index];
            
            NVisible++;
        }
    }
    
    return NVisible;
}

/*******************************************************************************
 ** Voronoi neighbours filter
 *******************************************************************************/
int voronoiNeighboursFilter(int NVisibleIn, int* visibleAtoms, int volumeDim, int *num_nebs_array, int minNebs, int maxNebs,
		  	            	int scalarsDim, double *scalars)
{
    int i, NVisible, index;
    
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (num_nebs_array[index] < minNebs || num_nebs_array[index] > maxNebs)
        {
            continue;
        }
        else
        {
            visibleAtoms[NVisible] = index;
            scalars[NVisible] = (double) num_nebs_array[index];
            
            NVisible++;
        }
    }
    
    return NVisible;
}

/*******************************************************************************
 **Q4 filter
 *******************************************************************************/
int Q4Filter(int NVisibleIn, int* visibleAtoms, int posDim, double *pos, double minQ4, double maxQ4, double maxBondDistance, 
		  	            int scalarsDim, double *scalars, double *minPos, double *maxPos, double *cellDims, int *PBC)
{
	int i, j, k, index, index2, NVisible, boxNebList[27];
	int *NBondsForAtom, boxIndex, visIndex, maxSep2, num_bonds;
	double Rx, Ry, Rz, approxBoxWidth, arg;
	double xposa, xposb, yposa, yposb, zposa, zposb;
	double *visiblePos, sepVec[3], sep2, sep;
	double *Q4m4, *Q4m3, *Q4m2, *Q4m1, *Q40, *Q41, *Q42, *Q43, *Q44;
	double Y4m4, Y4m3, Y4m2, Y4m1, Y40, Y41, Y42, Y43, Y44;
	double costheta, sintheta, cosphi, sinphi, minVal, maxVal;
	double Q4Param, Q4m4SQ, Q4m3SQ, Q4m2SQ, Q4m1SQ, Q40SQ, Q41SQ, Q42SQ, Q43SQ, Q44SQ;
	struct Boxes *boxes;
	
	
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
	
	/* only required for boxing */
	free(visiblePos);
	
	/* set up arrays for calculation */
	NBondsForAtom = calloc(NVisibleIn, sizeof(int));
	if (NBondsForAtom == NULL)
	{
		printf("ERROR: could not allocate NBondsForAtom\n");
		exit(50);
	}
	
	Q4m4 = calloc(NVisibleIn, sizeof(double));
	Q4m3 = calloc(NVisibleIn, sizeof(double));
	Q4m2 = calloc(NVisibleIn, sizeof(double));
	Q4m1 = calloc(NVisibleIn, sizeof(double));
	Q40 = calloc(NVisibleIn, sizeof(double));
	Q41 = calloc(NVisibleIn, sizeof(double));
	Q42 = calloc(NVisibleIn, sizeof(double));
	Q43 = calloc(NVisibleIn, sizeof(double));
	Q44 = calloc(NVisibleIn, sizeof(double));
	
	/* loop over atoms */
	num_bonds = 0;
	for (i = 0; i < NVisibleIn; i++)
	{
		/* atom index */
		index = visibleAtoms[i];
		
		/* position */
		xposa = pos[3*index];
		yposa = pos[3*index+1];
		zposa = pos[3*index+2];
		
		/* get box index of this atom */
		boxIndex = boxIndexOfAtom(xposa, yposa, zposa, boxes);
		
		/* find neighbouring boxes */
		getBoxNeighbourhood(boxIndex, boxNebList, boxes);
		
		/* loop over box neighbourhood */
		for (j=0; j<27; j++)
		{
			boxIndex = boxNebList[j];
			
			/* loop over atoms in box */
			for (k=0; k<boxes->boxNAtoms[boxIndex]; k++)
			{
				visIndex = boxes->boxAtoms[boxIndex][k];
				index2 = visibleAtoms[visIndex];
				
				if (index >= index2)
				{
					continue;
				}
				
				/* position */
				xposb = pos[3*index2];
				yposb = pos[3*index2+1];
				zposb = pos[3*index2+2];
				
				/* separation vector */
				atomSeparationVector(sepVec, xposa, yposa, zposa, xposb, yposb, zposb, cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
				
				sep2 = sepVec[0] * sepVec[0] + sepVec[1] * sepVec[1] + sepVec[2] * sepVec[2];
				
				if (sep2 < maxSep2)
				{
					sep = sqrt(sep2);
					
					Rx = -1.0 * sepVec[0];
					Ry = -1.0 * sepVec[1];
					Rz = -1.0 * sepVec[2];
					
					/* define cosine and sine for angles theta and phi */
					costheta = Rz / sep;
					arg = sqrt(Rx * Rx + Ry * Ry);
					sintheta = arg / sep;
					
					if (arg <= 0.0001)
					{
						cosphi = 0.0;
						sinphi = 0.0;
					}
					else
					{
						cosphi = Rx / arg;
						sinphi = Ry / arg;
					}
					
					/* spherical harmonics */
					Y4m4 = 3.0 / 4.0 * sqrt(35.0 / PI) * pow(sintheta, 4.0) * sinphi * cosphi * (2.0 * pow(cosphi, 2.0) - 1.0);
					Y4m3 = -3.0 / 4.0 * sqrt(35.0 / (2.0 * PI)) * pow(sintheta, 3.0) * costheta * (4.0 * pow(cosphi, 3.0) - 3.0 * cosphi);
					Y4m2 = 3.0 / 4.0 * sqrt(5.0 / PI) * pow(sintheta, 2.0) * (7.0 * pow(costheta, 2.0) - 1.0) * cosphi * sinphi;
					Y4m1 = -3.0 / 4.0 * sqrt(5.0 / (2.0 * PI)) * sintheta * (7.0 * pow(costheta, 3.0) - 3.0 * costheta) * cosphi;
					
					Y40 = 3.0 / (16.0 * sqrt(PI)) * (35.0 * pow(costheta, 4.0) - 30.0 * pow(costheta, 2.0) + 3.0);
					Y41 = -3.0 / 4.0 * sqrt(5.0 / (2.0*PI)) * sintheta * (7.0 * pow(costheta, 3.0) - 3.0 * costheta) * sinphi;
					Y42 = 3.0 / 8.0 * sqrt(5.0 / PI) * pow(sintheta, 2.0) * (7.0 * pow(costheta, 2.0) - 1.0) * (2.0 * pow(cosphi, 2.0) - 1.0);
					Y43 = -3.0 / 4.0 * sqrt(35.0 / (2.0 * PI)) * pow(sintheta, 3.0) * costheta * (3.0 * sinphi - 4.0 * pow(sinphi, 3.0));
					Y44 = 3.0 / 16.0 * sqrt(35.0 / PI) * pow(sintheta, 4.0) * (1.0 - 8.0 * pow(cosphi, 2.0) * pow(sinphi, 2.0));
					
					NBondsForAtom[i]++;
					Q4m4[i] += Y4m4;
					Q4m3[i] += Y4m3;
					Q4m2[i] += Y4m2;
					Q4m1[i] += Y4m1;
					Q40[i] += Y40;
					Q41[i] += Y41;
					Q42[i] += Y42;
					Q43[i] += Y43;
					Q44[i] += Y44;
					
					NBondsForAtom[visIndex]++;
					Q4m4[visIndex] += Y4m4;
					Q4m3[visIndex] += Y4m3;
					Q4m2[visIndex] += Y4m2;
					Q4m1[visIndex] += Y4m1;
					Q40[visIndex] += Y40;
					Q41[visIndex] += Y41;
					Q42[visIndex] += Y42;
					Q43[visIndex] += Y43;
					Q44[visIndex] += Y44;
					
					num_bonds += 2;
				}
			}
		}
	}
	
	NVisible = 0;
	minVal = 99999.0;
	maxVal = -99999.0;
	for (i = 0; i < NVisibleIn; i++)
	{
		Q4m4SQ = pow(Q4m4[i] / NBondsForAtom[i], 2.0);
		Q4m3SQ = pow(Q4m3[i] / NBondsForAtom[i], 2.0);
		Q4m2SQ = pow(Q4m2[i] / NBondsForAtom[i], 2.0);
		Q4m1SQ = pow(Q4m1[i] / NBondsForAtom[i], 2.0);
		
		Q40SQ = pow(Q40[i] / NBondsForAtom[i], 2.0);
		Q41SQ = pow(Q41[i] / NBondsForAtom[i], 2.0);
		Q42SQ = pow(Q42[i] / NBondsForAtom[i], 2.0);
		Q43SQ = pow(Q43[i] / NBondsForAtom[i], 2.0);
		Q44SQ = pow(Q44[i] / NBondsForAtom[i], 2.0);
		
		Q4Param = sqrt((4.0 * PI / 9.0) * (Q4m4SQ + Q4m3SQ + Q4m2SQ + Q4m1SQ + Q40SQ + Q41SQ + Q42SQ + Q43SQ + Q44SQ));
		
		if (Q4Param >= minQ4 && Q4Param <= maxQ4)
		{
			visibleAtoms[NVisible] = visibleAtoms[i];
			scalars[NVisible] = Q4Param;
			
			NVisible++;
		}
		
		if (i == 0)
		{
			minVal = Q4Param;
			maxVal = Q4Param;
		}
		else
		{
			minVal = (Q4Param < minVal) ? Q4Param : minVal;
			maxVal = (Q4Param > maxVal) ? Q4Param : maxVal;
		}
	}
	
	/* tidy up */
	free(Q4m4);
	free(Q4m3);
	free(Q4m2);
	free(Q4m1);
	free(Q40);
	free(Q41);
	free(Q42);
	free(Q43);
	free(Q44);
	free(NBondsForAtom);
	freeBoxes(boxes);
	
	return NVisible;
}



