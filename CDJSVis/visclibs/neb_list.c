
#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <math.h>
#include "boxeslib.h"
#include "utilities.h"
#include "neb_list.h"


static int addAtomToNebList(int, int, double, struct NeighbourList2 *);


struct NeighbourList * constructNeighbourList(int NAtoms, double *pos, struct Boxes *boxes, double *cellDims, int *PBC, double maxSep2)
{
    int i, j, k, boxIndex, indexb, newsize;
    int boxNebList[27];
    double rxa, rya, rza, rxb, ryb, rzb, sep2;
    struct NeighbourList *nebList;
    
    
    /* allocate neb list */
    nebList = malloc(NAtoms * sizeof(struct NeighbourList));
    if (nebList == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate nebList");
        return NULL;
    }
    
    /* initialise */
    for (i = 0; i < NAtoms; i++)
    {
        nebList[i].chunk = 16;
        nebList[i].neighbourCount = 0;
    }
    
    /* loop over atoms */
    for (i = 0; i < NAtoms; i++)
    {
        int boxNebListSize;
        
        /* atom position */
        rxa = pos[3*i];
        rya = pos[3*i+1];
        rza = pos[3*i+2];
        
        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(rxa, rya, rza, boxes);
        if (boxIndex < 0)
        {
            freeNeighbourList(nebList, NAtoms);
            return NULL;
        }
        
        /* find neighbouring boxes */
        boxNebListSize = getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over box neighbourhood */
        for (j = 0; j < boxNebListSize; j++)
        {
            boxIndex = boxNebList[j];
            
            /* loop over atoms in box */
            for (k = 0; k < boxes->boxNAtoms[boxIndex]; k++)
            {
                indexb = boxes->boxAtoms[boxIndex][k];
                
                if (indexb == i) continue;
                
                /* atom position */
                rxb = pos[3*indexb];
                ryb = pos[3*indexb+1];
                rzb = pos[3*indexb+2];
                
                /* separation */
                sep2 = atomicSeparation2(rxa, rya, rza, rxb, ryb, rzb, cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
                
                /* check if neighbour */
                if (sep2 < maxSep2)
                {
                    /* check if need to resize neighbour pointers */
                    if (nebList[i].neighbourCount == 0)
                    {
                        nebList[i].neighbour = malloc(nebList[i].chunk * sizeof(int));
                        if (nebList[i].neighbour == NULL)
                        {
                            char errstring[128];
                            sprintf(errstring, "Could not allocate nebList[%d].neighbour\n", i);
                            PyErr_SetString(PyExc_MemoryError, errstring);
                            freeNeighbourList(nebList, NAtoms);
                            return NULL;
                        }
                        nebList[i].neighbourSep = malloc(nebList[i].chunk * sizeof(double));
                        if (nebList[i].neighbourSep == NULL)
                        {
                            char errstring[128];
                            sprintf(errstring, "Could not allocate nebList[%d].neighbourSep\n", i);
                            PyErr_SetString(PyExc_MemoryError, errstring);
                            freeNeighbourList(nebList, NAtoms);
                            return NULL;
                        }
                    }
                    else if (nebList[i].neighbourCount % nebList[i].chunk == 0)
                    {
                        newsize = nebList[i].neighbourCount + nebList[i].chunk;
                        nebList[i].neighbour = realloc(nebList[i].neighbour, newsize * sizeof(int));
                        if (nebList[i].neighbour == NULL)
                        {
                            char errstring[128];
                            sprintf(errstring, "Could not reallocate nebList[%d].neighbour\n", i);
                            PyErr_SetString(PyExc_MemoryError, errstring);
                            freeNeighbourList(nebList, NAtoms);
                            return NULL;
                        }
                        nebList[i].neighbourSep = realloc(nebList[i].neighbourSep, newsize * sizeof(double));
                        if (nebList[i].neighbourSep == NULL)
                        {
                            char errstring[128];
                            sprintf(errstring, "Could not reallocate nebList[%d].neighbourSep\n", i);
                            PyErr_SetString(PyExc_MemoryError, errstring);
                            freeNeighbourList(nebList, NAtoms);
                            return NULL;
                        }
                    }
                    
                    /* add neighbour */
                    nebList[i].neighbour[nebList[i].neighbourCount] = indexb;
                    nebList[i].neighbourSep[nebList[i].neighbourCount] = sqrt(sep2);
                    nebList[i].neighbourCount++;
                    
//                    if (i < 1)
//                        printf("NEB OF %d: %d; sep %lf\n", i, indexb, sqrt(sep2));
                }
            }
        }
//        if (i < 1)
//            printf("NUM NEBS FOR %d: %d\n", i, nebList[i].neighbourCount);
    }
    
    return nebList;
}

void freeNeighbourList(struct NeighbourList *nebList, int size)
{
    int i;
    
    for (i = 0; i < size; i++)
    {
        if (nebList[i].neighbourCount > 0)
        {
            free(nebList[i].neighbour);
            free(nebList[i].neighbourSep);
        }
    }
    free(nebList);
}

/*************************************************/

static int addAtomToNebList(int mainIndex, int nebIndex, double sep, struct NeighbourList2 *nebList)
{
    int newsize;
    
    
    /* check if need to resize neighbour pointers */
    if (nebList[mainIndex].neighbourCount == 0)
    {
        nebList[mainIndex].neighbour = malloc(nebList[mainIndex].chunk * sizeof(struct Neighbour));
        if (nebList[mainIndex].neighbour == NULL)
        {
            char errstring[128];
            sprintf(errstring, "Could not allocate nebList[%d].neighbour\n", mainIndex);
            PyErr_SetString(PyExc_MemoryError, errstring);
            return 1;
        }
    }
    else if (nebList[mainIndex].neighbourCount % nebList[mainIndex].chunk == 0)
    {
        newsize = nebList[mainIndex].neighbourCount + nebList[mainIndex].chunk;
        nebList[mainIndex].neighbour = realloc(nebList[mainIndex].neighbour, newsize * sizeof(struct Neighbour));
        if (nebList[mainIndex].neighbour == NULL)
        {
            char errstring[128];
            sprintf(errstring, "Could not reallocate nebList[%d].neighbour\n", mainIndex);
            PyErr_SetString(PyExc_MemoryError, errstring);
            return 2;
        }
    }
    
    /* add neighbour */
    nebList[mainIndex].neighbour[nebList[mainIndex].neighbourCount].index = nebIndex;
    nebList[mainIndex].neighbour[nebList[mainIndex].neighbourCount].separation = sep;
    nebList[mainIndex].neighbourCount++;
    
    return 0;
}

struct NeighbourList2 * constructNeighbourList2(int NAtoms, double *pos, struct Boxes *boxes, double *cellDims, int *PBC, double maxSep2)
{
    int i, j, k, boxIndex, indexb;
    int boxNebList[27];
    double rxa, rya, rza, rxb, ryb, rzb, sep2;
    struct NeighbourList2 *nebList;
    
    
    /* allocate neb list */
    nebList = malloc(NAtoms * sizeof(struct NeighbourList2));
    if (nebList == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate nebList");
        return NULL;
    }
    
    /* initialise */
    for (i = 0; i < NAtoms; i++)
    {
        nebList[i].chunk = 16;
        nebList[i].neighbourCount = 0;
    }
    
    /* loop over atoms */
    for (i = 0; i < NAtoms; i++)
    {
        int boxNebListSize;
        
        /* atom position */
        rxa = pos[3*i];
        rya = pos[3*i+1];
        rza = pos[3*i+2];
        
        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(rxa, rya, rza, boxes);
        if (boxIndex < 0)
        {
            freeNeighbourList2(nebList, NAtoms);
            return NULL;
        }
        
        /* find neighbouring boxes */
        boxNebListSize = getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over box neighbourhood */
        for (j = 0; j < boxNebListSize; j++)
        {
            boxIndex = boxNebList[j];
            
            /* loop over atoms in box */
            for (k=0; k<boxes->boxNAtoms[boxIndex]; k++)
            {
                indexb = boxes->boxAtoms[boxIndex][k];
                
                if (indexb <= i) continue;
                
                /* atom position */
                rxb = pos[3*indexb];
                ryb = pos[3*indexb+1];
                rzb = pos[3*indexb+2];
                
                /* separation */
                sep2 = atomicSeparation2(rxa, rya, rza, rxb, ryb, rzb, cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
                
                /* check if neighbour */
                if (sep2 < maxSep2)
                {
                    int addstat;
                    double sep = sqrt(sep2);
                    
                    addstat = addAtomToNebList(i, indexb, sep, nebList);
                    if (addstat)
                    {
                        freeNeighbourList2(nebList, NAtoms);
                        return NULL;
                    }
                    addstat = addAtomToNebList(indexb, i, sep, nebList);
                    if (addstat)
                    {
                        freeNeighbourList2(nebList, NAtoms);
                        return NULL;
                    }
                }
            }
        }
    }
    
    return nebList;
}

void freeNeighbourList2(struct NeighbourList2 *nebList, int size)
{
    int i;
    
    for (i = 0; i < size; i++)
    {
        if (nebList[i].neighbourCount > 0)
        {
            free(nebList[i].neighbour);
        }
    }
    free(nebList);
}
