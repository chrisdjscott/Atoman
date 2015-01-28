
/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Functions associated with spatially decomposing a system of atoms into boxes
 *******************************************************************************/

#ifndef BOXESLIB_SET
#define BOXESLIB_SET

/* create structure for containing boxes data */
struct Boxes
{
    double minPos[3];
    double maxPos[3];
    int PBC[3];
    double cellDims[3];
    
    int *boxNAtoms;
    int **boxAtoms;
    
    int totNBoxes;
    int NBoxes[3];
    double boxWidth[3];
    
    int allocChunk;
};

/* available functions */
struct Boxes * setupBoxes(double, double *, double *, int *, double *);

int boxIndexOfAtom( double, double, double, struct Boxes *);

int putAtomsInBoxes(int, double *, struct Boxes *);

void freeBoxes(struct Boxes *);

void boxIJKIndices(int, int *, int, struct Boxes *);

int boxIndexFromIJK(int, int, int, struct Boxes *);

int getBoxNeighbourhood(int, int *, struct Boxes *);

#endif
