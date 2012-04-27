
/* create structure for containing boxes data */
struct Boxes
{
    double minPos[3];
    double maxPos[3];
    int PBC[3]; // not used at the moment
    
    int *boxNAtoms;
    int **boxAtoms;
    
    int totNBoxes;
    int NBoxes[3];
    double boxWidth[3];
    
    int maxAtomsPerBox;
};

/* available functions */
struct Boxes * setupBoxes(double, double *, double *, int *, int);
int boxIndexOfAtom( double, double, double, struct Boxes *);
void putAtomsInBoxes(int, double *, struct Boxes *);
void freeBoxes(struct Boxes *);
void boxIJKIndices(int, int *, int, struct Boxes *);
int boxIndexFromIJK(int, int, int, struct Boxes *);
void getBoxNeighbourhood(int, int *, struct Boxes *);
