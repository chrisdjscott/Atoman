

int *boxNAtoms;
int *boxAtoms;
int totNBoxes;
int NBoxes[3];
int maxAtomsPerBox;
double boxWidth[3];
double boxMinPos[3];
double boxMaxPos[3];

void setupBoxes( double*, double*, double );
int boxIndexOfAtom( double,  double, double );
void putAtomsInBoxes( int, double* );
void getBoxNeighbourhood( int, int* );
void cleanupBoxes( void );
