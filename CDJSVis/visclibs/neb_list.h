
/* create structure for containing boxes data */
struct NeighbourList
{
    int neighbourCount;
    int chunk;
    int *neighbour;
    double *neighbourSep;
};

struct NeighbourList * constructNeighbourList(int, double *, struct Boxes *, double *, int *, double);

void freeNeighbourList(struct NeighbourList *, int);
