
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


struct Neighbour
{
    int index;
    double sep2;
};

struct NeighbourList2
{
    int neighbourCount;
    int chunk;
    struct Neighbour *neighbour;
};

struct NeighbourList2 * constructNeighbourList2(int, double *, struct Boxes *, double *, int *, double);

void freeNeighbourList2(struct NeighbourList2 *, int);
