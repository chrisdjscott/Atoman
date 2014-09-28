
typedef struct {
    double volume;
    
    int numFaces;
    int *numFaceVertices;
    int **faceVertices;
    int numNeighbours;
    int *neighbours;
    
    int numVertices;
    double *vertices;
    
    double originalPos[3];
} vorores_t;

#ifdef __cplusplus
extern "C"
#endif
int computeVoronoiVoroPlusPlusWrapper(int, double*, int*, double*, double*, int, double*, vorores_t*);
