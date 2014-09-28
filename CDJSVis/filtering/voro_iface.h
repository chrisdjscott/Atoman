
typedef struct {
	double volume;
	
	int numFaces;
	int numNeighbours; // can be different (if add threshold)
	int *numFaceVertices;
	int **faceVertices;
	
	int numVertices;
	double *vertices;
} vorores_t;

#ifdef __cplusplus
extern "C"
#endif
int computeVoronoiVoroPlusPlusWrapper(int, double*, int*, double*, double*, int, double*, vorores_t*);
