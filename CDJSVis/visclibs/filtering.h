
#ifndef FILTERING_SET
#define FILTERING_SET

int specieFilter(int, int *, int, int *, int, int *, int, double *);

int sliceFilter(int, int *, int, double *, double, double, double, double, double, double, int, int, double *);

int cropSphereFilter(int, int *, int, double *, double, double, double, double, double *, int *, int, int, double *);

int cropFilter(int, int *, int, double*, double, double, double, double, double, double, int, int, int, int, int, double *);

int displacementFilter(int, int *, int, double *, int, double *, int, double *, double *, int *, double, double, int, double*, int, int, double*);

int KEFilter(int, int *, int, double *, double, double, int, double *);

int PEFilter(int, int *, int, double *, double, double, int, double *);

int chargeFilter(int, int *, int, double *, double, double, int, double *);

int coordNumFilter(int, int *, double *, int *, int, double *, double *, double, double *, int *, double *, double *, double *, int, int, int, double*, int);

int voronoiVolumeFilter(int, int*, int, double*, double, double, int, double*, int, double *, int);

int voronoiNeighboursFilter(int, int*, int, int*, int, int, int, double *, int, double *, int);

int Q4Filter(int, int*, int, double *, double, double, double, int, double *, double *, double *, double *, int *, int, double *, int);

int calculate_drift_vector(int, double*, double*, double*, int*, double*);

int atomIndexFilter(int, int*, int*, int, int, int, int, double*);

#endif
