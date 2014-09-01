
#ifndef OUTPUT_SET
#define OUTPUT_SET

int writeLattice(char*, int, int, int *, double *, char*, int*, double*, double*, int);

int writePOVRAYDefects(char *, int, int *, int, int *, int, int *, int, int *, int *, double *, int *, double *, double *, double *, double *, double *, int, int *);

typedef void*(*rgbcalc_t)(double);

int writePOVRAYAtoms(char*, int, int*, int*, double*, double*, double*, double*, double*, double*, int, int, rgbcalc_t);

#endif
