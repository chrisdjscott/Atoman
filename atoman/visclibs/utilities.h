
#include <time.h>

#ifndef UTILITIES_SET
#define UTILITIES_SET

double atomicSeparation2(double, double, double, double, double, double, double, double, double, int, int, int);
int getSpecieIndex(int, char*, char*);
void atomSeparationVector(double*, double, double, double, double, double, double, double, double, double, int, int, int);
double atomicSeparation2PBCCheck(double, double, double, double, double, double, double, double, double, int, int, int, int*);
char* specieListFromPyObject(PyObject*);

#define walltime() (double) clock() / (double) CLOCKS_PER_SEC

#endif
