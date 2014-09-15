
#ifndef ARRAY_UTILS_SET
#define ARRAY_UTILS_SET

double *pyvector_to_Cptr_double(PyArrayObject*);
int *pyvector_to_Cptr_int(PyArrayObject*);
int not_doubleVector(PyArrayObject*);
int not_intVector(PyArrayObject*);

double **pymatrix_to_Cptrs_double(PyArrayObject*);
int **pymatrix_to_Cptrs_int(PyArrayObject*);
void free_Cptrs_double(double**);
int not_doubleMatrix(PyArrayObject*);

#endif
