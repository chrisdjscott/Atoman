
#ifndef ARRAY_UTILS_SET
#define ARRAY_UTILS_SET

double *pyvector_to_Cptr_double(PyArrayObject*);
int *pyvector_to_Cptr_int(PyArrayObject*);
char *pyvector_to_Cptr_char(PyArrayObject*);
int not_doubleVector(PyArrayObject*);
int not_intVector(PyArrayObject*);

double **pymatrix_to_Cptrs_double(PyArrayObject*);
int **pymatrix_to_Cptrs_int(PyArrayObject*);
void free_Cptrs_double(double**);
int not_doubleMatrix(PyArrayObject*);

#define DIND1(a, i) *((double *) PyArray_GETPTR1(a, i))
#define DIND2(a, i, j) *((double *) PyArray_GETPTR2(a, i, j))
#define DIND3(a, i, j, k) *((double *) Py_Array_GETPTR3(a, i, j, k))

#define IIND1(a, i) *((int *) PyArray_GETPTR1(a, i))
#define IIND2(a, i, j) *((int *) PyArray_GETPTR2(a, i, j))
#define IIND3(a, i, j, k) *((int *) Py_Array_GETPTR3(a, i, j, k))

#endif
