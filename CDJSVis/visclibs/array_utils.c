
#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>


static double **ptrvector_double(long);


double *pyvector_to_Cptr_double(PyArrayObject *vectin)
{
    int n;
    
    n = vectin->dimensions[0];
    return (double *) vectin->data;
}


int *pyvector_to_Cptr_int(PyArrayObject *vectin)
{
    int n;
    
    n = vectin->dimensions[0];
    return (int *) vectin->data;
}


int not_doubleVector(PyArrayObject *vectin)
{
    if (vectin->descr->type_num != NPY_FLOAT64 || vectin->nd != 1)
    {
        PyErr_SetString(PyExc_ValueError, "In not_doubleVector: vector must be of type float and 1 dimensional");
        return 1;
    }
    
    return 0;
}


int not_intVector(PyArrayObject *vectin)
{
    if (vectin->descr->type_num != NPY_INT32 || vectin->nd != 1)
    {
        PyErr_SetString(PyExc_ValueError, "In not_intVector: vector must be of type int and 1 dimensional");
        return 1;
    }
    
    return 0;
}


static double **ptrvector_double(long n)
{
    double **v;
    
    v = (double **) malloc((size_t) (n * sizeof(double)));
    if (!v)
    {
        printf("In **ptrvector_double. Allocation of memory for double array failed.\n");
        exit(34);
    }
    
    return v;
}


double **pymatrix_to_Cptrs_double(PyArrayObject *arrayin)
{
    double **c, *a;
    int i, n, m;
    
    n = arrayin->dimensions[0];
    m = arrayin->dimensions[1];
    c = ptrvector(n);
    a = (double *) arrayin->data;
    for (i = 0; i < n; i++)
        c[i] = a + i * m;
    
    return c;
}


void free_Cptrs_double(double **v)
{
    free((char *) v);
}

