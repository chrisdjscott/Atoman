
/*******************************************************************************
 ** IO routines written in C to improve performance
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "visclibs/array_utils.h"

#if PY_MAJOR_VERSION >= 3
    #define PyString_FromFormat PyUnicode_FromFormat
    #define MOD_ERROR_VAL NULL
    #define MOD_SUCCESS_VAL(val) val
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
        static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);
#else
    #define MOD_ERROR_VAL
    #define MOD_SUCCESS_VAL(val)
    #define MOD_INIT(name) void init##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
        ob = Py_InitModule3(name, methods, doc);
#endif

static PyObject* readLatticeLBOMD(PyObject*, PyObject*);
static PyObject* readRef(PyObject*, PyObject*);
static PyObject* readLBOMDXYZ(PyObject*, PyObject*);

static int specieIndex(char*, int, char*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef module_methods[] = {
    {"readLatticeLBOMD", readLatticeLBOMD, METH_VARARGS, "Read LBOMD lattice format file"},
    {"readRef", readRef, METH_VARARGS, "Read LBOMD animation reference format file"},
    {"readLBOMDXYZ", readLBOMDXYZ, METH_VARARGS, "Read LBOMD XYZ format file"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
MOD_INIT(_input)
{
    PyObject *mod;
    
    MOD_DEF(mod, "_input", "Input C extension", module_methods)
    if (mod == NULL)
        return MOD_ERROR_VAL;
    
    import_array();
    
    return MOD_SUCCESS_VAL(mod);
}

/*******************************************************************************
 * Update specie list and counter
 *******************************************************************************/
static int specieIndex(char* sym, int NSpecies, char* specieList)
{
    int index, j, comp;
    
    
    index = NSpecies;
    for (j=0; j<NSpecies; j++)
    {
        comp = strcmp( &specieList[3*j], &sym[0] );
        if (comp == 0)
        {
            index = j;
            
            break;
        }
    }
    
    return index;
}


/*******************************************************************************
** read animation-reference file
*******************************************************************************/
static PyObject*
readRef(PyObject *self, PyObject *args)
{
    char *file;
    int *atomID, *specie, *specieCount_c;
    double *pos, *charge, *maxPos, *minPos, *KE, *PE;
    PyArrayObject *atomIDIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *chargeIn=NULL;
    PyObject *specieListPy=NULL;
    PyArrayObject *specieCount_cIn=NULL;
    PyArrayObject *maxPosIn=NULL;
    PyArrayObject *minPosIn=NULL;
    PyArrayObject *KEIn=NULL;
    PyArrayObject *PEIn=NULL;
    PyArrayObject *force=NULL;
    
    int i, NAtoms, specInd, stat;
    FILE *INFILE;
    double xdim, ydim, zdim;
    char symtemp[3];
    char* specieList;
    double xpos, ypos, zpos;
    double xforce, yforce, zforce;
    int id, index, NSpecies;
    double ketemp, petemp, chargetemp;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "sO!O!O!O!O!O!O!O!O!O!O!", &file, &PyArray_Type, &atomIDIn, &PyArray_Type, &specieIn, 
            &PyArray_Type, &posIn, &PyArray_Type, &chargeIn, &PyArray_Type, &KEIn, &PyArray_Type, &PEIn, 
            &PyArray_Type, &force, &PyList_Type, &specieListPy, &PyArray_Type, 
            &specieCount_cIn, &PyArray_Type, &maxPosIn, &PyArray_Type, &minPosIn))
        return NULL;
    
    if (not_intVector(atomIDIn)) return NULL;
    atomID = pyvector_to_Cptr_int(atomIDIn);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(chargeIn)) return NULL;
    charge = pyvector_to_Cptr_double(chargeIn);
    
    if (not_doubleVector(KEIn)) return NULL;
    KE = pyvector_to_Cptr_double(KEIn);
    
    if (not_doubleVector(PEIn)) return NULL;
    PE = pyvector_to_Cptr_double(PEIn);
    
    if (not_doubleVector(force)) return NULL;
    
    if (not_doubleVector(minPosIn)) return NULL;
    minPos = pyvector_to_Cptr_double(minPosIn);
    
    if (not_doubleVector(maxPosIn)) return NULL;
    maxPos = pyvector_to_Cptr_double(maxPosIn);
    
    if (not_intVector(specieCount_cIn)) return NULL;
    specieCount_c = pyvector_to_Cptr_int(specieCount_cIn);
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    
    /* open file */
    INFILE = fopen( file, "r" );
    if (INFILE == NULL)
    {
        printf("ERROR: could not open file: %s\n", file);
        printf("       reason: %s\n", strerror(errno));
        exit(35);
    }
    
    stat = fscanf(INFILE, "%d", &NAtoms);
    if (stat != 1)
    {
        fclose(INFILE);
        return Py_BuildValue("i", -3);
    }
    
    stat = fscanf(INFILE, "%lf%lf%lf", &xdim, &ydim, &zdim);
    if (stat != 3)
    {
        fclose(INFILE);
        return Py_BuildValue("i", -3);
    }
    
    specieList = malloc(3 * sizeof(char));
    
    minPos[0] = 1000000;
    minPos[1] = 1000000;
    minPos[2] = 1000000;
    maxPos[0] = -1000000;
    maxPos[1] = -1000000;
    maxPos[2] = -1000000;
    NSpecies = 0;
    for (i = 0; i < NAtoms; i++)
    {
        int ind3;
        
        stat = fscanf(INFILE, "%d%s%lf%lf%lf%lf%lf%lf%lf%lf%lf", &id, symtemp, &xpos, &ypos, &zpos, &ketemp, &petemp, &xforce, &yforce, &zforce, &chargetemp);
        if (stat != 11)
        {
            fclose(INFILE);
            free(specieList);
            return Py_BuildValue("i", -3);
        }
        
        /* index for storage is (id-1) */
        index = id - 1;
        ind3 = 3 * index;
        
        atomID[index] = id;
        
        pos[ind3    ] = xpos;
        pos[ind3 + 1] = ypos;
        pos[ind3 + 2] = zpos;
        
        KE[index] = ketemp;
        PE[index] = petemp;
        
        DIND2(force, index, 0) = xforce;
        DIND2(force, index, 1) = yforce;
        DIND2(force, index, 2) = zforce;
        
        charge[index] = chargetemp;
        
        /* find specie index */
        specInd = specieIndex(symtemp, NSpecies, specieList);
        specie[index] = specInd;
        
        if (specInd == NSpecies)
        {
            PyObject *sympy;
            
            /* new specie */
            specieList = realloc(specieList, 3 * (NSpecies+1) * sizeof(char));
            
            specieList[3 * specInd] = symtemp[0];
            specieList[3 * specInd + 1] = symtemp[1];
            specieList[3 * specInd + 2] = symtemp[2];
            
            sympy = PyString_FromFormat("%s", symtemp);
            PyList_Append(specieListPy, sympy);
            Py_DECREF(sympy);
            
            NSpecies++;
        }
        
        /* update specie counter */
        specieCount_c[specInd]++;
                
        /* max and min positions */
        if (xpos > maxPos[0]) maxPos[0] = xpos;
        if (ypos > maxPos[1]) maxPos[1] = ypos;
        if (zpos > maxPos[2]) maxPos[2] = zpos;
        if (xpos < minPos[0]) minPos[0] = xpos;
        if (ypos < minPos[1]) minPos[1] = ypos;
        if (zpos < minPos[2]) minPos[2] = zpos;
    }
    
    fclose(INFILE);
    free(specieList);
    
    return Py_BuildValue("i", 0);
}


/*******************************************************************************
** read xyz input file
*******************************************************************************/
static PyObject*
readLBOMDXYZ(PyObject *self, PyObject *args)
{
    char *file;
    int *atomID, xyzformat;
    double *pos, *charge, *maxPos, *minPos, *KE, *PE;
    PyArrayObject *atomIDIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *chargeIn=NULL;
    PyArrayObject *maxPosIn=NULL;
    PyArrayObject *minPosIn=NULL;
    PyArrayObject *KEIn=NULL;
    PyArrayObject *PEIn=NULL;
    PyArrayObject *velocity=NULL;
    PyArrayObject *specie=NULL;
    PyArrayObject *refSpecie=NULL;
    PyArrayObject *refCharge=NULL;
    
    FILE *INFILE;
    int i, NAtoms, stat;
    double simTime;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "sO!O!O!O!O!O!O!O!iO!O!O!", &file, &PyArray_Type, &atomIDIn, &PyArray_Type, &posIn,
            &PyArray_Type, &chargeIn, &PyArray_Type, &KEIn, &PyArray_Type, &PEIn, &PyArray_Type, &velocity,
            &PyArray_Type, &maxPosIn, &PyArray_Type, &minPosIn, &xyzformat, &PyArray_Type, &specie, &PyArray_Type,
            &refSpecie, &PyArray_Type, &refCharge))
        return NULL;
    
    if (not_intVector(atomIDIn)) return NULL;
    atomID = pyvector_to_Cptr_int(atomIDIn);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(chargeIn)) return NULL;
    charge = pyvector_to_Cptr_double(chargeIn);
    
    if (not_doubleVector(KEIn)) return NULL;
    KE = pyvector_to_Cptr_double(KEIn);
    
    if (not_doubleVector(PEIn)) return NULL;
    PE = pyvector_to_Cptr_double(PEIn);
    
    if (not_doubleVector(minPosIn)) return NULL;
    minPos = pyvector_to_Cptr_double(minPosIn);
    
    if (not_doubleVector(maxPosIn)) return NULL;
    maxPos = pyvector_to_Cptr_double(maxPosIn);
    
    if (not_intVector(specie)) return NULL;
    if (not_intVector(refSpecie)) return NULL;
    if (not_doubleVector(refCharge)) return NULL;
    if (not_doubleVector(velocity)) return NULL;
    
    /* open file */
    INFILE = fopen(file, "r");
    if (INFILE == NULL)
    {
        printf("ERROR: could not open file: %s\n", file);
        printf("       reason: %s\n", strerror(errno));
        exit(35);
    }
    
    /* read header */
    stat = fscanf(INFILE, "%d", &NAtoms);
    if (stat != 1) return Py_BuildValue("i", -3);
    
    stat = fscanf(INFILE, "%lf", &simTime);
    if (stat != 1) return Py_BuildValue("i", -3);
        
    /* read atoms */
    minPos[0] = 1000000;
    minPos[1] = 1000000;
    minPos[2] = 1000000;
    maxPos[0] = -1000000;
    maxPos[1] = -1000000;
    maxPos[2] = -1000000;
    
    if (xyzformat == 0)
    {
        for (i=0; i<NAtoms; i++)
        {
            int id, index, ind3;
            double xpos, ypos, zpos, KEtmp, PEtmp;
            
            stat = fscanf(INFILE, "%d %lf %lf %lf %lf %lf", &id, &xpos, &ypos, &zpos, &KEtmp, &PEtmp);
            if (stat != 6)
            {
                fclose(INFILE);
                return Py_BuildValue("i", -3);
            }
            
            index = id - 1;
            ind3 = index * 3;
            
            /* store data */
            atomID[index] = id;
            
            pos[ind3    ] = xpos;
            pos[ind3 + 1] = ypos;
            pos[ind3 + 2] = zpos;
            
            KE[index] = KEtmp;
            PE[index] = PEtmp;
            
            IIND1(specie, index) = IIND1(refSpecie, index);
            
            charge[index] = DIND1(refCharge, index);
            
            /* max and min positions */
            if (xpos > maxPos[0]) maxPos[0] = xpos;
            if (ypos > maxPos[1]) maxPos[1] = ypos;
            if (zpos > maxPos[2]) maxPos[2] = zpos;
            if (xpos < minPos[0]) minPos[0] = xpos;
            if (ypos < minPos[1]) minPos[1] = ypos;
            if (zpos < minPos[2]) minPos[2] = zpos;
        }
    }
    else if (xyzformat == 1)
    {
        for (i=0; i<NAtoms; i++)
        {
            int id, index, ind3;
            double xpos, ypos, zpos, KEtmp, PEtmp, chargetmp;
            
            stat = fscanf(INFILE, "%d%lf%lf%lf%lf%lf%lf", &id, &xpos, &ypos, &zpos, &KEtmp, &PEtmp, &chargetmp);
            if (stat != 7) 
            {
                fclose(INFILE);
                return Py_BuildValue("i", -3);
            }
            
            index = id - 1;
            ind3 = index * 3;
            
            /* store data */
            atomID[index] = id;
            
            pos[ind3    ] = xpos;
            pos[ind3 + 1] = ypos;
            pos[ind3 + 2] = zpos;
            
            KE[index] = KEtmp;
            PE[index] = PEtmp;
            
            IIND1(specie, index) = IIND1(refSpecie, index);
            
            charge[index] = chargetmp;
            
            /* max and min positions */
            if (xpos > maxPos[0]) maxPos[0] = xpos;
            if (ypos > maxPos[1]) maxPos[1] = ypos;
            if (zpos > maxPos[2]) maxPos[2] = zpos;
            if (xpos < minPos[0]) minPos[0] = xpos;
            if (ypos < minPos[1]) minPos[1] = ypos;
            if (zpos < minPos[2]) minPos[2] = zpos;
        }
    }
    else if (xyzformat == 2)
    {
        for (i=0; i<NAtoms; i++)
        {
            int id, index, ind3;
            double xpos, ypos, zpos, KEtmp, PEtmp;
            double xvel, yvel, zvel;
            
            stat = fscanf(INFILE, "%d %lf %lf %lf %lf %lf %lf %lf %lf", &id, &xpos, &ypos, &zpos, &KEtmp, &PEtmp, &xvel, &yvel, &zvel);
            if (stat != 9)
            {
                fclose(INFILE);
                return Py_BuildValue("i", -3);
            }
            
            index = id - 1;
            ind3 = index * 3;
            
            /* store data */
            atomID[index] = id;
            
            pos[ind3    ] = xpos;
            pos[ind3 + 1] = ypos;
            pos[ind3 + 2] = zpos;
            
            KE[index] = KEtmp;
            PE[index] = PEtmp;
            
            IIND1(specie, index) = IIND1(refSpecie, index);
            
            charge[index] = DIND1(refCharge, index);
            
            DIND2(velocity, index, 0) = xvel;
            DIND2(velocity, index, 1) = yvel;
            DIND2(velocity, index, 2) = zvel;
            
            /* max and min positions */
            if (xpos > maxPos[0]) maxPos[0] = xpos;
            if (ypos > maxPos[1]) maxPos[1] = ypos;
            if (zpos > maxPos[2]) maxPos[2] = zpos;
            if (xpos < minPos[0]) minPos[0] = xpos;
            if (ypos < minPos[1]) minPos[1] = ypos;
            if (zpos < minPos[2]) minPos[2] = zpos;
        }
    }
    else
    {
        char errstring[64];
        
        sprintf(errstring, "Unrecognised xyzformat for LBOMD XYZ file (%d)", xyzformat);
        PyErr_SetString(PyExc_RuntimeError, errstring);
        fclose(INFILE);
        return NULL;
    }
    
    fclose(INFILE);
    
    return Py_BuildValue("i", 0);
}


/*******************************************************************************
 * Read LBOMD lattice file
 *******************************************************************************/
static PyObject*
readLatticeLBOMD(PyObject *self, PyObject *args)
{
    char *file;
    int *atomID, *specie, *specieCount_c;
    double *pos, *charge, *maxPos, *minPos;
    PyArrayObject *atomIDIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *chargeIn=NULL;
    PyObject *specieListPy=NULL;
    PyArrayObject *specieCount_cIn=NULL;
    PyArrayObject *maxPosIn=NULL;
    PyArrayObject *minPosIn=NULL;
    
    FILE *INFILE;
    int i, NAtoms, specInd;
    double xdim, ydim, zdim;
    char symtemp[3];
    char* specieList;
    double xpos, ypos, zpos, chargetemp;
    int NSpecies, stat;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "sO!O!O!O!O!O!O!O!", &file, &PyArray_Type, &atomIDIn, &PyArray_Type, &specieIn, 
            &PyArray_Type, &posIn, &PyArray_Type, &chargeIn, &PyList_Type, &specieListPy, &PyArray_Type, 
            &specieCount_cIn, &PyArray_Type, &maxPosIn, &PyArray_Type, &minPosIn))
        return NULL;
    
    if (not_intVector(atomIDIn)) return NULL;
    atomID = pyvector_to_Cptr_int(atomIDIn);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(chargeIn)) return NULL;
    charge = pyvector_to_Cptr_double(chargeIn);
    
    if (not_doubleVector(minPosIn)) return NULL;
    minPos = pyvector_to_Cptr_double(minPosIn);
    
    if (not_doubleVector(maxPosIn)) return NULL;
    maxPos = pyvector_to_Cptr_double(maxPosIn);
    
    if (not_intVector(specieCount_cIn)) return NULL;
    specieCount_c = pyvector_to_Cptr_int(specieCount_cIn);
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    
    /* open file */
    INFILE = fopen(file, "r");
    if (INFILE == NULL)
    {
        printf("ERROR: could not open file: %s\n", file);
        printf("       reason: %s\n", strerror(errno));
        exit(35);
    }
    
    /* read header */
    stat = fscanf( INFILE, "%d", &NAtoms );
    if (stat != 1)
    {
        fclose(INFILE);
        return Py_BuildValue("i", -3);
    }
    
    stat = fscanf(INFILE, "%lf %lf %lf", &xdim, &ydim, &zdim);
    if (stat != 3)
    {
        fclose(INFILE);
        return Py_BuildValue("i", -3);
    }
    
    /* allocate specieList */
    specieList = malloc(3 * sizeof(char));
    
    /* read in atoms */
    minPos[0] = 1000000;
    minPos[1] = 1000000;
    minPos[2] = 1000000;
    maxPos[0] = -1000000;
    maxPos[1] = -1000000;
    maxPos[2] = -1000000;
    NSpecies = 0;
    for (i = 0; i < NAtoms; i++)
    {
        stat = fscanf(INFILE, "%s %lf %lf %lf %lf", symtemp, &xpos, &ypos, &zpos, &chargetemp);
        if (stat != 5)
        {
            fclose(INFILE);
            free(specieList);
            return Py_BuildValue("i", -3);
        }
        
        /* atom ID */
        atomID[i] = i + 1;
        
        /* store position and charge */
        pos[3 * i] = xpos;
        pos[3 * i + 1] = ypos;
        pos[3 * i + 2] = zpos;
        
        charge[i] = chargetemp;
        
        /* find specie index */
        specInd = specieIndex(symtemp, NSpecies, specieList);
        specie[i] = specInd;
        if (specInd == NSpecies)
        {
            PyObject *sympy;
            
            /* new specie */
            specieList = realloc(specieList, 3 * (NSpecies+1) * sizeof(char));
            
            specieList[3*specInd] = symtemp[0];
            specieList[3*specInd+1] = symtemp[1];
            specieList[3*specInd+2] = symtemp[2];
            
            sympy = PyString_FromFormat("%s", symtemp);
            PyList_Append(specieListPy, sympy);
            Py_DECREF(sympy);
            
            NSpecies++;
        }
        
        /* update specie counter */
        specieCount_c[specInd]++;
                
        /* max and min positions */
        if (xpos > maxPos[0]) maxPos[0] = xpos;
        if (ypos > maxPos[1]) maxPos[1] = ypos;
        if (zpos > maxPos[2]) maxPos[2] = zpos;
        if (xpos < minPos[0]) minPos[0] = xpos;
        if (ypos < minPos[1]) minPos[1] = ypos;
        if (zpos < minPos[2]) minPos[2] = zpos;
    }
    
    fclose(INFILE);
    free(specieList);
    
    return Py_BuildValue("i", 0);
}
