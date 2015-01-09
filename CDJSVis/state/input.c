
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** IO routines written in C to improve performance
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "array_utils.h"

static PyObject* readLatticeLBOMD(PyObject*, PyObject*);
static PyObject* readRef(PyObject*, PyObject*);
static PyObject* readLBOMDXYZ(PyObject*, PyObject*);
static PyObject* readGenericLatticeFile(PyObject*, PyObject*);

static int specieIndex(char*, int, char*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"readLatticeLBOMD", readLatticeLBOMD, METH_VARARGS, "Read LBOMD lattice format file"},
    {"readRef", readRef, METH_VARARGS, "Read LBOMD animation reference format file"},
    {"readLBOMDXYZ", readLBOMDXYZ, METH_VARARGS, "Read LBOMD XYZ format file"},
    {"readGenericLatticeFile", readGenericLatticeFile, METH_VARARGS, "Read generic Lattice file"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_input(void)
{
    (void)Py_InitModule("_input", methods);
    import_array();
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
	char *file, *specieList_c;
	int *atomID, *specie, *specieCount_c;
	double *pos, *charge, *maxPos, *minPos, *KE, *PE;
	PyArrayObject *atomIDIn=NULL;
	PyArrayObject *specieIn=NULL;
	PyArrayObject *posIn=NULL;
	PyArrayObject *chargeIn=NULL;
	PyArrayObject *specieList_cIn=NULL;
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
			&PyArray_Type, &force, &PyArray_Type, &specieList_cIn, &PyArray_Type, 
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
	
	// no test yet for nx2 char*
	specieList_c = pyvector_to_Cptr_char(specieList_cIn);
    
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
    for (i=0; i<NAtoms; i++)
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
            /* new specie */
            specieList = realloc( specieList, 3 * (NSpecies+1) * sizeof(char) );
            
            specieList[3*specInd] = symtemp[0];
            specieList[3*specInd+1] = symtemp[1];
            specieList[3*specInd+2] = symtemp[2];
            
            specieList_c[2*specInd] = symtemp[0];
            specieList_c[2*specInd+1] = symtemp[1];
            
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
    
    /* terminate specie list */
    specieList_c[2*NSpecies] = 'X';
    specieList_c[2*NSpecies+1] = 'X';
    
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
    char *file, *specieList_c;
    int *atomID, *specie, *specieCount_c;
    double *pos, *charge, *maxPos, *minPos;
    PyArrayObject *atomIDIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *chargeIn=NULL;
    PyArrayObject *specieList_cIn=NULL;
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
            &PyArray_Type, &posIn, &PyArray_Type, &chargeIn, &PyArray_Type, &specieList_cIn, &PyArray_Type, 
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
    
    // no test yet for nx2 char*
    specieList_c = pyvector_to_Cptr_char(specieList_cIn);
    
    /* open file */
    INFILE = fopen( file, "r" );
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
    for (i=0; i<NAtoms; i++)
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
        pos[3*i] = xpos;
        pos[3*i+1] = ypos;
        pos[3*i+2] = zpos;
        
        charge[i] = chargetemp;
        
        /* find specie index */
        specInd = specieIndex(symtemp, NSpecies, specieList);
        
        specie[i] = specInd;
        
        if (specInd == NSpecies)
        {
            /* new specie */
            specieList = realloc( specieList, 3 * (NSpecies+1) * sizeof(char) );
            
            specieList[3*specInd] = symtemp[0];
            specieList[3*specInd+1] = symtemp[1];
            specieList[3*specInd+2] = symtemp[2];
            
            specieList_c[2*specInd] = symtemp[0];
            specieList_c[2*specInd+1] = symtemp[1];
            
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
    
    /* terminate specie list */
    specieList_c[2*NSpecies] = 'X';
    specieList_c[2*NSpecies+1] = 'X';
        
    free(specieList);
    
    return Py_BuildValue("i", 0);
}

/*******************************************************************************
 * Read generic lattice file
 *******************************************************************************/
static PyObject*
readGenericLatticeFile(PyObject *self, PyObject *args)
{
    int atomIndexOffset;
    char *filename, *delimiter;
    FILE *INFILE=NULL;
    PyObject *headerList=NULL;
    PyObject *bodyList=NULL;
    PyObject *resultDict=NULL;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "sO!O!si", &filename, &PyList_Type, &headerList, &PyList_Type, &bodyList, &delimiter, &atomIndexOffset))
        return NULL;
    
    printf("GENREADER: reading file: '%s'\n", filename);
    printf("Delimiter is '%s'\n", delimiter);
    
    INFILE = fopen(filename, "r");
    if (INFILE == NULL)
    {
        char errstring[128];
        
        sprintf(errstring, "%s: '%s'", strerror(errno), filename);
        PyErr_SetString(PyExc_IOError, errstring);
        return NULL;
    }
    else
    {
        int atomIDFlag = 0;
        int haveSpecieOrSymbol = 0;
        int cellDimsFlag = 0;
        long i, numLines;
        long NAtoms = -1;
        PyObject *specieList=NULL;
        PyObject *specieCount=NULL;
        
        /* allocate result dict */
        resultDict = PyDict_New();
        if (resultDict == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not allocate resultDict");
            fclose(INFILE);
            return NULL;
        }
        
        /* allocate atom scalar/vector arrays and add them to the dict */
        
        
        /* number of header lines */
        numLines = PyList_Size(headerList);
        printf("HEADER NUM LINES: %ld\n", numLines);
        
        /* read header */
        for (i = 0; i < numLines; i++)
        {
            char line[512], *pch;
            long lineLength, count;
            PyObject *headerLine=NULL;
            
            /* this is a borrowed ref so no need to DECREF */
            headerLine = PyList_GetItem(headerList, i);
//             printf("Parsing header line %ld\n", i);
            lineLength = PyList_Size(headerLine);
//             printf("Header line %ld; length = %ld\n", i, lineLength);
            
            /* read line */
            if (fgets(line, sizeof(line), INFILE) == NULL)
            {
                PyErr_SetString(PyExc_IOError, "End of file reached while reading header");
                Py_DECREF(resultDict);
                fclose(INFILE);
                return NULL;
            }
            
            /* parse the line */
            pch = strtok(line, delimiter);
            count = 0;
            while (pch != NULL && count < lineLength)
            {
                char *key, *type;
                int stat, dim;
                PyObject *itemTuple=NULL;
                PyObject *value=NULL;
                
//                 printf("  Line value %ld: %s\n", count, pch);
                
                /* each item is a tuple: (key, type, dim) */
                itemTuple = PyList_GetItem(headerLine, count);
                if (!PyArg_ParseTuple(itemTuple, "ssi", &key, &type, &dim)) 
                {
                    // need to free arrays too...
                    fclose(INFILE);
                    Py_DECREF(resultDict);
                    return NULL;
                }
                
                /* parse */
//                 printf("    Key: '%s'; Type: '%s'; Dim: %d\n", keytmp, typetmp, dimtmp);
                
                if (strcmp("SKIP", key))
                {
                    if (!strcmp("xdim", key) || !strcmp("ydim", key) || !strcmp("zdim", key))
                        cellDimsFlag++;
                
                    if (!strcmp("i", type))
                    {
                        long tmpval;
                    
                        tmpval = atol(pch);
                        if (!strcmp("NAtoms", key)) NAtoms = tmpval;
                        value = Py_BuildValue(type, tmpval);
                    }
                    else if (!strcmp("d", type))
                        value = Py_BuildValue(type, atof(pch));
                    else if (!strcmp("s", type))
                        value = Py_BuildValue(type, pch);
                    else
                    {
                        char errstring[128];
                    
                        sprintf("Unrecognised type string: '%s'", type);
                        PyErr_SetString(PyExc_RuntimeError, errstring);
                        // need to free arrays too...
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        return NULL;
                    }
                
                    stat = PyDict_SetItemString(resultDict, key, value);
                
                    /* DO I NEED TO DECREF VALUE TOO? */
                    Py_DECREF(value);
                
                    if (stat)
                    {
                        char errstring[128];
                    
                        sprintf("Could not set item in dictionary: '%s'", key);
                        PyErr_SetString(PyExc_RuntimeError, errstring);
                        // need to free arrays too...
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        return NULL;
                    }
                }
                
                /* read next token */
                pch = strtok(NULL, delimiter);
                count++;
            }
            
            if (count != lineLength)
            {
                char errstring[128];
                
                sprintf(errstring, "Wrong length for header line %ld: %ld != %ld", i, count, lineLength);
                PyErr_SetString(PyExc_IOError, errstring);
                Py_DECREF(resultDict);
                // may have to free stuff in resultDict!
                fclose(INFILE);
                return NULL;
            }
        }
        
        /* we check that NAtoms was read during the header... */
        if (NAtoms == -1)
        {
            PyErr_SetString(PyExc_RuntimeError, "Cannot autodetect NAtoms at the moment...");
            Py_DECREF(resultDict);
            // may have to free stuff in resultDict!
            fclose(INFILE);
            return NULL;
        }
        
        /* cell dims */
        printf("Cell dims flag: %d\n", cellDimsFlag);
        
        
        
        
        
        printf("Preparing to read body; NAtoms = %ld\n", NAtoms);
        
        /* number of body lines per atom */
        numLines = PyList_Size(bodyList);
        printf("BODY NUM LINES PER ATOM: %ld\n", numLines);
        
        /* allocate the arrays for scalar/vector data */
        for (i = 0; i < numLines; i++)
        {
            long lineLength, j;
            PyObject *lineList=NULL;
            
            lineList = PyList_GetItem(bodyList, i);
            lineLength = PyList_Size(lineList);
            
            printf("Body line %ld; num items = %ld\n", i, lineLength);
            
            for (j = 0; j < lineLength; j++)
            {
                char *key, *type;
                int dim, stat, typenum;
                long shape_dim;
                npy_intp np_dims[2] = {NAtoms, 3};
                PyObject *itemTuple=NULL;
                PyArrayObject *data=NULL;
                
                itemTuple = PyList_GetItem(lineList, j);
                if (!PyArg_ParseTuple(itemTuple, "ssi", &key, &type, &dim)) 
                {
                    // need to free arrays too... PyDict_Clear???? will it free all items in the dict
                    fclose(INFILE);
                    Py_DECREF(resultDict);
                    return NULL;
                }
                
                if (strcmp("SKIP", key))
                {
                    if (!atomIDFlag && !strcmp("atomID", key)) atomIDFlag = 1;
    //                 if (!haveSpecieOrSymbol && (!strcmp("Symbol", key) || !strcmp("Specie", key)))
                    if (!haveSpecieOrSymbol && !strcmp("Symbol", key))
                        haveSpecieOrSymbol = 1;
                
                    if (!strcmp("i", type))
                        typenum = NPY_INT32;
                    else if (!strcmp("d", type))
                        typenum = NPY_FLOAT64;
                    else
                    {
                        char errstring[128];
                    
                        sprintf("Unrecognised type string (body prep): '%s'", type);
                        PyErr_SetString(PyExc_RuntimeError, errstring);
                        // need to free arrays too...
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        return NULL;
                    }
                
                    shape_dim = dim == 1 ? 1 : 2;
                    data = (PyArrayObject *) PyArray_SimpleNew(shape_dim, np_dims, typenum);
                    if (data == NULL)
                    {
                        char errstring[128];
                    
                        sprintf(errstring, "Could not allocate ndarray: '%s'", key);
                        PyErr_SetString(PyExc_MemoryError, errstring);
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        return NULL;
                    }
                
                    /* store in dict */
                    stat = PyDict_SetItemString(resultDict, key, PyArray_Return(data));
                
                    /* decrease ref count on data */
                    /* the documentation does not say PyDict_SetItem steals the reference
                    ** (like PyList_SetItem) so I assume it increases the ref count and that
                    ** I need to decrease it
                    */
                    Py_DECREF(data);
                
                    if (stat)
                    {
                        char errstring[128];
                    
                        sprintf("Could not set item in dictionary (body prep): '%s'", key);
                        PyErr_SetString(PyExc_RuntimeError, errstring);
                        // need to free arrays too...
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        return NULL;
                    }
                }
            }
        }
        
        printf("Atom ID flag: %d\n", atomIDFlag);
        printf("Have specie or symbol: %d\n", haveSpecieOrSymbol);
        
        /* if no atomID array was specified we create one ourselves... */
        if (!atomIDFlag)
        {
            int stat;
            PyObject *atomID=NULL;
            
            atomID = PyArray_Arange(1, NAtoms + 1, 1, NPY_INT32);
            if (atomID == NULL)
            {
                PyErr_SetString(PyExc_MemoryError, "Could not allocate atomID array");
                fclose(INFILE);
                Py_DECREF(resultDict);
                return NULL;
            }
            
            /* store in dict */
            stat = PyDict_SetItemString(resultDict, "atomID", atomID);
            
            /* give up our reference to array */
            Py_DECREF(atomID);
            
            if (stat)
            {
                PyErr_SetString(PyExc_RuntimeError, "Could not set atomID in dictionary");
                // need to free arrays too...
                fclose(INFILE);
                Py_DECREF(resultDict);
                return NULL;
            }
        }
        
        /* specie list/count */
        specieList = PyList_New(0);
        if (specieList == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not create specieList\n");
            fclose(INFILE);
            Py_DECREF(resultDict);
            return NULL;
        }
        
        specieCount = PyList_New(0);
        if (specieCount == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not create specieCount\n");
            fclose(INFILE);
            Py_DECREF(resultDict);
            Py_DECREF(specieList);
            return NULL;
        }
        
        /* read the body */
        printf("Reading body...\n");
        
        for (i = 0; i < NAtoms; i++)
        {
            long j, atomIndex = -1;
            char *atomLines[numLines];
            const int MAX_LINE_LENGTH = 512;
            
//             printf("Reading atom: %ld\n", i);
            
            /* read all of this atom's lines first... */
            for (j = 0; j < numLines; j++)
            {
                atomLines[j] = malloc(MAX_LINE_LENGTH * sizeof(char));
                if (atomLines[j] == NULL)
                {
                    long k;
                    
                    PyErr_SetString(PyExc_MemoryError, "Could not allocate atom line");
                    Py_DECREF(resultDict);
                    Py_DECREF(specieList);
                    Py_DECREF(specieCount);
                    fclose(INFILE);
                    for (k = 0; k < j; k++) free(atomLines[k]);
                    return NULL;
                }
                
                if (fgets(atomLines[j], MAX_LINE_LENGTH, INFILE) == NULL)
                {
                    long k;
                    char errstring[128];
                    
                    sprintf(errstring, "End of file reached while reading body (atom %ld)", i);
                    PyErr_SetString(PyExc_IOError, errstring);
                    Py_DECREF(resultDict);
                    Py_DECREF(specieList);
                    Py_DECREF(specieCount);
                    fclose(INFILE);
                    for (k = 0; k < j; k++) free(atomLines[k]);
                    return NULL;
                }
            }
            
            /* if atomID is in file, we should parse the line once and get the atom ID */
            /* Check that the atomID is <= NAtoms (depending if starts from one)
             * Make an input flag/option that says if atom ID starts from 1 or 0...
             */
            if (atomIDFlag)
            {
                int foundAtomID = 0;
                
                for (j = 0; j < numLines; j++)
                {
                    char *pch;
                    char line[MAX_LINE_LENGTH];
                    long lineLength, count;
                    PyObject *lineList=NULL;
                    
                    /* steals reference, no need to DECREF */
                    lineList = PyList_GetItem(bodyList, j);
                    lineLength = PyList_Size(lineList);
                    
                    /* parse for atomID */
                    /* make a copy of line as strtok modifies the original? */
                    strcpy(line, atomLines[j]);
                    pch = strtok(line, delimiter);
                    count = 0;
                    while (!foundAtomID && pch != NULL && count < lineLength)
                    {
                        char *key, *type;
                        int dim, dimcount;
                        PyObject *itemTuple;
                    
                        itemTuple = PyList_GetItem(lineList, count);
                        if (!PyArg_ParseTuple(itemTuple, "ssi", &key, &type, &dim)) 
                        {
                            long k;
                            
                            for (k = 0; k < numLines; k++) free(atomLines[k]);
                            fclose(INFILE);
                            Py_DECREF(resultDict);
                            Py_DECREF(specieList);
                            Py_DECREF(specieCount);
                            return NULL;
                        }
                    
                        dimcount = 0;
                        while (!foundAtomID && pch != NULL && dimcount < dim)
                        {
                            if (!strcmp("atomID", key))
                            {
                                atomIndex = (long) atoi(pch);
                                foundAtomID = 1;
                            }
                        
                            pch = strtok(NULL, delimiter);
                            dimcount++;
                        }
                    }
                    // TODO: should have err checking here on count/dimcount?
                    
                    /* check atomIndex in range */
                    if (foundAtomID)
                        atomIndex -= (long) atomIndexOffset;
                    
                    if (!foundAtomID || (atomIndex < 0 || atomIndex > NAtoms))
                    {
                        char errstring[128];
                        long k;
                        
                        if (foundAtomID) sprintf(errstring, "Atom index error: %ld out of range (atom %ld)", atomIndex, i);
                        else sprintf(errstring, "Atom index not in line (atom %ld)", i);
                        PyErr_SetString(PyExc_RuntimeError, errstring);
                        for (k = 0; k < numLines; k++) free(atomLines[k]);
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        Py_DECREF(specieList);
                        Py_DECREF(specieCount);
                    }
                }
            }
            else atomIndex = i;
            
            for (j = 0; j < numLines; j++)
            {
                char line[MAX_LINE_LENGTH], *pch;
                long lineLength, count;
                PyObject *lineList=NULL;
        
                lineList = PyList_GetItem(bodyList, j);
                lineLength = PyList_Size(lineList);
        
//                 printf("Body line %ld; num items = %ld\n", j, lineLength);
                
                /* read line */
                strcpy(line, atomLines[j]);

                /* parse the line */
                pch = strtok(line, delimiter);
                count = 0;
                while (pch != NULL && count < lineLength)
                {
                    char *key, *type;
                    int dim, dimcount;
                    PyObject *itemTuple=NULL;
                    PyArrayObject *array=NULL;
                
    //                 printf("  Line value %ld: %s\n", count, pch);
                
                    /* each item is a tuple: (key, type, dim) */
                    itemTuple = PyList_GetItem(lineList, count);
                    if (!PyArg_ParseTuple(itemTuple, "ssi", &key, &type, &dim)) 
                    {
                        long k;
                        
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        Py_DECREF(specieList);
                        Py_DECREF(specieCount);
                        for (k = 0; k < numLines; k++) free(atomLines[k]);
                        return NULL;
                    }
                    
                    /* get the array from the dictionary */
                    array = (PyArrayObject *) PyDict_GetItemString(resultDict, key);
                    
                    /* read one or three values */
                    dimcount = 0;
                    while (pch != NULL && dimcount < dim)
                    {
                        if (strcmp("SKIP", key))
                        {
                            /* symbol is special... */
                            if (!strcmp("Symbol", key))
                            {
                                int check, stat;
                                long value;
                                Py_ssize_t symlen, index;
                                PyObject *symin=NULL;
                                PyObject *valueObj=NULL;
                            
                                /* get the symbol */
                                symin = Py_BuildValue("s", pch);
                                symlen = PyString_Size(symin);
                                if (symlen == 1)
                                {
                                    Py_XDECREF(symin);
                                    symin = NULL;
                                    symin = PyString_FromFormat("%s_", pch);
                                }
                                else if (symlen != 2)
                                {
                                    char errstring[128];
                                    long k;
                                
                                    sprintf(errstring, "Cannot handle symbol of length %d", (int) symlen);
                                    PyErr_SetString(PyExc_RuntimeError, errstring);
                                    for (k = 0; k < numLines; k++) free(atomLines[k]);
                                    fclose(INFILE);
                                    Py_DECREF(resultDict);
                                    Py_DECREF(specieList);
                                    Py_DECREF(specieCount);
                                    Py_XDECREF(symin);
                                    return NULL;
                                }
                            
    //                            printf("Symbol: '%s'\n", PyString_AsString(symin));
                            
                                /* check if it already exists in the list */
                                check = PySequence_Contains(specieList, symin);
                            
                                /* error */
                                if (check == -1)
                                {
                                    long k;
                                    
                                    for (k = 0; k < numLines; k++) free(atomLines[k]);
                                    PyErr_SetString(PyExc_RuntimeError, "Checking if symbol in specie list failed");
                                    fclose(INFILE);
                                    Py_DECREF(resultDict);
                                    Py_DECREF(specieList);
                                    Py_DECREF(specieCount);
                                    Py_XDECREF(symin);
                                    return NULL;
                                }
                                /* symbol not in specie list */
                                else if (check == 0)
                                {
                                    PyObject *init=NULL;
                                
                                    printf("Add new specie\n");
                                
                                    /* add to list */
                                    stat = PyList_Append(specieList, symin);
                                    if (stat == -1)
                                    {
                                        long k;
                                        
                                        for (k = 0; k < numLines; k++) free(atomLines[k]);
                                        fclose(INFILE);
                                        Py_DECREF(resultDict);
                                        Py_DECREF(specieList);
                                        Py_DECREF(specieCount);
                                        Py_XDECREF(symin);
                                        return NULL;
                                    }
                                
                                    init = PyInt_FromLong(0);
                                    stat = PyList_Append(specieCount, init);
                                    Py_XDECREF(init);
                                    if (stat == -1)
                                    {
                                        long k;
                                        
                                        for (k = 0; k < numLines; k++) free(atomLines[k]);
                                        fclose(INFILE);
                                        Py_DECREF(resultDict);
                                        Py_DECREF(specieList);
                                        Py_DECREF(specieCount);
                                        Py_XDECREF(symin);
                                        return NULL;
                                    }
                                }
                            
                                /* increment specie counter */
                                index = PySequence_Index(specieList, symin);
                                if (index == -1)
                                {
                                    long k;
                                    
                                    for (k = 0; k < numLines; k++) free(atomLines[k]);
                                    PyErr_SetString(PyExc_RuntimeError, "Could not find symbol index in specieList");
                                    fclose(INFILE);
                                    Py_DECREF(resultDict);
                                    Py_DECREF(specieList);
                                    Py_DECREF(specieCount);
                                    Py_XDECREF(symin);
                                    return NULL;
                                }
                            
                                valueObj = PyList_GetItem(specieCount, index);
                                if (valueObj == NULL)
                                {
                                    long k;
                                    
                                    for (k = 0; k < numLines; k++) free(atomLines[k]);
                                    fclose(INFILE);
                                    Py_DECREF(resultDict);
                                    Py_DECREF(specieList);
                                    Py_DECREF(specieCount);
                                    Py_XDECREF(symin);
                                    return NULL;
                                }
                            
                                value = PyInt_AsLong(valueObj);
                                value++;
                            
                                stat = PyList_SetItem(specieCount, index, PyInt_FromLong(value));
                                if (stat == -1)
                                {
                                    long k;
                                    
                                    for (k = 0; k < numLines; k++) free(atomLines[k]);
                                    PyErr_SetString(PyExc_RuntimeError, "Could not set incremented specie count on list");
                                    fclose(INFILE);
                                    Py_DECREF(resultDict);
                                    Py_DECREF(specieList);
                                    Py_DECREF(specieCount);
                                    Py_XDECREF(symin);
                                    return NULL;
                                }
                            
                                Py_XDECREF(symin);
                            
                                /* set specie value */
                                IIND1(array, atomIndex) = (int) index;
                            }
                            else
                            {
                                if (!strcmp("i", type))
                                {
                                    int value;
                            
                                    value = atoi(pch);
                                    if (dim == 1) IIND1(array, atomIndex) = value;
                                    else IIND2(array, atomIndex, dimcount) = value;
                                }
                                else if (!strcmp("d", type))
                                {
                                    double value;
                                
                                    value = atof(pch);
                                    if (dim == 1) DIND1(array, atomIndex) = value;
                                    else DIND2(array, atomIndex, dimcount) = value;
                                }
                                else
                                {
                                    char errstring[128];
                                    long k;
                                    
                                    for (k = 0; k < numLines; k++) free(atomLines[k]);
                                    sprintf("Unrecognised type string (body): '%s'", type);
                                    PyErr_SetString(PyExc_RuntimeError, errstring);
                                    fclose(INFILE);
                                    Py_DECREF(resultDict);
                                    Py_DECREF(specieList);
                                    Py_DECREF(specieCount);
                                    return NULL;
                                }
                            }
                        }
                    
                        /* read next token */
                        pch = strtok(NULL, delimiter);
                        dimcount++;
                    }
                    
                    if (dimcount != dim)
                    {
                        char errstring[128];
                        long k;
                        
                        for (k = 0; k < numLines; k++) free(atomLines[k]);
                        sprintf(errstring, "Error during body line read (%ld:%ld): dim %d != %d", i, j, dimcount, dim);
                        PyErr_SetString(PyExc_IOError, errstring);
                        Py_DECREF(resultDict);
                        fclose(INFILE);
                        Py_DECREF(specieList);
                        Py_DECREF(specieCount);
                        return NULL;
                    }
                    
                    count++;
                }
                
                if (count != lineLength)
                {
                    char errstring[128];
                    long k;
                    
                    for (k = 0; k < numLines; k++) free(atomLines[k]);
                    sprintf(errstring, "Error during body line read (%ld:%ld): %ld != %ld", i, j, count, lineLength);
                    PyErr_SetString(PyExc_IOError, errstring);
                    Py_DECREF(resultDict);
                    fclose(INFILE);
                    Py_DECREF(specieList);
                    Py_DECREF(specieCount);
                    return NULL;
                }
            }
            
            for (j = 0; j < numLines; j++) free(atomLines[j]);
        }
        
        /* store specieList/Count */
        PyDict_SetItemString(resultDict, "specieList", specieList);
        Py_DECREF(specieList);
        PyDict_SetItemString(resultDict, "specieCount", specieCount);
        Py_DECREF(specieCount);
    }
    
    fclose(INFILE);
    
    printf("GENREADER: finished\n");
    
    return resultDict;
}
