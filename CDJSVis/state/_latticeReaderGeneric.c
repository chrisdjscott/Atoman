
/*******************************************************************************
 ** Copyright Chris Scott 2015
 ** Generic Lattice reader
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "array_utils.h"

#define DEBUG
#define MAX_LINE_LENGTH 512

struct BodyLineItem
{
    char *key;
    char *type;
    int dim;
};

struct BodyLine
{
    Py_ssize_t numItems;
    struct BodyLineItem *items;
};

struct Body
{
    Py_ssize_t numLines;
    struct BodyLine *lines;
};

static PyObject* readGenericLatticeFile(PyObject*, PyObject*);
static PyObject* getMinMaxPos(PyObject*, PyObject*);
static void freeBody(struct Body);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"readGenericLatticeFile", readGenericLatticeFile, METH_VARARGS, "Read generic Lattice file"},
    {"getMinMaxPos", getMinMaxPos, METH_VARARGS, "Get the min/max pos"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_latticeReaderGeneric(void)
{
    (void)Py_InitModule("_latticeReaderGeneric", methods);
    import_array();
}

/*******************************************************************************
 * Free body struct
 *******************************************************************************/
static void
freeBody(struct Body body)
{
    Py_ssize_t i;

    for (i = 0; i < body.numLines; i++)
        free(body.lines[i].items);

    free(body.lines);
}


/*******************************************************************************
 * Read generic lattice file
 *******************************************************************************/
static PyObject*
readGenericLatticeFile(PyObject *self, PyObject *args)
{
    int atomIndexOffset, linkedNAtoms;
    char *filename, *delimiter, *basename=NULL;
    FILE *INFILE=NULL;
    PyObject *headerList=NULL;
    PyObject *bodyList=NULL;
    PyObject *resultDict=NULL;
    PyObject *updateProgressCallback=NULL;

    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "sO!O!sii|Os", &filename, &PyList_Type, &headerList, &PyList_Type, &bodyList, &delimiter,
            &atomIndexOffset, &linkedNAtoms, &updateProgressCallback, &basename))
        return NULL;

#ifdef DEBUG
    printf("GENREADER: reading file: '%s'\n", filename);
    printf("Delimiter is '%s'\n", delimiter);
#endif
    
    /* check callback is callable */
    if (updateProgressCallback != NULL)
    {
        if (!PyCallable_Check(updateProgressCallback))
        {
            PyErr_SetString(PyExc_TypeError, "update progress callback must be callable");
            return NULL;
        }
#ifdef DEBUG
        printf("Have progress callback\n");
#endif
    }
    
    /* open the file for reading */
    INFILE = fopen(filename, "r");

    /* handle error */
    if (INFILE == NULL)
    {
        char errstring[128];

        sprintf(errstring, "%s: '%s'", strerror(errno), filename);
        PyErr_SetString(PyExc_IOError, errstring);
        return NULL;
    }
    /* continue to read */
    else
    {
        int atomIDFlag = 0;
        int haveSpecieOrSymbol = 0;
        long i, numLines, callbackInterval = 0;
        long NAtoms = -1;
        PyObject *specieList=NULL;
        PyObject *specieCount=NULL;
        struct Body bodyFormat;

        /* allocate result dict */
        resultDict = PyDict_New();
        if (resultDict == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not allocate resultDict");
            fclose(INFILE);
            return NULL;
        }

        /* number of header lines */
        numLines = PyList_Size(headerList);
#ifdef DEBUG
        printf("HEADER NUM LINES: %ld\n", numLines);
#endif

        /* read header */
        for (i = 0; i < numLines; i++)
        {
            char line[MAX_LINE_LENGTH], *pch;
            long lineLength, count;
            PyObject *headerLine=NULL;

            /* this is a borrowed ref so no need to DECREF */
            headerLine = PyList_GetItem(headerList, i);
            lineLength = PyList_Size(headerLine);

            /* read line */
            if (fgets(line, MAX_LINE_LENGTH, INFILE) == NULL)
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

                /* each item is a tuple: (key, type, dim) */
                itemTuple = PyList_GetItem(headerLine, count); // borrowed ref, no need to DECREF
                if (!PyArg_ParseTuple(itemTuple, "ssi", &key, &type, &dim))
                {
                    fclose(INFILE);
                    Py_DECREF(resultDict);
                    return NULL;
                }

                /* check if supposed to skip this item */
                if (strcmp("SKIP", key))
                {
                    /* build integer value */
                    if (!strcmp("i", type))
                    {
                        long tmpval;

                        tmpval = atol(pch);
                        if (!strcmp("NAtoms", key)) NAtoms = tmpval;
                        value = Py_BuildValue(type, tmpval);
                    }
                    /* build float value */
                    else if (!strcmp("d", type))
                        value = Py_BuildValue(type, atof(pch));
                    /* build string value */
                    else if (!strcmp("s", type))
                        value = Py_BuildValue(type, pch);
                    /* unrecognised type */
                    else
                    {
                        char errstring[128];

                        sprintf("Unrecognised type string: '%s'", type);
                        PyErr_SetString(PyExc_RuntimeError, errstring);
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        return NULL;
                    }

                    /* add the value to the dictionary */
                    stat = PyDict_SetItemString(resultDict, key, value);

                    /* Release our reference to value (the dict has a reference now too) */
                    Py_DECREF(value);

                    /* adding to dict failed? */
                    if (stat)
                    {
                        char errstring[128];

                        sprintf("Could not set item in dictionary: '%s'", key);
                        PyErr_SetString(PyExc_RuntimeError, errstring);
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        return NULL;
                    }
                }

                /* read next token */
                pch = strtok(NULL, delimiter);
                count++;
            }

            /* check we read the correct number of items from this line */
            if (count != lineLength)
            {
                char errstring[128];

                sprintf(errstring, "Wrong length for header line %ld: %ld != %ld", i, count, lineLength);
                PyErr_SetString(PyExc_IOError, errstring);
                Py_DECREF(resultDict);
                fclose(INFILE);
                return NULL;
            }
        }

        /* we check that NAtoms was read during the header... */
        if (NAtoms == -1)
        {
            PyErr_SetString(PyExc_RuntimeError, "Cannot autodetect NAtoms at the moment...");
            Py_DECREF(resultDict);
            fclose(INFILE);
            return NULL;
        }
        // we could do a pass through whole file to get NAtoms, then seek back to where we were...

        
        /* check NAtoms is same as linked */
        if (linkedNAtoms != -1)
        {
            if (NAtoms != linkedNAtoms)
            {
                char errstring[128];
                
                sprintf(errstring, "Number of atoms does not match linked lattice (%ld != %d)", NAtoms, linkedNAtoms);
                Py_DECREF(resultDict);
                fclose(INFILE);
                PyErr_SetString(PyExc_ValueError, errstring);
                return NULL;
            }
        }
        
#ifdef DEBUG
        printf("Preparing to read body; NAtoms = %ld\n", NAtoms);
#endif

        /* callback interval */
        if (updateProgressCallback != NULL)
        {
            /* callback interval */
            if (NAtoms < 500)
                callbackInterval = NAtoms;
            else
            {
                callbackInterval = (long) (NAtoms / 10);
                if (callbackInterval < 500)
                    callbackInterval = 500;
                else if (callbackInterval > 20000)
                    callbackInterval = 20000;
            }
#ifdef DEBUG
            printf("Callback interval = %ld atoms\n", callbackInterval);
#endif
        }
        
        /* body format (should be faster than parsing list/tuples) */
        /* number of body lines per atom */
        bodyFormat.numLines = PyList_Size(bodyList);
#ifdef DEBUG
        printf("Number of body lines per atom: %ld\n", numLines);
#endif

        /* allocate memory for body lines */
        bodyFormat.lines = malloc(bodyFormat.numLines * sizeof(struct BodyLine));
        if (bodyFormat.lines == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Cannot allocate bodyFormat.lines");
            Py_DECREF(resultDict);
            fclose(INFILE);
            return NULL;
        }

        /* allocate the arrays for scalar/vector data */
        for (i = 0; i < bodyFormat.numLines; i++)
        {
            long j;
            PyObject *lineList=NULL;

            /* list of items in this line */
            lineList = PyList_GetItem(bodyList, i); // steals ref, no need to DECREF
            bodyFormat.lines[i].numItems = PyList_Size(lineList);

            /* allocate memory for this line */
            bodyFormat.lines[i].items = malloc(bodyFormat.lines[i].numItems * sizeof(struct BodyLineItem));
            if (bodyFormat.lines[i].items == NULL)
            {
                PyErr_SetString(PyExc_MemoryError, "Cannot allocate bodyFormat.lines[].items");
                Py_DECREF(resultDict);
                fclose(INFILE);
                freeBody(bodyFormat);
                return NULL;
            }

#ifdef DEBUG
            printf("Body line %ld; num items = %ld\n", i, bodyFormat.lines[i].numItems);
#endif

            for (j = 0; j < bodyFormat.lines[i].numItems; j++)
            {
                char *key, *type;
                int dim, stat, typenum;
                long shape_dim;
                npy_intp np_dims[2] = {NAtoms, 3};
                PyObject *itemTuple=NULL;
                PyArrayObject *data=NULL;

                /* get and parse the tuple containing the item */
                itemTuple = PyList_GetItem(lineList, j);
                if (!PyArg_ParseTuple(itemTuple, "ssi", &key, &type, &dim))
                {
                    fclose(INFILE);
                    Py_DECREF(resultDict);
                    freeBody(bodyFormat);
                    return NULL;
                }

                /* store item in bodyFormat struct */
                bodyFormat.lines[i].items[j].key = key;
                bodyFormat.lines[i].items[j].type = type;
                bodyFormat.lines[i].items[j].dim = dim;

                /* check if we're supposed to ignore this value... */
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
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        freeBody(bodyFormat);
                        return NULL;
                    }

                    shape_dim = (dim == 1) ? 1 : 2;
                    data = (PyArrayObject *) PyArray_SimpleNew(shape_dim, np_dims, typenum);
                    if (data == NULL)
                    {
                        char errstring[128];

                        sprintf(errstring, "Could not allocate ndarray: '%s'", key);
                        PyErr_SetString(PyExc_MemoryError, errstring);
                        fclose(INFILE);
                        Py_DECREF(resultDict);
                        freeBody(bodyFormat);
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
                        freeBody(bodyFormat);
                        return NULL;
                    }
                }
            }
        }

#ifdef DEBUG
        printf("Atom ID flag: %d\n", atomIDFlag);
        printf("Have specie or symbol: %d\n", haveSpecieOrSymbol);
#endif

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
                freeBody(bodyFormat);
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
                freeBody(bodyFormat);
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
            freeBody(bodyFormat);
            return NULL;
        }

        specieCount = PyList_New(0);
        if (specieCount == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not create specieCount\n");
            fclose(INFILE);
            Py_DECREF(resultDict);
            Py_DECREF(specieList);
            freeBody(bodyFormat);
            return NULL;
        }
        
        /* read the body */
#ifdef DEBUG
        printf("Reading body...\n");
#endif

        /* loop over all atoms */
        numLines = bodyFormat.numLines;
        for (i = 0; i < NAtoms; i++)
        {
            long atomIndex = -1;
            Py_ssize_t j;
            char *atomLines[numLines];

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
                    freeBody(bodyFormat);
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
                    freeBody(bodyFormat);
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

                j = 0;
                while(!foundAtomID && j < numLines)
                {
                    char *pch;
                    char line[MAX_LINE_LENGTH];
                    Py_ssize_t numItems, count;

                    /* number of items in the line */
                    numItems = bodyFormat.lines[j].numItems;

                    /* parse for atomID */
                    /* make a copy of line as strtok might(?) modify the original */
                    strcpy(line, atomLines[j]);
                    pch = strtok(line, delimiter);
                    count = 0;
                    while (!foundAtomID && pch != NULL && count < numItems)
                    {
                        char *key;
                        int dim, dimcount;

                        /* unpack item */
                        key = bodyFormat.lines[j].items[count].key;
                        dim = bodyFormat.lines[j].items[count].dim;
                        
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

                        if (!foundAtomID && dimcount != dim)
                        {
                            char errstring[128];
                            long k;

                            for (k = 0; k < numLines; k++) free(atomLines[k]);
                            sprintf(errstring, "Error during body line read for atomID (%ld:%ld): dim %d != %d", i, j, dimcount, dim);
                            PyErr_SetString(PyExc_IOError, errstring);
                            Py_DECREF(resultDict);
                            fclose(INFILE);
                            Py_DECREF(specieList);
                            Py_DECREF(specieCount);
                            freeBody(bodyFormat);
                            return NULL;
                        }

                        count++;
                    }

                    if (!foundAtomID && count != numItems)
                    {
                        char errstring[128];
                        long k;

                        for (k = 0; k < numLines; k++) free(atomLines[k]);
                        sprintf(errstring, "Error during body line read for atomID (%ld:%ld): %ld != %ld", i, j, count, numItems);
                        PyErr_SetString(PyExc_IOError, errstring);
                        Py_DECREF(resultDict);
                        fclose(INFILE);
                        Py_DECREF(specieList);
                        Py_DECREF(specieCount);
                        freeBody(bodyFormat);
                        return NULL;
                    }
                    
                    j++;
                }

                /* check atomIndex in range */
                if (foundAtomID)
                    atomIndex -= (long) atomIndexOffset;
                
                if (!foundAtomID || (atomIndex < 0 || atomIndex >= NAtoms))
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
                    freeBody(bodyFormat);
                    return NULL;
                }
            }
            else atomIndex = i;

            /* now read the data for real */
            for (j = 0; j < numLines; j++)
            {
                char *line, *pch;
                Py_ssize_t numItems, count;

                /* number of items in the line */
                numItems = bodyFormat.lines[j].numItems;

                /* read line */
                line = atomLines[j];

                /* parse the line */
                pch = strtok(line, delimiter);
                count = 0;
                while (pch != NULL && count < numItems)
                {
                    char *key, *type;
                    int dim, dimcount;
                    PyArrayObject *array=NULL;

                    /* unpack item */
                    key = bodyFormat.lines[j].items[count].key;
                    type = bodyFormat.lines[j].items[count].type;
                    dim = bodyFormat.lines[j].items[count].dim;

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

                                    sprintf(errstring, "Cannot handle symbol of length %d: '%s'", (int) symlen, PyString_AsString(symin));
                                    PyErr_SetString(PyExc_RuntimeError, errstring);
                                    for (k = 0; k < numLines; k++) free(atomLines[k]);
                                    fclose(INFILE);
                                    Py_DECREF(resultDict);
                                    Py_DECREF(specieList);
                                    Py_DECREF(specieCount);
                                    Py_XDECREF(symin);
                                    freeBody(bodyFormat);
                                    return NULL;
                                }

//                                printf("Symbol: '%s'\n", PyString_AsString(symin));

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
                                    freeBody(bodyFormat);
                                    return NULL;
                                }
                                /* symbol not in specie list */
                                else if (check == 0)
                                {
                                    PyObject *init=NULL;

#ifdef DEBUG
                                    printf("Add new specie\n");
#endif

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
                                        freeBody(bodyFormat);
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
                                        freeBody(bodyFormat);
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
                                    freeBody(bodyFormat);
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
                                    freeBody(bodyFormat);
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
                                    freeBody(bodyFormat);
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
                                    freeBody(bodyFormat);
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
                        freeBody(bodyFormat);
                        return NULL;
                    }

                    count++;
                }

                if (count != numItems)
                {
                    char errstring[128];
                    long k;

                    for (k = 0; k < numLines; k++) free(atomLines[k]);
                    sprintf(errstring, "Error during body line read (%ld:%ld): %ld != %ld", i, j, count, numItems);
                    PyErr_SetString(PyExc_IOError, errstring);
                    Py_DECREF(resultDict);
                    fclose(INFILE);
                    Py_DECREF(specieList);
                    Py_DECREF(specieCount);
                    freeBody(bodyFormat);
                    return NULL;
                }
            }
            
            /* free lines */
            for (j = 0; j < numLines; j++) free(atomLines[j]);
            
            /* progress callback */
            if (updateProgressCallback != NULL && (i % callbackInterval == 0 || i == NAtoms - 1))
            {
                char message[512];
                PyObject *arglist;
                PyObject *cbres;
                
                /* callback */
#ifdef DEBUG
                printf("Progress callback at: %ld atoms\n", i + 1);
#endif
                sprintf(message, "Reading: '%s'", basename);
                arglist = Py_BuildValue("(iis)", (int) i, (int) NAtoms, message);
                cbres = PyObject_CallObject(updateProgressCallback, arglist);
                Py_DECREF(arglist);
                if (cbres == NULL) return NULL;
                Py_DECREF(cbres);
            }
        }

        /* store specieList/Count */
        PyDict_SetItemString(resultDict, "specieList", specieList);
        Py_DECREF(specieList);
        PyDict_SetItemString(resultDict, "specieCount", specieCount);
        Py_DECREF(specieCount);

        freeBody(bodyFormat);
    }

    fclose(INFILE);

#ifdef DEBUG
    printf("GENREADER: finished\n");
#endif

    return resultDict;
}

/*******************************************************************************
 * Get max/min pos
 *******************************************************************************/
static PyObject*
getMinMaxPos(PyObject *self, PyObject *args)
{
    Py_ssize_t np_dims[1] = {3};
    Py_ssize_t i, numAtoms;
    PyArrayObject *pos=NULL;
    PyArrayObject *minPos=NULL;
    PyArrayObject *maxPos=NULL;
    PyObject *tuple=NULL;

    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &pos))
        return NULL;
    
    /* number of atoms */
    numAtoms = pos->dimensions[0] / 3;
    
    /* allocate min/max pos */
    minPos = (PyArrayObject *) PyArray_SimpleNew(1, np_dims, NPY_FLOAT64);
    if (minPos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate minPos");
        return NULL;
    }
    
    maxPos = (PyArrayObject *) PyArray_SimpleNew(1, np_dims, NPY_FLOAT64);
    if (maxPos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate maxPos");
        Py_DECREF(minPos);
        return NULL;
    }
    
    /* initialise */
    for (i = 0; i < 3; i++)
    {
        DIND1(minPos, i) = 100000.0;
        DIND1(maxPos, i) = -100000.0;
    }
    
    /* loop over atoms */
    for (i = 0; i < numAtoms; i++)
    {
        Py_ssize_t j, i3 = 3 * i;
        
        for (j = 0; j < 3; j++)
        {
            DIND1(minPos, j) = (DIND1(pos, i3 + j) < DIND1(minPos, j)) ? DIND1(pos, i3 + j) : DIND1(minPos, j);
            DIND1(maxPos, j) = (DIND1(pos, i3 + j) > DIND1(maxPos, j)) ? DIND1(pos, i3 + j) : DIND1(maxPos, j);
        }
    }
    
    /* tuple for storing result */
    tuple = PyTuple_New(2);
    if (tuple == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate tuple");
        return NULL;
    }
    
    /* add min/max pos to tuple (steals ref) */
    PyTuple_SetItem(tuple, 0, PyArray_Return(minPos));
    PyTuple_SetItem(tuple, 1, PyArray_Return(maxPos));
    
    return tuple;
}
