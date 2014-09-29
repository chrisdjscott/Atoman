
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** Helper methods for computing Voronoi cells/volumes
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "array_utils.h"
#include "voro_iface.h"

/*******************************************************************************
 ** Define Voronoi object structure
 *******************************************************************************/
typedef struct {
    PyObject_HEAD
    vorores_t *voroResult;
    int voroResultSize;
} Voronoi;

/*******************************************************************************
 ** function prototypes
 *******************************************************************************/
static PyObject* makeVoronoiPoints(PyObject*, PyObject*);
static void free_vorores(Voronoi*);
static PyObject* Voronoi_computeVoronoi(Voronoi*, PyObject*);
static PyObject* Voronoi_atomVolume(Voronoi*, PyObject*);
static PyObject* Voronoi_atomNumNebs(Voronoi*, PyObject*);
static PyObject* Voronoi_atomNebList(Voronoi*, PyObject*);
static PyObject* Voronoi_getInputAtomPos(Voronoi*, PyObject*);
static PyObject* Voronoi_atomVertices(Voronoi*, PyObject*);
static PyObject* Voronoi_atomFaces(Voronoi*, PyObject*);
static PyObject* Voronoi_atomVolumesArray(Voronoi*);
static PyObject* Voronoi_atomNumNebsArray(Voronoi*);

/*******************************************************************************
 ** free vorores pointer
 *******************************************************************************/
static void free_vorores(Voronoi *self)
{
    int i;
    
    for (i = 0; i < self->voroResultSize; i++)
    {
        int j;
        
        for (j = 0; j < self->voroResult[i].numFaces; j++)
            free(self->voroResult[i].faceVertices[j]);
        free(self->voroResult[i].faceVertices);
        free(self->voroResult[i].numFaceVertices);
        free(self->voroResult[i].neighbours);
        free(self->voroResult[i].vertices);
    }
    
    free(self->voroResult);
    self->voroResult = NULL;
    self->voroResultSize = 0;
}

/*******************************************************************************
 ** Deallocation
 *******************************************************************************/
static void 
Voronoi_dealloc(Voronoi* self)
{
    free_vorores(self);
    self->ob_type->tp_free((PyObject*)self);
}

/*******************************************************************************
 ** New Voronoi object
 *******************************************************************************/
static PyObject *
Voronoi_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Voronoi *self;

    self = (Voronoi *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        /* set size = 0 initially */
        self->voroResultSize = 0;
        self->voroResult = NULL;
    }

    return (PyObject *)self;
}

/*******************************************************************************
 ** Return volume of atom
 *******************************************************************************/
static PyObject*
Voronoi_atomVolume(Voronoi *self, PyObject *args)
{
    int atomIndex;
    double volume;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "i", &atomIndex))
        return NULL;
    
    /* check index within range */
    if (atomIndex >= self->voroResultSize)
    {
        char msg[64];
        
        sprintf(msg, "Index is out of range (%d >= %d)", atomIndex, self->voroResultSize);
        PyErr_SetString(PyExc_IndexError, msg);
        return NULL;
    }
    
    /* get volume */
    volume = self->voroResult[atomIndex].volume;
    
    return Py_BuildValue("d", volume);
}

/*******************************************************************************
 ** Return array of atom volumes
 *******************************************************************************/
static PyObject*
Voronoi_atomVolumesArray(Voronoi *self)
{
    int i, size, dims[1];
    PyArrayObject *volumes=NULL;
    
    /* allocate volumes array */
    size = self->voroResultSize;
    dims[0] = size;
    volumes = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT64);
    if (volumes == NULL) return NULL;
    
    /* populate with volumes */
    for (i = 0; i < size; i++)
        DIND1(volumes, i) = self->voroResult[i].volume;
    
    return PyArray_Return(volumes);
}

/*******************************************************************************
 ** Return number of neighbours of an atom
 *******************************************************************************/
static PyObject*
Voronoi_atomNumNebs(Voronoi *self, PyObject *args)
{
    int i, atomIndex, numNebs;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "i", &atomIndex))
        return NULL;
    
    /* check index within range */
    if (atomIndex >= self->voroResultSize)
    {
        char msg[64];
        
        sprintf(msg, "Index is out of range (%d >= %d)", atomIndex, self->voroResultSize);
        PyErr_SetString(PyExc_IndexError, msg);
        return NULL;
    }
    
    /* check not infinite cell */
    for (i = 0; i < self->voroResult[atomIndex].numNeighbours; i++)
    {
        if (self->voroResult[atomIndex].neighbours[i] < 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Negative neighbour index (infinite cell?)");
            return NULL;
        }
    }
    
    /* get number of neighbours */
    numNebs = self->voroResult[atomIndex].numNeighbours;
    
    return Py_BuildValue("i", numNebs);
}

/*******************************************************************************
 ** Return array of number of neighbours for each atom
 *******************************************************************************/
static PyObject*
Voronoi_atomNumNebsArray(Voronoi *self)
{
    int i, size, dims[1];
    PyArrayObject *nebsArray=NULL;
    
    /* allocate numpy array */
    size = self->voroResultSize;
    dims[0] = size;
    nebsArray = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT32);
    if (nebsArray == NULL) return NULL;
    
    /* loop over atoms */
    for (i = 0; i < size; i++)
    {
        int j, numNebs;
        
        numNebs = self->voroResult[i].numNeighbours;
        
        /* check not infinite cell */
        for (j = 0; j < numNebs; j++)
        {
            if (self->voroResult[i].neighbours[j] < 0)
            {
                Py_DECREF(nebsArray);
                PyErr_SetString(PyExc_RuntimeError, "Negative neighbour index (infinite cell?)");
                return NULL;
            }
        }
        
        /* add to array */
        IIND1(nebsArray, i) = numNebs;
    }
    
    return PyArray_Return(nebsArray);
}

/*******************************************************************************
 ** Return neighbours of an atom
 *******************************************************************************/
static PyObject*
Voronoi_atomNebList(Voronoi *self, PyObject *args)
{
    int i, atomIndex, numNebs, dims[1];
    PyArrayObject *atomNebs=NULL;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "i", &atomIndex))
        return NULL;
    
    /* check index within range */
    if (atomIndex >= self->voroResultSize)
    {
        char msg[64];
        
        sprintf(msg, "Index is out of range (%d >= %d)", atomIndex, self->voroResultSize);
        PyErr_SetString(PyExc_IndexError, msg);
        return NULL;
    }
    
    /* allocate numpy array */
    numNebs = self->voroResult[atomIndex].numNeighbours;
    dims[0] = numNebs;
    atomNebs = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT32);
    if (atomNebs == NULL) return NULL;
    
    /* populate array */
    for (i = 0; i < numNebs; i++)
    {
        int nebidx;
        
        /* return exception if negative neighbour (infinite cell??) */
        nebidx = self->voroResult[atomIndex].neighbours[i];
        if (nebidx < 0)
        {
            Py_DECREF(atomNebs);
            PyErr_SetString(PyExc_RuntimeError, "Negative neighbour index (infinite cell?)");
            return NULL;
        }
        
        IIND1(atomNebs, i) = nebidx;
    }
    
    return PyArray_Return(atomNebs);
}

/*******************************************************************************
 ** Return position of input atom
 *******************************************************************************/
static PyObject*
Voronoi_getInputAtomPos(Voronoi *self, PyObject *args)
{
    int atomIndex, dims[1];
    PyArrayObject *pos=NULL;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "i", &atomIndex))
        return NULL;
    
    /* check index within range */
    if (atomIndex >= self->voroResultSize)
    {
        char msg[64];
        
        sprintf(msg, "Index is out of range (%d >= %d)", atomIndex, self->voroResultSize);
        PyErr_SetString(PyExc_IndexError, msg);
        return NULL;
    }
    
    /* allocate numpy array */
    dims[0] = 3;
    pos = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_FLOAT64);
    if (pos == NULL) return NULL;
    
    /* populate array */
    DIND1(pos, 0) = self->voroResult[atomIndex].originalPos[0];
    DIND1(pos, 1) = self->voroResult[atomIndex].originalPos[1];
    DIND1(pos, 2) = self->voroResult[atomIndex].originalPos[2];
    
    return PyArray_Return(pos);
}

/*******************************************************************************
 ** Return vertices of an atoms Voronoi cell
 *******************************************************************************/
static PyObject*
Voronoi_atomVertices(Voronoi *self, PyObject *args)
{
    int i, nverts, atomIndex, dims[2];
    PyArrayObject *vertices=NULL;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "i", &atomIndex))
        return NULL;
    
    /* check index within range */
    if (atomIndex >= self->voroResultSize)
    {
        char msg[64];
        
        sprintf(msg, "Index is out of range (%d >= %d)", atomIndex, self->voroResultSize);
        PyErr_SetString(PyExc_IndexError, msg);
        return NULL;
    }
    
    /* allocate numpy array */
    nverts = self->voroResult[atomIndex].numVertices;
    dims[0] = nverts;
    dims[1] = 3;
    vertices = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT64);
    if (vertices == NULL) return NULL;
    
    /* populate array */
    for (i = 0; i < nverts; i++)
    {
        DIND2(vertices, i, 0) = self->voroResult[atomIndex].vertices[3*i];
        DIND2(vertices, i, 1) = self->voroResult[atomIndex].vertices[3*i+1];
        DIND2(vertices, i, 2) = self->voroResult[atomIndex].vertices[3*i+2];
    }
    
    return PyArray_Return(vertices);
}

/*******************************************************************************
 ** Return the faces of the Voronoi cell of the given atom, as a list of 
 ** indexes of its vertices
 *******************************************************************************/
static PyObject*
Voronoi_atomFaces(Voronoi *self, PyObject *args)
{
    int i, atomIndex, nfaces;
    PyObject *faceList=NULL;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "i", &atomIndex))
        return NULL;
    
    /* check index within range */
    if (atomIndex >= self->voroResultSize)
    {
        char msg[64];
        
        sprintf(msg, "Index is out of range (%d >= %d)", atomIndex, self->voroResultSize);
        PyErr_SetString(PyExc_IndexError, msg);
        return NULL;
    }
    
    /* check not infinite cell */
    for (i = 0; i < self->voroResult[atomIndex].numNeighbours; i++)
    {
        if (self->voroResult[atomIndex].neighbours[i] < 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Negative neighbour index (infinite cell?)");
            return NULL;
        }
    }
    
    /* allocate list and loop over faces */
    nfaces = self->voroResult[atomIndex].numFaces;
    faceList = PyList_New(nfaces);
    for (i = 0; i < nfaces; i++)
    {
        int j, nverts, dims[1];
        PyArrayObject *vertArray=NULL;
        
        /* number of vertices making up this face */
        nverts = self->voroResult[atomIndex].numFaceVertices[i];
        
        /* allocate numpy array for storing vertices */
        dims[0] = nverts;
        vertArray = (PyArrayObject *) PyArray_FromDims(1, dims, NPY_INT32);
        
        /* add vertices to array */
        for (j = 0; j < nverts; j++)
            IIND1(vertArray, j) = self->voroResult[atomIndex].faceVertices[i][j];
        
        /* add array to list */
        PyList_SetItem(faceList, i, PyArray_Return(vertArray));
    }
    
    return faceList;
}

/*******************************************************************************
 * Compute Voronoi using Voro++
 *******************************************************************************/
static PyObject*
Voronoi_computeVoronoi(Voronoi *self, PyObject *args)
{
    int *specie, *PBC, NAtoms, useRadii;
    double *pos, *minPos, *maxPos, *cellDims, *specieCovalentRadius, dispersion;
    PyArrayObject *posIn=NULL;
    PyArrayObject *minPosIn=NULL;
    PyArrayObject *maxPosIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *specieCovalentRadiusIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *PBCIn=NULL;
    int i, status;
    double bound_lo[3], bound_hi[3];
    double *radii;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!di", &PyArray_Type, &posIn, &PyArray_Type, &minPosIn, &PyArray_Type, 
            &maxPosIn, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &PyArray_Type, &specieIn, &PyArray_Type, 
            &specieCovalentRadiusIn, &dispersion, &useRadii))
        return NULL;
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    NAtoms = ((int) posIn->dimensions[0]) / 3;
    
    if (not_doubleVector(minPosIn)) return NULL;
    minPos = pyvector_to_Cptr_double(minPosIn);
    
    if (not_doubleVector(maxPosIn)) return NULL;
    maxPos = pyvector_to_Cptr_double(maxPosIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_doubleVector(specieCovalentRadiusIn)) return NULL;
    specieCovalentRadius = pyvector_to_Cptr_double(specieCovalentRadiusIn);
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    /* prepare for Voro call */
    for (i = 0; i < 3; i++)
    {
        if (PBC[i])
        {
            bound_lo[i] = 0.0;
            bound_hi[i] = cellDims[i];
        }
        else
        {
            bound_lo[i] = minPos[i] - dispersion;
            bound_hi[i] = maxPos[i] + dispersion;
        }
    }
    
    if (useRadii)
    {
        radii = malloc(NAtoms * sizeof(double));
        if (radii == NULL)
        {
            printf("ERROR: could not allocate radii\n");
            exit(55);
        }
        for (i = 0; i < NAtoms; i++)
            radii[i] = specieCovalentRadius[specie[i]];
    }
    
    /* deallocate if ran previously */
    free_vorores(self);
    
    /* allocate structure for holding results */
    self->voroResult = malloc(NAtoms * sizeof(vorores_t));
    if (self->voroResult == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate voroResult pointer");
        return NULL;
    }
    self->voroResultSize = NAtoms;
    
    /* call voro++ wrapper */
    status = computeVoronoiVoroPlusPlusWrapper(NAtoms, pos, PBC, bound_lo, bound_hi, useRadii, radii, self->voroResult);
    
    /* if status if positive we should dealloc everything and return error */
    if (status)
    {
        free_vorores(self);
        PyErr_SetString(PyExc_RuntimeError, "Error creating VoronoiResult structure");
        return NULL;
    }
    
    if (useRadii) free(radii);
    
    Py_INCREF(Py_None);
    return Py_None;
}

/*******************************************************************************
 ** List of methods on Voronoi object
 *******************************************************************************/
static PyMethodDef Voronoi_methods[] = {
    {"atomVolume", (PyCFunction)Voronoi_atomVolume, METH_VARARGS, 
            "Return the volume of the given atom"
    },
    {"computeVoronoi", (PyCFunction)Voronoi_computeVoronoi, METH_VARARGS, 
            "Compute Voronoi volumes of the atoms using Voro++ interface"
    },
    {"atomNumNebs", (PyCFunction)Voronoi_atomNumNebs, METH_VARARGS, 
            "Return the number of neighbours of the given atom"
    },
    {"atomNebList", (PyCFunction)Voronoi_atomNebList, METH_VARARGS, 
                "Return the neighbours of the given atom"
    },
    {"getInputAtomPos", (PyCFunction)Voronoi_getInputAtomPos, METH_VARARGS, 
                    "Return the original position of the given atom"
    },
    {"atomVertices", (PyCFunction)Voronoi_atomVertices, METH_VARARGS, 
                        "Return the positions of the vertices of the Voronoi cell of the given atom"
    },
    {"atomFaces", (PyCFunction)Voronoi_atomFaces, METH_VARARGS, 
                            "Return the list of indexes of the vertices that make up the faces of an atoms Voronoi cell"
    },
    {"atomVolumesArray", (PyCFunction)Voronoi_atomVolumesArray, METH_NOARGS, 
                "Return array of atom volumes"
    },
    {"atomNumNebsArray", (PyCFunction)Voronoi_atomNumNebsArray, METH_NOARGS, 
                    "Return array of number of neighbours of atoms"
    },
    {NULL}  /* Sentinel */
};

/*******************************************************************************
 ** Voronoi object type
 *******************************************************************************/
static PyTypeObject VoronoiType = {
    PyObject_HEAD_INIT(NULL)
    0,                                  /*ob_size*/
    "_voronoi.Voronoi",                 /*tp_name*/
    sizeof(Voronoi),                    /*tp_basicsize*/
    0,                                  /*tp_itemsize*/
    (destructor)Voronoi_dealloc,        /*tp_dealloc*/
    0,                                  /*tp_print*/
    0,                                  /*tp_getattr*/
    0,                                  /*tp_setattr*/
    0,                                  /*tp_compare*/
    0,                                  /*tp_repr*/
    0,                                  /*tp_as_number*/
    0,                                  /*tp_as_sequence*/
    0,                                  /*tp_as_mapping*/
    0,                                  /*tp_hash */
    0,                                  /*tp_call*/
    0,                                  /*tp_str*/
    0,                                  /*tp_getattro*/
    0,                                  /*tp_setattro*/
    0,                                  /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Object for computing the Voronoi cells of atoms in a Lattice", /* tp_doc */
    0,                                  /* tp_traverse */
    0,                                  /* tp_clear */
    0,                                  /* tp_richcompare */
    0,                                  /* tp_weaklistoffset */
    0,                                  /* tp_iter */
    0,                                  /* tp_iternext */
    Voronoi_methods,                    /* tp_methods */
    0,                                  /* tp_members */
    0,                                  /* tp_getset */
    0,                                  /* tp_base */
    0,                                  /* tp_dict */
    0,                                  /* tp_descr_get */
    0,                                  /* tp_descr_set */
    0,                                  /* tp_dictoffset */
    0,                                  /* tp_init */
    0,                                  /* tp_alloc */
    Voronoi_new,                        /* tp_new */
};

/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"makeVoronoiPoints", makeVoronoiPoints, METH_VARARGS, "Make points array for passing to Voronoi method"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_voronoi(void)
{
    PyObject *m;
    
    VoronoiType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&VoronoiType) < 0)
        return;
    
    m = Py_InitModule3("_voronoi", methods, "Module for computing Voronoi cells of atoms");
    import_array();
    
    Py_INCREF(&VoronoiType);
    PyModule_AddObject(m, "Voronoi", (PyObject *)&VoronoiType);
}

/*******************************************************************************
 * Make points array for passing to Voronoi method
 *******************************************************************************/
static PyObject*
makeVoronoiPoints(PyObject *self, PyObject *args)
{
    int *PBC, NAtoms;
    double *pos, *cellDims, skin;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!d", &PyArray_Type, &posIn, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &skin))
        return NULL;
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    NAtoms = ((int) posIn->dimensions[0]) / 3;
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (PBC[0] || PBC[1] || PBC[2])
    {
        int i, addCount, dims[2], count;
        double halfDims[3];
        PyArrayObject *pts = NULL;
        
        /* first pass to get total number of points (only if PBCs) */
        addCount = 0;
        for (i = 0; i < NAtoms; i++)
        {
            int rxb, ryb, rzb;
            double rx, ry, rz;
            
            rx = pos[3*i];
            ry = pos[3*i+1];
            rz = pos[3*i+2];
            
            rxb = PBC[0] && (rx < skin || rx > cellDims[0] - skin);
            ryb = PBC[1] && (ry < skin || ry > cellDims[1] - skin);
            rzb = PBC[2] && (rz < skin || rz > cellDims[2] - skin);
            
            if (rxb)
            {
                addCount++;
                
                if (ryb)
                {
                    addCount += 2;
                    
                    if (rzb) addCount += 4;
                }
                else if (rzb) addCount += 2;
            }
            
            else if (ryb)
            {
                addCount++;
                
                if (rzb) addCount += 2;
            }
            
            else if (rzb) addCount++;
        }
        
        printf("Adding %d ghost atoms (skin = %lf)\n", addCount, skin);
        
        /* second pass to make the pts array */
        dims[0] = NAtoms + addCount;
        dims[1] = 3;
        pts = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT64);
        
        /* first add real atoms */
        for (i = 0; i < NAtoms; i++)
        {
            DIND2(pts, i, 0) = pos[3*i+0];
            DIND2(pts, i, 1) = pos[3*i+1];
            DIND2(pts, i, 2) = pos[3*i+2];
        }
        
        /* now add ghost atoms */
        halfDims[0] = cellDims[0] * 0.5;
        halfDims[1] = cellDims[1] * 0.5;
        halfDims[2] = cellDims[2] * 0.5;
        count = NAtoms;
        for (i = 0; i < NAtoms; i++)
        {
            int rxb, ryb, rzb;
            double rx, ry, rz;
            double rxmod, rymod, rzmod;
            
            rx = pos[3*i];
            ry = pos[3*i+1];
            rz = pos[3*i+2];
            
            rxb = PBC[0] && (rx < skin || rx > cellDims[0] - skin);
            ryb = PBC[1] && (ry < skin || ry > cellDims[1] - skin);
            rzb = PBC[2] && (rz < skin || rz > cellDims[2] - skin);
            
            if (rxb)
            {
                rxmod = (rx < halfDims[0]) ? cellDims[0] : -1 * cellDims[0];
                
                DIND2(pts, count, 0) = rx + rxmod;
                DIND2(pts, count, 1) = ry;
                DIND2(pts, count++, 2) = rz;
                
                if (ryb)
                {
                    rymod = (ry < halfDims[1]) ? cellDims[1] : -1 * cellDims[1];
                    
                    DIND2(pts, count, 0) = rx + rxmod;
                    DIND2(pts, count, 1) = ry + rymod;
                    DIND2(pts, count++, 2) = rz;
                    
                    DIND2(pts, count, 0) = rx;
                    DIND2(pts, count, 1) = ry + rymod;
                    DIND2(pts, count++, 2) = rz;
                    
                    if (rzb)
                    {
                        rzmod = (rz < halfDims[2]) ? cellDims[2] : -1 * cellDims[2];
                        
                        DIND2(pts, count, 0) = rx + rxmod;
                        DIND2(pts, count, 1) = ry + rymod;
                        DIND2(pts, count++, 2) = rz + rzmod;
                        
                        DIND2(pts, count, 0) = rx;
                        DIND2(pts, count, 1) = ry + rymod;
                        DIND2(pts, count++, 2) = rz + rzmod;
                        
                        DIND2(pts, count, 0) = rx + rxmod;
                        DIND2(pts, count, 1) = ry;
                        DIND2(pts, count++, 2) = rz + rzmod;
                        
                        DIND2(pts, count, 0) = rx;
                        DIND2(pts, count, 1) = ry;
                        DIND2(pts, count++, 2) = rz + rzmod;
                    }
                }
                else if (rzb)
                {
                    rzmod = (rz < halfDims[2]) ? cellDims[2] : -1 * cellDims[2];
                    
                    DIND2(pts, count, 0) = rx;
                    DIND2(pts, count, 1) = ry;
                    DIND2(pts, count++, 2) = rz + rzmod;
                    
                    DIND2(pts, count, 0) = rx + rxmod;
                    DIND2(pts, count, 1) = ry;
                    DIND2(pts, count++, 2) = rz + rzmod;
                }
            }
            
            else if (ryb)
            {
                rymod = (ry < halfDims[1]) ? cellDims[1] : -1 * cellDims[1];
                
                DIND2(pts, count, 0) = rx;
                DIND2(pts, count, 1) = ry + rymod;
                DIND2(pts, count++, 2) = rz;
                
                if (rzb)
                {
                    rzmod = (rz < halfDims[2]) ? cellDims[2] : -1 * cellDims[2];
                    
                    DIND2(pts, count, 0) = rx;
                    DIND2(pts, count, 1) = ry;
                    DIND2(pts, count++, 2) = rz + rzmod;
                    
                    DIND2(pts, count, 0) = rx;
                    DIND2(pts, count, 1) = ry + rymod;
                    DIND2(pts, count++, 2) = rz + rzmod;
                }
            }
            
            else if (rzb)
            {
                rzmod = (rz < halfDims[2]) ? cellDims[2] : -1 * cellDims[2];
                
                DIND2(pts, count, 0) = rx;
                DIND2(pts, count, 1) = ry;
                DIND2(pts, count++, 2) = rz + rzmod;
            }
        }
        
        return PyArray_Return(pts);
    }
    else
    {
        int i, dims[2];
        PyArrayObject *pts = NULL;
        
        /* second pass to make the pts array */
        dims[0] = NAtoms;
        dims[1] = 3;
        pts = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_FLOAT64);
        
        /* just add real atoms */
        for (i = 0; i < NAtoms; i++)
        {
            DIND2(pts, i, 0) = pos[3*i+0];
            DIND2(pts, i, 1) = pos[3*i+1];
            DIND2(pts, i, 2) = pos[3*i+2];
        }
        
        return PyArray_Return(pts);
    }
}
