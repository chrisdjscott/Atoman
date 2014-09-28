
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
 ** Define object structure
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

/*******************************************************************************
 ** free vorores pointer
 *******************************************************************************/
static void free_vorores(Voronoi *self)
{
    int i;
    
    for (i = 0; i < self->voroResultSize; i++)
    {
        if (self->voroResult[i].numFaces > 0)
        {
            int j;
            
            for (j = 0; j < self->voroResult[i].numFaces; j++)
                free(self->voroResult[i].faceVertices[j]);
            free(self->voroResult[i].faceVertices);
            free(self->voroResult[i].numFaceVertices);
        }
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
 ** Return number of neighbours of an atom
 *******************************************************************************/
static PyObject*
Voronoi_atomNumNebs(Voronoi *self, PyObject *args)
{
    int atomIndex;
    int numNebs;
    
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
    numNebs = self->voroResult[atomIndex].numNeighbours;
    
    return Py_BuildValue("i", numNebs);
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
    
    return Py_BuildValue("i", 0);
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
//    {"volumesArray", 
//    
//    }
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
    "Voronoi objects",                  /* tp_doc */
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
