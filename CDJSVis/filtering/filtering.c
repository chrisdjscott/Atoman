
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** Filtering routines written in C to improve performance
 *******************************************************************************/

#define DEBUG

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "utilities.h"
#include "boxeslib.h"
#include "array_utils.h"


static PyObject* calculate_drift_vector(PyObject*, PyObject*);
static PyObject* specieFilter(PyObject*, PyObject*);
static PyObject* sliceFilter(PyObject*, PyObject*);
static PyObject* cropSphereFilter(PyObject*, PyObject*);
static PyObject* cropFilter(PyObject*, PyObject*);
static PyObject* KEFilter(PyObject*, PyObject*);
static PyObject* PEFilter(PyObject*, PyObject*);
static PyObject* chargeFilter(PyObject*, PyObject*);
static PyObject* atomIndexFilter(PyObject*, PyObject*);
static PyObject* voronoiVolumeFilter(PyObject*, PyObject*);
static PyObject* voronoiNeighboursFilter(PyObject*, PyObject*);
static PyObject* coordNumFilter(PyObject*, PyObject*);
static PyObject* displacementFilter(PyObject*, PyObject*);
static PyObject* slipFilter(PyObject*, PyObject*);
static PyObject* genericScalarFilter(PyObject *, PyObject *);
static PyObject* cropDefectsFilter(PyObject *self, PyObject *args);
static PyObject* sliceDefectsFilter(PyObject *self, PyObject *args);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"calculate_drift_vector", calculate_drift_vector, METH_VARARGS, "Calculate the drift vector between two systems"},
    {"specieFilter", specieFilter, METH_VARARGS, "Filter by specie"},
    {"sliceFilter", sliceFilter, METH_VARARGS, "Slice filter"},
    {"cropSphereFilter", cropSphereFilter, METH_VARARGS, "Crop sphere filter"},
    {"cropFilter", cropFilter, METH_VARARGS, "Crop filter"},
    {"KEFilter", KEFilter, METH_VARARGS, "Kinetic energy filter"},
    {"PEFilter", PEFilter, METH_VARARGS, "Potential energy filter"},
    {"chargeFilter", chargeFilter, METH_VARARGS, "Charge filter"},
    {"atomIndexFilter", atomIndexFilter, METH_VARARGS, "Atom index filter"},
    {"voronoiVolumeFilter", voronoiVolumeFilter, METH_VARARGS, "Voronoi volume filter"},
    {"voronoiNeighboursFilter", voronoiNeighboursFilter, METH_VARARGS, "Voronoi neighbours filter"},
    {"coordNumFilter", coordNumFilter, METH_VARARGS, "Coordination number filter"},
    {"displacementFilter", displacementFilter, METH_VARARGS, "Displacement filter"},
    {"slipFilter", slipFilter, METH_VARARGS, "Slip filter"},
    {"genericScalarFilter", genericScalarFilter, METH_VARARGS, "Generic scalar filter"},
    {"cropDefectsFilter", cropDefectsFilter, METH_VARARGS, "Crop defects filter"},
    {"sliceDefectsFilter", sliceDefectsFilter, METH_VARARGS, "Slice defects filter"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_filtering(void)
{
    (void)Py_InitModule("_filtering", methods);
    import_array();
}

/*******************************************************************************
 ** calculate drift
 *******************************************************************************/
static PyObject* 
calculate_drift_vector(PyObject *self, PyObject *args)
{
    int NAtoms, *PBC;
    double *pos, *refPos, *cellDims, *driftVector;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *refPosIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *driftVectorIn=NULL;
    
    int i;
    double sepVec[3];
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!", &NAtoms, &PyArray_Type, &posIn, &PyArray_Type, &refPosIn, 
            &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &PyArray_Type, &driftVectorIn))
        return NULL;
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(refPosIn)) return NULL;
    refPos = pyvector_to_Cptr_double(refPosIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(driftVectorIn)) return NULL;
    driftVector = pyvector_to_Cptr_double(driftVectorIn);
    
    /* compute drift vector */
    driftVector[0] = 0.0;
    driftVector[1] = 0.0;
    driftVector[2] = 0.0;
    for (i = 0; i < NAtoms; i++)
    {
        atomSeparationVector(sepVec, refPos[3*i], refPos[3*i+1], refPos[3*i+2], pos[3*i], pos[3*i+1], pos[3*i+2], 
                             cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
        
        driftVector[0] += sepVec[0];
        driftVector[1] += sepVec[1];
        driftVector[2] += sepVec[2];
    }
    
    driftVector[0] /= (double) NAtoms;
    driftVector[1] /= (double) NAtoms;
    driftVector[2] /= (double) NAtoms;
    
    return Py_BuildValue("i", 0);
}

/*******************************************************************************
 ** Specie filter
 *******************************************************************************/
static PyObject* 
specieFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, visSpecDim, *visSpec, *specie, NScalars, NVectors;
    double *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *visSpecIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, NVisible;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!iO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &visSpecIn, 
            &PyArray_Type, &specieIn, &NScalars, &PyArray_Type, &fullScalarsIn, &NVectors, &PyArray_Type, 
            &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_intVector(visSpecIn)) return NULL;
    visSpec = pyvector_to_Cptr_int(visSpecIn);
    visSpecDim = (int) visSpecIn->dimensions[0];
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    /* run */
    NVisible = 0;
    for (i = 0; i < NVisibleIn; i++)
    {
        int j, index, match;
        
        index = visibleAtoms[i];
        
        match = 0;
        for (j = 0; j < visSpecDim; j++)
        {
            if (specie[index] == visSpec[j])
            {
                match = 1;
                break;
            }
        }
        
        if (match)
        {
            int k;
            
            for (k = 0; k < NScalars; k++)
                fullScalars[NVisibleIn * k + NVisible] = fullScalars[NVisibleIn * k + i];
            
            for (k = 0; k < NVectors; k++)
            {
                DIND2(fullVectors, NVisibleIn * k + NVisible, 0) = DIND2(fullVectors, NVisibleIn * k + i, 0);
                DIND2(fullVectors, NVisibleIn * k + NVisible, 1) = DIND2(fullVectors, NVisibleIn * k + i, 1);
                DIND2(fullVectors, NVisibleIn * k + NVisible, 2) = DIND2(fullVectors, NVisibleIn * k + i, 2);
            }
            
            visibleAtoms[NVisible++] = index;
        }
    }
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 ** Slice filter
 *******************************************************************************/
static PyObject* 
sliceFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, invert, NScalars, NVectors;
    double *pos, x0, y0, z0, xn, yn, zn, *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, j, NVisible, index;
    double mag, xd, yd, zd, dotProd, distanceToPlane;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ddddddiiO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, 
            &x0, &y0, &z0, &xn, &yn, &zn, &invert, &NScalars, &PyArray_Type, &fullScalarsIn, &NVectors, 
            &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    /* normalise (xn, yn, zn) */
    mag = sqrt(xn * xn + yn * yn + zn * zn);
    xn = xn / mag;
    yn = yn / mag;
    zn = zn / mag;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        xd = pos[3*index] - x0;
        yd = pos[3*index+1] - y0;
        zd = pos[3*index+2] - z0;
        
        dotProd = xd * xn + yd * yn + zd * zn;
        distanceToPlane = dotProd / mag;
        
        if ((invert && distanceToPlane > 0) || (!invert && distanceToPlane < 0))
        {
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
                fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
            for (j = 0; j < NVectors; j++)
            {
                DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
            }
            
            visibleAtoms[NVisible++] = index;
        }
    }
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 ** Slice defects filter
 *******************************************************************************/
static PyObject*
sliceDefectsFilter(PyObject *self, PyObject *args)
{
    int invert;
    double x0, y0, z0, xn, yn, zn;
    PyArrayObject *interstitials=NULL;
    PyArrayObject *vacancies=NULL;
    PyArrayObject *antisites=NULL;
    PyArrayObject *onAntisites=NULL;
    PyArrayObject *splitInterstitials=NULL;
    PyArrayObject *pos=NULL;
    PyArrayObject *refPos=NULL;
    PyObject *result=NULL;
    
    
    /* parse and check arguments from Python */
    if (PyArg_ParseTuple(args, "O!O!O!O!O!O!O!ddddddi", &PyArray_Type, &interstitials, &PyArray_Type, &vacancies, &PyArray_Type, &antisites,
            &PyArray_Type, &onAntisites, &PyArray_Type, &splitInterstitials, &PyArray_Type, &pos, &PyArray_Type, &refPos, &x0, &y0, &z0, &xn,
            &yn, &zn, &invert))
    {
        int i;
        int NVacsIn, NIntsIn, NAntsIn, NSplitsIn;
        int NVacs, NInts, NAnts, NSplits;
        double mag;
        
        /* check types */
        if (not_intVector(interstitials)) return NULL;
        NIntsIn = (int) interstitials->dimensions[0];
        if (not_intVector(vacancies)) return NULL;
        NVacsIn = (int) vacancies->dimensions[0];
        if (not_intVector(antisites)) return NULL;
        NAntsIn = (int) antisites->dimensions[0];
        if (not_intVector(splitInterstitials)) return NULL;
        NSplitsIn = (int) (splitInterstitials->dimensions[0] / 3);
        if (not_doubleVector(pos)) return NULL;
        if (not_doubleVector(refPos)) return NULL;
        
        /* normalise (xn, yn, zn) */
        mag = sqrt(xn*xn + yn*yn + zn*zn);
        xn = xn / mag;
        yn = yn / mag;
        zn = zn / mag;
        
        /* vacancies */
        NVacs = 0;
        for (i = 0; i < NVacsIn; i++)
        {
            int index = IIND1(vacancies, i);
            int ind3 = index * 3;
            double xd = DIND1(refPos, ind3    ) - x0;
            double yd = DIND1(refPos, ind3 + 1) - y0;
            double zd = DIND1(refPos, ind3 + 2) - z0;
            double dotProd = xd*xn + yd*yn + zd*zn;
            double distanceToPlane = dotProd / mag;
            
            if ((invert && distanceToPlane > 0) || (!invert && distanceToPlane < 0))
                IIND1(vacancies, NVacs++) = index;
        }
        
        /* antisites */
        NAnts = 0;
        for (i = 0; i < NAntsIn; i++)
        {
            int index = IIND1(antisites, i);
            int ind3 = index * 3;
            double xd = DIND1(refPos, ind3    ) - x0;
            double yd = DIND1(refPos, ind3 + 1) - y0;
            double zd = DIND1(refPos, ind3 + 2) - z0;
            double dotProd = xd*xn + yd*yn + zd*zn;
            double distanceToPlane = dotProd / mag;
            
            if ((invert && distanceToPlane > 0) || (!invert && distanceToPlane < 0))
            {
                IIND1(antisites, NAnts) = index;
                IIND1(onAntisites, NAnts++) = IIND1(onAntisites, i);
            }
        }
        
        /* interstitials */
        NInts = 0;
        for (i = 0; i < NIntsIn; i++)
        {
            int index = IIND1(interstitials, i);
            int ind3 = index * 3;
            double xd = DIND1(pos, ind3    ) - x0;
            double yd = DIND1(pos, ind3 + 1) - y0;
            double zd = DIND1(pos, ind3 + 2) - z0;
            double dotProd = xd*xn + yd*yn + zd*zn;
            double distanceToPlane = dotProd / mag;
            
            if ((invert && distanceToPlane > 0) || (!invert && distanceToPlane < 0))
                IIND1(interstitials, NInts++) = index;
        }
        
        /* split interstitials */
        NSplits = 0;
        for (i = 0; i < NSplitsIn; i++)
        {
            int i3 = 3 * i;
            int index = IIND1(splitInterstitials, i3);
            int ind3 = index * 3;
            double xd = DIND1(refPos, ind3    ) - x0;
            double yd = DIND1(refPos, ind3 + 1) - y0;
            double zd = DIND1(refPos, ind3 + 2) - z0;
            double dotProd = xd*xn + yd*yn + zd*zn;
            double distanceToPlane = dotProd / mag;
            
            if ((invert && distanceToPlane > 0) || (!invert && distanceToPlane < 0))
            {
                int nsplit3 = 3 * NSplits;
                IIND1(splitInterstitials, nsplit3    ) = index;
                IIND1(splitInterstitials, nsplit3 + 1) = IIND1(splitInterstitials, i3 + 1);
                IIND1(splitInterstitials, nsplit3 + 2) = IIND1(splitInterstitials, i3 + 2);
                NSplits++;
            }
        }
        
        /* result (setItem steals ownership) */
        result = PyTuple_New(4);
        PyTuple_SetItem(result, 0, Py_BuildValue("i", NInts));
        PyTuple_SetItem(result, 1, Py_BuildValue("i", NVacs));
        PyTuple_SetItem(result, 2, Py_BuildValue("i", NAnts));
        PyTuple_SetItem(result, 3, Py_BuildValue("i", NSplits));
    }
    
    return result;
}


/*******************************************************************************
 ** Crop sphere filter
 *******************************************************************************/
static PyObject* 
cropSphereFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, *PBC, invertSelection, NScalars, NVectors;
    double *pos, xCentre, yCentre, zCentre, radius, *cellDims, *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    
    int i, j, NVisible, index;
    double radius2, sep2;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ddddO!O!iiO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, 
            &xCentre, &yCentre, &zCentre, &radius, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn,
            &invertSelection, &NScalars, &PyArray_Type, &fullScalarsIn, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    radius2 = radius * radius;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                 xCentre, yCentre, zCentre, 
                                 cellDims[0], cellDims[1], cellDims[2], 
                                 PBC[0], PBC[1], PBC[2]);
        
        if (sep2 < radius2)
        {
            if (invertSelection)
            {
                /* handle full scalars array */
                for (j = 0; j < NScalars; j++)
                    fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
                
                for (j = 0; j < NVectors; j++)
                {
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
                }
                
                visibleAtoms[NVisible++] = index;
            }
        }
        else
        {
            if (!invertSelection)
            {
                /* handle full scalars array */
                for (j = 0; j < NScalars; j++)
                    fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
                
                for (j = 0; j < NVectors; j++)
                {
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
                }
                
                visibleAtoms[NVisible++] = index;
            }
        }
    }
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 ** Crop defects filter
 *******************************************************************************/
static PyObject*
cropDefectsFilter(PyObject *self, PyObject *args)
{
    int xEnabled, yEnabled, zEnabled, invertSelection;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    PyArrayObject *interstitials=NULL;
    PyArrayObject *vacancies=NULL;
    PyArrayObject *antisites=NULL;
    PyArrayObject *onAntisites=NULL;
    PyArrayObject *splitInterstitials=NULL;
    PyArrayObject *pos=NULL;
    PyArrayObject *refPos=NULL;
    PyObject *result=NULL;
    
    
    /* parse and check arguments from Python */
    if (PyArg_ParseTuple(args, "O!O!O!O!O!O!O!ddddddiiii", &PyArray_Type, &interstitials, &PyArray_Type, &vacancies, 
            &PyArray_Type, &antisites, &PyArray_Type, &onAntisites, &PyArray_Type, &splitInterstitials,
            &PyArray_Type, &pos, &PyArray_Type, &refPos, &xmin, &xmax, &ymin, &ymax, &zmin, &zmax, &xEnabled,
            &yEnabled, &zEnabled, &invertSelection))
    {
        int i;
        int NVacsIn, NIntsIn, NAntsIn, NSplitsIn;
        int NVacs, NInts, NAnts, NSplits;
        
        
        /* check types */
        if (not_intVector(interstitials)) return NULL;
        NIntsIn = (int) interstitials->dimensions[0];
        if (not_intVector(vacancies)) return NULL;
        NVacsIn = (int) vacancies->dimensions[0];
        if (not_intVector(antisites)) return NULL;
        NAntsIn = (int) antisites->dimensions[0];
        if (not_intVector(splitInterstitials)) return NULL;
        NSplitsIn = (int) (splitInterstitials->dimensions[0] / 3);
        if (not_doubleVector(pos)) return NULL;
        if (not_doubleVector(refPos)) return NULL;
        
        /* vacancies */
        NVacs = 0;
        for (i = 0; i < NVacsIn; i++)
        {
            int index = IIND1(vacancies, i);
            int ind3 = 3 * index;
            double rx = DIND1(refPos, ind3    );
            double ry = DIND1(refPos, ind3 + 1);
            double rz = DIND1(refPos, ind3 + 2);
            int add = 1;
            
            /* check each direction */
            if (xEnabled && (rx < xmin || rx > xmax)) add = 0;
            if (add && yEnabled && (ry < ymin || ry > ymax)) add = 0;
            if (add && zEnabled && (rz < zmin || rz > zmax)) add = 0;
            
            /* is this atom visible? */
            if ((add && !invertSelection) || (!add && invertSelection))
                IIND1(vacancies, NVacs++) = index;
        }
        
        /* antisites */
        NAnts = 0;
        for (i = 0; i < NAntsIn; i++)
        {
            int index = IIND1(antisites, i);
            int ind3 = 3 * index;
            double rx = DIND1(refPos, ind3    );
            double ry = DIND1(refPos, ind3 + 1);
            double rz = DIND1(refPos, ind3 + 2);
            int add = 1;
            
            /* check each direction */
            if (xEnabled && (rx < xmin || rx > xmax)) add = 0;
            if (add && yEnabled && (ry < ymin || ry > ymax)) add = 0;
            if (add && zEnabled && (rz < zmin || rz > zmax)) add = 0;
            
            /* is this atom visible? */
            if ((add && !invertSelection) || (!add && invertSelection))
            {
                IIND1(antisites, NAnts) = index;
                IIND1(onAntisites, NAnts++) = IIND1(onAntisites, i);
            }
        }
        
        /* interstitials */
        NInts = 0;
        for (i = 0; i < NIntsIn; i++)
        {
            int index = IIND1(interstitials, i);
            int ind3 = 3 * index;
            double rx = DIND1(pos, ind3    );
            double ry = DIND1(pos, ind3 + 1);
            double rz = DIND1(pos, ind3 + 2);
            int add = 1;
            
            /* check each direction */
            if (xEnabled && (rx < xmin || rx > xmax)) add = 0;
            if (add && yEnabled && (ry < ymin || ry > ymax)) add = 0;
            if (add && zEnabled && (rz < zmin || rz > zmax)) add = 0;
            
            /* is this atom visible? */
            if ((add && !invertSelection) || (!add && invertSelection))
                IIND1(interstitials, NInts++) = index;
        }
        
        /* split interstitials */
        NSplits = 0;
        for (i = 0; i < NSplitsIn; i++)
        {
            int i3 = 3 * i;
            int index = IIND1(splitInterstitials, i3);   // we use the position of the vacancy
            int ind3 = 3 * index;
            double rx = DIND1(refPos, ind3    );
            double ry = DIND1(refPos, ind3 + 1);
            double rz = DIND1(refPos, ind3 + 2);
            int add = 1;
            
            /* check each direction */
            if (xEnabled && (rx < xmin || rx > xmax)) add = 0;
            if (add && yEnabled && (ry < ymin || ry > ymax)) add = 0;
            if (add && zEnabled && (rz < zmin || rz > zmax)) add = 0;
            
            /* is this atom visible? */
            if ((add && !invertSelection) || (!add && invertSelection))
            {
                int nsplit3 = 3 * NSplits;
                IIND1(splitInterstitials, nsplit3    ) = index;
                IIND1(splitInterstitials, nsplit3 + 1) = IIND1(splitInterstitials, i3 + 1);
                IIND1(splitInterstitials, nsplit3 + 2) = IIND1(splitInterstitials, i3 + 2);
                NSplits++;
            }
        }
        
        /* result (setItem steals ownership) */
        result = PyTuple_New(4);
        PyTuple_SetItem(result, 0, Py_BuildValue("i", NInts));
        PyTuple_SetItem(result, 1, Py_BuildValue("i", NVacs));
        PyTuple_SetItem(result, 2, Py_BuildValue("i", NAnts));
        PyTuple_SetItem(result, 3, Py_BuildValue("i", NSplits));
    }
    
    return result;
}


/*******************************************************************************
 ** Crop filter
 *******************************************************************************/
static PyObject* 
cropFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, xEnabled, yEnabled, zEnabled, invertSelection, NScalars, NVectors;
    double *pos, xmin, xmax, ymin, ymax, zmin, zmax, *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, j, index, NVisible, add;
    double rx, ry, rz;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ddddddiiiiiO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, 
            &xmin, &xmax, &ymin, &ymax, &zmin, &zmax, &xEnabled, &yEnabled, &zEnabled, &invertSelection, 
            &NScalars, &PyArray_Type, &fullScalarsIn, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        rx = pos[3*index];
        ry = pos[3*index+1];
        rz = pos[3*index+2];
        
        add = 1;
        if (xEnabled == 1)
        {
            if (rx < xmin || rx > xmax)
            {
                add = 0;
            }
        }
        
        if (add && yEnabled == 1)
        {
            if (ry < ymin || ry > ymax)
            {
                add = 0;
            }
        }
        
        if (add && zEnabled == 1)
        {
            if (rz < zmin || rz > zmax)
            {
                add = 0;
            }
        }
        
        if ((add && !invertSelection) || (!add && invertSelection))
        {
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
                fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
            for (j = 0; j < NVectors; j++)
            {
                DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
            }
            
            visibleAtoms[NVisible++] = index;
        }
    }
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 ** Displacement filter
 *******************************************************************************/
static PyObject* 
displacementFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, *PBC, NScalars, filteringEnabled, driftCompensation, refPosDim, NVectors;
    double *scalars, *pos, *refPosIn, *cellDims, minDisp, maxDisp, *fullScalars, *driftVector;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *refPosIn_np=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *scalarsIn=NULL;
    PyArrayObject *driftVectorIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, NVisible, index, j;
    double sep2, maxDisp2, minDisp2;
    double *refPos;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!ddiO!iiO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &scalarsIn, 
            &PyArray_Type, &posIn, &PyArray_Type, &refPosIn_np, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, 
            &minDisp, &maxDisp, &NScalars, &PyArray_Type, &fullScalarsIn, &filteringEnabled, &driftCompensation, 
            &PyArray_Type, &driftVectorIn, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(refPosIn_np)) return NULL;
    refPosIn = pyvector_to_Cptr_double(refPosIn_np);
    refPosDim = (int) refPosIn_np->dimensions[0];
    
    if (not_doubleVector(scalarsIn)) return NULL;
    scalars = pyvector_to_Cptr_double(scalarsIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(driftVectorIn)) return NULL;
    driftVector = pyvector_to_Cptr_double(driftVectorIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    /* drift compensation? */
    if (driftCompensation)
    {
        refPos = malloc(refPosDim * sizeof(double));
        if (refPos == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate refPos");
            return NULL;
        }
        
        for (i = 0; i < refPosDim / 3; i++)
        {
            refPos[3*i] = refPosIn[3*i] + driftVector[0];
            refPos[3*i+1] = refPosIn[3*i+1] + driftVector[1];
            refPos[3*i+2] = refPosIn[3*i+2] + driftVector[2];
        }
    }
    else
    {
        refPos = refPosIn;
    }
    
    minDisp2 = minDisp * minDisp;
    maxDisp2 = maxDisp * maxDisp;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                 refPos[3*index], refPos[3*index+1], refPos[3*index+2], 
                                 cellDims[0], cellDims[1], cellDims[2], 
                                 PBC[0], PBC[1], PBC[2]);
        
        if (!filteringEnabled || (sep2 <= maxDisp2 && sep2 >= minDisp2))
        {
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
                fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
            for (j = 0; j < NVectors; j++)
            {
                DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
            }
            
            visibleAtoms[NVisible] = index;
            scalars[NVisible++] = sqrt(sep2);
        }
    }
    
    if (driftCompensation) free(refPos);
    else refPos = NULL;
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 ** Kinetic energy filter
 *******************************************************************************/
static PyObject* 
KEFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, NScalars, NVectors;
    double *KE, minKE, maxKE, *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *KEIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ddiO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &KEIn, 
            &minKE, &maxKE, &NScalars, &PyArray_Type, &fullScalarsIn, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(KEIn)) return NULL;
    KE = pyvector_to_Cptr_double(KEIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (KE[index] < minKE || KE[index] > maxKE)
        {
            continue;
        }
        else
        {
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
                fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
            for (j = 0; j < NVectors; j++)
            {
                DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
            }
            
            visibleAtoms[NVisible++] = index;
        }
    }
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 ** Potential energy filter
 *******************************************************************************/
static PyObject* 
PEFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, NScalars, NVectors;
    double *PE, minPE, maxPE, *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *PEIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ddiO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &PEIn, 
            &minPE, &maxPE, &NScalars, &PyArray_Type, &fullScalarsIn, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(PEIn)) return NULL;
    PE = pyvector_to_Cptr_double(PEIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (PE[index] < minPE || PE[index] > maxPE)
        {
            continue;
        }
        else
        {
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
                fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
            for (j = 0; j < NVectors; j++)
            {
                DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
            }
            
            visibleAtoms[NVisible++] = index;
        }
    }
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 ** Charge energy filter
 *******************************************************************************/
static PyObject* 
chargeFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, NScalars, NVectors;
    double *charge, minCharge, maxCharge, *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *chargeIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ddiO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &chargeIn, 
            &minCharge, &maxCharge, &NScalars, &PyArray_Type, &fullScalarsIn, &NVectors, &PyArray_Type, 
            &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(chargeIn)) return NULL;
    charge = pyvector_to_Cptr_double(chargeIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (charge[index] < minCharge || charge[index] > maxCharge)
        {
            continue;
        }
        else
        {
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
                fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
            for (j = 0; j < NVectors; j++)
            {
                DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
            }
            
            visibleAtoms[NVisible++] = index;
        }
    }
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 * Calculate coordination number
 *******************************************************************************/
static PyObject* 
coordNumFilter(PyObject *self, PyObject *args)
{
    int NVisible, *visibleAtoms, *specie, NSpecies, *PBC, minCoordNum, maxCoordNum, NScalars, filteringEnabled, NVectors;
    double *pos, *bondMinArray, *bondMaxArray, approxBoxWidth, *cellDims, *coordArray, *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *coordArrayIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *bondMinArrayIn=NULL;
    PyArrayObject *bondMaxArrayIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, j, k, index, index2, visIndex;
    int speca, specb, count, NVisibleNew;
    int boxIndex, boxNebList[27], boxstat;
    double *visiblePos, sep2, sep;
    struct Boxes *boxes;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!iO!O!dO!O!O!iiiO!iiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn,
            &PyArray_Type, &specieIn, &NSpecies, &PyArray_Type, &bondMinArrayIn, &PyArray_Type, &bondMaxArrayIn,
            &approxBoxWidth, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &PyArray_Type, &coordArrayIn,
            &minCoordNum, &maxCoordNum, &NScalars, &PyArray_Type, &fullScalarsIn, &filteringEnabled, &NVectors,
            &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisible = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    
    if (not_doubleVector(bondMinArrayIn)) return NULL;
    bondMinArray = pyvector_to_Cptr_double(bondMinArrayIn);
    
    if (not_doubleVector(bondMaxArrayIn)) return NULL;
    bondMaxArray = pyvector_to_Cptr_double(bondMaxArrayIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(coordArrayIn)) return NULL;
    coordArray = pyvector_to_Cptr_double(coordArrayIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
#ifdef DEBUG
    printf("COORDNUM CLIB\n");
    printf("N VIS: %d\n", NVisible);
    
    for (i=0; i<NSpecies; i++)
    {
        for (j=i; j<NSpecies; j++)
        {
            printf("%d - %d: %lf -> %lf\n", i, j, bondMinArray[i*NSpecies+j], bondMaxArray[i*NSpecies+j]);
        }
    }
#endif
    
    /* construct visible pos array */
    visiblePos = malloc(3 * NVisible * sizeof(double));
    if (visiblePos == NULL)
    {
        printf("ERROR: could not allocate visiblePos\n");
        exit(50);
    }
    
    for (i=0; i<NVisible; i++)
    {
        index = visibleAtoms[i];
        int i3 = i * 3;
        int ind3 = index * 3;
        visiblePos[i3    ] = pos[ind3    ];
        visiblePos[i3 + 1] = pos[ind3 + 1];
        visiblePos[i3 + 2] = pos[ind3 + 2];
    }
    
    /* box visible atoms */
    boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
    if (boxes == NULL)
    {
        free(visiblePos);
        return NULL;
    }
    boxstat = putAtomsInBoxes(NVisible, visiblePos, boxes);
    
    /* free visible pos */
    free(visiblePos);
    
    if (boxstat) return NULL;
    
    /* zero coord array */
    for (i=0; i<NVisible; i++) coordArray[i] = 0;
    
    /* loop over visible atoms */
    count = 0;
    for (i=0; i<NVisible; i++)
    {
        int boxNebListSize;
        
        index = visibleAtoms[i];
        
        speca = specie[index];
        
        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(pos[3*index], pos[3*index+1], pos[3*index+2], boxes);
        if (boxIndex < 0)
        {
            freeBoxes(boxes);
            return NULL;
        }
        
        /* find neighbouring boxes */
        boxNebListSize = getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over box neighbourhood */
        for (j = 0; j < boxNebListSize; j++)
        {
            boxIndex = boxNebList[j];
            
            for (k=0; k<boxes->boxNAtoms[boxIndex]; k++)
            {
                visIndex = boxes->boxAtoms[boxIndex][k];
                index2 = visibleAtoms[visIndex];
                
                if (index >= index2)
                {
                    continue;
                }
                
                specb = specie[index2];
                
                if (bondMinArray[speca*NSpecies+specb] == 0.0 && bondMaxArray[speca*NSpecies+specb] == 0.0)
                {
                    continue;
                }
                
                /* atomic separation */
                sep2 = atomicSeparation2(pos[3*index], pos[3*index+1], pos[3*index+2], 
                                         pos[3*index2], pos[3*index2+1], pos[3*index2+2], 
                                         cellDims[0], cellDims[1], cellDims[2], 
                                         PBC[0], PBC[1], PBC[2]);
                
                sep = sqrt(sep2);
                
                /* check if these atoms are bonded */
                if (sep >= bondMinArray[speca*NSpecies+specb] && sep <= bondMaxArray[speca*NSpecies+specb])
                {
                    coordArray[i]++;
                    coordArray[visIndex]++;
                    
                    count++;
                }
            }
        }
    }
    
    /* filter */
    if (filteringEnabled)
    {
        NVisibleNew = 0;
        for (i=0; i<NVisible; i++)
        {
            if (coordArray[i] >= minCoordNum && coordArray[i] <= maxCoordNum)
            {
                visibleAtoms[NVisibleNew] = visibleAtoms[i];
                coordArray[NVisibleNew] = coordArray[i];
                
                /* handle full scalars array */
                for (j = 0; j < NScalars; j++)
                {
                    fullScalars[NVisible * j + NVisibleNew] = fullScalars[NVisible * j + i];
                }
                
                for (j = 0; j < NVectors; j++)
                {
                    DIND2(fullVectors, NVisible * j + NVisibleNew, 0) = DIND2(fullVectors, NVisible * j + i, 0);
                    DIND2(fullVectors, NVisible * j + NVisibleNew, 1) = DIND2(fullVectors, NVisible * j + i, 1);
                    DIND2(fullVectors, NVisible * j + NVisibleNew, 2) = DIND2(fullVectors, NVisible * j + i, 2);
                }
                
                NVisibleNew++;
            }
        }
    }
    else
    {
        NVisibleNew = NVisible;
    }
    
    /* free */
    freeBoxes(boxes);
    
    return Py_BuildValue("i", NVisibleNew);
}

/*******************************************************************************
 ** Voronoi volume filter
 *******************************************************************************/
static PyObject* 
voronoiVolumeFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, NScalars, filteringEnabled, NVectors;
    double *volume, minVolume, maxVolume, *fullScalars, *scalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *volumeIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *scalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ddO!iO!iiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &volumeIn, 
            &minVolume, &maxVolume, &PyArray_Type, &scalarsIn, &NScalars, &PyArray_Type, &fullScalarsIn,
            &filteringEnabled, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(volumeIn)) return NULL;
    volume = pyvector_to_Cptr_double(volumeIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(scalarsIn)) return NULL;
    scalars = pyvector_to_Cptr_double(scalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (!filteringEnabled || (volume[index] >= minVolume && volume[index] <= maxVolume))
        {
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
                fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
            for (j = 0; j < NVectors; j++)
            {
                DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
            }
            
            visibleAtoms[NVisible] = index;
            scalars[NVisible++] = volume[index];
        }
    }
    
    return Py_BuildValue("i", NVisible);
}

/*******************************************************************************
 ** Voronoi neighbours filter
 *******************************************************************************/
static PyObject* 
voronoiNeighboursFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, NScalars, filteringEnabled, *num_nebs_array, minNebs, maxNebs, NVectors;
    double *fullScalars, *scalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *num_nebs_arrayIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *scalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!iiO!iO!iiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &num_nebs_arrayIn, 
            &minNebs, &maxNebs, &PyArray_Type, &scalarsIn, &NScalars, &PyArray_Type, &fullScalarsIn,
            &filteringEnabled, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_intVector(num_nebs_arrayIn)) return NULL;
    num_nebs_array = pyvector_to_Cptr_int(num_nebs_arrayIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(scalarsIn)) return NULL;
    scalars = pyvector_to_Cptr_double(scalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (!filteringEnabled || (num_nebs_array[index] >= minNebs && num_nebs_array[index] <= maxNebs))
        {
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
                fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
            for (j = 0; j < NVectors; j++)
            {
                DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
            }
            
            visibleAtoms[NVisible] = index;
            scalars[NVisible++] = num_nebs_array[index];
        }
    }
    
    return Py_BuildValue("i", NVisible);
}

/*******************************************************************************
 ** Atom index filter
 *******************************************************************************/
static PyObject* 
atomIndexFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, NScalars, *atomID, NVectors, numr;
    double *fullScalars;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *atomIDIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;
    PyArrayObject *rangeArray=NULL;
    
    int i, NVisible;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!iO!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &atomIDIn, 
            &PyArray_Type, &rangeArray, &NScalars, &PyArray_Type, &fullScalarsIn, &NVectors, 
            &PyArray_Type, &fullVectors))
        return NULL;
    
    /* check array types */
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_intVector(atomIDIn)) return NULL;
    atomID = pyvector_to_Cptr_int(atomIDIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    if (not_doubleVector(fullVectors)) return NULL;
    
    if (not_intVector(rangeArray)) return NULL;
    numr = (int) rangeArray->dimensions[0];
    
    NVisible = 0;
    for (i = 0; i < NVisibleIn; i++)
    {
        int j;
        int index = visibleAtoms[i];
        int id = atomID[index];
        int visible = 0;
        
        /* loop over ranges, looking to see if this atom is visible */
        for (j = 0; j < numr; j++)
        {
            int minid = IIND2(rangeArray, j, 0);
            int maxid = IIND2(rangeArray, j, 1);
            if (id >= minid && id <= maxid)
            {
                visible = 1;
                break;
            }
        }
        
        if (visible)
        {
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
            {
                fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            }
        
            for (j = 0; j < NVectors; j++)
            {
                DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
            }
        
            visibleAtoms[NVisible++] = index;
        }
    }
    
    return Py_BuildValue("i", NVisible);
}

/*******************************************************************************
 ** Generic scalar filter
 ** The scalars array should be NAtoms in size, not NVisibleIn
 *******************************************************************************/
static PyObject* 
genericScalarFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, NScalars, NVectors;
    double minVal, maxVal;
    PyArrayObject *visibleAtoms=NULL;
    PyArrayObject *scalars=NULL;
    PyArrayObject *fullScalars=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, NVisible;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!ddiO!iO!", &PyArray_Type, &visibleAtoms, &PyArray_Type, &scalars, &minVal, 
            &maxVal, &NScalars, &PyArray_Type, &fullScalars, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtoms)) return NULL;
    NVisibleIn = (int) visibleAtoms->dimensions[0];
    if (not_doubleVector(scalars)) return NULL;
    if (not_doubleVector(fullScalars)) return NULL;
    if (not_doubleVector(fullVectors)) return NULL;
    
    /* loop over visible atoms */
    NVisible = 0;
    for (i = 0; i < NVisibleIn; i++)
    {
        int index;
        double scalarVal;
        
        index = IIND1(visibleAtoms, i);
        scalarVal = DIND1(scalars, index);
        
        if (!(scalarVal < minVal || scalarVal > maxVal))
        {
            int j;
            
            /* handle full scalars array */
            for (j = 0; j < NScalars; j++)
            {
                int nj = NVisibleIn * j;
                
                DIND1(fullScalars, nj + NVisible) = DIND1(fullScalars, nj + i);
            }
            
            for (j = 0; j < NVectors; j++)
            {
                int nj, njn, nji;
                
                nj = NVisibleIn * j;
                njn = nj + NVisible;
                nji = nj + i;
                
                DIND2(fullVectors, njn, 0) = DIND2(fullVectors, nji, 0);
                DIND2(fullVectors, njn, 1) = DIND2(fullVectors, nji, 1);
                DIND2(fullVectors, njn, 2) = DIND2(fullVectors, nji, 2);
            }
            
            IIND1(visibleAtoms, NVisible++) = IIND1(visibleAtoms, i);
        }
    }
    
    return Py_BuildValue("i", NVisible);
}

/*******************************************************************************
 ** Slip filter
 *******************************************************************************/
static PyObject *
slipFilter(PyObject *self, PyObject *args)
{
    // we need: refPos, pos, minPos, maxPos, cellDims, PBC, visibleAtoms
    int NVisibleIn, *PBC, NScalars, filteringEnabled, driftCompensation, refPosDim, NVectors;
    double minSlip, maxSlip, *cellDims;
    PyArrayObject *visibleAtoms=NULL;
    PyArrayObject *refPosOrig=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *pos=NULL;
    PyArrayObject *scalars=NULL;
    PyArrayObject *driftVector=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *fullScalars=NULL;
    PyArrayObject *fullVectors=NULL;
    
    int i, NVisible, boxstat;
    int *slippedAtoms;
    double *refPos, *visiblePos, *slipx, *slipy, *slipz;
    double approxBoxWidth;
    struct Boxes *boxes;
    
#ifdef DEBUG
    printf("SLIPC: Entering slip C lib\n");
#endif
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!ddiO!iiO!iO!", &PyArray_Type, &visibleAtoms, &PyArray_Type, &scalars, 
            &PyArray_Type, &pos, &PyArray_Type, &refPosOrig, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, 
            &minSlip, &maxSlip, &NScalars, &PyArray_Type, &fullScalars, &filteringEnabled, &driftCompensation, 
            &PyArray_Type, &driftVector, &NVectors, &PyArray_Type, &fullVectors))
        return NULL;
    
    if (not_intVector(visibleAtoms)) return NULL;
    NVisibleIn = (int) visibleAtoms->dimensions[0];
    
    if (not_doubleVector(pos)) return NULL;
    
    if (not_doubleVector(refPosOrig)) return NULL;
    refPosDim = (int) refPosOrig->dimensions[0];
    
    if (not_doubleVector(scalars)) return NULL;
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(fullScalars)) return NULL;
    
    if (not_doubleVector(driftVector)) return NULL;
    
    if (not_doubleVector(fullVectors)) return NULL;
    
#ifdef DEBUG
    printf("SLIPC: Parsed args\n");
    printf("SLIPC: NVisibleIn = %d\n", NVisibleIn);
#endif
    
    /* drift compensation */
    if (driftCompensation)
    {
        refPos = malloc(refPosDim * sizeof(double));
        if (refPos == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Could not allocate refPos in slipFilter");
            return NULL;
        }
        
        for (i = 0; i < refPosDim / 3; i++)
        {
            int j, i3;
            
            i3 = 3 * i;
            for (j = 0; j < 3; j++) refPos[i3 + j] = DIND1(refPosOrig, i3 + j) + DIND1(driftVector, j);
        }
    }
    else
    {
        refPos = pyvector_to_Cptr_double(refPosOrig);
    }
    
    /* visible pos */
    visiblePos = malloc(3 * NVisibleIn * sizeof(double));
    if (visiblePos == NULL)
    {
        if (driftCompensation) free(refPos);
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate visiblePos in slipFilter");
        return NULL;
    }
    for (i = 0; i < NVisibleIn; i++)
    {
        int index3, i3;
        
        i3 = 3 * i;
        index3 = IIND1(visibleAtoms, i) * 3;
        visiblePos[i3    ] = DIND1(pos, index3    );
        visiblePos[i3 + 1] = DIND1(pos, index3 + 1);
        visiblePos[i3 + 2] = DIND1(pos, index3 + 2);
    }
    
    /* boxAtoms */
    approxBoxWidth = 5.0; // detect automatically!!
    boxes = setupBoxes(approxBoxWidth, PBC, cellDims);
    if (boxes == NULL)
    {
        free(visiblePos);
        if (driftCompensation) free(refPos);
        return NULL;
    }
    boxstat = putAtomsInBoxes(NVisibleIn, visiblePos, boxes);
    free(visiblePos);
    if (boxstat)
    {
        if (driftCompensation) free(refPos);
        freeBoxes(boxes);
        return NULL;
    }
    
    /* allocate slip arrays */
    slipx = calloc(NVisibleIn, sizeof(double));
    if (slipx == NULL);
    {
        int err = errno;
        char errstring[512];
        
        if (driftCompensation) free(refPos);
        freeBoxes(boxes);
        sprintf(errstring, "Allocate slipx (slipFilter) failed: '%s'", strerror(err));
        PyErr_SetString(PyExc_RuntimeError, errstring);
        return NULL;
    }
    
    slipy = calloc(NVisibleIn, sizeof(double));
    if (slipy == NULL);
    {
        if (driftCompensation) free(refPos);
        free(slipx);
        freeBoxes(boxes);
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate slipy in slipFilter");
        return NULL;
    }
    
    slipz = calloc(NVisibleIn, sizeof(double));
    if (slipz == NULL);
    {
        if (driftCompensation) free(refPos);
        free(slipx);
        free(slipy);
        freeBoxes(boxes);
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate slipz in slipFilter");
        return NULL;
    }
    
    slippedAtoms = calloc(NVisibleIn, sizeof(int));
    if (slippedAtoms == NULL);
    {
        if (driftCompensation) free(refPos);
        free(slipx);
        free(slipy);
        free(slipz);
        freeBoxes(boxes);
        PyErr_SetString(PyExc_RuntimeError, "Could not allocate slippedAtoms in slipFilter");
        return NULL;
    }
    
    /* loop over visible atoms */
    for (i = 0; i < NVisibleIn; i++)
    {
        int j, index, index3, boxIndex, boxNebList[27];
        double refxposi, refyposi, refzposi;
        
        index = IIND1(visibleAtoms, i);
        index3 = index * 3;
        refxposi = DIND1(refPos, index3    );
        refyposi = DIND1(refPos, index3 + 1);
        refzposi = DIND1(refPos, index3 + 2);
        
        /* find box for ref pos of this visible atom */
        boxIndex = boxIndexOfAtom(refxposi, refyposi, refzposi, boxes);
        if (boxIndex < 0)
        {
            if (driftCompensation) free(refPos);
            free(slipx);
            free(slipy);
            free(slipz);
            free(slippedAtoms);
            freeBoxes(boxes);
            return NULL;
        }
        
        /* box neighbourhood */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over boxes */
        for (j = 0; j < 27; j++)
        {
            int k;
            
            boxIndex = boxNebList[j];
            if (boxIndex < 0)
            {
                if (driftCompensation) free(refPos);
                free(slipx);
                free(slipy);
                free(slipz);
                free(slippedAtoms);
                freeBoxes(boxes);
                return NULL;
            }
            
            /* loop over atoms in box */
            for (k = 0; k < boxes->boxNAtoms[boxIndex]; k++)
            {
                int visIndex, index2;
                
                visIndex = boxes->boxAtoms[boxIndex][k];
                index2 = IIND1(visibleAtoms, visIndex);
                
                if (index < index2)
                {
                    int index23;
                    double refxposj, refyposj, refzposj, sep2;
                    
                    index23 = index2 * 3;
                    refxposj = refPos[index23    ];
                    refyposj = refPos[index23 + 1];
                    refzposj = refPos[index23 + 2];
                    
                    /* separation between reference positions */
                    sep2 = atomicSeparation2(refxposi, refyposi, refzposi, refxposj, refyposj, refzposj,
                            cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
                    
                    /* why 9, should this be an input and/or linked to approxBoxWidth!!?? */
                    /* we only compare to atoms that were local in the reference */
                    if (sep2 < 9.0)
                    {
                        double sepVeci[3], sepVecj[3];
                        double dslipx, dslipy, dslipz, slipMag;
                        
                        /* separation vectors (input - ref) */
                        atomSeparationVector(sepVeci, refPos[index3], refPos[index3 + 1], refPos[index3 + 2],
                                DIND1(pos, index3), DIND1(pos, index3 + 1), DIND1(pos, index3 + 2),
                                cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
                        
                        atomSeparationVector(sepVecj, DIND1(pos, index23), DIND1(pos, index23 + 1), DIND1(pos, index23 + 2),
                                refPos[index23], refPos[index23 + 1], refPos[index23 + 2], cellDims[0], cellDims[1], cellDims[2], 
                                PBC[0], PBC[1], PBC[2]);
                        
                        /* slip */
                        dslipx = sepVeci[0] - sepVecj[0];
                        dslipy = sepVeci[1] - sepVecj[1];
                        dslipz = sepVeci[2] - sepVecj[2];
                        
                        slipMag = dslipx * dslipx + dslipy * dslipy + dslipz * dslipz;
                        if (slipMag > 0.09) // why 0.09!!??
                        {
                            slipx[i] += dslipx;
                            slipy[i] += dslipy;
                            slipz[i] += dslipz;
                            slippedAtoms[i]++;
                            
                            slipx[visIndex] += dslipx;
                            slipy[visIndex] += dslipy;
                            slipz[visIndex] += dslipz;
                            slippedAtoms[visIndex]++;
                        }
                    }
                }
            }
        }
    }
    
    /* store slip value */
    for (i = 0; i < NVisibleIn; i++)
    {
        if (slippedAtoms[i])
        {
            DIND1(scalars, i) = sqrt(slipx[i] * slipx[i] + slipy[i] * slipy[i] + slipz[i] * slipz[i]) / ((double) slippedAtoms[i]);
        }
        else
        {
            DIND1(scalars, i) = 0.0;
        }
    }
    
    /* filtering */
    if (filteringEnabled)
    {
        NVisible = 0;
        for (i = 0; i < NVisibleIn; i++)
        {
            if (DIND1(scalars, i) >= minSlip && DIND1(scalars, i) < maxSlip)
            {
                int j;
                
                /* handle full scalars array */
                for (j = 0; j < NScalars; j++)
                {
                    DIND1(fullScalars, NVisibleIn * j + NVisible) = DIND1(fullScalars, NVisibleIn * j + i);
                }
                
                /* handle full vectors array */
                for (j = 0; j < NVectors; j++)
                {
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 0) = DIND2(fullVectors, NVisibleIn * j + i, 0);
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 1) = DIND2(fullVectors, NVisibleIn * j + i, 1);
                    DIND2(fullVectors, NVisibleIn * j + NVisible, 2) = DIND2(fullVectors, NVisibleIn * j + i, 2);
                }
                
                DIND1(scalars, NVisible) = DIND1(scalars, i);
                IIND1(visibleAtoms, NVisible++) = IIND1(visibleAtoms, i);
            }
        }
    }
    
    /* free */
    free(slipx);
    free(slipy);
    free(slipz);
    free(slippedAtoms);
    freeBoxes(boxes);
    if (driftCompensation) free(refPos);
    else refPos = NULL;
    
#ifdef DEBUG
    printf("SLIPC: Leaving slip C lib\n");
#endif
    
    return Py_BuildValue("i", NVisible);
}
