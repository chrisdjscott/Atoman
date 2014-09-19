
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** Filtering routines written in C to improve performance
 *******************************************************************************/

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
	int NVisibleIn, *visibleAtoms, visSpecDim, *visSpec, *specie, NScalars;
	double *fullScalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *visSpecIn=NULL;
	PyArrayObject *specieIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	
    int i, NVisible;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!O!iO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &visSpecIn, 
			&PyArray_Type, &specieIn, &NScalars, &PyArray_Type, &fullScalarsIn))
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
	int NVisibleIn, *visibleAtoms, invert, NScalars;
	double *pos, x0, y0, z0, xn, yn, zn, *fullScalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *posIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	
    int i, j, NVisible, index;
    double mag, xd, yd, zd, dotProd, distanceToPlane;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!ddddddiiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, 
			&x0, &y0, &z0, &xn, &yn, &zn, &invert, &NScalars, &PyArray_Type, &fullScalarsIn))
		return NULL;
	
	if (not_intVector(visibleAtomsIn)) return NULL;
	visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
	NVisibleIn = (int) visibleAtomsIn->dimensions[0];
	
	if (not_doubleVector(posIn)) return NULL;
	pos = pyvector_to_Cptr_double(posIn);
	
	if (not_doubleVector(fullScalarsIn)) return NULL;
	fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
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
            
			visibleAtoms[NVisible++] = index;
        }
    }
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 ** Crop sphere filter
 *******************************************************************************/
static PyObject* 
cropSphereFilter(PyObject *self, PyObject *args)
{
	int NVisibleIn, *visibleAtoms, *PBC, invertSelection, NScalars;
	double *pos, xCentre, yCentre, zCentre, radius, *cellDims, *fullScalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *posIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	PyArrayObject *PBCIn=NULL;
	PyArrayObject *cellDimsIn=NULL;
	
    int i, j, NVisible, index;
    double radius2, sep2;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!ddddO!O!iiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, 
			&xCentre, &yCentre, &zCentre, &radius, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn,
			&invertSelection, &NScalars, &PyArray_Type, &fullScalarsIn))
		return NULL;
	
	if (not_intVector(visibleAtomsIn)) return NULL;
	visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
	NVisibleIn = (int) visibleAtomsIn->dimensions[0];
	
	if (not_doubleVector(posIn)) return NULL;
	pos = pyvector_to_Cptr_double(posIn);
	
	if (not_doubleVector(fullScalarsIn)) return NULL;
	fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
	
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
                
				visibleAtoms[NVisible++] = index;
            }
        }
    }
    
    return Py_BuildValue("i", NVisible);
}


/*******************************************************************************
 ** Crop filter
 *******************************************************************************/
static PyObject* 
cropFilter(PyObject *self, PyObject *args)
{
	int NVisibleIn, *visibleAtoms, xEnabled, yEnabled, zEnabled, invertSelection, NScalars;
	double *pos, xmin, xmax, ymin, ymax, zmin, zmax, *fullScalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *posIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	
    int i, j, index, NVisible, add;
    double rx, ry, rz;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!ddddddiiiiiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, 
			&xmin, &xmax, &ymin, &ymax, &zmin, &zmax, &xEnabled, &yEnabled, &zEnabled, &invertSelection, 
			&NScalars, &PyArray_Type, &fullScalarsIn))
		return NULL;
	
	if (not_intVector(visibleAtomsIn)) return NULL;
	visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
	NVisibleIn = (int) visibleAtomsIn->dimensions[0];
	
	if (not_doubleVector(posIn)) return NULL;
	pos = pyvector_to_Cptr_double(posIn);
	
	if (not_doubleVector(fullScalarsIn)) return NULL;
	fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
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
	int NVisibleIn, *visibleAtoms, *PBC, NScalars, filteringEnabled, driftCompensation, refPosDim;
	double *scalars, *pos, *refPosIn, *cellDims, minDisp, maxDisp, *fullScalars, *driftVector;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *refPosIn_np=NULL;
	PyArrayObject *PBCIn=NULL;
	PyArrayObject *posIn=NULL;
	PyArrayObject *scalarsIn=NULL;
	PyArrayObject *driftVectorIn=NULL;
	PyArrayObject *cellDimsIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	
    int i, NVisible, index, j;
    double sep2, maxDisp2, minDisp2;
    double *refPos;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!ddiO!iiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &scalarsIn, 
			&PyArray_Type, &posIn, &PyArray_Type, &refPosIn_np, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, 
			&minDisp, &maxDisp, &NScalars, &PyArray_Type, &fullScalarsIn, &filteringEnabled, &driftCompensation, 
			&PyArray_Type, &driftVectorIn))
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
    
    /* drift compensation? */
    if (driftCompensation)
    {
        refPos = malloc(refPosDim * sizeof(double));
        if (refPos == NULL)
        {
            printf("ERROR: could not allocate refPos\n");
            exit(34);
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
	int NVisibleIn, *visibleAtoms, NScalars;
	double *KE, minKE, maxKE, *fullScalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *KEIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!ddiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &KEIn, 
			&minKE, &maxKE, &NScalars, &PyArray_Type, &fullScalarsIn))
		return NULL;
	
	if (not_intVector(visibleAtomsIn)) return NULL;
	visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
	NVisibleIn = (int) visibleAtomsIn->dimensions[0];
	
	if (not_doubleVector(KEIn)) return NULL;
	KE = pyvector_to_Cptr_double(KEIn);
	
	if (not_doubleVector(fullScalarsIn)) return NULL;
	fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
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
	int NVisibleIn, *visibleAtoms, NScalars;
	double *PE, minPE, maxPE, *fullScalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *PEIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!ddiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &PEIn, 
			&minPE, &maxPE, &NScalars, &PyArray_Type, &fullScalarsIn))
		return NULL;
	
	if (not_intVector(visibleAtomsIn)) return NULL;
	visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
	NVisibleIn = (int) visibleAtomsIn->dimensions[0];
	
	if (not_doubleVector(PEIn)) return NULL;
	PE = pyvector_to_Cptr_double(PEIn);
	
	if (not_doubleVector(fullScalarsIn)) return NULL;
	fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
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
	int NVisibleIn, *visibleAtoms, NScalars;
	double *charge, minCharge, maxCharge, *fullScalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *chargeIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!ddiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &chargeIn, 
			&minCharge, &maxCharge, &NScalars, &PyArray_Type, &fullScalarsIn))
		return NULL;
	
	if (not_intVector(visibleAtomsIn)) return NULL;
	visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
	NVisibleIn = (int) visibleAtomsIn->dimensions[0];
	
	if (not_doubleVector(chargeIn)) return NULL;
	charge = pyvector_to_Cptr_double(chargeIn);
	
	if (not_doubleVector(fullScalarsIn)) return NULL;
	fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
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
	int NVisible, *visibleAtoms, *specie, NSpecies, *PBC, minCoordNum, maxCoordNum, NScalars, filteringEnabled;
	double *pos, *bondMinArray, *bondMaxArray, approxBoxWidth, *cellDims, *minPos, *maxPos, *coordArray, *fullScalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *specieIn=NULL;
	PyArrayObject *PBCIn=NULL;
	PyArrayObject *coordArrayIn=NULL;
	PyArrayObject *posIn=NULL;
	PyArrayObject *bondMinArrayIn=NULL;
	PyArrayObject *bondMaxArrayIn=NULL;
	PyArrayObject *cellDimsIn=NULL;
	PyArrayObject *minPosIn=NULL;
	PyArrayObject *maxPosIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	
    int i, j, k, index, index2, visIndex;
    int speca, specb, count, NVisibleNew;
    int boxIndex, boxNebList[27];
    double *visiblePos, sep2, sep;
    struct Boxes *boxes;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!O!iO!O!dO!O!O!O!O!iiiO!i", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, 
			&PyArray_Type, &specieIn, &NSpecies, &PyArray_Type, &bondMinArrayIn, &PyArray_Type, &bondMaxArrayIn, 
			&approxBoxWidth, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &PyArray_Type, &minPosIn, 
			&PyArray_Type, &maxPosIn, &PyArray_Type, &coordArrayIn, &minCoordNum, &maxCoordNum, &NScalars, 
			&PyArray_Type, &fullScalarsIn, &filteringEnabled))
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
	
	if (not_doubleVector(minPosIn)) return NULL;
	minPos = pyvector_to_Cptr_double(minPosIn);
	
	if (not_doubleVector(maxPosIn)) return NULL;
	maxPos = pyvector_to_Cptr_double(maxPosIn);
	
	if (not_doubleVector(cellDimsIn)) return NULL;
	cellDims = pyvector_to_Cptr_double(cellDimsIn);
	
	if (not_intVector(PBCIn)) return NULL;
	PBC = pyvector_to_Cptr_int(PBCIn);
	
	if (not_doubleVector(coordArrayIn)) return NULL;
	coordArray = pyvector_to_Cptr_double(coordArrayIn);
    
	if (not_doubleVector(fullScalarsIn)) return NULL;
	fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
//    printf("BONDS CLIB\n");
//    printf("N VIS: %d\n", NVisible);
//    
//    for (i=0; i<NSpecies; i++)
//    {
//        for (j=i; j<NSpecies; j++)
//        {
//            printf("%d - %d: %lf -> %lf\n", i, j, bondMinArray[i*NSpecies+j], bondMaxArray[i*NSpecies+j]);
//        }
//    }
    
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
        
        visiblePos[3*i] = pos[3*index];
        visiblePos[3*i+1] = pos[3*index+1];
        visiblePos[3*i+2] = pos[3*index+2];
    }
    
    /* box visible atoms */
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(NVisible, visiblePos, boxes);
    
    /* free visible pos */
    free(visiblePos);
    
    /* zero coord array */
    for (i=0; i<NVisible; i++)
    {
        coordArray[i] = 0;
    }
    
    /* loop over visible atoms */
    count = 0;
    for (i=0; i<NVisible; i++)
    {
        index = visibleAtoms[i];
        
        speca = specie[index];
        
        /* get box index of this atom */
        boxIndex = boxIndexOfAtom(pos[3*index], pos[3*index+1], pos[3*index+2], boxes);
        
        /* find neighbouring boxes */
        getBoxNeighbourhood(boxIndex, boxNebList, boxes);
        
        /* loop over box neighbourhood */
        for (j=0; j<27; j++)
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
	int NVisibleIn, *visibleAtoms, NScalars, filteringEnabled;
	double *volume, minVolume, maxVolume, *fullScalars, *scalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *volumeIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	PyArrayObject *scalarsIn=NULL;
	
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!ddO!iO!i", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &volumeIn, 
			&minVolume, &maxVolume, &PyArray_Type, &scalarsIn, &NScalars, &PyArray_Type, &fullScalarsIn,
			&filteringEnabled))
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
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (!filteringEnabled || (volume[index] >= minVolume && volume[index] <= maxVolume))
        {
            /* handle full scalars array */
			for (j = 0; j < NScalars; j++)
				fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
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
	int NVisibleIn, *visibleAtoms, NScalars, filteringEnabled, *num_nebs_array, minNebs, maxNebs;
	double *fullScalars, *scalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *num_nebs_arrayIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	PyArrayObject *scalarsIn=NULL;
	
    int i, j, NVisible, index;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!iiO!iO!i", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &num_nebs_arrayIn, 
			&minNebs, &maxNebs, &PyArray_Type, &scalarsIn, &NScalars, &PyArray_Type, &fullScalarsIn,
			&filteringEnabled))
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
    
    NVisible = 0;
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        if (!filteringEnabled || (num_nebs_array[index] >= minNebs && num_nebs_array[index] <= maxNebs))
        {
            /* handle full scalars array */
			for (j = 0; j < NScalars; j++)
				fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
            
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
	int NVisibleIn, *visibleAtoms, NScalars, filteringEnabled, *atomID, minVal, maxVal;
	double *fullScalars;
	PyArrayObject *visibleAtomsIn=NULL;
	PyArrayObject *atomIDIn=NULL;
	PyArrayObject *fullScalarsIn=NULL;
	
    int i, id, index, NVisible, j;
    
    /* parse and check arguments from Python */
	if (!PyArg_ParseTuple(args, "O!O!iiiiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &atomIDIn, 
			&filteringEnabled, &minVal, &maxVal, &NScalars, &PyArray_Type, &fullScalarsIn))
		return NULL;
	
	if (not_intVector(visibleAtomsIn)) return NULL;
	visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
	NVisibleIn = (int) visibleAtomsIn->dimensions[0];
	
	if (not_intVector(atomIDIn)) return NULL;
	atomID = pyvector_to_Cptr_int(atomIDIn);
	
	if (not_doubleVector(fullScalarsIn)) return NULL;
	fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    NVisible = 0;
    for (i = 0; i < NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        id = atomID[index];
        if (filteringEnabled && (id < minVal || id > maxVal))
            continue;
        
        /* handle full scalars array */
        for (j = 0; j < NScalars; j++)
        {
            fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
        }
        
        visibleAtoms[NVisible++] = index;
    }
    
    return Py_BuildValue("i", NVisible);
}
