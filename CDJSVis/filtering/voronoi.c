
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** Helper methods for computing Voronoi cells/volumes
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "array_utils.h"
#include "voro_iface.h"


static PyObject* makeVoronoiPoints(PyObject*, PyObject*);
//static PyObject* computeVolumes(PyObject*, PyObject*);
static PyObject* computeVoronoiVoroPlusPlus(PyObject*, PyObject*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"makeVoronoiPoints", makeVoronoiPoints, METH_VARARGS, "Make points array for passing to Voronoi method"},
//    {"computeVolumes", computeVolumes, METH_VARARGS, "Compute Voronoi volumes of the atoms"},
    {"computeVoronoiVoroPlusPlus", computeVoronoiVoroPlusPlus, METH_VARARGS, "Compute Voronoi volumes of the atoms using Voro++ interface"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_voronoi(void)
{
    (void)Py_InitModule("_voronoi", methods);
    import_array();
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

/*******************************************************************************
 * Compute Voronoi volumes of atoms
 *******************************************************************************/
//static PyObject*
//computeVolumes(PyObject *self, PyObject *args)
//{
//    int NAtoms, int *point_region, 
//    
//    
//    
//    
//    
//    
//}

/*******************************************************************************
 * Compute Voronoi using Voro++
 *******************************************************************************/
static PyObject*
computeVoronoiVoroPlusPlus(PyObject *self, PyObject *args)
{
    int *specie, *PBC, *nebCounts, NAtoms, useRadii;
    double *pos, *minPos, *maxPos, *cellDims, *specieCovalentRadius, dispersion, *volumes;
    PyArrayObject *posIn=NULL;
    PyArrayObject *minPosIn=NULL;
    PyArrayObject *maxPosIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *specieCovalentRadiusIn=NULL;
    PyArrayObject *volumesIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *nebCountsIn=NULL;
    int i, status;
    double bound_lo[3], bound_hi[3];
    double *radii;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!diO!O!", &PyArray_Type, &posIn, &PyArray_Type, &minPosIn, &PyArray_Type, 
            &maxPosIn, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &PyArray_Type, &specieIn, &PyArray_Type, 
            &specieCovalentRadiusIn, &dispersion, &useRadii, &PyArray_Type, &volumesIn, &PyArray_Type, &nebCountsIn))
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
    
    if (not_doubleVector(volumesIn)) return NULL;
    volumes = pyvector_to_Cptr_double(volumesIn);
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_intVector(nebCountsIn)) return NULL;
    nebCounts = pyvector_to_Cptr_int(nebCountsIn);
    
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
    
    /* call voro++ wrapper */
    /* need to pass extra stuff eventually, eg radii etc... */
    status = computeVoronoiVoroPlusPlusWrapper(NAtoms, pos, PBC, bound_lo, bound_hi, volumes, nebCounts);
    
    
    
    
    
    
    
    if (useRadii) free(radii);
    
    return Py_BuildValue("i", 0);
}


