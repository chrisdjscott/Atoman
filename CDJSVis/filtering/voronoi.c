
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** Helper methods for computing Voronoi cells/volumes
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "array_utils.h"


static PyObject* makeVoronoiPoints(PyObject*, PyObject*);


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
    (void)Py_InitModule("_voronoi", methods);
    import_array();
}

/*******************************************************************************
 * Calculate bonds
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
