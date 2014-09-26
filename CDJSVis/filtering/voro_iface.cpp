
#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
//#include <numpy/arrayobject.h>
#include "math.h"
#include "voro_iface.h"
#include "voro++.hh"

#include <vector>

using namespace voro;

static void processAtomCell(voronoicell_neighbor&, int, double*, double*, int*, PyObject*);


/*******************************************************************************
 * Main interface function to be called from C
 *******************************************************************************/
extern "C" int computeVoronoiVoroPlusPlusWrapper(int NAtoms, double *pos, int *PBC, double *bound_lo, double *bound_hi, 
        double *volumes, int *nebCounts, PyObject *resultList)
{
    int i;
	
    /* number of cells for spatial decomposition */
    double n[3];
    for (i = 0; i < 3; i++) n[i] = bound_hi[i] - bound_lo[i];
    double V = n[0] * n[1] * n[2];
    for (i = 0; i < 3; i++)
    {
        n[i] = round(n[i] * pow(double(NAtoms) / (V * 8.0), 0.333333));
        n[i] = n[i] == 0 ? 1 : n[i];
        printf("DEBUG: n[%d] = %lf\n", i, n[i]);
    }
    
    /* initialise voro++ container, preallocates 8 atoms per cell */
    voronoicell_neighbor c;
    
    // monodisperse voro++ container
	container con(bound_lo[0], bound_hi[0],
				  bound_lo[1], bound_hi[1],
				  bound_lo[2], bound_hi[2],
				  int(n[0]),int(n[1]),int(n[2]),
				  bool(PBC[0]), bool(PBC[1]), bool(PBC[2]), 8); 

	// pass coordinates for local and ghost atoms to voro++
	for (i = 0; i < NAtoms; i++)
		con.put(i, pos[3*i], pos[3*i+1], pos[3*i+2]);

	// invoke voro++ and fetch results for owned atoms in group
	c_loop_all cl(con);
	if (cl.start()) do if (con.compute_cell(c,cl)) {
	  i = cl.pid();
	  processAtomCell(c, i, pos, volumes, nebCounts, resultList);
	} while (cl.inc());
    
    return 0;
}

/*******************************************************************************
 * Process cell; compute volume, num neighbours, facets etc.
 *******************************************************************************/
static void processAtomCell(voronoicell_neighbor &c, int i, double *pos, double *volumes, int *nebCounts, PyObject *resultList)
{
	/* create dictionary for this atoms results */
	PyObject *dict=NULL;
	dict = PyDict_New();
	
	// volume
	double vol = c.volume();
	volumes[i] = vol;
	
	PyObject *volPy=NULL;
	volPy = PyFloat_FromDouble(vol);
	PyDict_SetItemString(dict, "volume", volPy);
	Py_DECREF(volPy);
	
	// number of cell faces (should add threshold)
	std::vector<int> neighbours;
	c.neighbors(neighbours);
	nebCounts[i] = neighbours.size();
	
	
	// vertices
	std::vector<double> vertices;
	c.vertices(pos[3*i], pos[3*i+1], pos[3*i+2], vertices);
	int nvertices = c.p;
	
	PyObject *verticesList=NULL;
	verticesList = PyList_New(nvertices);
	for (int j = 0; j < nvertices; j++)
	{
	    PyObject *ptlist=NULL;
	    ptlist = PyList_New(3);
	    
	    PyObject *rx=NULL;
	    PyObject *ry=NULL;
	    PyObject *rz=NULL;
	    rx = PyFloat_FromDouble(vertices[3*j]);
	    ry = PyFloat_FromDouble(vertices[3*j+1]);
	    rz = PyFloat_FromDouble(vertices[3*j+2]);
	    
	    PyList_SetItem(ptlist, 0, rx);
	    PyList_SetItem(ptlist, 1, ry);
	    PyList_SetItem(ptlist, 2, rz);
	    
	    PyList_SetItem(verticesList, j, ptlist);
	}
	PyDict_SetItemString(dict, "vertices", verticesList);
	Py_DECREF(verticesList);
	
	// faces
	std::vector<int> faceVertices;
	c.face_vertices(faceVertices);
	int nfaces = c.number_of_faces();
	
	PyObject *facesList=NULL;
	facesList = PyList_New(nfaces);
	int count = 0;
	for (int j = 0; j < nfaces; j++)
	{
	    PyObject *facedict=NULL;
	    facedict = PyDict_New();
	    
	    int nfaceverts = faceVertices[count++];
	    PyObject *vertlist=NULL;
	    vertlist = PyList_New(nfaceverts);
	    for (int k = 0; k < nfaceverts; k++)
	    {
	        PyObject *vertindPy=NULL;
	        vertindPy = PyInt_FromLong(long(faceVertices[count++]));
	        PyList_SetItem(vertlist, k, vertindPy);
	    }
	    PyDict_SetItemString(facedict, "vertices", vertlist);
	    Py_DECREF(vertlist);
	    
	    PyObject *adjCellIndex=NULL;
	    adjCellIndex = PyInt_FromLong(long(neighbours[j]));
	    PyDict_SetItemString(facedict, "adjacent_cell", adjCellIndex);
	    Py_DECREF(adjCellIndex);
	    
	    PyList_SetItem(facesList, j, facedict);
	}
	PyDict_SetItemString(dict, "faces", facesList);
	Py_DECREF(facesList);
	
	// original position
	PyObject *origPos=NULL;
	origPos = PyList_New(3);
	PyObject *xpos=NULL;
	PyObject *ypos=NULL;
	PyObject *zpos=NULL;
	xpos = PyFloat_FromDouble(pos[3*i]);
	ypos = PyFloat_FromDouble(pos[3*i+1]);
	zpos = PyFloat_FromDouble(pos[3*i+2]);
	PyList_SetItem(origPos, 0, xpos);
	PyList_SetItem(origPos, 1, xpos);
	PyList_SetItem(origPos, 2, xpos);
	PyDict_SetItemString(dict, "original", origPos);
	
	/* add dict to result list (steals ref to dict) */
	PyList_SetItem(resultList, i, dict);
}

