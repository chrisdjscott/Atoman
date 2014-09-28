
//#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
//#include <numpy/arrayobject.h>
#include "stdlib.h"
#include "math.h"
#include "voro_iface.h"
#include "voro++.hh"
#include <vector>

using namespace voro;

static int processAtomCell(voronoicell_neighbor&, int, double*, vorores_t*);


/*******************************************************************************
 * Main interface function to be called from C
 *******************************************************************************/
extern "C" int computeVoronoiVoroPlusPlusWrapper(int NAtoms, double *pos, int *PBC, 
        double *bound_lo, double *bound_hi, int useRadii, double *radii, vorores_t *voroResult)
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
    
    // voro cell with neighbour information
    voronoicell_neighbor c;
    
    // use radii or not
    if (useRadii)
    {
        /* initialise voro++ container, preallocates 8 atoms per cell */
        container_poly con(bound_lo[0], bound_hi[0],
                           bound_lo[1], bound_hi[1],
                           bound_lo[2], bound_hi[2],
                           int(n[0]),int(n[1]),int(n[2]),
                           bool(PBC[0]), bool(PBC[1]), bool(PBC[2]), 8); 
    
        // pass coordinates for local and ghost atoms to voro++
        for (i = 0; i < NAtoms; i++)
            con.put(i, pos[3*i], pos[3*i+1], pos[3*i+2], radii[i]);
        
        // invoke voro++ and fetch results for owned atoms in group
        int count = 0;
        c_loop_all cl(con);
        if (cl.start()) do if (con.compute_cell(c,cl))
        {
            i = cl.pid();
            processAtomCell(c, i, pos, voroResult);
            count++;
        } while (cl.inc());
        
        printf("DEBUG: COUNT = %d (%d atoms)\n", count, NAtoms);
    }
    else
    {
        /* initialise voro++ container, preallocates 8 atoms per cell */
        container con(bound_lo[0], bound_hi[0],
                      bound_lo[1], bound_hi[1],
                      bound_lo[2], bound_hi[2],
                      int(n[0]),int(n[1]),int(n[2]),
                      bool(PBC[0]), bool(PBC[1]), bool(PBC[2]), 8); 
    
        // pass coordinates for local and ghost atoms to voro++
        for (i = 0; i < NAtoms; i++)
            con.put(i, pos[3*i], pos[3*i+1], pos[3*i+2]);
        
        // invoke voro++ and fetch results for owned atoms in group
        int count = 0;
        int errcnt = 0;
        c_loop_all cl(con);
        if (cl.start()) do if (con.compute_cell(c,cl))
        {
            int retval;
            
            i = cl.pid();
            retval = processAtomCell(c, i, pos, voroResult);
            
            if (retval)
            {
                errcnt++;
            }
            
            count++;
        } while (cl.inc());
        printf("DEBUG: COUNT = %d (%d atoms)\n", count, NAtoms);
        
        /* return error */
        if (errcnt) return -1;
    }
    
    return 0;
}

/*******************************************************************************
 * Process cell; compute volume, num neighbours, facets etc.
 *******************************************************************************/
static int processAtomCell(voronoicell_neighbor &c, int i, double *pos, vorores_t *voroResult)
{
    /* initialise result */
    voroResult[i].numFaces = 0;
    voroResult[i].numNeighbours = 0;
    voroResult[i].numVertices = 0;
    
    /* volume */
    voroResult[i].volume = c.volume();
    
    /* number of neighbours (should add threshold) */
    std::vector<int> neighbours;
    c.neighbors(neighbours);
    int nnebs = neighbours.size(); // this will change if add threshold...
    voroResult[i].neighbours = (int*) malloc(nnebs * sizeof(int));
    if (voroResult[i].neighbours == NULL) return -1;
    for (int j = 0; j < nnebs; j++)
        voroResult[i].neighbours[j] = neighbours[j];
    voroResult[i].numNeighbours = nnebs;
    
    /* vertices */
    std::vector<double> vertices;
    c.vertices(pos[3*i], pos[3*i+1], pos[3*i+2], vertices);
    int nvertices = c.p;
    voroResult[i].vertices = (double*) malloc(3 * nvertices * sizeof(double));
    if (voroResult[i].vertices == NULL) return -1;
    for (int j = 0; j < nvertices; j++)
    {
        int j3 = 3 * j;
        voroResult[i].vertices[j3] = vertices[j3];
        voroResult[i].vertices[j3+1] = vertices[j3+1];
        voroResult[i].vertices[j3+2] = vertices[j3+2];
    }
    voroResult[i].numVertices = nvertices;
    
    /* faces */
    std::vector<int> faceVertices;
    c.face_vertices(faceVertices);
    int nfaces = c.number_of_faces();
    
    voroResult[i].numFaceVertices = (int*) calloc(nfaces, sizeof(int));
    if (voroResult[i].numFaceVertices == NULL) return -1;
    
    voroResult[i].faceVertices = (int**) malloc(nfaces * sizeof(int*));
    if (voroResult[i].faceVertices == NULL) return -1;
    
    voroResult[i].numFaces = nfaces;
    
    int count = 0;
    for (int j = 0; j < nfaces; j++)
    {
        int nfaceverts = faceVertices[count++];
        
        /* allocate space for this faces vertices */
        voroResult[i].faceVertices[j] = (int*) malloc(nfaceverts * sizeof(int));
        if (voroResult[i].faceVertices[j] == NULL) return -1;
        voroResult[i].numFaceVertices[j] = nfaceverts;
        
        for (int k = 0; k < nfaceverts; k++)
            voroResult[i].faceVertices[j][k] = faceVertices[count++];
    }
    
    /* original position */
    voroResult[i].originalPos[0] = pos[3*i];
    voroResult[i].originalPos[1] = pos[3*i+1];
    voroResult[i].originalPos[2] = pos[3*i+2];
    
    return 0;
}
