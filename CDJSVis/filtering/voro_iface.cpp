
#include "stdlib.h"
#include "math.h"
#include "voro_iface.h"
#include "voro++.hh"

#include <vector>

using namespace voro;

static void processCell(voronoicell_neighbor&, int, double*, int*);


/*******************************************************************************
 * Main interface function to be called from C
 *******************************************************************************/
extern "C" int computeVoronoiVoroPlusPlusWrapper(int NAtoms, double *pos, int *PBC, double *bound_lo, double *bound_hi, 
        double *volumes, int *nebCounts)
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
	  processCell(c, i, volumes, nebCounts);
	} while (cl.inc());
    
    return 0;
}

/*******************************************************************************
 * Process cell; compute volume, num neighbours, facets etc.
 *******************************************************************************/
static void processCell(voronoicell_neighbor &c, int i, double *volumes, int *nebCounts)
{
	std::vector<int> neigh;
	
	// volume
	volumes[i] = c.volume();
	
	// number of cell faces (should add threshold)
	c.neighbors(neigh);
	nebCounts[i] = neigh.size();
}
