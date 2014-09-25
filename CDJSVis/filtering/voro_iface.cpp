
#include "stdlib.h"
#include "math.h"
#include "voro_iface.h"
#include "voro++.hh"

#include <vector>

using namespace voro;


extern "C" int computeVoronoiVoroPlusPlusWrapper(int NAtoms, double *pos, int *PBC, double *bounds_lo, double *bounds_hi, 
        double *volumes, int *nebCounts)
{
    
    /* number of cells for spatial decomposition */
    double n[3];
    for (int i = 0; i < 3; i++) n[i] = bounds_hi[i] - bounds_lo[i];
    double V = n[0] * n[1] * n[2];
    for (int i = 0; i < 3; i++)
    {
        n[i] = round(n[i] * pow(double(NAtoms) / (V * 8.0), 0.333333));
        n[i] = n[i] == 0 ? 1 : n[i];
        printf("DEBUG: n[%d] = %d\n", i, n[i]);
    }
    
    /* initialise voro++ container, preallocates 8 atoms per cell */
    
    
    
    return 0;
}
