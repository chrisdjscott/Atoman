
/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** Utility functions
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "utilities.h"


/*******************************************************************************
 ** returns specie index of given specie in the specie list
 *******************************************************************************/
int getSpecieIndex( int NSpecies, char* specie, char* specieList )
{
    int i, comp, specieIndex;
    
    for ( i=0; i<NSpecies; i++ )
    {
        comp = strcmp( &specie[0], &specieList[3*i] );
        if ( comp == 0 )
        {
            specieIndex = i;
            break;
        }
    }
    
    return specieIndex;
}


/*******************************************************************************
 * Find separation vector between two atoms
 *******************************************************************************/
void atomSeparationVector( double *vector3, double ax, double ay, double az, double bx, double by, double bz, double xdim, double ydim, double zdim, int pbcx, int pbcy, int pbcz )
{
    double dx, dy, dz;
    
    
    /* calculate separation */
    dx = bx - ax;
    dy = by - ay;
    dz = bz - az;
    
    /* handle PBCs here if required */
    if ( pbcx == 1 )
    {
        dx = dx - round( dx / xdim ) * xdim;
    }
    if ( pbcy == 1 )
    {
        dy = dy - round( dy / ydim ) * ydim;
    }
    if ( pbcz == 1 )
    {
        dz = dz - round( dz / zdim ) * zdim;
    }
    
    vector3[0] = dx;
    vector3[1] = dy;
    vector3[2] = dz;
}


/*******************************************************************************
 ** return atomic separation squared
 *******************************************************************************/
double atomicSeparation2( double ax, double ay, double az, double bx, double by, double bz, double xdim, double ydim, double zdim, int pbcx, int pbcy, int pbcz )
{
    double rx, ry, rz, r2;
    
    /* calculate separation */
    rx = ax - bx;
    ry = ay - by;
    rz = az - bz;
    
    /* handle PBCs here if required */
    if ( pbcx == 1 )
    {
        rx = rx - round( rx / xdim ) * xdim;
    }
    if ( pbcy == 1 )
    {
        ry = ry - round( ry / ydim ) * ydim;
    }
    if ( pbcz == 1 )
    {
        rz = rz - round( rz / zdim ) * zdim;
    }
    
    /* separation squared */
    r2 = rx * rx + ry * ry + rz * rz;
    
    return r2;
}


/*******************************************************************************
 ** return atomic separation squared, with check if PBCs were applied
 *******************************************************************************/
double atomicSeparation2PBCCheck( double ax, double ay, double az, 
                          double bx, double by, double bz, 
                          double xdim, double ydim, double zdim, 
                          int pbcx, int pbcy, int pbcz,
                          int *appliedPBCs )
{
    double rx, ry, rz, r2;
    double rxini, ryini, rzini;
    
    /* calculate separation */
    rxini = ax - bx;
    ryini = ay - by;
    rzini = az - bz;
    
    /* handle PBCs here if required */
    if ( pbcx == 1 )
    {
        rx = rxini - round( rxini / xdim ) * xdim;
    }
    else
    {
        rx = rxini;
    }
    
    if ( pbcy == 1 )
    {
        ry = ryini - round( ryini / ydim ) * ydim;
    }
    else
    {
        ry = ryini;
    }
    
    if ( pbcz == 1 )
    {
        rz = rzini - round( rzini / zdim ) * zdim;
    }
    else
    {
        rz = rzini;
    }
    
    if (rx != rxini)
    {
        appliedPBCs[0] = 1;
    }
    else
    {
        appliedPBCs[0] = 0;
    }
    
    if (ry != ryini)
    {
        appliedPBCs[1] = 1;
    }
    else
    {
        appliedPBCs[1] = 0;
    }
    
    if (rz != rzini)
    {
        appliedPBCs[2] = 1;
    }
    else
    {
        appliedPBCs[2] = 0;
    }
    
    /* separation squared */
    r2 = rx * rx + ry * ry + rz * rz;
    
    return r2;
}
