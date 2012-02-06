
/*******************************************************************************
 ** Copyright Chris Scott 2011
 ** IO routines written in C to improve performance
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


/*******************************************************************************
** read animation-reference file
*******************************************************************************/
void readRef( char* file, int dim6, char* sym, int dim1, double* pos, int dim2, double* charge, int dim3, double* KE, int dim4, double* PE, int dim5, double* force, int dim7, char* specieList_c, int dim8, int* specieCount_c, int dim15, double* maxPos, int dim16, double* minPos )
{
    int i, j, NAtoms;
    FILE *INFILE;
    double xdim, ydim, zdim;
    char* symtemp;
    char* specieList;
    double xpos, ypos, zpos;
    double xforce, yforce, zforce;
    int id, index;
    double ketemp, petemp, chargetemp;
    int NSpecies, comp, specieMatch;
    
    printf("CLIB: reading ref file %s\n", file);
    
    INFILE = fopen( file, "r" );
    
    fscanf( INFILE, "%d", &NAtoms );
    printf("  %d atoms\n", NAtoms);
    
    fscanf(INFILE, "%lf%lf%lf", &xdim, &ydim, &zdim);
    
    symtemp = malloc(3 * sizeof(char));
    specieList = malloc( 3 * sizeof(char) );
    
    minPos[0] = 1000000;
    minPos[1] = 1000000;
    minPos[2] = 1000000;
    maxPos[0] = -1000000;
    maxPos[1] = -1000000;
    maxPos[2] = -1000000;
    NSpecies = 0;
    for (i=0; i<NAtoms; i++)
    {
        fscanf(INFILE, "%d%s%lf%lf%lf%lf%lf%lf%lf%lf%lf", &id, symtemp, &xpos, &ypos, &zpos, &ketemp, &petemp, &xforce, &yforce, &zforce, &chargetemp);
        
        /* index for storage is (id-1) */
        index = id - 1;
        
        /* store atom info */
        sym[2*index] = symtemp[0];
        sym[2*index+1] = symtemp[1];
        
        pos[3*index] = xpos;
        pos[3*index+1] = ypos;
        pos[3*index+2] = zpos;
        
        KE[index] = ketemp;
        PE[index] = petemp;
        
//        force[3*index] = xforce;
//        force[3*index+1] = yforce;
//        force[3*index+2] = zforce;
        
        charge[index] = chargetemp;
        
        /* max and min positions */
        if ( xpos > maxPos[0] )
        {
            maxPos[0] = xpos;
        }
        if ( ypos > maxPos[1] )
        {
            maxPos[1] = ypos;
        }
        if ( zpos > maxPos[2] )
        {
            maxPos[2] = zpos;
        }
        if ( xpos < minPos[0] )
        {
            minPos[0] = xpos;
        }
        if ( ypos < minPos[1] )
        {
            minPos[1] = ypos;
        }
        if ( zpos < minPos[2] )
        {
            minPos[2] = zpos;
        }
        
        /* update specie list if required */
        if ( NSpecies == 0 )
        {
            specieList[0] = symtemp[0];
            specieList[1] = symtemp[1];
            specieList[2] = symtemp[2];
            
            specieList_c[0] = symtemp[0];
            specieList_c[1] = symtemp[1];
            
            specieCount_c[0] = 1;
            
            printf("  found 1st specie: %s\n", &specieList[3*NSpecies]);
            NSpecies++;
        }
        else
        {
            specieMatch = 0;
            for (j=0; j<NSpecies; j++)
            {
                comp = strcmp( &specieList[3*j], &symtemp[0] );
                if (comp == 0)
                {
                    specieMatch++;
                    
                    specieCount_c[j]++;
                    
                    break;
                }
            }
            if ( specieMatch == 0 )
            {
                /* new specie */
                specieList = realloc( specieList, 3 * (NSpecies+1) * sizeof(char) );
                specieList[3*NSpecies] = symtemp[0];
                specieList[3*NSpecies+1] = symtemp[1];
                specieList[3*NSpecies+2] = symtemp[2];
                
                specieList_c[2*NSpecies] = symtemp[0];
                specieList_c[2*NSpecies+1] = symtemp[1];
                
                specieCount_c[NSpecies] = 1;
                
                printf("  found new specie: %s\n", &specieList[3*NSpecies]);
                NSpecies++;
            }
        }
    }
    
    fclose(INFILE);
    
    /* terminate specie list */
    specieList_c[2*NSpecies] = 'X';
    specieList_c[2*NSpecies+1] = 'X';
    
    free(symtemp);
    free(specieList);
    
    printf("  x range is %f -> %f\n", minPos[0], maxPos[0]);
    printf("  y range is %f -> %f\n", minPos[1], maxPos[1]);
    printf("  z range is %f -> %f\n", minPos[2], maxPos[2]);
    
    printf("END CLIB\n");
}


/*******************************************************************************
 * Read LBOMD lattice file
 *******************************************************************************/
void readLatticeLBOMD( char* file, int dim6, char* sym, int dim1, double* pos, int dim2, double* charge, int dim7, char* specieList_c, int dim8, int* specieCount_c, int dim15, double* maxPos, int dim16, double* minPos, int verboseLevel )
{
    FILE *INFILE;
    int i, j, NAtoms;
    double xdim, ydim, zdim;
    char symtemp[3];
    char* specieList;
    double xpos, ypos, zpos, chargetemp;
    int NSpecies, comp, specieMatch;
        
    /* open file */
    INFILE = fopen( file, "r" );
    
    /* read header */
    fscanf( INFILE, "%d", &NAtoms );
    if (verboseLevel >= 2)
        printf("  %d atoms\n", NAtoms);
    fscanf(INFILE, "%lf%lf%lf", &xdim, &ydim, &zdim);
    
    /* allocate specieList */
    specieList = malloc( 3 * sizeof(char) );
    
    /* read in atoms */
    minPos[0] = 1000000;
    minPos[1] = 1000000;
    minPos[2] = 1000000;
    maxPos[0] = -1000000;
    maxPos[1] = -1000000;
    maxPos[2] = -1000000;
    NSpecies = 0;
    for (i=0; i<NAtoms; i++)
    {
        fscanf(INFILE, "%s%lf%lf%lf%lf", symtemp, &xpos, &ypos, &zpos, &chargetemp);
                
        /* store atom info */
        sym[2*i] = symtemp[0];
        sym[2*i+1] = symtemp[1];
        
        pos[3*i] = xpos;
        pos[3*i+1] = ypos;
        pos[3*i+2] = zpos;
        
        charge[i] = chargetemp;
        
        /* max and min positions */
        if ( xpos > maxPos[0] )
        {
            maxPos[0] = xpos;
        }
        if ( ypos > maxPos[1] )
        {
            maxPos[1] = ypos;
        }
        if ( zpos > maxPos[2] )
        {
            maxPos[2] = zpos;
        }
        if ( xpos < minPos[0] )
        {
            minPos[0] = xpos;
        }
        if ( ypos < minPos[1] )
        {
            minPos[1] = ypos;
        }
        if ( zpos < minPos[2] )
        {
            minPos[2] = zpos;
        }
        
        /* update specie list if required */
        if ( NSpecies == 0 )
        {
            specieList[0] = symtemp[0];
            specieList[1] = symtemp[1];
            specieList[2] = symtemp[2];
            
            specieList_c[0] = symtemp[0];
            specieList_c[1] = symtemp[1];
            
            specieCount_c[0] = 1;
            
            if (verboseLevel >= 2)
                printf("  found 1st specie: %s\n", &specieList[3*NSpecies]);
            NSpecies++;
        }
        else
        {
            specieMatch = 0;
            for (j=0; j<NSpecies; j++)
            {
                comp = strcmp( &specieList[3*j], &symtemp[0] );
                if (comp == 0)
                {
                    specieMatch++;
                    
                    specieCount_c[j]++;
                    
                    break;
                }
            }
            if ( specieMatch == 0 )
            {
                /* new specie */
                specieList = realloc( specieList, 3 * (NSpecies+1) * sizeof(char) );
                specieList[3*NSpecies] = symtemp[0];
                specieList[3*NSpecies+1] = symtemp[1];
                specieList[3*NSpecies+2] = symtemp[2];
                
                specieList_c[2*NSpecies] = symtemp[0];
                specieList_c[2*NSpecies+1] = symtemp[1];
                
                specieCount_c[NSpecies] = 1;
                
                if (verboseLevel >= 2)
                    printf("  found new specie: %s\n", &specieList[3*NSpecies]);
                NSpecies++;
            }
        }
    }
    
    fclose(INFILE);
    
    /* terminate specie list */
    specieList_c[2*NSpecies] = 'X';
    specieList_c[2*NSpecies+1] = 'X';
        
    free(specieList);
    
    if (verboseLevel >= 2)
        printf("END CLIB\n");
}


/*******************************************************************************
** write LBOMD lattice file
*******************************************************************************/
void writeLatticeLBOMD( char* file, int NAtoms, double xdim, double ydim, double zdim, int dim6, char* sym, int dim1, double* pos, int dim2, double* charge )
{
    int i;
    FILE *OUTFILE;
    char symtemp[3];
    
    
//    printf("CLIB: writing lattice file: %s\n", file);
    
    OUTFILE = fopen( file, "w" );
    
    fprintf(OUTFILE, "%d\n", NAtoms);
    fprintf(OUTFILE, "%f %f %f\n", xdim, ydim, zdim);
    
    for ( i=0; i<NAtoms; i++ )
    {
        symtemp[0] = sym[2*i];
        symtemp[1] = sym[2*i+1];
        symtemp[2] = '\0';
        
        fprintf( OUTFILE, "%s %f %f %f %f\n", &symtemp[0], pos[3*i], pos[3*i+1], pos[3*i+2], charge[i] );
    }
    
    fclose(OUTFILE);
    
//    printf("END CLIB\n");
}
