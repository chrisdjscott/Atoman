
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** IO routines written in C to improve performance
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include "output.h"


static void addPOVRAYSphere(FILE *, double, double, double, double, double, double, double);
static void addPOVRAYCube(FILE *, double, double, double, double, double, double, double, double);
static void addPOVRAYCellFrame(FILE *, double, double, double, double, double, double, double, double, double);


/*******************************************************************************
 ** write sphere to pov-ray file
 *******************************************************************************/
static void addPOVRAYSphere(FILE *fp, double xpos, double ypos, double zpos, double radius, double R, double G, double B)
{
    fprintf(fp, "sphere { <%lf,%lf,%lf>, %lf pigment { color rgb <%lf,%lf,%lf> } finish { ambient %f phong %f } }\n",
            xpos, ypos, zpos, radius, R, G, B, 0.25, 0.9);
}


/*******************************************************************************
 ** write cube to pov-ray file
 *******************************************************************************/
static void addPOVRAYCube(FILE *fp, double xpos, double ypos, double zpos, double radius, double R, double G, double B, double transparency)
{
    fprintf(fp, "box { <%lf,%lf,%lf>,<%lf,%lf,%lf> pigment { color rgbt <%lf,%lf,%lf,%lf> } finish {diffuse %lf ambient %lf phong %lf } }\n",
            xpos - radius, ypos - radius, zpos - radius, xpos + radius, ypos + radius, zpos + radius, R, G, B,
            transparency, 0.4, 0.25, 0.9);
}


/*******************************************************************************
 ** write cell frame to pov-ray file
 *******************************************************************************/
static void addPOVRAYCellFrame(FILE *fp, double xposa, double yposa, double zposa, double xposb, double yposb, double zposb, 
                               double R, double G, double B)
{
    fprintf( fp, "#declare R = 0.1;\n" );
    fprintf( fp, "#declare myObject = union {\n" );
    fprintf( fp, "  sphere { <%.2f,%.2f,%.2f>, R }\n", xposa, yposa, zposa );
    fprintf( fp, "  sphere { <%.2f,%.2f,%.2f>, R }\n", xposb, yposa, zposa );
    fprintf( fp, "  sphere { <%.2f,%.2f,%.2f>, R }\n", xposa, yposa, zposb );
    fprintf( fp, "  sphere { <%.2f,%.2f,%.2f>, R }\n", xposb, yposa, zposb );
    fprintf( fp, "  sphere { <%.2f,%.2f,%.2f>, R }\n", xposa, yposb, zposa );
    fprintf( fp, "  sphere { <%.2f,%.2f,%.2f>, R }\n", xposb, yposb, zposa );
    fprintf( fp, "  sphere { <%.2f,%.2f,%.2f>, R }\n", xposa, yposb, zposb );
    fprintf( fp, "  sphere { <%.2f,%.2f,%.2f>, R }\n", xposb, yposb, zposb );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposa, yposa, zposa, xposb, yposa, zposa );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposa, yposa, zposb, xposb, yposa, zposb );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposa, yposb, zposa, xposb, yposb, zposa );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposa, yposb, zposb, xposb, yposb, zposb );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposa, yposa, zposa, xposa, yposb, zposa );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposa, yposa, zposb, xposa, yposb, zposb );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposb, yposa, zposa, xposb, yposb, zposa );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposb, yposa, zposb, xposb, yposb, zposb );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposa, yposa, zposa, xposa, yposa, zposb );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposa, yposb, zposa, xposa, yposb, zposb );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposb, yposa, zposa, xposb, yposa, zposb );
    fprintf( fp, "  cylinder { <%.2f,%.2f,%.2f>, <%.2f,%.2f,%.2f>, R }\n", xposb, yposb, zposa, xposb, yposb, zposb );
    fprintf( fp, "  texture { pigment { color rgb <%.2f,%.2f,%.2f> }\n", R, G, B );
    fprintf( fp, "            finish { diffuse 0.9 phong 1 } } }\n" );
    fprintf( fp, "object{myObject}\n" );
 
}


/*******************************************************************************
 ** write visible atoms to pov-ray file
 *******************************************************************************/
int writePOVRAYAtoms(char* filename, int NVisible, int *visibleAtoms, int* specie, double* pos, 
                     double* specieCovRad, double* PE, double* KE, double* charge, double* scalars, 
                     int scalarType, int heightAxis, rgbcalc_t rgbcalc)
{
    int i, index, specieIndex;
    double *rgb, scalar;
    FILE *OUTFILE;
    
    /* open file */
    OUTFILE = fopen(filename, "w");
    if (OUTFILE == NULL)
    {
        printf("ERROR: could not open file: %s\n", filename);
        printf("       reason: %s\n", strerror(errno));
        exit(35);
    }
    
    printf("Scalar type %d\n", scalarType);
    
    /* loop over visible atoms */
    for (i=0; i<NVisible; i++)
    {
        index = visibleAtoms[i];
        specieIndex = specie[index];
        
        /* scalar */
        if (scalarType == 0)
        {
            scalar = specieIndex;
        }
        else if (scalarType == 1)
        {
            scalar = pos[3*index+heightAxis];
        }
        else if (scalarType == 2)
        {
            scalar = KE[index];
        }
        else if (scalarType == 3)
        {
            scalar = PE[index];
        }
        else if (scalarType == 4)
        {
            scalar = charge[index];
        }
        else
        {
            scalar = scalars[i];
        }
        
        /* get rgb */
        rgb = rgbcalc(scalar);
        
        printf("VIS ATOM %d; scalar %lf; rgb (%lf, %lf, %lf)\n", index, scalar, rgb[0], rgb[1], rgb[2]);
        
        /* write atom */
        addPOVRAYSphere(OUTFILE, - pos[3*index], pos[3*index+1], pos[3*index+2], 
                        specieCovRad[specieIndex], rgb[0], rgb[1], rgb[2]);
    }
    
    fclose(OUTFILE);
    
    return 0;
}


/*******************************************************************************
 ** write defects to povray file
 *******************************************************************************/
int writePOVRAYDefects(char *filename, int vacsDim, int *vacs, int intsDim, int *ints, int antsDim, int *ants, int onAntsDim, int *onAnts,
                       int *specie, double *pos, int *refSpecie, double *refPos, double *specieRGB, double *specieCovRad, double *refSpecieRGB,
                       double* refSpecieCovRad, int splitIntsDim, int *splitInts)
{
    int i, index, specieIndex;
    FILE *OUTFILE;


    /* open file */
    OUTFILE = fopen(filename, "w");
    if (OUTFILE == NULL)
    {
        printf("ERROR: could not open file: %s\n", filename);
        printf("       reason: %s\n", strerror(errno));
        exit(35);
    }

    /* write interstitials */
    for (i=0; i<intsDim; i++)
    {
        index = ints[i];
        specieIndex = specie[index];

        addPOVRAYSphere(OUTFILE, - pos[3*index], pos[3*index+1], pos[3*index+2], specieCovRad[specieIndex],
                        specieRGB[specieIndex*3+0], specieRGB[specieIndex*3+1],
                        specieRGB[specieIndex*3+2]);
    }

    /* write vacancies */
    for (i=0; i<vacsDim; i++)
    {
        index = vacs[i];
        specieIndex = refSpecie[index];

        addPOVRAYCube(OUTFILE, - refPos[3*index], refPos[3*index+1], refPos[3*index+2], refSpecieCovRad[specieIndex],
                        refSpecieRGB[specieIndex*3+0], refSpecieRGB[specieIndex*3+1],
                        refSpecieRGB[specieIndex*3+2], 0.2);
    }

    /* write antisites */
    for (i=0; i<antsDim; i++)
    {
        index = ants[i];
        specieIndex = refSpecie[index];

        addPOVRAYCellFrame(OUTFILE, - refPos[3*index] - specieCovRad[specieIndex], refPos[3*index+1] - specieCovRad[specieIndex],
                           refPos[3*index+2] - specieCovRad[specieIndex], - refPos[3*index] + specieCovRad[specieIndex],
                           refPos[3*index+1] + specieCovRad[specieIndex], refPos[3*index+2] + specieCovRad[specieIndex],
                           refSpecieRGB[specieIndex*3+0], refSpecieRGB[specieIndex*3+1],
                           refSpecieRGB[specieIndex*3+2]);
    }

    /* write antisites occupying atom */
    for (i=0; i<onAntsDim; i++)
    {
        index = onAnts[i];
        specieIndex = specie[index];

        addPOVRAYSphere(OUTFILE, - pos[3*index], pos[3*index+1], pos[3*index+2], specieCovRad[specieIndex],
                        specieRGB[specieIndex*3+0], specieRGB[specieIndex*3+1],
                        specieRGB[specieIndex*3+2]);
    }

    /* write split interstitials */
    for (i=0; i<splitIntsDim; i++)
    {
        /* vacancy/bond */
        index = splitInts[3*i];
        specieIndex = refSpecie[index];
        
        addPOVRAYCube(OUTFILE, - refPos[3*index], refPos[3*index+1], refPos[3*index+2], refSpecieCovRad[specieIndex],
                                refSpecieRGB[specieIndex*3+0], refSpecieRGB[specieIndex*3+1],
                                refSpecieRGB[specieIndex*3+2], 0.2);
        
        /* first interstitial atom */
        index = splitInts[3*i+1];
        specieIndex = specie[index];
        
        addPOVRAYSphere(OUTFILE, - pos[3*index], pos[3*index+1], pos[3*index+2], specieCovRad[specieIndex],
                                specieRGB[specieIndex*3+0], specieRGB[specieIndex*3+1],
                                specieRGB[specieIndex*3+2]);
        
        /* second interstitial atom */
        index = splitInts[3*i+2];
        specieIndex = specie[index];
        
        addPOVRAYSphere(OUTFILE, - pos[3*index], pos[3*index+1], pos[3*index+2], specieCovRad[specieIndex],
                                specieRGB[specieIndex*3+0], specieRGB[specieIndex*3+1],
                                specieRGB[specieIndex*3+2]);
    }

    fclose(OUTFILE);
    
    return 0;
}

/*******************************************************************************
** write lattice file
*******************************************************************************/
int writeLattice(char* file, int NAtoms, int NVisible, int *visibleAtoms, double *cellDims, 
                 char* specieList, int* specie, double* pos, double* charge, int writeFullLattice)
{
    int i, index, NAtomsWrite;
    FILE *OUTFILE;
    char symtemp[3];
    
    
    OUTFILE = fopen(file, "w");
    if (OUTFILE == NULL)
    {
        printf("ERROR: could not open file: %s\n", file);
        printf("       reason: %s\n", strerror(errno));
        exit(35);
    } 
    
    NAtomsWrite = (writeFullLattice) ? NAtoms : NVisible;
    
    fprintf(OUTFILE, "%d\n", NAtomsWrite);
    fprintf(OUTFILE, "%f %f %f\n", cellDims[0], cellDims[1], cellDims[2]);
    
    if (writeFullLattice)
    {
        for (i=0; i<NAtoms; i++)
        {
            symtemp[0] = specieList[2*specie[i]];
            symtemp[1] = specieList[2*specie[i]+1];
            symtemp[2] = '\0';
            
            fprintf(OUTFILE, "%s %f %f %f %f\n", &symtemp[0], pos[3*i], pos[3*i+1], pos[3*i+2], charge[i]);
        }
    }
    else
    {
        for (i=0; i<NVisible; i++)
        {
            index = visibleAtoms[i];
            
            symtemp[0] = specieList[2*specie[index]];
            symtemp[1] = specieList[2*specie[index]+1];
            symtemp[2] = '\0';
            
            fprintf(OUTFILE, "%s %f %f %f %f\n", &symtemp[0], pos[3*index], pos[3*index+1], pos[3*index+2], charge[index]);
        }
    }
        
    fclose(OUTFILE);
    
    return 0;
}


