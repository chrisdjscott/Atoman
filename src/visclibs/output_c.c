
/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** IO routines written in C to improve performance
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>


void addPOVRAYSphere(FILE *, double, double, double, double, double, double, double);




/*******************************************************************************
 ** write sphere to pov-ray file
 *******************************************************************************/
void addPOVRAYSphere(FILE *fp, double xpos, double ypos, double zpos, double radius, double R, double G, double B)
{
    fprintf(fp, "sphere { <%lf,%lf,%lf>, %lf pigment { color rgb <%lf,%lf,%lf> } finish { ambient %f phong %f } }\n",
            xpos, ypos, zpos, radius, R, G, B, 0.25, 0.9);
}


/*******************************************************************************
 ** write visible atoms to pov-ray file
 *******************************************************************************/
void writePOVRAYAtoms(char *filename, int specieDim, int *specie, int posDim, double *pos, 
                      int visibleAtomsDim, int *visibleAtoms, int specieRGBDim1, int specieRGBDim2, 
                      double *specieRGB, int specieCovRadDim, double *specieCovRad)
{
    int i, index, specieIndex;
    FILE *OUTFILE;
    double xpos, ypos, zpos, R, G, B, rad;
    
    /* open file */
    OUTFILE = fopen(filename, "w");
    if (OUTFILE == NULL)
    {
        printf("ERROR: could not open file: %s\n", filename);
        printf("       reason: %s\n", strerror(errno));
        exit(35);
    }
    
    for (i=0; i<visibleAtomsDim; i++)
    {
        index = visibleAtoms[i];
        specieIndex = specie[index];
        
        addPOVRAYSphere(OUTFILE, - pos[3*index], pos[3*index+1], pos[3*index+2], specieCovRad[specieIndex],
                        specieRGB[specieIndex*specieRGBDim2+0], specieRGB[specieIndex*specieRGBDim2+1],
                        specieRGB[specieIndex*specieRGBDim2+2]);
    }
    
    fclose(OUTFILE);
}

