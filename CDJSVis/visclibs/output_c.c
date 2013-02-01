
/*******************************************************************************
 ** Copyright Chris Scott 2012
 ** IO routines written in C to improve performance
 *******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>


void addPOVRAYSphere(FILE *, double, double, double, double, double, double, double);
void addPOVRAYCube(FILE *, double, double, double, double, double, double, double, double);
void addPOVRAYCellFrame(FILE *, double, double, double, double, double, double, double, double, double);



/*******************************************************************************
 ** write sphere to pov-ray file
 *******************************************************************************/
void addPOVRAYSphere(FILE *fp, double xpos, double ypos, double zpos, double radius, double R, double G, double B)
{
    fprintf(fp, "sphere { <%lf,%lf,%lf>, %lf pigment { color rgb <%lf,%lf,%lf> } finish { ambient %f phong %f } }\n",
            xpos, ypos, zpos, radius, R, G, B, 0.25, 0.9);
}


/*******************************************************************************
 ** write cube to pov-ray file
 *******************************************************************************/
void addPOVRAYCube(FILE *fp, double xpos, double ypos, double zpos, double radius, double R, double G, double B, double transparency)
{
	fprintf(fp, "box { <%lf,%lf,%lf>,<%lf,%lf,%lf> pigment { color rgbt <%lf,%lf,%lf,%lf> } finish {diffuse %lf ambient %lf phong %lf } }\n",
			xpos - radius, ypos - radius, zpos - radius, xpos + radius, ypos + radius, zpos + radius, R, G, B,
			transparency, 0.4, 0.25, 0.9);
}


/*******************************************************************************
 ** write cell frame to pov-ray file
 *******************************************************************************/
void addPOVRAYCellFrame(FILE *fp, double xposa, double yposa, double zposa, double xposb, double yposb, double zposb, 
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


/*******************************************************************************
 ** write defects to povray file
 *******************************************************************************/
void writePOVRAYDefects(char *filename, int vacsDim, int *vacs, int intsDim, int *ints, int antsDim, int *ants, int onAntsDim, int *onAnts,
						int specieDim, int *specie, int posDim, double *pos, int refSpecieDim, int *refSpecie, int refPosDim,
						double *refPos, int specieRGBDim1, int specieRGBDim2, double *specieRGB, int specieCovRadDim,
						double *specieCovRad, int refSpecieRGBDim1, int refSpecieRGBDim2, double *refSpecieRGB,
                        int refSpecieCovRadDim, double* refSpecieCovRad)
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
						specieRGB[specieIndex*specieRGBDim2+0], specieRGB[specieIndex*specieRGBDim2+1],
						specieRGB[specieIndex*specieRGBDim2+2]);
	}

	/* write vacancies */
	for (i=0; i<vacsDim; i++)
	{
		index = vacs[i];
		specieIndex = refSpecie[index];

		addPOVRAYCube(OUTFILE, - refPos[3*index], refPos[3*index+1], refPos[3*index+2], refSpecieCovRad[specieIndex],
						refSpecieRGB[specieIndex*refSpecieRGBDim2+0], refSpecieRGB[specieIndex*refSpecieRGBDim2+1],
						refSpecieRGB[specieIndex*refSpecieRGBDim2+2], 0.2);
	}

	/* write antisites */
	for (i=0; i<antsDim; i++)
    {
        index = ants[i];
        specieIndex = refSpecie[index];

        addPOVRAYCellFrame(OUTFILE, - refPos[3*index] - specieCovRad[specieIndex], refPos[3*index+1] - specieCovRad[specieIndex],
                           refPos[3*index+2] - specieCovRad[specieIndex], - refPos[3*index] + specieCovRad[specieIndex],
                           refPos[3*index+1] + specieCovRad[specieIndex], refPos[3*index+2] + specieCovRad[specieIndex],
                           refSpecieRGB[specieIndex*refSpecieRGBDim2+0], refSpecieRGB[specieIndex*refSpecieRGBDim2+1],
                           refSpecieRGB[specieIndex*refSpecieRGBDim2+2]);
    }

	/* write antisites occupying atom */
	for (i=0; i<onAntsDim; i++)
	{
		index = onAnts[i];
		specieIndex = specie[index];

		addPOVRAYSphere(OUTFILE, - pos[3*index], pos[3*index+1], pos[3*index+2], specieCovRad[specieIndex],
						specieRGB[specieIndex*specieRGBDim2+0], specieRGB[specieIndex*specieRGBDim2+1],
						specieRGB[specieIndex*specieRGBDim2+2]);
	}


	fclose(OUTFILE);
}

/*******************************************************************************
** write lattice file
*******************************************************************************/
void writeLattice(char* file, int visibleAtomsDim, int *visibleAtoms, double xdim, double ydim, double zdim, int speclistDim, 
                  char* specieList_c, int specieDim, int* specie, int posDim, double* pos, int chargeDim, 
                  double* charge)
{
    int i, index;
    FILE *OUTFILE;
    char symtemp[3];
    
    
    OUTFILE = fopen(file, "w");
    if (OUTFILE == NULL)
    {
        printf("ERROR: could not open file: %s\n", file);
        printf("       reason: %s\n", strerror(errno));
        exit(35);
    } 
    
    fprintf(OUTFILE, "%d\n", visibleAtomsDim);
    fprintf(OUTFILE, "%f %f %f\n", xdim, ydim, zdim);
    
    for (i=0; i<visibleAtomsDim; i++)
    {
        index = visibleAtoms[i];
        
        symtemp[0] = specieList_c[2*specie[index]];
        symtemp[1] = specieList_c[2*specie[index]+1];
        symtemp[2] = '\0';
        
        fprintf(OUTFILE, "%s %f %f %f %f\n", &symtemp[0], pos[3*index], pos[3*index+1], pos[3*index+2], charge[index]);
    }
    
    fclose(OUTFILE);
}


