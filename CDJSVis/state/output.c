
/*******************************************************************************
 ** Copyright Chris Scott 2014
 ** IO routines written in C to improve performance
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include "array_utils.h"


//static PyObject* writePOVRAYAtoms(PyObject*, PyObject*);
static PyObject* writePOVRAYDefects(PyObject*, PyObject*);
static PyObject* writeLattice(PyObject*, PyObject*);
static void addPOVRAYSphere(FILE *, double, double, double, double, double, double, double);
static void addPOVRAYCube(FILE *, double, double, double, double, double, double, double, double);
static void addPOVRAYCellFrame(FILE *, double, double, double, double, double, double, double, double, double);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
//    {"writePOVRAYAtoms", writePOVRAYAtoms, METH_VARARGS, "Write atoms to POV-Ray file"},
    {"writePOVRAYDefects", writePOVRAYDefects, METH_VARARGS, "Write defects to POV-Ray file"},
    {"writeLattice", writeLattice, METH_VARARGS, "Write (visible) atoms to lattice file"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
init_output(void)
{
    (void)Py_InitModule("_output", methods);
    import_array();
}

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
//static PyObject*
//writePOVRAYAtoms(PyObject *self, PyObject *args)
//{
//    char *filename;
//    int *visibleAtoms, *specie, scalarType, heightAxis, 
//    double *pos, *specieCovRad, *PE, *KE, *charge, *scalars;
//    
//    
////int writePOVRAYAtoms(char* filename, int NVisible, int *visibleAtoms, int* specie, double* pos, 
////                     double* specieCovRad, double* PE, double* KE, double* charge, double* scalars, 
////                     int scalarType, int heightAxis, rgbcalc_t rgbcalc)
////{
//    int i, index, specieIndex;
//    double *rgb, scalar;
//    FILE *OUTFILE;
//    
//    /* open file */
//    OUTFILE = fopen(filename, "w");
//    if (OUTFILE == NULL)
//    {
//        printf("ERROR: could not open file: %s\n", filename);
//        printf("       reason: %s\n", strerror(errno));
//        exit(35);
//    }
//    
//    printf("Scalar type %d\n", scalarType);
//    
//    /* loop over visible atoms */
//    for (i=0; i<NVisible; i++)
//    {
//        index = visibleAtoms[i];
//        specieIndex = specie[index];
//        
//        /* scalar */
//        if (scalarType == 0)
//        {
//            scalar = specieIndex;
//        }
//        else if (scalarType == 1)
//        {
//            scalar = pos[3*index+heightAxis];
//        }
//        else if (scalarType == 2)
//        {
//            scalar = KE[index];
//        }
//        else if (scalarType == 3)
//        {
//            scalar = PE[index];
//        }
//        else if (scalarType == 4)
//        {
//            scalar = charge[index];
//        }
//        else
//        {
//            scalar = scalars[i];
//        }
//        
//        /* get rgb */
//        rgb = rgbcalc(scalar);
//        
//        printf("VIS ATOM %d; scalar %lf; rgb (%lf, %lf, %lf)\n", index, scalar, rgb[0], rgb[1], rgb[2]);
//        
//        /* write atom */
//        addPOVRAYSphere(OUTFILE, - pos[3*index], pos[3*index+1], pos[3*index+2], 
//                        specieCovRad[specieIndex], rgb[0], rgb[1], rgb[2]);
//    }
//    
//    fclose(OUTFILE);
//    
//    return 0;
//}


/*******************************************************************************
 ** write defects to povray file
 *******************************************************************************/
static PyObject*
writePOVRAYDefects(PyObject *self, PyObject *args)
{
    char *filename;
    int vacsDim, *vacs, intsDim, *ints, antsDim, *ants, onAntsDim, *onAnts, *specie, *refSpecie, splitIntsDim, *splitInts;  
    double *pos, *refPos, *specieRGB, *specieCovRad, *refSpecieRGB, *refSpecieCovRad;
    PyArrayObject *vacsIn=NULL;
    PyArrayObject *intsIn=NULL;
    PyArrayObject *antsIn=NULL;
    PyArrayObject *onAntsIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *refSpecieIn=NULL;
    PyArrayObject *splitIntsIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *refPosIn=NULL;
    PyArrayObject *specieRGBIn=NULL;
    PyArrayObject *specieCovRadIn=NULL;
    PyArrayObject *refSpecieRGBIn=NULL;
    PyArrayObject *refSpecieCovRadIn=NULL;
    
    int i, index, specieIndex;
    FILE *OUTFILE;
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "sO!O!O!O!O!O!O!O!O!O!O!O!O!", &filename, &PyArray_Type, &vacsIn, &PyArray_Type, &intsIn, &PyArray_Type, &antsIn,
            &PyArray_Type, &onAntsIn, &PyArray_Type, &specieIn, &PyArray_Type, &posIn, &PyArray_Type, &refSpecieIn, &PyArray_Type, &refPosIn,
            &PyArray_Type, &specieRGBIn, &PyArray_Type, &specieCovRadIn, &PyArray_Type, &refSpecieRGBIn, &PyArray_Type, &refSpecieCovRadIn,
            &PyArray_Type, &splitIntsIn))
        return NULL;
    
    if (not_intVector(vacsIn)) return NULL;
    vacs = pyvector_to_Cptr_int(vacsIn);
    vacsDim = (int) vacsIn->dimensions[0];
    
    if (not_intVector(intsIn)) return NULL;
    ints = pyvector_to_Cptr_int(intsIn);
    intsDim = (int) intsIn->dimensions[0];
    
    if (not_intVector(antsIn)) return NULL;
    ants = pyvector_to_Cptr_int(antsIn);
    antsDim = (int) antsIn->dimensions[0];
    
    if (not_intVector(onAntsIn)) return NULL;
    onAnts = pyvector_to_Cptr_int(onAntsIn);
    onAntsDim = (int) onAntsIn->dimensions[0];
    
    if (not_intVector(splitIntsIn)) return NULL;
    splitInts = pyvector_to_Cptr_int(splitIntsIn);
    splitIntsDim = (int) splitIntsIn->dimensions[0];
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    
    if (not_intVector(refSpecieIn)) return NULL;
    refSpecie = pyvector_to_Cptr_int(refSpecieIn);
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(refPosIn)) return NULL;
    refPos = pyvector_to_Cptr_double(refPosIn);
    
    if (not_doubleVector(specieRGBIn)) return NULL;
    specieRGB = pyvector_to_Cptr_double(specieRGBIn);
    
    if (not_doubleVector(refSpecieRGBIn)) return NULL;
    refSpecieRGB = pyvector_to_Cptr_double(refSpecieRGBIn);
    
    if (not_doubleVector(specieCovRadIn)) return NULL;
    specieCovRad = pyvector_to_Cptr_double(specieCovRadIn);
    
    if (not_doubleVector(refSpecieCovRadIn)) return NULL;
    refSpecieCovRad = pyvector_to_Cptr_double(refSpecieCovRadIn);

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
    
    return Py_BuildValue("i", 0);
}

/*******************************************************************************
** write lattice file
*******************************************************************************/
static PyObject*
writeLattice(PyObject *self, PyObject *args)
{
    char *filename, *specieList;
    int NAtoms, NVisible, *visibleAtoms, *specie, writeFullLattice;
    double *cellDims, *pos, *charge;
    PyArrayObject *specieListIn=NULL;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *specieIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *posIn=NULL;
    PyArrayObject *chargeIn=NULL;

    int i, index, NAtomsWrite;
    FILE *OUTFILE;
    char symtemp[3];
    
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "sO!O!O!O!O!O!i", &filename, &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &cellDimsIn, 
            &PyArray_Type, &specieListIn, &PyArray_Type, &specieIn, &PyArray_Type, &posIn, &PyArray_Type, &chargeIn,
            &writeFullLattice))
        return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisible = (int) visibleAtomsIn->dimensions[0];
    
    if (not_intVector(specieIn)) return NULL;
    specie = pyvector_to_Cptr_int(specieIn);
    NAtoms = (int) specieIn->dimensions[0];
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_doubleVector(chargeIn)) return NULL;
    charge = pyvector_to_Cptr_double(chargeIn);
    
    specieList = pyvector_to_Cptr_char(specieListIn);
    
    /* open file */
    OUTFILE = fopen(filename, "w");
    if (OUTFILE == NULL)
    {
        printf("ERROR: could not open file: %s\n", filename);
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
    
    return Py_BuildValue("i", 0);
}
