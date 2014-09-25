
/*******************************************************************************
 ** Calculate Steinhardt order parameters
 ** W. Lechner and C. Dellago. J. Chem.Phys. 129, 114707 (2008)
 ** Copyright Chris Scott 2014
 *******************************************************************************/

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_legendre.h>
#include <omp.h>
#include "boxeslib.h"
#include "neb_list.h"
#include "utilities.h"
#include "array_utils.h"


struct AtomStructureResults
{
    double Q6;
    double Q4;
    double realQ4[9];
    double imgQ4[9];
    double realQ6[13];
    double imgQ6[13];
};


static PyObject* bondOrderFilter(PyObject*, PyObject*);
static void Ylm(int, int, double, double, double*, double*);
static void convertToSphericalCoordinates(double, double, double, double, double*, double*);
static void complex_qlm(int, int*, struct NeighbourList*, double*, double*, int*, struct AtomStructureResults*);
static void calculate_Q(int, struct AtomStructureResults*);


/*******************************************************************************
 ** List of python methods available in this module
 *******************************************************************************/
static struct PyMethodDef methods[] = {
    {"bondOrderFilter", bondOrderFilter, METH_VARARGS, "Run the bond order filter (calculates Steinhardt order parameters)"},
    {NULL, NULL, 0, NULL}
};

/*******************************************************************************
 ** Module initialisation function
 *******************************************************************************/
PyMODINIT_FUNC
initbond_order(void)
{
    (void)Py_InitModule("bond_order", methods);
    import_array();
}

/*******************************************************************************
 ** Compute Y_lm (spherical harmonics)
 *******************************************************************************/
static void Ylm(int l, int m, double theta, double phi, double *realYlm, double *imgYlm)
{
    double factor, arg;
    
    
    /* call GSL spherical P_lm function */
    factor = gsl_sf_legendre_sphPlm(l, m, cos(theta));
    
    arg = (double) m * phi;
    *realYlm = factor * cos(arg);
    *imgYlm = factor * sin(arg);
}

/*******************************************************************************
 ** Convert to spherical coordinates
 *******************************************************************************/
static void convertToSphericalCoordinates(double xdiff, double ydiff, double zdiff, double sep, double *phi, double *theta)
{
    *theta = acos(zdiff / sep);
    *phi = atan2(ydiff, xdiff);
}

/*******************************************************************************
 ** Compute complex q_lm (sum over eq. 3 from Stutowski paper), for each atom
 *******************************************************************************/
static void complex_qlm(int NVisibleIn, int *visibleAtoms, struct NeighbourList *nebList, double *pos, double *cellDims, int *PBC, struct AtomStructureResults *results)
{
    int visIndex;
    double looptime;
    
    
    /* loop over atoms */
#pragma omp parallel for
    for (visIndex = 0; visIndex < NVisibleIn; visIndex++)
    {
        int index, m;
        double xpos1, ypos1, zpos1;
        
        /* pos 1 */
        index = visibleAtoms[visIndex];
        xpos1 = pos[3*index];
        ypos1 = pos[3*index+1];
        zpos1 = pos[3*index+2];
        
        /* loop over m, l = 6 */
        for (m = -6; m < 7; m++)
        {
            int i;
            double real_part, img_part;
            
            real_part = 0.0;
            img_part = 0.0;
            for (i = 0; i < nebList[visIndex].neighbourCount; i++)
            {
                int visIndex2, index2;
                double xpos2, ypos2, zpos2, sepVec[3];
                double theta, phi, realYlm, complexYlm;
                
                /* pos 2 */
                visIndex2 = nebList[visIndex].neighbour[i];
                index2 = visibleAtoms[visIndex2];
                xpos2 = pos[3*index2];
                ypos2 = pos[3*index2+1];
                zpos2 = pos[3*index2+2];
                
                /* separation vector */
                atomSeparationVector(sepVec, xpos1, ypos1, zpos1, xpos2, ypos2, zpos2, cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
                
                /* convert to spherical coordinates */
                convertToSphericalCoordinates(sepVec[0], sepVec[1], sepVec[2], nebList[visIndex].neighbourSep[i], &phi, &theta);
                
                /* calc Ylm */
                if (m < 0)
                {
                    Ylm(6, abs(m), theta, phi, &realYlm, &complexYlm);
                    realYlm = pow(-1.0, m) * realYlm;
                    complexYlm = pow(-1.0, m) * complexYlm;
                }
                else
                {
                    Ylm(6, m, theta, phi, &realYlm, &complexYlm);
                }
                
                /* sum */
                real_part += realYlm;
                img_part += complexYlm;
            }
            
            /* divide by num nebs */
            results[visIndex].realQ6[m+6] = real_part / ((double) nebList[visIndex].neighbourCount);
            results[visIndex].imgQ6[m+6] = img_part / ((double) nebList[visIndex].neighbourCount);
        }
        
        /* loop over m, l = 4 */
        for (m = -4; m < 5; m++)
        {
            int i;
            double real_part, img_part;
            
            real_part = 0.0;
            img_part = 0.0;
            for (i = 0; i < nebList[visIndex].neighbourCount; i++)
            {
                int visIndex2, index2;
                double xpos2, ypos2, zpos2, sepVec[3];
                double theta, phi, realYlm, complexYlm;
                
                /* pos 2 */
                visIndex2 = nebList[visIndex].neighbour[i];
                index2 = visibleAtoms[visIndex2];
                xpos2 = pos[3*index2];
                ypos2 = pos[3*index2+1];
                zpos2 = pos[3*index2+2];
                
                /* separation vector */
                atomSeparationVector(sepVec, xpos1, ypos1, zpos1, xpos2, ypos2, zpos2, cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);
                
                /* convert to spherical coordinates */
                convertToSphericalCoordinates(sepVec[0], sepVec[1], sepVec[2], nebList[visIndex].neighbourSep[i], &phi, &theta);
                
                /* calc Ylm */
                if (m < 0)
                {
                    Ylm(4, abs(m), theta, phi, &realYlm, &complexYlm);
                    realYlm = pow(-1.0, m) * realYlm;
                    complexYlm = pow(-1.0, m) * complexYlm;
                }
                else
                {
                    Ylm(4, m, theta, phi, &realYlm, &complexYlm);
                }
                
                /* sum */
                real_part += realYlm;
                img_part += complexYlm;
            }
            
            /* divide by num nebs */
            results[visIndex].realQ4[m+4] = real_part / ((double) nebList[visIndex].neighbourCount);
            results[visIndex].imgQ4[m+4] = img_part / ((double) nebList[visIndex].neighbourCount);
        }
    }
}

/*******************************************************************************
 ** calculate Q4/6 from complex q_lm's
 *******************************************************************************/
static void calculate_Q(int NVisibleIn, struct AtomStructureResults *results)
{
    int i, m;
    double sumQ6, sumQ4;
    
    
    for (i = 0; i < NVisibleIn; i++)
    {
        sumQ6 = 0.0;
        for (m = 0; m < 13; m++)
        {
            sumQ6 += results[i].realQ6[m] * results[i].realQ6[m] + results[i].imgQ6[m] * results[i].imgQ6[m];
        }
        results[i].Q6 = pow(((4.0 * M_PI / 13.0) * sumQ6), 0.5);
        
        sumQ4 = 0.0;
        for (m = 0; m < 9; m++)
        {
            sumQ4 += results[i].realQ4[m] * results[i].realQ4[m] + results[i].imgQ4[m] * results[i].imgQ4[m];
        }
        results[i].Q4 = pow(((4.0 * M_PI / 9.0) * sumQ4), 0.5);
    }
}






/*******************************************************************************
 ** bond order filter
 *******************************************************************************/
static PyObject*
bondOrderFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, *PBC, NScalars, filterQ4Enabled, filterQ6Enabled;
    double maxBondDistance, *scalarsQ4, *scalarsQ6, *minPos, *maxPos, *cellDims;
    double *pos, *fullScalars, minQ4, maxQ4, minQ6, maxQ6;
    PyArrayObject *posIn=NULL;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *scalarsQ4In=NULL;
    PyArrayObject *scalarsQ6In=NULL;
    PyArrayObject *minPosIn=NULL;
    PyArrayObject *maxPosIn=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    
    int i, j, index, NVisible;
    int maxSep2;
    double approxBoxWidth, q4, q6;
    double *visiblePos;
    struct Boxes *boxes;
    struct NeighbourList *nebList;
    struct AtomStructureResults *results;
    
    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!dO!O!O!O!O!O!iO!iddidd", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, &maxBondDistance, 
            &PyArray_Type, &scalarsQ4In, &PyArray_Type, &scalarsQ6In, &PyArray_Type, &minPosIn, &PyArray_Type, &maxPosIn, 
            &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &NScalars, &PyArray_Type, &fullScalarsIn, &filterQ4Enabled, &minQ4, 
            &maxQ4, &filterQ6Enabled, &minQ6, &maxQ6))
            return NULL;
    
    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) visibleAtomsIn->dimensions[0];
    
    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);
    
    if (not_doubleVector(scalarsQ4In)) return NULL;
    scalarsQ4 = pyvector_to_Cptr_double(scalarsQ4In);
    
    if (not_doubleVector(scalarsQ6In)) return NULL;
    scalarsQ6 = pyvector_to_Cptr_double(scalarsQ6In);
    
    if (not_doubleVector(minPosIn)) return NULL;
    minPos = pyvector_to_Cptr_double(minPosIn);
    
    if (not_doubleVector(maxPosIn)) return NULL;
    maxPos = pyvector_to_Cptr_double(maxPosIn);
    
    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);
    
    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);
    
    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);
    
    /* construct visible pos array */
    visiblePos = malloc(3 * NVisibleIn * sizeof(double));
    if (visiblePos == NULL)
    {
        printf("ERROR: could not allocate visiblePos\n");
        exit(50);
    }
    
    for (i=0; i<NVisibleIn; i++)
    {
        index = visibleAtoms[i];
        
        visiblePos[3*i] = pos[3*index];
        visiblePos[3*i+1] = pos[3*index+1];
        visiblePos[3*i+2] = pos[3*index+2];
    }
    
    /* box visible atoms */
    approxBoxWidth = maxBondDistance;
    maxSep2 = maxBondDistance * maxBondDistance;
    boxes = setupBoxes(approxBoxWidth, minPos, maxPos, PBC, cellDims);
    putAtomsInBoxes(NVisibleIn, visiblePos, boxes);
    
    /* build neighbour list (this should be separate function) */
    nebList = constructNeighbourList(NVisibleIn, visiblePos, boxes, cellDims, PBC, maxSep2);
    
    /* only required for building neb list */
    free(visiblePos);
    freeBoxes(boxes);
    
    /* allocate results structure */
    results = malloc(NVisibleIn * sizeof(struct AtomStructureResults));
    if (results == NULL)
    {
        printf("ERROR: could not allocate results\n");
        exit(50);
    }
    
    /* first calc q_lm for each atom over all m values */
    complex_qlm(NVisibleIn, visibleAtoms, nebList, pos, cellDims, PBC, results);
    
    /* calculate Q4 and Q6 */
    calculate_Q(NVisibleIn, results);
    
    /* do filtering here, storing results along the way */
    NVisible = 0;
    for (i = 0; i < NVisibleIn; i++)
    {
        q4 = results[i].Q4;
        q6 = results[i].Q6;
        
        if (filterQ4Enabled && (q4 < minQ4 || q4 > maxQ4))
            continue;
        
        if (filterQ6Enabled && (q6 < minQ6 || q6 > maxQ6))
            continue;
        
        /* add */
        visibleAtoms[NVisible] = visibleAtoms[i];
        
        /* store scalars */
        scalarsQ4[NVisible] = q4;
        scalarsQ6[NVisible] = q6;
        
        /* handle full scalars array */
        for (j = 0; j < NScalars; j++)
        {
            fullScalars[NVisibleIn * j + NVisible] = fullScalars[NVisibleIn * j + i];
        }
        
        NVisible++;
    }
    
    /* free */
    freeNeighbourList(nebList, NVisibleIn);
    free(results);
    
    return Py_BuildValue("i", NVisible);
}
