
/*******************************************************************************
 ** Calculate Steinhardt order parameters
 ** W. Lechner and C. Dellago. J. Chem.Phys. 129, 114707 (2008)
 *******************************************************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h> // includes stdio.h, string.h, errno.h, stdlib.h
#include <numpy/arrayobject.h>
#include <math.h>
#include "visclibs/boxeslib.h"
#include "visclibs/neb_list.h"
#include "visclibs/utilities.h"
#include "visclibs/array_utils.h"
#include "visclibs/constants.h"
#include "gui/preferences.h"


/* structure for storing the result for an atom */
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

const double factorials[] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0, 39916800.0, 479001600.0};


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
init_bond_order(void)
{
    (void)Py_InitModule("_bond_order", methods);
    import_array();
}

/*******************************************************************************
 ** Compute P_lm (numerical recipes Ch.8 p.254)
 *******************************************************************************/
double Plm(int l, int m, double x)
{
    double pmm;


    if (m < 0 || m > l || fabs(x) > 1.0)
    {
        fprintf(stderr, "Bad arguments to in routine Plm\n");
        exit(1);
    }
    pmm = 1.0;
    if (m > 0)
    {
        int i;
        double somx2, fact;

        somx2 = sqrt((1.0 - x) * (1.0 + x));
        fact = 1.0;
        for (i = 1; i <= m; i++)
        {
            pmm *= - fact * somx2;
            fact += 2.0;
        }
    }
    if (l == m)
        return pmm;
    else
    {
        double pmmp1;

        pmmp1 = x * (2.0 * m + 1.0) * pmm;
        if (l == m + 1)
            return pmmp1;
        else
        {
            int ll;
            double pll = 0.0;

            for (ll = m + 2; ll <= l; ll++)
            {
                pll = (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
                pmm = pmmp1;
                pmmp1 = pll;
            }
            return pll;
        }
    }
}


/*******************************************************************************
 ** Compute Y_lm (spherical harmonics)
 *******************************************************************************/
static void Ylm(int l, int m, double theta, double phi, double *realYlm, double *imgYlm)
{
    double factor, arg, val;


    val = Plm(l, m, cos(theta));
    factor = ((2.0 * (double)l + 1.0) * factorials[l - m]) / (4.0 * CONST_PI * factorials[l + m]);
    factor = sqrt(factor);

    arg = (double) m * phi;
    *realYlm = factor * val * cos(arg);
    *imgYlm = factor * val *  sin(arg);
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
 ** Compute complex q_lm (sum over eq. 3 from Stukowski paper), for each atom
 *******************************************************************************/
static void complex_qlm(int NVisibleIn, int *visibleAtoms, struct NeighbourList *nebList, double *pos, double *cellDims,
        int *PBC, struct AtomStructureResults *results)
{
    int visIndex;


    /* loop over atoms */
    #pragma omp parallel for num_threads(prefs_numThreads)
    for (visIndex = 0; visIndex < NVisibleIn; visIndex++)
    {
        int index, m;
        double xpos1, ypos1, zpos1;

        /* atom 1 position */
        index = visibleAtoms[visIndex];
        xpos1 = pos[3*index];
        ypos1 = pos[3*index+1];
        zpos1 = pos[3*index+2];

        /* loop over m, l = 6 */
        for (m = -6; m < 7; m++)
        {
            int i;
            double real_part, img_part;

            /* loop over neighbours */
            real_part = 0.0;
            img_part = 0.0;
            for (i = 0; i < nebList[visIndex].neighbourCount; i++)
            {
                int visIndex2, index2;
                double xpos2, ypos2, zpos2, sepVec[3];
                double theta, phi, realYlm, complexYlm;

                /* atom 2 position */
                visIndex2 = nebList[visIndex].neighbour[i];
                index2 = visibleAtoms[visIndex2];
                xpos2 = pos[3*index2];
                ypos2 = pos[3*index2+1];
                zpos2 = pos[3*index2+2];

                /* calculate separation vector between atoms */
                atomSeparationVector(sepVec, xpos1, ypos1, zpos1, xpos2, ypos2, zpos2, cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);

                /* convert to spherical coordinates */
                convertToSphericalCoordinates(sepVec[0], sepVec[1], sepVec[2], nebList[visIndex].neighbourSep[i], &phi, &theta);

                /* calculate Ylm */
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

            /* divide by number of neighbours */
            results[visIndex].realQ6[m+6] = real_part / ((double) nebList[visIndex].neighbourCount);
            results[visIndex].imgQ6[m+6] = img_part / ((double) nebList[visIndex].neighbourCount);
        }

        /* loop over m, l = 4 */
        for (m = -4; m < 5; m++)
        {
            int i;
            double real_part, img_part;

            /* loop over neighbours */
            real_part = 0.0;
            img_part = 0.0;
            for (i = 0; i < nebList[visIndex].neighbourCount; i++)
            {
                int visIndex2, index2;
                double xpos2, ypos2, zpos2, sepVec[3];
                double theta, phi, realYlm, complexYlm;

                /* atom 2  position */
                visIndex2 = nebList[visIndex].neighbour[i];
                index2 = visibleAtoms[visIndex2];
                xpos2 = pos[3*index2];
                ypos2 = pos[3*index2+1];
                zpos2 = pos[3*index2+2];

                /* calculate separation vector */
                atomSeparationVector(sepVec, xpos1, ypos1, zpos1, xpos2, ypos2, zpos2, cellDims[0], cellDims[1], cellDims[2], PBC[0], PBC[1], PBC[2]);

                /* convert to spherical coordinates */
                convertToSphericalCoordinates(sepVec[0], sepVec[1], sepVec[2], nebList[visIndex].neighbourSep[i], &phi, &theta);

                /* calculate Ylm */
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

            /* divide by number of neighbours */
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
    const double pi13 = 4.0 * CONST_PI / 13.0;
    const double pi9 = 4.0 * CONST_PI / 9.0;


    /* loop over atoms and compute the Q4 and Q6 values */
    for (i = 0; i < NVisibleIn; i++)
    {
        sumQ6 = 0.0;
        for (m = 0; m < 13; m++)
        {
            sumQ6 += results[i].realQ6[m] * results[i].realQ6[m] + results[i].imgQ6[m] * results[i].imgQ6[m];
        }
        results[i].Q6 = pow(pi13 * sumQ6, 0.5);

        sumQ4 = 0.0;
        for (m = 0; m < 9; m++)
        {
            sumQ4 += results[i].realQ4[m] * results[i].realQ4[m] + results[i].imgQ4[m] * results[i].imgQ4[m];
        }
        results[i].Q4 = pow(pi9 * sumQ4, 0.5);
    }
}

/*******************************************************************************
 ** Calculate the bond order parameters and filter atoms (if required).
 **
 ** Inputs:
 **     - visibleAtoms: the list of atoms that are currently visible
 **     - pos: positions of all the atoms
 **     - maxBondDistance: the maximum bond distance to consider
 **     - scalarsQ4: array to store the Q4 parameter value
 **     - scalarsQ6: array to store the Q6 parameter value
 **     - cellDims: simulation cell dimensions
 **     - PBC: periodic boundaries conditions
 **     - NScalars: the number of previously calculated scalar values
 **     - fullScalars: the full list of previously calculated scalars
 **     - NVectors: the number of previously calculated vector values
 **     - fullVectors: the full list of previously calculated vectors
 **     - filterQ4: filter atoms by the Q4 parameter
 **     - minQ4: the minimum Q4 for an atom to be visible
 **     - maxQ4: the maximum Q4 for an atom to be visible
 **     - filterQ6: filter atoms by the Q6 parameter
 **     - minQ6: the minimum Q6 for an atom to be visible
 **     - maxQ6: the maximum Q6 for an atom to be visible
 *******************************************************************************/
static PyObject*
bondOrderFilter(PyObject *self, PyObject *args)
{
    int NVisibleIn, *visibleAtoms, *PBC, NScalars, filterQ4Enabled, filterQ6Enabled;
    int NVectors;
    double maxBondDistance, *scalarsQ4, *scalarsQ6, *cellDims;
    double *pos, *fullScalars, minQ4, maxQ4, minQ6, maxQ6;
    PyArrayObject *posIn=NULL;
    PyArrayObject *visibleAtomsIn=NULL;
    PyArrayObject *PBCIn=NULL;
    PyArrayObject *scalarsQ4In=NULL;
    PyArrayObject *scalarsQ6In=NULL;
    PyArrayObject *cellDimsIn=NULL;
    PyArrayObject *fullScalarsIn=NULL;
    PyArrayObject *fullVectors=NULL;

    int i, NVisible, boxstat;
    double *visiblePos, maxSep2;
    struct Boxes *boxes;
    struct NeighbourList *nebList;
    struct AtomStructureResults *results;

    /* parse and check arguments from Python */
    if (!PyArg_ParseTuple(args, "O!O!dO!O!O!O!iO!iddiddiO!", &PyArray_Type, &visibleAtomsIn, &PyArray_Type, &posIn, &maxBondDistance,
            &PyArray_Type, &scalarsQ4In, &PyArray_Type, &scalarsQ6In, &PyArray_Type, &cellDimsIn, &PyArray_Type, &PBCIn, &NScalars,
            &PyArray_Type, &fullScalarsIn, &filterQ4Enabled, &minQ4, &maxQ4, &filterQ6Enabled, &minQ6, &maxQ6, &NVectors,
            &PyArray_Type, &fullVectors))
        return NULL;

    if (not_intVector(visibleAtomsIn)) return NULL;
    visibleAtoms = pyvector_to_Cptr_int(visibleAtomsIn);
    NVisibleIn = (int) PyArray_DIM(visibleAtomsIn, 0);

    if (not_doubleVector(posIn)) return NULL;
    pos = pyvector_to_Cptr_double(posIn);

    if (not_doubleVector(scalarsQ4In)) return NULL;
    scalarsQ4 = pyvector_to_Cptr_double(scalarsQ4In);

    if (not_doubleVector(scalarsQ6In)) return NULL;
    scalarsQ6 = pyvector_to_Cptr_double(scalarsQ6In);

    if (not_doubleVector(cellDimsIn)) return NULL;
    cellDims = pyvector_to_Cptr_double(cellDimsIn);

    if (not_intVector(PBCIn)) return NULL;
    PBC = pyvector_to_Cptr_int(PBCIn);

    if (not_doubleVector(fullScalarsIn)) return NULL;
    fullScalars = pyvector_to_Cptr_double(fullScalarsIn);

    if (not_doubleVector(fullVectors)) return NULL;

    /* construct array of positions of visible atoms */
    visiblePos = malloc(3 * NVisibleIn * sizeof(double));
    if (visiblePos == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate visiblePos");
        return NULL;
    }
    for (i = 0; i < NVisibleIn; i++)
    {
        int index = visibleAtoms[i];
        int ind3 = 3 * index;
        int i3 = 3 * i;
        visiblePos[i3    ] = pos[ind3    ];
        visiblePos[i3 + 1] = pos[ind3 + 1];
        visiblePos[i3 + 2] = pos[ind3 + 2];
    }

    /* box visible atoms */
    boxes = setupBoxes(maxBondDistance, PBC, cellDims);
    if (boxes == NULL)
    {
        free(visiblePos);
        return NULL;
    }
    boxstat = putAtomsInBoxes(NVisibleIn, visiblePos, boxes);
    if (boxstat)
    {
        free(visiblePos);
        return NULL;
    }

    /* build neighbour list */
    maxSep2 = maxBondDistance * maxBondDistance;
    nebList = constructNeighbourList(NVisibleIn, visiblePos, boxes, cellDims, PBC, maxSep2);

    /* only required for building neb list */
    free(visiblePos);
    freeBoxes(boxes);

    /* return if failed to build the neighbour list */
    if (nebList == NULL) return NULL;

    /* allocate results structure */
    results = malloc(NVisibleIn * sizeof(struct AtomStructureResults));
    if (results == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate results");
        freeNeighbourList(nebList, NVisibleIn);
        return NULL;
    }

    /* first calc q_lm for each atom over all m values */
    complex_qlm(NVisibleIn, visibleAtoms, nebList, pos, cellDims, PBC, results);

    /* free neighbour list */
    freeNeighbourList(nebList, NVisibleIn);

    /* calculate Q4 and Q6 */
    calculate_Q(NVisibleIn, results);

    /* do filtering here, storing results along the way */
    NVisible = 0;
    for (i = 0; i < NVisibleIn; i++)
    {
        int j;
        double q4 = results[i].Q4;
        double q6 = results[i].Q6;

        /* skip if not within the valid range */
        if (filterQ4Enabled && (q4 < minQ4 || q4 > maxQ4))
            continue;
        if (filterQ6Enabled && (q6 < minQ6 || q6 > maxQ6))
            continue;

        /* store in visible atoms array */
        visibleAtoms[NVisible] = visibleAtoms[i];

        /* store calculated values */
        scalarsQ4[NVisible] = q4;
        scalarsQ6[NVisible] = q6;

        /* update full scalars/vectors arrays */
        for (j = 0; j < NScalars; j++)
        {
            int nj = j * NVisibleIn;
            fullScalars[nj + NVisible] = fullScalars[nj + i];
        }

        for (j = 0; j < NVectors; j++)
        {
            int nj = j * NVisibleIn;
            DIND2(fullVectors, nj + NVisible, 0) = DIND2(fullVectors, nj + i, 0);
            DIND2(fullVectors, nj + NVisible, 1) = DIND2(fullVectors, nj + i, 1);
            DIND2(fullVectors, nj + NVisible, 2) = DIND2(fullVectors, nj + i, 2);
        }

        NVisible++;
    }

    /* free results memory */
    free(results);

    return Py_BuildValue("i", NVisible);
}
