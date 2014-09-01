
#ifndef BOND_ORDER_SET
#define BOND_ORDER_SET

int bondOrderFilter(int, int*, int, double *, double, int, double *, double *, double *, double *, 
                    double *, int *, int, double *, int, double, double, int, double, double);

struct AtomStructureResults
{
    double Q6;
    double Q4;
    double realQ4[9];
    double imgQ4[9];
    double realQ6[13];
    double imgQ6[13];
};

#endif
