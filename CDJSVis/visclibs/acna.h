
#ifndef ACNA_SET
#define ACNA_SET

int adaptiveCommonNeighbourAnalysis(int, int*, int, double*, int, double*, double*, double*, double*, int*, int, double*, double, int*, int, int*);

#define MAX_REQUIRED_NEBS 14
#define MIN_REQUIRED_NEBS 12

/* structure types */
enum AtomStructureType {
	ATOM_STRUCTURE_DISORDERED 		= 0,
	ATOM_STRUCTURE_FCC				= 1,
	ATOM_STRUCTURE_HCP				= 2,
	ATOM_STRUCTURE_BCC				= 3,
	ATOM_STRUCTURE_ICOSAHEDRAL		= 4,
	ATOM_STRUCTURE_SIGMA11_TILT1	= 5,
	ATOM_STRUCTURE_SIGMA11_TILT2	= 6
};

#endif
