
"""
Provides information about elements.

Eg. atomic mass, radius, etc.

@author: Chris Scott

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
import logging

from ..visutils.utilities import dataPath, createDataFile
import six


################################################################################
class Elements:
    """
    Structure to hold atomic data
    ie mass, radius, etc
    
    """
    def __init__(self):
        self.atomicNumberDict = {}
        self.atomicMassDict = {}
        self.atomNameDict = {}
        self.covalentRadiusDict = {}
        self.RGBDict = {}
        self.bondDict = {}
    
    def read(self, filename):
        """
        Read in atomic information from given file.
        
        """
        f = open( filename, "r" )
        
        # read into dictionaries    
        for line in f:
            line = line.strip()
            
            if not len(line):
                continue
            
            array = line.split()
            
            if not len(array) == 8:
                print("ERROR LEN WRONG")
                continue
            
            key = array[3]
            if len(key) == 1:
                key = key + '_'
            
            self.atomicNumberDict[key] = int(array[0])
            self.atomicMassDict[key] = float(array[1])
            self.atomNameDict[key] = array[2]
            self.covalentRadiusDict[key] = float(array[4])
            self.RGBDict[key] = [float(array[5]), float(array[6]), float(array[7])]
        
        f.close()
    
    def listElements(self):
        """
        Return a list of elements
        
        """
        return sorted(self.atomNameDict.keys())
    
    def readBonds(self, filename):
        """
        Read bonds file.
        
        """
        if filename is None or not os.path.exists(filename):
            return
        
        f = open(filename)
        
        for line in f:
            line = line.strip()
            
            if not len(line):
                continue
            
            array = line.split()
            
            if len(array) != 4:
                print("BAD LEN")
                continue
            
            keya = array[0]
            keyb = array[1]
            bondMin = float(array[2])
            bondMax = float(array[3])
            
            self.addBond(keya, keyb, bondMin, bondMax)
        
        f.close()
        
#        for keya in self.bondDict:
#            for keyb, val in self.bondDict[keya].items():
#                print "%s - %s: %f -> %f" % (keya, keyb, val[0], val[1])
    
    def addBond(self, keya, keyb, bondMin, bondMax):
        """
        Add a new bond with given values
        
        """
        # ensure bondMin >= 0 and bondMax >= bondMin
        bondMin = max(bondMin, 0)
        bondMax = max(bondMin, bondMax)
        
        if len(keya) == 1:
            keya += "_"
        if len(keyb) == 1:
            keyb += "_"
        
        if not keya in self.bondDict:
            self.bondDict[keya] = {}
        if not keyb in self.bondDict:
            self.bondDict[keyb] = {}
        
        self.bondDict[keya][keyb] = (bondMin, bondMax)
        self.bondDict[keyb][keya] = (bondMin, bondMax)
    
    def write(self, filename):
        """
        Write new atomic information file.
        
        """
        f = open(filename, "w")
        
        for key, value in sorted(six.iteritems(self.atomicNumberDict), key=lambda k_v: (k_v[1], k_v[0])):
            if key[1] == "_":
                sym = key[0]
            
            else:
                sym = key
            
            string = "%-3d  %-6f  %-20s  %-2s  %-6f  %-6f  %-6f  %-6f\n" % (value, self.atomicMassDict[key], self.atomNameDict[key],
                                                                            sym, self.covalentRadiusDict[key], self.RGBDict[key][0],
                                                                            self.RGBDict[key][1], self.RGBDict[key][2])
            
            f.write(string)
        
        f.close()
    
    def writeBonds(self, filename):
        """
        Write new bonds file.
        
        """
        f = open(filename, "w")
        
        for syma, d in list(self.bondDict.items()):
            if syma[1] == "_":
                syma = syma[0]
            else:
                syma = syma
            
            for symb, range in list(d.items()):
                if symb[1] == "_":
                    symb = symb[0]
                else:
                    symb = symb
                
                if syma > symb:
                    continue
                
                string = "%-2s  %-2s  %-6f  %-6f\n" % (syma, symb, range[0], range[1])
                
                f.write(string)
        
        f.close()
    
    def atomicNumber(self, sym):
        """
        Return atomic number of given element.
        
        """
        try:
            value = self.atomicNumberDict[sym]
        except KeyError:
            sys.exit(__name__+": ERROR: no atomic number for "+sym)
        
        return value

    def atomicMass(self, sym):
        """
        Return atomic mass of given element.
        
        """
        try:
            value = self.atomicMassDict[sym]
        except KeyError:
            sys.exit(__name__+": ERROR: no atomic mass for "+sym)
        
        return value
    
    def atomName(self, sym):
        """
        Return name of given element.
        
        """
        try:
            value = self.atomNameDict[sym]
        except KeyError:
            sys.exit(__name__+": ERROR: no atom name for "+sym)
        
        return value
    
    def covalentRadius(self, sym):
        """
        Return covalent radius of given element.
        
        """
        try:
            value = self.covalentRadiusDict[sym]
        except KeyError:
            sys.exit(__name__+": ERROR: no covalent radius for "+sym)
        
        return value
    
    def RGB(self, sym):
        """
        Return RGB of given element.
        
        """
        try:
            value = self.RGBDict[sym]
        except KeyError:
            sys.exit(__name__+": ERROR: no RGB for "+sym)
        
        return value
    
    def updateRGB(self, sym, R, G, B):
        """
        Update the RGB values for given specie.
        
        """
        self.RGBDict[sym][0] = R
        self.RGBDict[sym][1] = G
        self.RGBDict[sym][2] = B
    
    def updateCovalentRadius(self, sym, radius):
        """
        Update covalent radius values for given specie.
        
        """
        self.covalentRadiusDict[sym] = radius


################################################################################
# create global Atomic information object
elements = Elements()

################################################################################
# default atoms data

_defaultAtomsFile = """1    1.008000  Hydrogen              H   0.310000  0.000000  1.000000  0.000000
2      4.002602  Helium                He  0.280000  1.000000  0.772549  0.000000
3      6.940000  Lithium               Li  1.280000  0.800000  0.500000  1.000000
4      9.012183  Beryllium             Be  0.960000  1.000000  0.000000  0.000000
5     10.810000  Boron                 B   0.840000  0.149000  0.490000  0.670000
6     12.011200  Carbon                C   0.770000  0.200000  0.200000  0.200000
7     14.007000  Nitrogen              N   0.710000  1.000000  0.000000  0.000000
8     15.999000  Oxygen                O   0.660000  1.000000  0.050000  0.050000
9     18.998403  Fluorine              F   0.640000  0.000000  1.000000  1.000000
10    20.179700  Neon                  Ne  0.580000  1.000000  0.000000  0.000000
11    22.989769  Sodium                Na  1.660000  0.670588  0.360784  0.949019
12    24.305000  Magnesium             Mg  1.410000  0.870000  0.870000  0.820000
13    26.981539  Aluminum              Al  1.210000  0.750000  0.760000  0.760000
14    28.085500  Silicon               Si  1.110000  1.000000  0.900000  0.100000
15    30.973762  Phosphorus            P   1.070000  0.650000  0.000000  0.650000
16    32.060000  Sulfur                S   1.050000  1.000000  0.000000  0.000000
17    35.450000  Chlorine              Cl  1.020000  1.000000  0.000000  0.000000
18    39.948000  Argon                 Ar  1.060000  0.000000  1.000000  0.000000
19    39.098300  Potassium             K   2.030000  1.000000  0.000000  0.000000
20    40.078000  Calcium               Ca  1.760000  0.120000  0.840000  0.000000
21    44.955908  Scandium              Sc  1.700000  1.000000  0.000000  0.000000
22    47.867000  Titanium              Ti  1.600000  0.900000  0.900000  0.900000
23    50.941500  Vanadium              V   1.530000  1.000000  0.000000  0.000000
24    51.996100  Chromium              Cr  1.390000  1.000000  0.000000  0.000000
25    54.938044  Manganese             Mn  1.390000  1.000000  0.000000  0.000000
26    55.845000  Iron                  Fe  1.320000  0.000000  0.720000  0.000000
27    58.933194  Cobalt                Co  1.260000  1.000000  0.000000  0.000000
28    58.693400  Nickel                Ni  1.240000  1.000000  0.000000  0.000000
29    63.546000  Copper                Cu  1.320000  1.000000  0.000000  0.000000
30    65.380000  Zinc                  Zn  1.220000  0.520000  0.800000  0.980000
31    69.723000  Gallium               Ga  1.220000  0.000000  0.000000  1.000000
32    72.630000  Germanium             Ge  1.220000  1.000000  0.000000  0.000000
33    74.921595  Arsenic               As  1.190000  1.000000  0.000000  0.000000
34    78.971000  Selenium              Se  1.200000  1.000000  0.000000  0.000000
35    79.904000  Bromine               Br  1.200000  1.000000  0.000000  0.000000
36    83.798000  Krypton               Kr  1.160000  1.000000  0.000000  0.000000
37    85.467800  Rubidium              Rb  2.200000  1.000000  0.000000  0.000000
38    87.620000  Strontium             Sr  1.950000  1.000000  0.000000  0.000000
39    88.905840  Yttrium               Y   1.900000  1.000000  0.000000  0.000000
40    91.224000  Zirconium             Zr  1.750000  0.580392  0.878431  0.878431
41    92.906370  Niobium               Nb  1.640000  0.460000  0.590000  0.620000
42    95.950000  Molybdenum            Mo  1.540000  1.000000  0.000000  0.000000
43    98.000000  Technetium            Tc  1.470000  1.000000  0.000000  0.000000
44   101.070000  Ruthenium             Ru  1.460000  1.000000  0.000000  0.000000
45   102.905500  Rhodium               Rh  1.420000  1.000000  0.000000  0.000000
46   106.420000  Palladium             Pd  1.390000  1.000000  0.000000  0.000000
47   107.868200  Silver                Ag  1.450000  0.753000  0.753000  0.753000
48   112.414000  Cadmium               Cd  1.440000  0.000000  1.000000  0.000000
49   114.818000  Indium                In  1.420000  1.000000  0.000000  0.000000
50   118.710000  Tin                   Sn  1.390000  1.000000  0.000000  0.000000
51   121.760000  Antimony              Sb  1.390000  1.000000  0.000000  0.000000
52   127.600000  Tellurium             Te  1.380000  1.000000  0.000000  0.000000
53   126.904470  Iodine                I   1.390000  1.000000  0.000000  0.000000
54   131.293000  Xenon                 Xe  1.400000  1.000000  0.000000  1.000000
55   132.905452  Cesium                Cs  2.440000  1.000000  0.000000  0.000000
56   137.327000  Barium                Ba  2.150000  1.000000  0.000000  0.000000
57   138.905470  Lanthanum             La  2.070000  1.000000  0.000000  0.000000
58   140.116000  Cerium                Ce  2.040000  1.000000  1.000000  0.780392
59   140.907660  Praseodymium          Pr  2.030000  1.000000  0.000000  0.000000
60   144.242000  Neodymium             Nd  2.010000  1.000000  0.000000  0.000000
61   145.000000  Promethium            Pm  1.990000  1.000000  0.000000  0.000000
62   150.360000  Samarium              Sm  1.980000  1.000000  0.000000  0.000000
63   151.964000  Europium              Eu  1.980000  1.000000  0.000000  0.000000
64   157.250000  Gadolinium            Gd  1.960000  1.000000  0.000000  0.000000
65   158.925350  Terbium               Tb  1.940000  1.000000  0.000000  0.000000
66   162.500000  Dysprosium            Dy  1.920000  1.000000  0.000000  0.000000
67   164.930330  Holmium               Ho  1.920000  1.000000  0.000000  0.000000
68   167.259000  Erbium                Er  1.890000  1.000000  0.964706  0.149020
69   168.934220  Thulium               Tm  1.900000  1.000000  0.000000  0.000000
70   173.045000  Ytterbium             Yb  1.870000  1.000000  0.000000  0.000000
71   174.966800  Lutetium              Lu  1.870000  1.000000  0.000000  0.000000
72   178.490000  Hafnium               Hf  1.750000  0.500000  0.500000  0.000000
73   180.947880  Tantalum              Ta  1.700000  1.000000  0.000000  0.000000
74   183.840000  Tungsten              W   1.620000  1.000000  0.000000  0.000000
75   186.207000  Rhenium               Re  1.510000  1.000000  0.000000  0.000000
76   190.230000  Osmium                Os  1.440000  1.000000  0.000000  0.000000
77   192.217000  Iridium               Ir  1.410000  1.000000  0.000000  0.000000
78   195.084000  Platinum              Pt  1.360000  1.000000  0.000000  0.000000
79   196.966569  Gold                  Au  1.360000  1.000000  0.840000  0.000000
80   200.592000  Mercury               Hg  1.320000  1.000000  0.000000  0.000000
81   204.380000  Thallium              Tl  1.450000  1.000000  0.000000  0.000000
82   207.200000  Lead                  Pb  1.460000  1.000000  0.000000  0.000000
83   208.980400  Bismuth               Bi  1.480000  1.000000  0.000000  0.000000
84   209.000000  Polonium              Po  1.400000  1.000000  0.000000  0.000000
85   210.000000  Astatine              At  1.500000  1.000000  0.000000  0.000000
86   222.000000  Radon                 Rn  1.500000  1.000000  0.000000  0.000000
87   223.000000  Francium              Fr  2.600000  1.000000  0.000000  0.000000
88   226.000000  Radium                Ra  2.210000  1.000000  0.000000  0.000000
89   227.000000  Actinium              Ac  2.150000  1.000000  0.000000  0.000000
90   232.037700  Thorium               Th  2.060000  1.000000  0.000000  0.000000
91   231.035880  Protactinium          Pa  2.000000  1.000000  0.000000  0.000000
92   238.028910  Uranium               U   1.960000  1.000000  0.000000  0.000000
93   237.000000  Neptunium             Np  1.900000  1.000000  0.000000  0.000000
94   244.000000  Plutonium             Pu  1.870000  1.000000  0.000000  0.000000
95   243.000000  Americium             Am  1.800000  1.000000  0.000000  0.000000
96   247.000000  Curium                Cm  1.690000  1.000000  0.000000  0.000000
97   247.000000  Berkelium             Bk  1.700000  1.000000  0.000000  0.000000
98   251.000000  Californium           Cf  1.000000  1.000000  0.000000  0.000000
99   252.000000  Einsteinium           Es  1.000000  1.000000  0.000000  0.000000
100  257.000000  Fermium               Fm  1.000000  1.000000  0.000000  0.000000
101  258.000000  Mendelevium           Md  1.000000  1.000000  0.000000  0.000000
102  259.000000  Nobelium              No  1.000000  1.000000  0.000000  0.000000
103  266.000000  Lawrencium            Lr  1.000000  1.000000  0.000000  0.000000
104  267.000000  Rutherfordium         Rf  1.000000  1.000000  0.000000  0.000000
105  268.000000  Dubnium               Db  1.000000  1.000000  0.000000  0.000000
106  269.000000  Seaborgium            Sg  1.000000  1.000000  0.000000  0.000000
107  270.000000  Bohrium               Bh  1.000000  1.000000  0.000000  0.000000
108  269.000000  Hassium               Hs  1.000000  1.000000  0.000000  0.000000
109  278.000000  Meitnerium            Mt  1.000000  1.000000  0.000000  0.000000
110  281.000000  Darmstadtium          Ds  1.280000  1.000000  0.000000  0.000000
111  282.000000  Roentgenium           Rg  1.210000  1.000000  0.000000  0.000000
112  285.000000  Copernicium           Cn  1.220000  1.000000  0.000000  0.000000
113  270.000000  Hydrogend(Water)      HW  0.800000  1.000000  1.000000  1.000000
114  271.000000  Oxygen(Water)         OW  1.200000  1.000000  0.000000  0.000000
115   55.845000  Iron(Fe3+)            FF  1.500000  0.000000  1.000000  0.882353
"""

################################################################################
# default bonds file
_defaultBondsFile = """H   O   0.000000  1.100000
Pu  Pu  0.000000  3.700000
Nb  O   1.600000  2.500000
Nb  Nb  0.000000  0.000000
HW  OW  0.000000  1.100000
Ti  Ti  0.000000  3.000000
Te  Te  0.000000  4.000000
Hf  O   0.000000  2.500000
Hf  Hf  0.000000  0.000000
Hf  Mg  0.000000  0.000000
Mg  O   0.000000  2.500000
Mg  Mg  0.000000  0.000000
Zn  Zn  1.000000  3.000000
Ag  O   0.000000  2.500000
Ag  Zn  1.000000  2.000000
Ag  Ag  1.000000  3.000000
Ag  Ti  1.000000  3.000000
Ca  O   0.000000  3.000000
Al  O   1.600000  2.500000
Al  Nb  0.000000  0.000000
Al  Al  0.000000  0.000000
Cd  Te  0.000000  4.000000
Cd  Cd  0.000000  4.000000
O   Zn  1.000000  3.500000
O   Ti  0.000000  2.600000
Au  O   0.000000  0.000000
Au  Hf  0.000000  0.000000
Au  Mg  0.000000  0.000000
Au  Au  2.500000  3.000000
Ga  Ga  0.000000  3.700000
Ga  Pu  0.000000  3.700000
C   C   0.000000  1.800000
B   O   0.000000  2.100000
Si  O   0.000000  2.100000
FF  O   0.000000  2.500000
Fe  O   0.000000  2.500000
P   O   0.000000  2.500000
Ce  O   0.000000  2.800000
Fe  Fe  2.000000  2.600000
Fe  P   2.000000  2.600000
O   O   0.800000  0.800000
"""

################################################################################
def resetBonds(fname="bonds.IN"):
    """Write the default bonds file to the data location."""
    logger = logging.getLogger(__name__)
    logger.debug("Resetting the bonds settings")
    createDataFile(fname, _defaultBondsFile)
    bondfile = dataPath("bonds.IN")
    elements.readBonds(bondfile)

################################################################################
def resetAtoms(fname="atoms.IN"):
    """Write the default atoms file to the data location."""
    logger = logging.getLogger(__name__)
    logger.debug("Resetting the atoms settings")
    createDataFile(fname, _defaultAtomsFile)
    atomfile = dataPath("atoms.IN")
    elements.read(atomfile)
    
################################################################################
def initialise():
    """
    Initialise the module.
    
    Create and read in Elements object.
    
    """
    logger = logging.getLogger(__name__)
    logger.debug("Initialising elements/bonds")
    
    # atoms data
    atomfile = dataPath("atoms.IN")
    if not os.path.exists(atomfile):
        # create the default version
        logger.debug("Atoms file does not exist in data dir; creating: '{0}'".format(atomfile))
        createDataFile("atoms.IN", _defaultAtomsFile)
    logger.debug("Reading atoms file: '{0}'".format(atomfile))
    elements.read(atomfile)
    
    # bonds data
    bondfile = dataPath("bonds.IN")
    if not os.path.exists(bondfile):
        # create the default version
        logger.debug("Bonds file does not exist in data dir; creating: '{0}'".format(bondfile))
        createDataFile("bonds.IN", _defaultBondsFile)
    logger.debug("Reading bonds file: '{0}'".format(bondfile))
    elements.readBonds(bondfile)

################################################################################
if __name__ == '__main__':
    pass
else:
    initialise()
