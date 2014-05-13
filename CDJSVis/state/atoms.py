
"""
Provides information about elements.

Eg. atomic mass, radius, etc.

@author: Chris Scott

"""
import os
import sys

from ..visutils.utilities import resourcePath


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
                print "ERROR LEN WRONG"
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
                print "BAD LEN"
                continue
            
            keya = array[0]
            keyb = array[1]
            bondMin = float(array[2])
            bondMax = float(array[3])
            
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
        
        f.close()
        
#        for keya in self.bondDict:
#            for keyb, val in self.bondDict[keya].items():
#                print "%s - %s: %f -> %f" % (keya, keyb, val[0], val[1])
    
    def write(self, filename):
        """
        Write new atomic information file.
        
        """
        f = open(filename, "w")
        
        for key, value in sorted(self.atomicNumberDict.iteritems(), key=lambda (k, v): (v, k)):
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
        
        for syma, d in self.bondDict.items():
            if syma[1] == "_":
                syma = syma[0]
            else:
                syma = syma
            
            for symb, range in d.items():
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
def initialise():
    """
    Initialise the module.
    
    Create and read in Elements object.
    
    """
    filename = resourcePath("atoms.IN")
    bondfile = resourcePath("bonds.IN")
    
    elements.read(filename)
    elements.readBonds(bondfile)
    

################################################################################
if __name__ == '__main__':
    pass

else:
    initialise()
