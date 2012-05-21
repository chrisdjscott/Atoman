
"""
Provides information about elements.

Eg. atomic mass, radius, etc.

@author: Chris Scott

"""
import sys

from utilities import resourcePath


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
        self.NElements = 0
    
    def read(self, filename):
        """
        Read in atomic information from given file.
        
        """
        f = open( filename, "r" )
        
        # read into dictionaries    
        count = 0    
        for line in f:
            line = line.strip()
            
            array = line.split()
            
            key = array[3]
            if len(key) == 1:
                key = key + '_'
            
            self.atomicNumberDict[key] = int(array[0])
            self.atomicMassDict[key] = float(array[1])
            self.atomNameDict[key] = array[2]
            self.covalentRadiusDict[key] = float(array[4])
            self.RGBDict[key] = [float(array[5]), float(array[6]), float(array[7])]
            
            count += 1
        
        self.NElements = count
        
        f.close()
    
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
    filename = resourcePath("data/atoms.IN")
    
    elements.read(filename)
    

################################################################################
if __name__ == '__main__':
    pass

else:
    initialise()
