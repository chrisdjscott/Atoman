
################################################################################
## Copyright Chris Scott 2011
## Requires Atoms.IN parameter file
## Provides:
##        atomicNumber()
##        atomicMass()
##        atomName()
##        covalentRadius()
##        RGB()
################################################################################

import sys
import os

from utilities import resourcePath



# atomic number
def atomicNumber( sym ):
    global atomicNumberDict
    
    try:
        value = atomicNumberDict[sym]
    except:
        sys.exit(__name__+": ERROR: no atomic number for "+sym)
    
    return value


# atomic mass
def atomicMass(sym):
    global atomicMassDict
    
    try:
        value = atomicMassDict[sym]
    except:
        sys.exit(__name__+": ERROR: no atomic mass for "+sym)
    
    return value


# name of atom
def atomName(sym):
    global atomNameDict
        
    try:
        value = atomNameDict[sym]
    except:
        sys.exit(__name__+": ERROR: no atom name for "+sym)
    
    return value


# covalent radius
def covalentRadius(sym):
    global covalentRadiusDict
    
    try:
        value = covalentRadiusDict[sym]
    except:
        sys.exit(__name__+": ERROR: no covalent radius for "+sym)
    
    return value


# RGB values
def RGB(sym):
    global RGBDict
    
    try:
        value = RGBDict[sym]
    except:
        sys.exit(__name__+": ERROR: no RGB for "+sym)
    
    return value





# read atom data
def initialise():
    global atomicNumberDict, atomicMassDict, atomNameDict, covalentRadiusDict, RGBDict
    
    filename = resourcePath("data/atoms.IN")
    
    if os.path.exists( filename ):
        try:
            f = open( filename, "r" )
        except:
            sys.exit('error: could not open atoms file: ' + filename)
    else:
        sys.exit('error: could not find atoms file: ' + filename)    
    
    # read into dictionaries
    atomicNumberDict = {}
    atomicMassDict = {}
    atomNameDict = {}
    covalentRadiusDict = {}
    RGBDict = {}
    
    count = 0
    for line in f:
        line = line.strip()
        
        array = line.split()
        
        key = array[3]
        if len(key) == 1:
            key = key + '_'
        
        atomicNumberDict[key] = int(array[0])
        atomicMassDict[key] = float(array[1])
        atomNameDict[key] = array[2]
        covalentRadiusDict[key] = float(array[4])
        RGBDict[key] = [float(array[5]), float(array[6]), float(array[7])]
        
        count += 1
        
    f.close()
        


if __name__ == '__main__':
    pass
else:
    initialise()



