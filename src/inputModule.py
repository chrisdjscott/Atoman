
"""
File input methods.

author: Chris Scott
last edited: February 2012
"""

import os
import sys
import shutil

try:
    import numpy as np
except:
    sys.exit(__name__ +": ERROR: numpy not found")

try:
    import utilities
except:
    sys.exit(__name__+ ": ERROR: utilities not found")
try:
    from visclibs import input_c
except:
    sys.exit(__name__ +": ERROR: c_input not found")
try:
    import atoms
except:
    sys.exit(__name__ +": ERROR: c_input not found")



################################################################################
def readFile(filename, tmpLocation, lattice, fileType, state):
    
    # read file
    loc = checkForZipped(filename, tmpLocation)
    filename = os.path.join(loc, filename)
    
    print "READING", filename, fileType, state
    
    # first read the header
    if fileType == "LBOMD":
        
        if state == "ref":
            readLBOMDRef(filename, tmpLocation, lattice, fileType, state) 
        
        elif state == "input":
            readLBOMDInput(filename, tmpLocation, lattice, fileType, state)
            
    elif fileType == "DAT":
        
        readLattice(filename, tmpLocation, lattice, fileType, state)
        
        
    
    cleanUnzipped(loc)


################################################################################
def readLattice(filename, tmpLocation, lattice, fileType, state):
    
    f = open(filename)
            
    line = f.readline().strip()
    NAtoms = int(line)
    
    line = f.readline().strip()
    array = line.split()
    lattice.setDims(array)
    
    f.close()
    
    lattice.reset(NAtoms)
    
    # need temporary specie list and counter arrays
    maxNumSpecies = 20
    dt = np.dtype((str, 2))
    specieListTemp = np.empty( maxNumSpecies+1, dt ) 
    specieCountTemp = np.empty( maxNumSpecies+1, np.int32 )
    
    # call c lib
    input_c.readLatticeLBOMD( filename, lattice.sym, lattice.pos, lattice.charge, specieListTemp, specieCountTemp, lattice.maxPos, lattice.minPos )
    
    # build specie list and counter in lattice object
    print  __name__, "Building specie list"
    for i in range(maxNumSpecies):
        if specieListTemp[i] == 'XX':
            break
        else:
            lattice.specieList.append( specieListTemp[i] )
            lattice.specieCount.append( specieCountTemp[i] )
            print "  new specie: " + specieListTemp[i] + " (" + atoms.atomName(specieListTemp[i]) + ")"
            print "   ", str(specieCountTemp[i]) + " " + atoms.atomName(specieListTemp[i]) + " atoms"



################################################################################
def readLBOMDInput(filename, tmpLocation, lattice, fileType, state):
    
    f = open(filename)
            
    line = f.readline().strip()
    NAtoms = int(line)
    
    line = f.readline().strip()
    simTime = float(line)
    
    f.close()
    


################################################################################
def readLBOMDRef(filename, tmpLocation, lattice, fileType, state):
    
    f = open(filename)
            
    line = f.readline().strip()
    NAtoms = int(line)
    
    line = f.readline().strip()
    array = line.split()
    lattice.setDims(array)
    
    f.close()
    
    lattice.reset(NAtoms)
    
    # temporary specie list and counter arrays
    maxNumSpecies = 20 ## if there are more than 20 species these must be changed
    dt = np.dtype((str, 2))
    specieListTemp = np.empty( maxNumSpecies+1, dt ) 
    specieCountTemp = np.empty( maxNumSpecies+1, np.int32 )
    
    tmpForceArray = np.empty(3, np.float64)
    
    # call c lib
    input_c.readRef( filename, lattice.sym, lattice.pos, lattice.charge, lattice.KE, lattice.PE, tmpForceArray, specieListTemp, specieCountTemp, lattice.maxPos, lattice.minPos )
        
    # build specie list and counter in lattice object
    print "  Building specie list in python"
    for i in range(maxNumSpecies):
        if specieListTemp[i] == 'XX':
            break
        else:
            lattice.specieList.append( specieListTemp[i] )
            lattice.specieCount.append( specieCountTemp[i] )
            print "    found new specie: " + specieListTemp[i] + " (" + atoms.atomName(specieListTemp[i]) + ")"
            print "      " + str(specieCountTemp[i]) + " " + atoms.atomName(specieListTemp[i]) + " atoms"





################################################################################
def checkForZipped(filename, tmpLocation):
    
    if os.path.exists(filename):
        fileLocation = '.'
    else:
        if os.path.exists(filename + '.bz2'):
            print 'INFO: bzip exists'
            command = "bzcat -k %s.bz2 > " % (filename)
            zippedFile = file+'.bz2'
        elif os.path.exists(filename + '.gz'):
            print 'INFO: gzip exists'
            command = "zcat %s.gz > " % (filename)
            zippedFile = filename+'.gz'
        else:
            print "WARNING: cannot find file:", filename
            return -1
            
        fileLocation = tmpLocation
        command = command + os.path.join(fileLocation, filename)
        os.system(command)
                
    return fileLocation



################################################################################
def cleanUnzipped(fileLocation):
    
    if fileLocation == '.':
        pass
    else:
        shutil.rmtree(fileLocation)


