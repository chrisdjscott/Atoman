
"""
File input methods.

author: Chris Scott
last edited: February 2012
"""

import os
import sys
import shutil

import numpy as np

import utilities
from visclibs import input_c
import atoms



################################################################################
def readFile(filename, tmpLocation, lattice, fileType, state, log):
    
    # read file
    loc = checkForZipped(filename, tmpLocation)
    if loc == -1:
        return -1
    
    log("Reading file %s (%s, %s)" % (filename, fileType, state))
    
    filename = os.path.join(loc, filename)
    
    # first read the header
    if fileType == "LBOMD":
        
        if state == "ref":
            readLBOMDRef(filename, tmpLocation, lattice, fileType, state, log) 
        
        elif state == "input":
            readLBOMDInput(filename, tmpLocation, lattice, fileType, state, log)
            
    elif fileType == "DAT":
        
        readLattice(filename, tmpLocation, lattice, fileType, state, log)
    
    cleanUnzipped(loc)
    
    return 0


################################################################################
def readLattice(filename, tmpLocation, lattice, fileType, state, log):
    
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
    log("Building specie list", 2, 1)
    for i in range(maxNumSpecies):
        if specieListTemp[i] == 'XX':
            break
        else:
            lattice.specieList.append( specieListTemp[i] )
            lattice.specieCount.append( specieCountTemp[i] )
            log("new specie: "+specieListTemp[i] +" (" + atoms.atomName(specieListTemp[i]) + ")", 2, 2)
            log(str(specieCountTemp[i]) + " " + atoms.atomName(specieListTemp[i]) + " atoms", 2, 3)



################################################################################
def readLBOMDInput(filename, tmpLocation, lattice, fileType, state, log):
    
    f = open(filename)
            
    line = f.readline().strip()
    NAtoms = int(line)
    
    line = f.readline().strip()
    simTime = float(line)
    
    f.close()
    


################################################################################
def readLBOMDRef(filename, tmpLocation, lattice, fileType, state, log):
    
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
    log("Building specie list", 2, 1)
    for i in range(maxNumSpecies):
        if specieListTemp[i] == 'XX':
            break
        else:
            lattice.specieList.append( specieListTemp[i] )
            lattice.specieCount.append( specieCountTemp[i] )
            log("new specie: "+specieListTemp[i] +" (" + atoms.atomName(specieListTemp[i]) + ")", 2, 2)
            log(str(specieCountTemp[i]) + " " + atoms.atomName(specieListTemp[i]) + " atoms", 2, 3)





################################################################################
def checkForZipped(filename, tmpLocation):
    
    if os.path.exists(filename):
        fileLocation = '.'
    else:
        if os.path.exists(filename + '.bz2'):
            command = "bzcat -k %s.bz2 > " % (filename)
            zippedFile = file+'.bz2'
        elif os.path.exists(filename + '.gz'):
            command = "zcat %s.gz > " % (filename)
            zippedFile = filename+'.gz'
        else:
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


