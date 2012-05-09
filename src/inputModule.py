
"""
File input methods.

@author: Chris Scott

"""

import os
import sys
import shutil
import copy

import numpy as np

import utilities
from visclibs import input_c
import atoms



################################################################################
def readFile(filename, tmpLocation, lattice, fileType, state, log, refLattice=None):
    """
    Read file.
    
    Status:
        0  : success
        -1 : could not find file
        -2 : LBOMD XYZ NAtoms not matching
        -3 : unrecognised LBOMD XYZ format
        
    """
    # read file
    filepath, zipFlag = checkForZipped(filename, tmpLocation)
    if zipFlag == -1:
        return -1
    
    log("Reading file %s (%s, %s)" % (filename, fileType, state))
    
    # first read the header
    status = 0
    if fileType == "LBOMD":
        
        if state == "ref":
            status = readLBOMDRef(filepath, tmpLocation, lattice, fileType, state, log) 
        
        elif state == "input":
            if refLattice is None:
                print "MUST PASS REF LATTICE WHEN READING LBOMD XYZ INPUT"
                sys.exit(35)
            
            status = readLBOMDInput(filepath, tmpLocation, lattice, fileType, state, log, refLattice)
            
    elif fileType == "DAT":
        
        status = readLattice(filepath, tmpLocation, lattice, fileType, state, log)
    
    cleanUnzipped(filepath, zipFlag)
    
    return status


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
    input_c.readLatticeLBOMD( filename, lattice.specie, lattice.pos, lattice.charge, specieListTemp, 
                              specieCountTemp, lattice.maxPos, lattice.minPos )
    
    # build specie list and counter in lattice object
    log("Building specie list", 2, 1)
    NSpecies = 0
    for i in range(maxNumSpecies):
        if specieListTemp[i] == 'XX':
            break
        else:
            NSpecies += 1
            
    # allocate specieList/Counter arrays
    dt = np.dtype((str, 2))
    lattice.specieList = np.empty(NSpecies, dt)
    lattice.specieCount = np.empty(NSpecies, np.int32)
    lattice.specieMass = np.empty(NSpecies, np.float64)
    lattice.specieCovalentRadius = np.empty(NSpecies, np.float64)
    lattice.specieRGB = np.empty((NSpecies, 3), np.float64)
    for i in xrange(NSpecies):
        lattice.specieList[i] = specieListTemp[i]
        lattice.specieCount[i] = specieCountTemp[i]
        
        lattice.specieMass[i] = atoms.atomicMass(lattice.specieList[i])
        lattice.specieCovalentRadius[i] = atoms.covalentRadius(lattice.specieList[i])
        rgbtemp = atoms.RGB(lattice.specieList[i])
        lattice.specieRGB[i][0] = rgbtemp[0]
        lattice.specieRGB[i][1] = rgbtemp[1]
        lattice.specieRGB[i][2] = rgbtemp[2]
        
        log("new specie: " + specieListTemp[i] + " (" + atoms.atomName(specieListTemp[i]) + ")", 2, 2)
        log(str(specieCountTemp[i]) + " " + atoms.atomName(specieListTemp[i]) + " atoms", 2, 3)
    
    return 0

################################################################################
def readLBOMDInput(filename, tmpLocation, lattice, fileType, state, log, refLattice):
    
    f = open(filename)
            
    line = f.readline().strip()
    NAtoms = int(line)
    
    if NAtoms != refLattice.NAtoms:
        return -2
    
    # simulation time
    line = f.readline().strip()
    simTime = float(line)
    
    # read first line to get format
    line = f.readline().strip()
    array = line.split()
    if len(array) == 6:
        xyzformat = 0
    else:
        return -3
    
    f.close()
    
    lattice.reset(NAtoms)
    lattice.simTime = simTime
    
    tmpForceArray = np.empty(3, np.float64)
    
    # call clib
    input_c.readLBOMDXYZ(filename, lattice.pos, lattice.charge, lattice.KE, lattice.PE, tmpForceArray, 
                         lattice.maxPos, lattice.minPos, xyzformat)
    
    
    # copy charge if not included in xyz
    for i in xrange(NAtoms):
        if xyzformat == 0:
            lattice.charge[i] = refLattice.charge[i]
        
        lattice.specie[i] = refLattice.specie[i]
    
    # copy specie arrays for refLattice
    lattice.specieList = copy.deepcopy(refLattice.specieList)
    lattice.specieCount = copy.deepcopy(refLattice.specieCount)
    lattice.specieMass = copy.deepcopy(refLattice.specieMass)
    lattice.specieCovalentRadius = copy.deepcopy(refLattice.specieCovalentRadius)
    lattice.specieRGB = copy.deepcopy(refLattice.specieRGB)
    
    return 0


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
    specieCountTemp = np.zeros( maxNumSpecies+1, np.int32 )
    
    tmpForceArray = np.empty(3, np.float64)
    
    # call c lib
    input_c.readRef( filename, lattice.specie, lattice.pos, lattice.charge, lattice.KE, lattice.PE, tmpForceArray, 
                     specieListTemp, specieCountTemp, lattice.maxPos, lattice.minPos )
    
    # build specie list and counter in lattice object
    log("Building specie list", 2, 1)
    NSpecies = 0
    for i in range(maxNumSpecies):
        if specieListTemp[i] == 'XX':
            break
        else:
            NSpecies += 1
            
    # allocate specieList/Counter arrays
    dt = np.dtype((str, 2))
    lattice.specieList = np.empty(NSpecies, dt)
    lattice.specieCount = np.empty(NSpecies, np.int32)
    lattice.specieMass = np.empty(NSpecies, np.float64)
    lattice.specieCovalentRadius = np.empty(NSpecies, np.float64)
    lattice.specieRGB = np.empty((NSpecies, 3), np.float64)
    for i in xrange(NSpecies):
        lattice.specieList[i] = specieListTemp[i]
        lattice.specieCount[i] = specieCountTemp[i]
        
        lattice.specieMass[i] = atoms.atomicMass(lattice.specieList[i])
        lattice.specieCovalentRadius[i] = atoms.covalentRadius(lattice.specieList[i])
        rgbtemp = atoms.RGB(lattice.specieList[i])
        lattice.specieRGB[i][0] = rgbtemp[0]
        lattice.specieRGB[i][1] = rgbtemp[1]
        lattice.specieRGB[i][2] = rgbtemp[2]
        
        log("new specie: " + specieListTemp[i] + " (" + atoms.atomName(specieListTemp[i]) + ")", 2, 2)
        log(str(specieCountTemp[i]) + " " + atoms.atomName(specieListTemp[i]) + " atoms", 2, 3)

    return 0


################################################################################
def checkForZipped(filename, tmpLocation):
    
    if os.path.exists(filename):
        fileLocation = '.'
        zipFlag = 0
    
    else:
        if os.path.exists(filename + '.bz2'):
            command = "bzcat -k %s.bz2 > " % (filename)
        
        elif os.path.exists(filename + '.gz'):
            command = "gzcat %s.gz > " % (filename)
            
        else:
            return (None, -1)
            
        fileLocation = tmpLocation
        command = command + os.path.join(fileLocation, filename)
        os.system(command)
        zipFlag = 1
        
    filepath = os.path.join(fileLocation, filename)
    if not os.path.exists(filepath):
        return (None, -1)
        
    return (filepath, zipFlag)



################################################################################
def cleanUnzipped(filepath, zipFlag):
    
    if zipFlag:
        os.unlink(filepath)


