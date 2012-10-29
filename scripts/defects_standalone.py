#!/usr/bin/env python

"""
Standalone defects scripts using vis clibs.

@author: Chris Scott

"""
import os
import sys
dirname = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(dirname, "..", "src")))
import copy
import tempfile
import shutil

import numpy as np

import lattice as Lattice
import inputModule
from atoms import elements

from visclibs import input_c
from visclibs import defects_c


################################################################################

def readRef(filename, tmpDir):
    lattice = Lattice.Lattice()
    
    filepath, zipFlag = inputModule.checkForZipped(filename, tmpDir)
    
    f = open(filepath)
            
    line = f.readline().strip()
    NAtoms = int(line)
    
    line = f.readline().strip()
    array = line.split()
    lattice.setDims(array)
    
    f.close()
    
    lattice.reset(NAtoms)
    
    print "%d atoms" % (NAtoms,)
    
    # temporary specie list and counter arrays
    maxNumSpecies = 20 ## if there are more than 20 species these must be changed
    dt = np.dtype((str, 2))
    specieListTemp = np.empty( maxNumSpecies+1, dt ) 
    specieCountTemp = np.zeros( maxNumSpecies+1, np.int32 )
    
    tmpForceArray = np.empty(3, np.float64)
    
    # call c lib
    input_c.readRef( filepath, lattice.specie, lattice.pos, lattice.charge, lattice.KE, lattice.PE, tmpForceArray, 
                     specieListTemp, specieCountTemp, lattice.maxPos, lattice.minPos )
    
    inputModule.cleanUnzipped(filepath, zipFlag)
    
    # build specie list and counter in lattice object
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
        
        lattice.specieMass[i] = elements.atomicMass(lattice.specieList[i])
        lattice.specieCovalentRadius[i] = elements.covalentRadius(lattice.specieList[i])
        rgbtemp = elements.RGB(lattice.specieList[i])
        lattice.specieRGB[i][0] = rgbtemp[0]
        lattice.specieRGB[i][1] = rgbtemp[1]
        lattice.specieRGB[i][2] = rgbtemp[2]
        
        print "%d %s (%s) atoms" % (specieCountTemp[i], specieListTemp[i], elements.atomName(specieListTemp[i]))
    
    return lattice

################################################################################

def readLBOMDInput(filename, refLattice, tmpDir):
    
    lattice = Lattice.Lattice()
    
    filepath, zipFlag = inputModule.checkForZipped(filename, tmpDir)
    
    f = open(filepath)
            
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
        
    elif len(array) == 7:
        xyzformat = 1
    
    else:
        return -3
    
    f.close()
    
    lattice.reset(NAtoms)
    lattice.simTime = simTime
    
    tmpForceArray = np.empty(3, np.float64)
    
    # call clib
    input_c.readLBOMDXYZ(filepath, lattice.pos, lattice.charge, lattice.KE, lattice.PE, tmpForceArray, 
                         lattice.maxPos, lattice.minPos, xyzformat)
    
    inputModule.cleanUnzipped(filepath, zipFlag)
    
    # copy charge if not included in xyz
    for i in xrange(refLattice.NAtoms):
        if xyzformat == 0:
            lattice.charge[i] = refLattice.charge[i]
        
        lattice.specie[i] = refLattice.specie[i]
    
    # copy specie arrays from refLattice
    lattice.specieList = copy.deepcopy(refLattice.specieList)
    lattice.specieCount = copy.deepcopy(refLattice.specieCount)
    lattice.specieMass = copy.deepcopy(refLattice.specieMass)
    lattice.specieCovalentRadius = copy.deepcopy(refLattice.specieCovalentRadius)
    lattice.specieRGB = copy.deepcopy(refLattice.specieRGB)
    
    return lattice

################################################################################

def main(fileOutput=True, a=0, b=200, verbose=True):
    # tmp dir
    tmpDirectory = tempfile.mkdtemp(dir="/tmp")
    
    try:
        # read ref
        refLattice = readRef("animation-reference.xyz", tmpDirectory)
        
        f = open("defect_counts.csv", "w")
        f.write("Time (fs),Total defects,Vacancies,Interstitials,Pu-Ga split interstitials\n")
        f.close()
        
        for ii in xrange(a, b):
            fn = "PuGaH%04d.xyz" % ii
            
            # read input
            inputLattice = readLBOMDInput(fn, refLattice, tmpDirectory)
            
            # set up arrays
            interstitials = np.empty(inputLattice.NAtoms, np.int32)
            splitInterstitials = np.empty(inputLattice.NAtoms, np.int32)
            vacancies = np.empty(refLattice.NAtoms, np.int32)
            antisites = np.empty(refLattice.NAtoms, np.int32)
            onAntisites = np.empty(refLattice.NAtoms, np.int32)
            
            # set up excluded specie arrays
            visibleSpecieList = ["Pu", "Ga"]
            exclSpecs = []
            for i in xrange(len(inputLattice.specieList)):
                spec = inputLattice.specieList[i]
                if spec not in visibleSpecieList:
                    exclSpecs.append(i)
            exclSpecsInput = np.empty(len(exclSpecs), np.int32)
            for i in xrange(len(exclSpecs)):
                exclSpecsInput[i] = exclSpecs[i]
            
            exclSpecs = []
            for i in xrange(len(refLattice.specieList)):
                spec = refLattice.specieList[i]
                if spec not in visibleSpecieList:
                    exclSpecs.append(i)
            exclSpecsRef = np.empty(len(exclSpecs), np.int32)
            for i in xrange(len(exclSpecs)):
                exclSpecsRef[i] = exclSpecs[i]
            
            # specie counter arrays
            vacSpecCount = np.zeros( len(refLattice.specieList), np.int32 )
            intSpecCount = np.zeros( len(inputLattice.specieList), np.int32 )
            antSpecCount = np.zeros( len(refLattice.specieList), np.int32 )
            onAntSpecCount = np.zeros( (len(refLattice.specieList), len(inputLattice.specieList)), np.int32 )
            splitIntSpecCount = np.zeros( (len(inputLattice.specieList), len(inputLattice.specieList)), np.int32 )
            
            NDefectsByType = np.zeros(6, np.int32)
            
            # set min/max pos to lattice (for boxing)
            minPos = refLattice.minPos
            maxPos = refLattice.maxPos
            
            defectCluster = np.empty(0, np.int32)
            
            # call C library
            defects_c.findDefects(1, 1, 0, NDefectsByType, vacancies, 
                                  interstitials, antisites, onAntisites, exclSpecsInput, exclSpecsRef, inputLattice.NAtoms, inputLattice.specieList,
                                  inputLattice.specie, inputLattice.pos, refLattice.NAtoms, refLattice.specieList, refLattice.specie, 
                                  refLattice.pos, refLattice.cellDims, np.ones(3, np.int32), 1.3, minPos, maxPos, 
                                  0, 3.5, defectCluster, vacSpecCount, intSpecCount, antSpecCount,
                                  onAntSpecCount, splitIntSpecCount, 3, 10, splitInterstitials, 
                                  1)
            
            # summarise
            NDef = NDefectsByType[0]
            NVac = NDefectsByType[1]
            NInt = NDefectsByType[2]
            NAnt = NDefectsByType[3]
            NSplit = NDefectsByType[5]
            vacancies.resize(NVac)
            interstitials.resize(NInt)
            antisites.resize(NAnt)
            onAntisites.resize(NAnt)
            splitInterstitials.resize(NSplit*3)
            
            # report counters
            if verbose:
                print "Found %d defects" % (NDef,)
                
                if 1:
                    print "%d vacancies" % (NVac,)
                    for i in xrange(len(refLattice.specieList)):
                        print "%d %s vacancies" % (vacSpecCount[i], refLattice.specieList[i])
                
                if 1:
                    print "%d interstitials" % (NInt + NSplit,)
                    for i in xrange(len(inputLattice.specieList)):
                        print "%d %s interstitials" % (intSpecCount[i], inputLattice.specieList[i])
                
                    if 1:
                        print "%d split interstitials" % (NSplit,)
                        for i in xrange(len(inputLattice.specieList)):
                            for j in xrange(i, len(inputLattice.specieList)):
                                if j == i:
                                    N = splitIntSpecCount[i][j]
                                else:
                                    N = splitIntSpecCount[i][j] + splitIntSpecCount[j][i]
                                print "%d %s - %s split interstitials" % (N, inputLattice.specieList[i], inputLattice.specieList[j])
            
            if fileOutput:
                f = open("defect_counts.csv", "a")
                
                for i in xrange(len(inputLattice.specieList)):
                    for j in xrange(i, len(inputLattice.specieList)):
                        if (inputLattice.specieList[i] == "Pu" and inputLattice.specieList[j] == "Ga") or (inputLattice.specieList[i] == "Ga" and inputLattice.specieList[j] == "Pu"):
                            NPuGaSplit = splitIntSpecCount[i][j] + splitIntSpecCount[j][i]
                            break
                
                f.write("%f,%d,%d,%d,%d\n" % (inputLattice.simTime, NDef, NVac, NInt+NSplit, NPuGaSplit))
                
                f.close()
    
    finally:
        shutil.rmtree(tmpDirectory)
    
    return NDef

################################################################################

def commandLineArgs():
    """
    Parse command line arguments.
    
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Defects script.")
    
#    parser.add_argument('NSimulations', type=int, help="Number of simulations.")
#    parser.add_argument('-l', dest="lkmcInputFile", default="lkmcInput.IN", help="LKMC input file (default=lkmcInput.IN)")
    parser.add_argument('-o', dest="outputDir", default="output.dir", help="Output directory (default=output.dir)")
    parser.add_argument('-d', dest="vacancyRadius", type=float, default=1.3, help="Vacancy radius (default=1.3)")
    parser.add_argument('-r', dest="renderDefects", action="store_true", default=False, help="Render defects (default=False)")
#    parser.add_argument('-m', dest="mdThermDir", default="therminput.dir", help="Directory containing thermalisation input files (default=therminput.dir)")
#    parser.add_argument('-d', dest="mdPostThermDir", default="posttherminput.dir", help="Directory containing post thermalisation input files (default=posttherminput.dir)")
#    parser.add_argument('-i', dest="runInputDir", default="runinput.dir", help="Directory containing simulation input files (default=runinput.dir)")
#    parser.add_argument('-n', dest="nodefile", default="nodes.IN", help="Node file (default=nodes.IN)")
    
    return parser.parse_args()

################################################################################

if __name__ == "__main__":
    args = commandLineArgs()
    main(args)
