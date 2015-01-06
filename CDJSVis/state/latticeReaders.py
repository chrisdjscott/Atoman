
"""
Lattice reader objects.

@author: Chris Scott

"""
import os
import copy
import re
import logging

import numpy as np

from . import _input as input_c
from .atoms import elements
from ..visutils import utilities
from .lattice import Lattice


################################################################################

def basic_displayWarning(message):
    print "WARNING: %s" % message

def basic_displayError(message):
    print "ERROR: %s" % message

def basic_log(message):
    print message

################################################################################

class GenericLatticeReader(object):
    """
    Base lattice reader object.
    
    """
    def __init__(self, tmpLocation, log, displayWarning, displayError):
        self.tmpLocation = tmpLocation
        self.log = log
        self.currentFile = None
        self.displayWarning = displayWarning
        self.displayError = displayError
        self.requiresRef = False
        self.logger = logging.getLogger(__name__)
        self.formatIdentifiers = []
    
    def checkForZipped(self, filename):
        """
        Check if file exists (unzip if required)
        
        """
        if os.path.exists(filename) and (filename.endswith(".gz") or filename.endswith(".bz2")):
            if filename.endswith(".gz"):
                filename = filename[:-3]
                command = 'gzip -dc "%s.gz" > ' % filename
            
            elif filename.endswith(".bz2"):
                filename = filename[:-4]
                command = 'bzcat -k "%s.bz2" > ' % filename
            
            fileLocation = self.tmpLocation
            command = command + os.path.join(fileLocation, filename)
            os.system(command)
            zipFlag = 1
        
        elif os.path.exists(filename):
            fileLocation = '.'
            zipFlag = 0
        
        else:
            if os.path.exists(filename + '.bz2'):
                command = 'bzcat -k "%s.bz2" > ' % filename
            
            elif os.path.exists(filename + '.gz'):
                command = 'gzip -dc "%s.gz" > ' % filename
                
            else:
                return (None, -1)
                
            fileLocation = self.tmpLocation
            command = command + os.path.join(fileLocation, filename)
            os.system(command)
            zipFlag = 1
            
        filepath = os.path.join(fileLocation, filename)
        if not os.path.exists(filepath):
            return (None, -1)
            
        return filepath, zipFlag
    
    def cleanUnzipped(self, filepath, zipFlag):
        """
        Clean up unzipped file.
        
        """
        if zipFlag:
            os.unlink(filepath)
    
    def readFile(self, filename, rouletteIndex=None):
        """
        Read file.
        
        Status:
         0 : success
        -1 : could not find file
        -2 : LBOMD XYZ NAtoms not matching
        -3 : unrecognised LBOMD XYZ format
        -4 : file already loaded
        
        """
#        if os.path.abspath(filename) == self.currentFile:
#            print "ALREADY LOADED"
#            return -4, None
        
        self.logger.info("Reading file: '%s'", filename)
        
        # strip gz/bz2 extension
        if filename.endswith(".bz2"):
            filename = filename[:-4]
        elif filename.endswith(".gz"):
            filename = filename[:-3]
        
        filepath, zipFlag = self.checkForZipped(filename)
        if zipFlag == -1:
            self.displayWarning("Could not find file: "+filename)
            self.logger.warning("Could not find file: %s", filename)
            return -1, None
        
        status, state = self.readFileMain(filepath, rouletteIndex)
        
        self.cleanUnzipped(filepath, zipFlag)
        
        if status:
            if status == -1:
                self.displayWarning("Could not find file: "+filename)
                self.logger.warning("Could not find file: %s", filename)
            
            elif status == -2:
                self.displayWarning("LBOMD XYZ input NAtoms does not match reference!")
                self.logger.warning("LBOMD XYZ input NAtoms does not match reference!")
            
            elif status == -3:
                self.displayWarning("Unrecognised format for input file!")
                self.logger.warning("Unrecognised format for input file!")
            
            else:
                self.displayWarning("Input file read failed with error code: %s" % str(status))
                self.logger.warning("Input file read failed with error code: %s", str(status))
        
        elif state is not None:
            self.currentFile = os.path.abspath(filename)
        
        return status, state
    
    def readFileMain(self, filename):
        """
        Main read file routine (to be overriden).
        
        """
        return 1, None

################################################################################

class LbomdXYZReader(GenericLatticeReader):
    """
    Read LBOMD XYZ files.
    
    This is harder since they must be linked with a reference!
    
    """
    def __init__(self, tmpLocation, log, displayWarning, displayError):
        super(LbomdXYZReader, self).__init__(tmpLocation, log, displayWarning, displayError)
        
        self.formatIdentifiers.append([1, 1, 6])
        self.formatIdentifiers.append([1, 1, 7])
        self.formatIdentifiers.append([1, 1, 9])
        
        self.requiresRef = True
    
    def readFile(self, xyzfilename, refState, rouletteIndex=None):
        """
        We override the main readFile function too, so that
        we can pass the ref file name in.
        
        """
#        if os.path.abspath(xyzfilename) == self.currentFile:
#            print "ALREADY LOADED"
#            return -4, None
        
        self.logger.info("Reading file: '%s'", xyzfilename)
        
        # check input exists, unzip if necessary
        filepath, zipFlag = self.checkForZipped(xyzfilename)
        if zipFlag == -1:
            self.displayWarning("Could not find file: "+xyzfilename)
            self.logger.warning("Could not find file: %s", xyzfilename)
            return -1, None
        
        # read input
        status, state = self.readFileMain(filepath, refState, rouletteIndex)
        
        self.cleanUnzipped(filepath, zipFlag)
        
        if status:
            if status == -1:
                self.displayWarning("Could not find file: "+xyzfilename)
                self.logger.warning("Could not find file: %s", xyzfilename)
            
            elif status == -2:
                self.displayWarning("LBOMD XYZ input NAtoms does not match reference!")
                self.logger.warning("LBOMD XYZ input NAtoms does not match reference!")
            
            elif status == -3:
                self.displayWarning("Unrecognised format for LBOMD XYZ input file!")
                self.logger.warning("Unrecognised format for input file!")
        
        elif state is not None:
            self.currentFile = os.path.abspath(xyzfilename)
        
        return status, state
    
    def readFileMain(self, filename, refLattice, rouletteIndex):
        """
        Main read file routine.
        
        """
        state = Lattice()
        
        f = open(filename)
            
        line = f.readline().strip()
        NAtoms = int(line)
        
        if NAtoms != refLattice.NAtoms:
            return -2, None
        
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
        
        elif len(array) == 9:
            xyzformat = 2
        
        else:
            return -3, None
        
        f.close()
        
        state.reset(NAtoms)
        state.simTime = simTime
        
        self.logger.info("  %d atoms", NAtoms)
        
        tmpForceArray = np.empty(3, np.float64)
        
        # call clib
        status = input_c.readLBOMDXYZ(filename, state.atomID, state.pos, state.charge, state.KE, state.PE, tmpForceArray, 
                                      state.maxPos, state.minPos, xyzformat)
        
        if status:
            return status, None
        
        # copy charge if not included in xyz
        for i in xrange(refLattice.NAtoms):
            if xyzformat == 0:
                state.charge[i] = refLattice.charge[i]
            
            state.specie[i] = refLattice.specie[i]
        
        state.setDims(refLattice.cellDims)
        
        # copy specie arrays from refLattice
        state.specieList = copy.deepcopy(refLattice.specieList)
        state.specieCount = copy.deepcopy(refLattice.specieCount)
        state.specieMass = copy.deepcopy(refLattice.specieMass)
        state.specieCovalentRadius = copy.deepcopy(refLattice.specieCovalentRadius)
        state.specieRGB = copy.deepcopy(refLattice.specieRGB)
        state.specieAtomicNumber = copy.deepcopy(refLattice.specieAtomicNumber)
        
        for i in xrange(len(state.specieList)):
            self.logger.info("    %d %s (%s) atoms", state.specieCount[i], state.specieList[i], elements.atomName(state.specieList[i]))
        
        return 0, state
        

################################################################################

class LbomdRefReader(GenericLatticeReader):
    """
    Read LBOMD animation reference files.
    
    """
    def __init__(self, tmpLocation, log, displayWarning, displayError):
        super(LbomdRefReader, self).__init__(tmpLocation, log, displayWarning, displayError)
        
        self.formatIdentifiers.append([1, 3, 11])
    
    def readFileMain(self, filename, rouletteIndex):
        """
        Read file.
        
        Status:
         0 : success
        -1 : could not find file
        -2 : LBOMD XYZ NAtoms not matching
        -3 : unrecognised LBOMD XYZ format
        
        """
        state = Lattice()
        
        f = open(filename)
            
        line = f.readline().strip()
        NAtoms = int(line)
        
        line = f.readline().strip()
        dims_array = line.split()
        
        f.close()
        
        if len(dims_array) != 3:
            return -3, None
        
        state.reset(NAtoms)
        
        state.setDims(dims_array)
        
        self.logger.info("  %d atoms", NAtoms)
        
        # temporary specie list and counter arrays
        maxNumSpecies = 20 ## if there are more than 20 species these must be changed
        dt = np.dtype((str, 2))
        specieListTemp = np.empty( maxNumSpecies+1, dt ) 
        specieCountTemp = np.zeros( maxNumSpecies+1, np.int32 )
        
        tmpForceArray = np.empty(3, np.float64)
        
        # call c lib
        status = input_c.readRef(filename, state.atomID, state.specie, state.pos, state.charge, state.KE, state.PE, tmpForceArray, 
                                 specieListTemp, specieCountTemp, state.maxPos, state.minPos)
        
        if status:
            return -3, None
        
        # build specie list and counter in lattice object
        NSpecies = 0
        for i in range(maxNumSpecies):
            if specieListTemp[i] == 'XX':
                break
            else:
                NSpecies += 1
                
        # allocate specieList/Counter arrays
        dt = np.dtype((str, 2))
        state.specieList = np.empty(NSpecies, dt)
        state.specieCount = np.empty(NSpecies, np.int32)
        state.specieMass = np.empty(NSpecies, np.float64)
        state.specieCovalentRadius = np.empty(NSpecies, np.float64)
        state.specieAtomicNumber = np.empty(NSpecies, np.int32)
        state.specieRGB = np.empty((NSpecies, 3), np.float64)
        for i in xrange(NSpecies):
            state.specieList[i] = specieListTemp[i]
            state.specieCount[i] = specieCountTemp[i]
            
            state.specieMass[i] = elements.atomicMass(state.specieList[i])
            state.specieCovalentRadius[i] = elements.covalentRadius(state.specieList[i])
            state.specieAtomicNumber[i] = elements.atomicNumber(state.specieList[i])
            rgbtemp = elements.RGB(state.specieList[i])
            state.specieRGB[i][0] = rgbtemp[0]
            state.specieRGB[i][1] = rgbtemp[1]
            state.specieRGB[i][2] = rgbtemp[2]
            
            self.logger.info("    %d %s (%s) atoms", specieCountTemp[i], specieListTemp[i], elements.atomName(specieListTemp[i]))
    
        return 0, state


################################################################################

class LbomdDatReader(GenericLatticeReader):
    """
    Reads LBOMD lattice files.
    
    """
    def __init__(self, tmpLocation, log, displayWarning, displayError):
        super(LbomdDatReader, self).__init__(tmpLocation, log, displayWarning, displayError)
        
        self.formatIdentifiers.append([1, 3, 5])
        
        self.intRegex = re.compile(r'[0-9]+')
    
    def readFileMain(self, filename, rouletteIndex):
        """
        Read file.
        
        Status:
         0 : success
        -1 : could not find file
        -2 : LBOMD XYZ NAtoms not matching
        -3 : unrecognised LBOMD XYZ format
        
        """
        state = Lattice()
        
        f = open(filename)
        
        line = f.readline().strip()
        NAtoms = int(line)
        
        line = f.readline().strip()
        dims_array = line.split()
        
        f.close()
        
        state.reset(NAtoms)
        
        state.setDims(dims_array)
        
        self.logger.info("  %d atoms", NAtoms)
        
        # need temporary specie list and counter arrays
        maxNumSpecies = 20
        dt = np.dtype((str, 2))
        specieListTemp = np.empty( maxNumSpecies+1, dt ) 
        specieCountTemp = np.zeros( maxNumSpecies+1, np.int32 )
        
        # call c lib
        status = input_c.readLatticeLBOMD(filename, state.atomID, state.specie, state.pos, state.charge, specieListTemp, 
                                          specieCountTemp, state.maxPos, state.minPos)
        
        if status:
            return status, None
        
        # build specie list and counter in lattice object
        NSpecies = 0
        for i in range(maxNumSpecies):
            if specieListTemp[i] == 'XX':
                break
            else:
                NSpecies += 1
                
        # allocate specieList/Counter arrays
        dt = np.dtype((str, 2))
        state.specieList = np.empty(NSpecies, dt)
        state.specieCount = np.empty(NSpecies, np.int32)
        state.specieMass = np.empty(NSpecies, np.float64)
        state.specieCovalentRadius = np.empty(NSpecies, np.float64)
        state.specieAtomicNumber = np.empty(NSpecies, np.int32)
        state.specieRGB = np.empty((NSpecies, 3), np.float64)
        for i in xrange(NSpecies):
            state.specieList[i] = specieListTemp[i]
            state.specieCount[i] = specieCountTemp[i]
            
            state.specieMass[i] = elements.atomicMass(state.specieList[i])
            state.specieCovalentRadius[i] = elements.covalentRadius(state.specieList[i])
            state.specieAtomicNumber[i] = elements.atomicNumber(state.specieList[i])
            rgbtemp = elements.RGB(state.specieList[i])
            state.specieRGB[i][0] = rgbtemp[0]
            state.specieRGB[i][1] = rgbtemp[1]
            state.specieRGB[i][2] = rgbtemp[2]
            
            self.logger.info("    %d %s (%s) atoms", specieCountTemp[i], specieListTemp[i], elements.atomName(specieListTemp[i]))
        
        # guess roulette
        stepNumber = None
        if rouletteIndex is None:
            # file name
            basename = os.path.basename(filename)
            
            # look for integers in the name
            result = self.intRegex.findall(basename)
            
            if len(result):
                try:
                    index = int(result[0])
                except ValueError:
                    rouletteIndex = None
                else:
                    stepNumber = index
                    if index > 0:
                        rouletteIndex = index - 1
        
        # attempt to read roulette file
        if rouletteIndex is not None:
            # different path?
            head = os.path.dirname(filename)
            if len(head):
                testpath = head
            else:
                testpath = None
            
            # step number
            if stepNumber is None:
                stepNumber = rouletteIndex + 1
            state.kmcStep = stepNumber
            self.logger.info("Detected KMC step as: %d", state.kmcStep)
            
            # read simulation time
            simTime = utilities.getTimeFromRoulette(rouletteIndex, testpath=testpath)
            
            if simTime is not None:
                state.simTime = simTime
                self.logger.info("Detected simulation time as: %f", state.simTime)
            
            # get barrier
            state.barrier = utilities.getBarrierFromRoulette(rouletteIndex, testpath=testpath)
            if state.barrier is not None:
                self.logger.info("Detected barrier as: %f", state.barrier)
        
        return 0, state
