
"""
Lattice reader objects.

@author: Chris Scott

"""
import os

import numpy as np

from .visclibs import input_c
from .atoms import elements
from .visutils import utilities
from .lattice import Lattice





################################################################################

class GenericLatticeReader(object):
    """
    Base lattice reader object.
    
    """
    def __init__(self, tmpLocation, log, displayWarning):
        self.tmpLocation = tmpLocation
        self.log = log
        self.currentFile = None
        self.displayWarning = displayWarning
    
    def checkForZipped(self, filename):
        """
        Check if file exists (unzip if required)
        
        """
        if os.path.exists(filename):
            fileLocation = '.'
            zipFlag = 0
        
        else:
            if os.path.exists(filename + '.bz2'):
                command = "bzcat -k %s.bz2 > " % (filename)
            
            elif os.path.exists(filename + '.gz'):
                command = "gzip -dc %s.gz > " % (filename)
                
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
        if os.path.abspath(filename) == self.currentFile:
            print "ALREADY LOADED"
#            return -4, None
        
        filepath, zipFlag = self.checkForZipped(filename)
        if zipFlag == -1:
            self.displayWarning("Could not find file: "+filename)
            return -1, None
        
        status, state = self.readFileMain(filepath, rouletteIndex)
        
        self.cleanUnzipped(filepath, zipFlag)
        
        if status:
            if status == -1:
                self.displayWarning("Could not find file: "+filename)
            
            elif status == -2:
                self.displayWarning("LBOMD XYZ input NAtoms does not match reference!")
            
            elif status == -3:
                self.displayWarning("Unrecognised format for LBOMD XYZ input file!")
            
            return None, None
        
        elif state is not None:
            self.currentFile = os.path.abspath(filename)
            print "CURRENT FILE", self.currentFile
        
        return status, state
    
    def readFileMain(self, filename):
        """
        Main read file routine (to be overriden).
        
        """
        return 1, None

################################################################################

class LbomdRefReader(GenericLatticeReader):
    """
    Read LBOMD animation reference files.
    
    """
    def __init__(self, tmpLocation, log, displayWarning):
        super(LbomdRefReader, self).__init__(tmpLocation, log, displayWarning)
    
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
        array = line.split()
        state.setDims(array)
        
        f.close()
        
        state.reset(NAtoms)
        
        self.log("%d atoms" % (NAtoms,), 0, 1)
        
        # temporary specie list and counter arrays
        maxNumSpecies = 20 ## if there are more than 20 species these must be changed
        dt = np.dtype((str, 2))
        specieListTemp = np.empty( maxNumSpecies+1, dt ) 
        specieCountTemp = np.zeros( maxNumSpecies+1, np.int32 )
        
        tmpForceArray = np.empty(3, np.float64)
        
        # call c lib
        input_c.readRef( filename, state.specie, state.pos, state.charge, state.KE, state.PE, tmpForceArray, 
                         specieListTemp, specieCountTemp, state.maxPos, state.minPos )
        
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
        state.specieRGB = np.empty((NSpecies, 3), np.float64)
        for i in xrange(NSpecies):
            state.specieList[i] = specieListTemp[i]
            state.specieCount[i] = specieCountTemp[i]
            
            state.specieMass[i] = elements.atomicMass(state.specieList[i])
            state.specieCovalentRadius[i] = elements.covalentRadius(state.specieList[i])
            rgbtemp = elements.RGB(state.specieList[i])
            state.specieRGB[i][0] = rgbtemp[0]
            state.specieRGB[i][1] = rgbtemp[1]
            state.specieRGB[i][2] = rgbtemp[2]
            
            self.log("%d %s (%s) atoms" % (specieCountTemp[i], specieListTemp[i], elements.atomName(specieListTemp[i])), 0, 2)
    
        return 0, state


################################################################################

class LbomdDatReader(GenericLatticeReader):
    """
    Reads LBOMD lattice files.
    
    """
    def __init__(self, tmpLocation, log, displayWarning):
        super(LbomdDatReader, self).__init__(tmpLocation, log, displayWarning)
    
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
        array = line.split()
        state.setDims(array)
        
        f.close()
        
        state.reset(NAtoms)
        
        self.log("%d atoms" % (NAtoms,), 0, 1)
        
        # need temporary specie list and counter arrays
        maxNumSpecies = 20
        dt = np.dtype((str, 2))
        specieListTemp = np.empty( maxNumSpecies+1, dt ) 
        specieCountTemp = np.zeros( maxNumSpecies+1, np.int32 )
        
        # call c lib
        input_c.readLatticeLBOMD( filename, state.specie, state.pos, state.charge, specieListTemp, 
                                  specieCountTemp, state.maxPos, state.minPos )
        
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
        state.specieRGB = np.empty((NSpecies, 3), np.float64)
        for i in xrange(NSpecies):
            state.specieList[i] = specieListTemp[i]
            state.specieCount[i] = specieCountTemp[i]
            
            state.specieMass[i] = elements.atomicMass(state.specieList[i])
            state.specieCovalentRadius[i] = elements.covalentRadius(state.specieList[i])
            rgbtemp = elements.RGB(state.specieList[i])
            state.specieRGB[i][0] = rgbtemp[0]
            state.specieRGB[i][1] = rgbtemp[1]
            state.specieRGB[i][2] = rgbtemp[2]
            
            self.log("%d %s (%s) atoms" % (specieCountTemp[i], specieListTemp[i], elements.atomName(specieListTemp[i])), 0, 2)
        
        # attempt to read time from roulette
        if rouletteIndex is not None:
            simTime = utilities.getTimeFromRoulette(rouletteIndex)
            
            if simTime is not None:
                state.simTime = simTime
            
            # get barrier
            state.barrier = utilities.getBarrierFromRoulette(rouletteIndex)
        
        return 0, state
    
    
    


