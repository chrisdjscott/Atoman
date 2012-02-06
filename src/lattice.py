
"""
The Lattice object.
Stores positions etc

author: Chris Scott
last edited: February 2012
"""

import os
import sys

try:
    import numpy as np
except:
    sys.exit(__name__+": ERROR: could not import numpy")



class Lattice:
    def __init__(self):
        
        self.NAtoms = 0
        self.NVisible = 0
        
        self.cellDims = np.zeros(3, np.float64)
        
        self.visibleAtoms = []
        
        self.visibleSpecieList = []
        self.visibleSpecieCount = []
        
        self.specieList = []
        self.specieCount = []
        
        self.pos = []
        self.KE = []
        self.PE = []
        self.charge = []
    
    def reset(self, NAtoms):
        
#        if self.NAtoms != 0:
#            del self.pos
#            del self.KE
#            del self.PE
#            del self.charge
        
        self.NAtoms = NAtoms
        self.pos = np.empty(3 * NAtoms, np.float64)
        self.KE = np.empty(NAtoms, np.float64)
        self.PE = np.empty(NAtoms, np.float64)
        self.charge = np.empty(NAtoms, np.float64)
    
    def setDims(self, dimsarray):
        
        self.cellDims[0] = float(dimsarray[0])
        self.cellDims[1] = float(dimsarray[1])
        self.cellDims[2] = float(dimsarray[2])
    
    # function to find specie index in specie list
    def specieIndex( self, sym ):
        if not len(self.specieList):
            sys.exit(__name__+": specieIndex: error: specie list is empty")
        
        match = 0
        for i in range(len(self.specieList)):
            if sym == self.specieList[i]:
                match = 1
                specieIndex = i
                break
        
        if not match:
            sys.exit(__name__+": specieIndex: error: could not find "+sym+" in specie list")
        
        return specieIndex
    
