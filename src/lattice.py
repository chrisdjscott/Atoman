
"""
The Lattice object.
Stores positions etc

@author: Chris Scott

"""

import os
import sys

import numpy as np



class Lattice:
    def __init__(self):
        
        self.NAtoms = 0
        
        self.simTime = 0.0
        
        self.cellDims = np.empty(3, np.float64)
        
        self.specieList = []
        self.specieCount = []
        self.specieMass = []
        self.specieCovalentRadius = []
        self.specieRGB = []
        
        self.minPos = np.empty(3, np.float64)
        self.maxPos = np.empty(3, np.float64)
        
        self.specie = []
        self.pos = []
        self.KE = []
        self.PE = []
        self.charge = []
    
    def reset(self, NAtoms):
        """
        Reinitialise arrays and counters
        
        """
        self.NAtoms = NAtoms
        
        self.specie = np.empty(NAtoms, np.int32)
        self.pos = np.empty(3 * NAtoms, np.float64)
        self.KE = np.zeros(NAtoms, np.float64)
        self.PE = np.zeros(NAtoms, np.float64)
        self.charge = np.zeros(NAtoms, np.float64)
        
        self.specieList = []
        self.specieCount = []
        self.specieMass = []
        self.specieCovalentRadius = []
        self.specieRGB = []
        
#         self.minPos = np.empty(3, np.float64)
#         self.maxPos = np.empty(3, np.float64)
#         
#         self.cellDims = np.zeros(3, np.float64)
        
        self.simTime = 0.0
    
    def setDims(self, dimsarray):
        
        self.cellDims[0] = float(dimsarray[0])
        self.cellDims[1] = float(dimsarray[1])
        self.cellDims[2] = float(dimsarray[2])
    
    def clone(self, lattice):
        """
        Copy given lattice into this instance
        
        """
        if lattice.NAtoms != self.NAtoms:
            self.reset(lattice.NAtoms)
        
        NAtoms = lattice.NAtoms
        
        self.simTime = lattice.simTime
        
        # copy dims
        self.cellDims[0] = lattice.cellDims[0]
        self.cellDims[1] = lattice.cellDims[1]
        self.cellDims[2] = lattice.cellDims[2]
        
        # specie stuff
        NSpecies = len(lattice.specieList)
        dt = np.dtype((str, 2))
        self.specieList = np.empty(NSpecies, dtype=dt)
        self.specieCount = np.zeros(NSpecies, np.int32)
        self.specieMass = np.empty(NSpecies, np.float64)
        self.specieCovalentRadius = np.empty(NSpecies, np.float64)
        self.specieRGB = np.empty((NSpecies, 3), np.float64)
        for i in xrange(NSpecies):
            self.specieList[i] = lattice.specieList[i]
            self.specieCount[i] = lattice.specieCount[i]
            self.specieMass[i] = lattice.specieMass[i]
            self.specieCovalentRadius[i] = lattice.specieCovalentRadius[i]
            for j in xrange(3):
                self.specieRGB[i][j] = lattice.specieRGB[i][j]
        
        # atom data
        self.specie = np.empty(NAtoms, np.int32)
        self.pos = np.empty(3 * NAtoms, np.float64)
        self.KE = np.empty(NAtoms, np.float64)
        self.PE = np.empty(NAtoms, np.float64)
        self.charge = np.empty(NAtoms, np.float64)
        for i in xrange(NAtoms):
            self.specie[i] = lattice.specie[i]
            self.KE[i] = lattice.KE[i]
            self.PE[i] = lattice.PE[i]
            self.charge[i] = lattice.charge[i]
            for j in xrange(3):
                self.pos[3*i+j] = lattice.pos[3*i+j]
        
        self.minPos[0] = lattice.minPos[0]
        self.minPos[1] = lattice.minPos[1]
        self.minPos[2] = lattice.minPos[2]
        self.maxPos[0] = lattice.maxPos[0]
        self.maxPos[1] = lattice.maxPos[1]
        self.maxPos[2] = lattice.maxPos[2]
