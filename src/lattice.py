
"""
The Lattice object.
Stores positions etc

author: Chris Scott
last edited: February 2012
"""

import os
import sys

import numpy as np



class Lattice:
    def __init__(self):
        
        self.NAtoms = 0
        self.NVisible = 0
        
        self.simTime = 0.0
        
        self.cellDims = np.empty(3, np.float64)
        
#        self.visibleAtoms = []
#        self.visibleType = []
#        
#        self.visibleSpecieList = []
#        self.visibleSpecieCount = []
        
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
        self.NVisible = NAtoms
        
        self.specie = np.empty(NAtoms, np.int32)
        self.pos = np.empty(3 * NAtoms, np.float64)
        self.KE = np.empty(NAtoms, np.float64)
        self.PE = np.empty(NAtoms, np.float64)
        self.charge = np.empty(NAtoms, np.float64)
#        self.visible = np.arange(NAtoms, dtype=np.int32)
#        self.visibleType = np.zeros(NAtoms, np.int32)
        
        self.specieList = []
        self.specieCount = []
#        self.visibleSpecieList = []
#        self.visibleSpecieCount = []
        
#         self.minPos = np.empty(3, np.float64)
#         self.maxPos = np.empty(3, np.float64)
#         
#         self.cellDims = np.zeros(3, np.float64)
        
        self.simTime = 0.0
    
    def setDims(self, dimsarray):
        
        self.cellDims[0] = float(dimsarray[0])
        self.cellDims[1] = float(dimsarray[1])
        self.cellDims[2] = float(dimsarray[2])
    
    

