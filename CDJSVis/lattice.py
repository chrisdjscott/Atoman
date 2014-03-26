
"""
Lattice module, with Lattice object and utilities

@author: Chris Scott

"""
import logging

import numpy as np

from .atoms import elements
from .md import forces


################################################################################

class Lattice(object):
    """
    The Lattice object.
    
    """
    def __init__(self):
        self.NAtoms = 0
        
        self.simTime = 0.0
        self.barrier = None
        
        self.cellDims = np.empty(3, np.float64)
        
        dt = np.dtype((str, 2))
        self.specieList = np.empty(0, dt)
        self.specieCount = np.empty(0, np.int32)
        self.specieMass = np.empty(0, np.float64)
        self.specieCovalentRadius = np.empty(0, np.float64)
        self.specieRGB = np.empty((0,3), np.float64)
        self.specieAtomicNumber = np.empty(0, np.int32)
        
        self.minPos = np.empty(3, np.float64)
        self.maxPos = np.empty(3, np.float64)
        
        self.atomID = np.empty(0, np.int32)
        self.specie = np.empty(0, np.int32)
        self.pos = np.empty(0, np.float64)
        self.KE = np.empty(0, np.float64)
        self.PE = np.empty(0, np.float64)
        self.charge = np.empty(0, np.float64)
        
        self.voronoiDict = {}
    
    def reset(self, NAtoms):
        """
        Reinitialise arrays and counters
        
        """
        self.NAtoms = NAtoms
        
        self.atomID = np.empty(NAtoms, np.int32)
        self.specie = np.empty(NAtoms, np.int32)
        self.pos = np.empty(3 * NAtoms, np.float64)
        self.KE = np.zeros(NAtoms, np.float64)
        self.PE = np.zeros(NAtoms, np.float64)
        self.charge = np.zeros(NAtoms, np.float64)
        
        dt = np.dtype((str, 2))
        self.specieList = np.empty(0, dt)
        self.specieCount = np.empty(0, np.int32)
        self.specieMass = np.empty(0, np.float64)
        self.specieCovalentRadius = np.empty(0, np.float64)
        self.specieRGB = np.empty((0,3), np.float64)
        self.specieAtomicNumber = np.empty(0, np.int32)
        
        self.minPos = np.empty(3, np.float64)
        self.maxPos = np.empty(3, np.float64)
         
        self.cellDims = np.zeros(3, np.float64)
        
        self.simTime = 0.0
        self.barrier = None
        self.voronoiDict = {}
    
    def addSpecie(self, sym, count=None):
        """
        Add specie to specie list
        
        """
        if sym in self.specieList:
            if count is not None:
                specInd = self.specieIndex(sym)
                self.specieCount[specInd] = count
            
            return
        
        if count is None:
            count = 0
        
        self.specieList = np.append(self.specieList, sym)
        self.specieCount = np.append(self.specieCount, np.int32(count))
        
        self.specieMass = np.append(self.specieMass, elements.atomicMass(sym))
#         self.specieMassAMU = np.append(self.specieMassAMU, Atoms.atomicMassAMU(sym))
        self.specieCovalentRadius = np.append(self.specieCovalentRadius, elements.covalentRadius(sym))
        rgbtemp = elements.RGB(sym)
        rgbnew = np.empty((1,3), np.float64)
        rgbnew[0][0] = rgbtemp[0]
        rgbnew[0][1] = rgbtemp[1]
        rgbnew[0][2] = rgbtemp[2]            
        self.specieRGB = np.append(self.specieRGB, rgbnew, axis=0)
    
    def addAtom(self, sym, pos, charge, atomID=None):
        """
        Add an atom to the lattice
        
        """
        if sym not in self.specieList:
            self.addSpecie(sym)
        
        # atom ID
        if atomID is None:
            atomID = self.NAtoms
        
        specInd = self.getSpecieIndex(sym)
        
        self.specieCount[specInd] += 1
        
        pos = np.asarray(pos, dtype=np.float64)
        
        self.atomID = np.append(self.specie, np.int32(atomID))
        self.specie = np.append(self.specie, np.int32(specInd))
        self.pos = np.append(self.pos, pos)
        self.charge = np.append(self.charge, charge)
#         self.force = np.append(self.force, np.zeros(3, np.float64))
        self.KE = np.append(self.KE, 0.0)
        self.PE = np.append(self.PE, 0.0)
        
        # wrap positions
        
        
        # min/max pos!!??
        if pos[0] < self.minPos[0]:
            self.minPos[0] = pos[0]
        if pos[1] < self.minPos[1]:
            self.minPos[1] = pos[2]
        if pos[2] < self.minPos[1]:
            self.minPos[2] = pos[2]
        if pos[0] > self.maxPos[0]:
            self.maxPos[0] = pos[0]
        if pos[1] > self.maxPos[1]:
            self.maxPos[1] = pos[2]
        if pos[2] > self.maxPos[1]:
            self.maxPos[2] = pos[2]
        
        self.voronoiDict = {}
        self.NAtoms += 1
    
    def removeAtom(self, index):
        """
        Remove an atom
        
        """
        specInd = self.specie[index]
        self.atomID = np.delete(self.atomID, index)
        self.specie = np.delete(self.specie, index)
        self.pos = np.delete(self.pos, [3*index,3*index+1,3*index+2])
        self.charge = np.delete(self.charge, index)
        self.force = np.delete(self.force, [3*index,3*index+1,3*index+2])
        self.NAtoms -= 1
        
        # modify specie list / counter if required
        self.specieCount[specInd] -= 1
        if self.specieCount[specInd] == 0:
            self.removeSpecie(specInd)
        
        self.KE = np.delete(self.KE, index)
        self.PE = np.delete(self.PE, index)
        
        self.voronoiDict = {}
    
    def removeSpecie(self, index):
        """
        Remove a specie from the specie list.
        
        """
        self.specieCount = np.delete(self.specieCount, index)
        self.specieList = np.delete(self.specieList, index)
        self.specieCovalentRadius = np.delete(self.specieCovalentRadius, index)
        self.specieMass = np.delete(self.specieMass, index)
#         self.specieMassAMU = np.delete(self.specieMassAMU, index)
        self.specieRGB = np.delete(self.specieRGB, index, axis=0)
        
        for i in xrange(self.NAtoms):
            if self.specie[i] > index:
                self.specie[i] -= 1
    
    def calcForce(self, forceConfig):
        """
        Calculate force on lattice.
        
        """
        if type(forceConfig) is not forces.ForceConfig:
            print "FORCE CONFIG WRONG TYPE"
            return 113
        
        return forces.calc_force(self, forceConfig)
    
    def atomPos(self, index):
        """
        Return pointer to atom position within pos array: [xpos, ypos, zpos].
        
        """
        atomPos = None
        if index < self.NAtoms:
            atomPos = self.pos[3*index:3*index+3]
        
        return atomPos
    
    def atomSym(self, index):
        """
        Returns symbol of given atom.
        
        """
        atomSym = None
        if index < self.NAtoms:
            atomSym = self.specieList[self.specie[index]]
        
        return atomSym
    
    def getSpecieIndex(self, sym):
        """
        Return index of specie in specie list.
        
        """
        index = None
        for i in xrange(len(self.specieList)):
            if self.specieList[i] == sym:
                index = i
                break
        
        return index
    
    def setDims(self, dimsarray):
        
        self.cellDims[0] = float(dimsarray[0])
        self.cellDims[1] = float(dimsarray[1])
        self.cellDims[2] = float(dimsarray[2])
    
    def refreshElementProperties(self):
        """
        Refresh element properties.
        
        """
        for i, sym in enumerate(self.specieList):
            self.specieMass[i] = elements.atomicMass(sym)
            self.specieCovalentRadius[i] = elements.covalentRadius(sym)
            self.specieAtomicNumber[i] = elements.atomicNumber(sym)
            rgbtemp = elements.RGB(sym)
            self.specieRGB[i][0] = rgbtemp[0]
            self.specieRGB[i][1] = rgbtemp[1]
            self.specieRGB[i][2] = rgbtemp[2]
    
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
        
        self.voronoiDict = lattice.voronoiDict
        
        # specie stuff
        NSpecies = len(lattice.specieList)
        dt = np.dtype((str, 2))
        self.specieList = np.empty(NSpecies, dtype=dt)
        self.specieCount = np.zeros(NSpecies, np.int32)
        self.specieMass = np.empty(NSpecies, np.float64)
        self.specieCovalentRadius = np.empty(NSpecies, np.float64)
        self.specieAtomicNumber = np.zeros(NSpecies, np.int32)
        self.specieRGB = np.empty((NSpecies, 3), np.float64)
        for i in xrange(NSpecies):
            self.specieList[i] = lattice.specieList[i]
            self.specieCount[i] = lattice.specieCount[i]
            self.specieMass[i] = lattice.specieMass[i]
            self.specieCovalentRadius[i] = lattice.specieCovalentRadius[i]
            self.specieAtomicNumber[i] = lattice.specieAtomicNumber[i]
            for j in xrange(3):
                self.specieRGB[i][j] = lattice.specieRGB[i][j]
        
        # atom data
        self.atomID = np.empty(NAtoms, np.int32)
        self.specie = np.empty(NAtoms, np.int32)
        self.pos = np.empty(3 * NAtoms, np.float64)
        self.KE = np.empty(NAtoms, np.float64)
        self.PE = np.empty(NAtoms, np.float64)
        self.charge = np.empty(NAtoms, np.float64)
        for i in xrange(NAtoms):
            self.atomID[i] = lattice.atomID[i]
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
