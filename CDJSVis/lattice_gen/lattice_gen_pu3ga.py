
"""
Generate Ga stabilised delta-Pu lattice using Pu3Ga technique

@author: Chris Scott

"""
from __future__ import division
import logging

import numpy as np

from ..state.lattice import Lattice
from . import _lattice_gen_pu3ga

################################################################################

class Args(object):
    """
    NCells: 3-tuple containing number of unit cells in each direction (default=(10,10,10))
    percGa: atomic percent Ga (max 25) (default=5)
    a0: lattice constant (default=4.64)
    f: output filename
    x,y,z: PBCs in each direction (default=True)
    
    """
    def __init__(self, NCells=[10,10,10], percGa=5, a0=4.64, pbcx=True, pbcy=True, pbcz=True):
        self.NCells = NCells
        self.percGa = percGa
        self.a0 = a0
        self.pbcx = pbcx
        self.pbcy = pbcy
        self.pbcz = pbcz

################################################################################

class Pu3GaLatticeGenerator(object):
    """
    Pu3Ga lattice generator.
    
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generateLattice(self, args):
        """
        Generate the lattice.
        
        """
        self.logger.info("Generating Pu-Ga lattice")
        
        # numpy arrays
        numCells = np.asarray(args.NCells, dtype=np.int32)
        pbc = np.asarray([args.pbcx, args.pbcy, args.pbcz], dtype=np.int32)
        
        # lattice constants
        a0 = args.a0
        
        # lattice dimensions
        dims = [a0 * numCells[0], a0 * numCells[1], a0 * numCells[2]]
        
        # generate lattice data
        NAtoms, specie, pos, charge, specieCount = _lattice_gen_pu3ga.generatePu3GaLattice(numCells, pbc, a0, args.percGa)
        
        # lattice structure
        lattice = Lattice()
        
        # set up correctly for this number of atoms
        lattice.reset(NAtoms)
        
        # set dimensions
        lattice.setDims(dims)
        
        # set data
        lattice.addSpecie("Pu", count=specieCount[0])
        lattice.addSpecie("Ga", count=specieCount[1])
        lattice.specie = specie
        lattice.pos = pos
        lattice.charge = charge
        lattice.NAtoms = NAtoms
        
        # min/max pos
        for i in xrange(3):
            lattice.minPos[i] = np.min(lattice.pos[i::3])
            lattice.maxPos[i] = np.max(lattice.pos[i::3])
        
        # atom ID
        lattice.atomID = np.arange(1, lattice.NAtoms + 1, dtype=np.int32)
        
        self.logger.info("  Number of atoms: %d", NAtoms)
        self.logger.info("  Dimensions: %s", str(dims))
        self.logger.info("  Ga concentration: %f %%", lattice.specieCount[1] / lattice.NAtoms * 100.0)
        
        return 0, lattice
