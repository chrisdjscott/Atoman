
"""
Generate Rock Salt lattice

@author: Chris Scott

"""
import logging

import numpy as np

from ..state.lattice import Lattice
from . import lattice_gen_utils


################################################################################

class Args(object):
    """
    NCells: 3-tuple containing number of unit cells in each direction (default=(10,10,10))
    percGa: atomic percent Ga (max 25) (default=5)
    a0: lattice constant (default=4.64)
    f: output filename
    x,y,z: PBCs in each direction (default=True)
    quiet: suppress stdout
    
    """
    def __init__(self, sym1="Mg", sym2="O_", charge1=2.0, charge2=-2.0, 
                 NCells=[8,8,8], a0=4.19853, pbcx=True, pbcy=True, pbcz=True, quiet=False):
        self.sym1 = sym1
        self.sym2 = sym2
        self.charge1 = charge1
        self.charge2 = charge2
        self.NCells = NCells
        self.a0 = a0
        self.pbcx = pbcx
        self.pbcy = pbcy
        self.pbcz = pbcz
        self.quiet = quiet

################################################################################

class RockSaltLatticeGenerator(object):
    """
    Rock Salt lattice generator.
    
    """
    def __init__(self, log=None):
        self.logger = log
    
    def log(self, message, level=0, indent=0):
        """
        Write log message.
        
        """
        if self.logger is not None:
            self.logger(message, level=level, indent=indent)
    
    def generateLattice(self, args):
        """
        Generate the lattice.
        
        """
        logger = logging.getLogger(__name__)
        logger.info("Generating Rock Salt lattice")
        
        # lattice constants
        a0 = args.a0
        a1 = a0 / 2.0
        
        # define primitive cell
        # symbols
        sym_uc = [args.sym1, args.sym2, args.sym1, args.sym2, 
                  args.sym1, args.sym2, args.sym1, args.sym2]
        
        # positions
        pos_uc = np.empty(3 * 8, np.float64)
        pos_uc[0] = 0.0; pos_uc[1] = 0.0; pos_uc[2] = 0.0
        pos_uc[3] = a1; pos_uc[4] = 0.0; pos_uc[5] = 0.0
        pos_uc[6] = a1; pos_uc[7] = a1; pos_uc[8] = 0.0
        pos_uc[9] = 0.0; pos_uc[10] = a1; pos_uc[11] = 0.0
        pos_uc[12] = a1; pos_uc[13] = 0.0; pos_uc[14] = a1
        pos_uc[15] = 0.0; pos_uc[16] = 0.0; pos_uc[17] = a1
        pos_uc[18] = 0.0; pos_uc[19] = a1; pos_uc[20] = a1
        pos_uc[21] = a1; pos_uc[22] = a1; pos_uc[23] = a1
        
        # charges
        q_uc = np.empty(8, np.float64)
        q_uc[0] = args.charge1
        q_uc[1] = args.charge2
        q_uc[2] = args.charge1
        q_uc[3] = args.charge2
        q_uc[4] = args.charge1
        q_uc[5] = args.charge2
        q_uc[6] = args.charge1
        q_uc[7] = args.charge2
        
        # handle PBCs
        if args.pbcx:
            iStop = args.NCells[0]
        else:
            iStop = args.NCells[0] + 1
        
        if args.pbcy:
            jStop = args.NCells[1]
        else:
            jStop = args.NCells[1] + 1
        
        if args.pbcz:
            kStop = args.NCells[2]
        else:
            kStop = args.NCells[2] + 1
        
        # lattice dimensions
        dims = [a0*args.NCells[0], a0*args.NCells[1], a0*args.NCells[2]]
        
        # lattice structure
        lattice = Lattice()
        
        # set dimensions
        lattice.setDims(dims)
        
        # generate lattice
        count = 0
        totalQ = 0.0
        for i in xrange(iStop):
            for j in xrange(jStop):
                for k in xrange(kStop):
                    for l in xrange(8):
                        # position of new atom
                        rx_tmp = pos_uc[3*l+0] + i * a0
                        ry_tmp = pos_uc[3*l+1] + j * a0
                        rz_tmp = pos_uc[3*l+2] + k * a0
                        
                        # skip if outside lattice (ie when making extra cell to get surface for non-periodic boundaries)
                        if (rx_tmp > dims[0]+0.0001) or (ry_tmp > dims[1]+0.0001) or (rz_tmp > dims[2]+0.0001):
                            continue
                        
                        # add to lattice structure
                        lattice.addAtom(sym_uc[l], (rx_tmp, ry_tmp, rz_tmp), q_uc[l])
                        
                        totalQ += q_uc[l]
                        count += 1
        
        NAtoms = count
        
        assert NAtoms == lattice.NAtoms
        
        logger.info("  Number of atoms: %d", NAtoms)
        logger.info("  Dimensions: %s", str(dims))
        logger.info("  Total charge: %f", totalQ)
        
        # sort out charges with fixed boundaries
        if not args.pbcx and not args.pbcy and not args.pbcz:
            if args.charge1 != 0.0 or args.charge2 != 0:
                logger.info("Fixing charges on fixed boundaries")
                
                totalQ = lattice_gen_utils.fixChargesOnFixedBoundaries(lattice)
                
                logger.info("  Total charge after mofification: %f", totalQ)
        
        return 0, lattice
