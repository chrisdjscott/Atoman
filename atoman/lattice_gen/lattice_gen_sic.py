
"""
Generate Fluorite lattice

@author: Chris Scott

"""
import logging

import numpy as np

from ..system.lattice import Lattice
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
    def __init__(self, sym1="Si", sym2="C_", charge1=0.0, charge2=0.0, 
                 NCells=[8,8,8], a0=4.321, pbcx=True, pbcy=True, pbcz=True, quiet=False):
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

class SiC4HLatticeGenerator(object):
    """
    SiC 4H lattice generator.
    
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
        logger.info("Generating Fluorite lattice")
        
        # lattice constants
        a0 = args.a0
        b0 = a0 / 4.0
        b1 = a0 / 2.0
        b2 = 3.0 * b0
        
        # define primitive cell
        # symbols
        sym_uc = [args.sym1, args.sym2, args.sym1, args.sym2, 
                  args.sym1, args.sym1, args.sym2, args.sym2]
        
        # positions
        pos_uc = np.empty(3 * 12, np.float64)
        pos_uc[0] = 0.0; pos_uc[1] = 0.0; pos_uc[2] = 0.0
        pos_uc[3] = b0; pos_uc[4] = b0; pos_uc[5] = b0
        pos_uc[6] = b1; pos_uc[7] = b1; pos_uc[8] = 0.0
        pos_uc[9] = b2; pos_uc[10] = b2; pos_uc[11] = b0
        pos_uc[12] = b1; pos_uc[13] = 0.0; pos_uc[14] = b1
        pos_uc[15] = 0.0; pos_uc[16] = b1; pos_uc[17] = b1
        pos_uc[18] = b2; pos_uc[19] = b0; pos_uc[20] = b2
        pos_uc[21] = b0; pos_uc[22] = b2; pos_uc[23] = b2
        
        # charges
        q_uc = np.empty(12, np.float64)
        q_uc[0] = args.charge1
        q_uc[1] = args.charge2
        q_uc[2] = args.charge1
        q_uc[3] = args.charge2
        q_uc[4] = args.charge1
        q_uc[5] = args.charge1
        q_uc[6] = args.charge2
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
            ifac = i * a0
            
            for j in xrange(jStop):
                jfac = j * a0
                
                for k in xrange(kStop):
                    kfac = k * a0
                    
                    for l in xrange(8):
                        # position of new atom
                        l3 = 3 * l
                        rx_tmp = pos_uc[l3    ] + ifac
                        ry_tmp = pos_uc[l3 + 1] + jfac
                        rz_tmp = pos_uc[l3 + 2] + kfac
                        
                        # skip if outside lattice (ie when making extra cell to get surface for non-periodic boundaries)
                        if (rx_tmp > dims[0]+0.0001) or (ry_tmp > dims[1]+0.0001) or (rz_tmp > dims[2]+0.0001):
                            continue
                        
                        # add to lattice structure
                        lattice.addAtom(sym_uc[l], (rx_tmp, ry_tmp, rz_tmp), q_uc[l])
                        
                        totalQ += q_uc[l]
                        count += 1
        
        NAtoms = count
        
        assert NAtoms == lattice.NAtoms
        
        # periodic boundaries
        lattice.PBC[0] = int(args.pbcx)
        lattice.PBC[1] = int(args.pbcy)
        lattice.PBC[2] = int(args.pbcz)
        
        logger.info("  Number of atoms: %d", NAtoms)
        logger.info("  Dimensions: %s", str(dims))
        logger.info("  Total charge: %f", totalQ)
        
        # sort out charges with fixed boundaries
        if not args.pbcx and not args.pbcy and not args.pbcz:
            if args.charge1 != 0.0 or args.charge2 != 0:
                logger.info("Fixing charges on fixed boundaries")
                
                totalQ = lattice_gen_utils.fixChargesOnFixedBoundaries(lattice)
                
                logger.info("  Total charge after modification: %f", totalQ)
        
        return 0, lattice
