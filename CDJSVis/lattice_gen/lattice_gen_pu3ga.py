
"""
Generate Ga stabilised delta-Pu lattice using Pu3Ga technique

@author: Chris Scott

"""
import logging
import random

from ..state.lattice import Lattice


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
        
        # lattice constants
        a0 = args.a0
        a1 = a0 / 2.0
        
        # define primitive cell (4 atoms)
        # corner atom is Ga
        UC_sym = ["Pu", "Pu", "Pu", "Ga"]
        UC_rx = [0.0, 0.0, a1,  a1]
        UC_ry = [0.0, a1,  0.0, a1]
        UC_rz = [0.0, a1,  a1,  0.0]
        
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
        GaIndexes = []
        count = 0
        for i in xrange(iStop):
            for j in xrange(jStop):
                for k in xrange(kStop):
                    for l in xrange(4):
                        # position of new atom
                        rx_tmp = UC_rx[l] + i * a0
                        ry_tmp = UC_ry[l] + j * a0
                        rz_tmp = UC_rz[l] + k * a0
                        
                        # skip if outside lattice (ie when making extra cell to get surface for non-periodic boundaries)
                        if (rx_tmp > dims[0]+0.0001) or (ry_tmp > dims[1]+0.0001) or (rz_tmp > dims[2]+0.0001):
                            continue
                        
                        # add to lattice structure
                        lattice.addAtom(UC_sym[l], (rx_tmp, ry_tmp, rz_tmp), 0.0)
                        
                        if UC_sym[l] == "Ga":
                            GaIndexes.append(count)
                        
                        count += 1
        
        NAtoms = count
        NGa = len(GaIndexes)
        
        assert NAtoms == lattice.NAtoms
        assert NGa == lattice.specieCount[lattice.getSpecieIndex("Ga")]
        
        self.logger.info("  Number of atoms: %d", NAtoms)
        self.logger.info("  Dimensions: %s", str(dims))
        
        # obtain correct percentage of Ga
        self.fixGaPercentage(lattice, GaIndexes, args)
        
        return 0, lattice
    
    def fixGaPercentage(self, lattice, GaIndexes, args):
        """
        Get correct percentage of Ga in the lattice.
        
        """
        NGa = len(GaIndexes)
        
        self.logger.debug("  Fixing Ga percentage to %f %%", args.percGa)
            
        currentPercGa = float(NGa) / float(lattice.NAtoms) * 100.0
        self.logger.debug("    Current Ga percentage: %f %%", currentPercGa)
        
        newNGa = int(args.percGa * 0.01 * float(lattice.NAtoms))
        
        diff = NGa - newNGa
        if diff > 0:
            # remove Ga until right number
            self.logger.debug("    Removing %d Ga atoms", diff)
            
            count = 0
            while count < diff:
                GaIndex = random.randint(0, NGa-1)
                
                index = GaIndexes[GaIndex]
                GaIndexes.pop(GaIndex)
                NGa -= 1
                
                assert lattice.atomSym(index) == "Ga"
                
                specIndPu = lattice.getSpecieIndex("Pu")
                specIndGa = lattice.getSpecieIndex("Ga")
                lattice.specie[index] = specIndPu
                lattice.specieCount[specIndGa] -= 1
                lattice.specieCount[specIndPu] += 1
                
                count += 1
        else:
            # add Ga until right number
            pass
        
        currentPercGa = float(NGa) / float(lattice.NAtoms) * 100.0
        
        assert ((currentPercGa - args.percGa) / args.percGa) < 0.01
