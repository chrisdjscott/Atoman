
"""
Generate Carbon Diamond Lattice

@author: Kenny Jolley
Derived from similar form by Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

import numpy as np

from ..system.lattice import Lattice
from . import lattice_gen_utils
from six.moves import range


################################################################################

class Args(object):
    """
    NCells: 3-tuple containing number of unit cells in each direction (default=(6,6,6))
    charge is defaulted to zero, and should not be changed for pure carbon systems
    a0: lattice 'a' constant (default=3.556717) (AIREBO)
    pbcx,pbcy,pbcz: PBCs in each direction (default=True)
    
    """
    def __init__(self, sym1="C_", charge1=0.0, NCells=[6,6,6], a0=3.556717,
                 pbcx=True, pbcy=True, pbcz=True, quiet=False):
        self.sym1 = sym1
        self.charge1 = charge1
        self.NCells = NCells
        self.a0 = a0
        self.pbcx = pbcx
        self.pbcy = pbcy
        self.pbcz = pbcz

################################################################################

class DiamondLatticeGenerator(object):
    """
    Carbon Diamond lattice generator.

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
        # atom symbol (this defaults to carbon)
        sym_uc = args.sym1

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

        # atom charge (this should be zero)
        q_uc = args.charge1

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

        # create specie list
        lattice.addSpecie(args.sym1)


        lattice.specie = np.zeros(iStop*jStop*kStop*8, dtype=np.int32)
        lattice.charge = np.zeros(iStop*jStop*kStop*8, dtype=np.float64)
        lattice.pos = np.zeros((iStop*jStop*kStop*8*3), dtype=np.float64)

        # generate lattice
        count = 0
        totalQ = 0.0
        for i in range(iStop):
            ifac = i * a0

            for j in range(jStop):
                jfac = j * a0

                for k in range(kStop):
                    kfac = k * a0

                    for l in range(8):
                        # position of new atom
                        l3 = 3 * l
                        rx_tmp = pos_uc[l3    ] + ifac
                        ry_tmp = pos_uc[l3 + 1] + jfac
                        rz_tmp = pos_uc[l3 + 2] + kfac

                        # skip if outside lattice (ie when making extra cell to get surface for non-periodic boundaries)
                        if (rx_tmp > dims[0]+0.0001) or (ry_tmp > dims[1]+0.0001) or (rz_tmp > dims[2]+0.0001):
                            continue

                        # add to lattice structure
                        #lattice.addAtom(sym_uc[l], (rx_tmp, ry_tmp, rz_tmp), q_uc[l])

                        specInd = lattice.getSpecieIndex(sym_uc)
                        lattice.specieCount[specInd] += 1

                        #pos = np.asarray((rx_tmp, ry_tmp, rz_tmp), dtype=np.float64)

                        #lattice.atomID = np.append(lattice.atomID, np.int32(count+1))
                        #lattice.specie = np.append(lattice.specie, np.int32(specInd))
                        #lattice.pos = np.append(lattice.pos, pos)
                        #lattice.charge = np.append(lattice.charge, np.float64(q_uc[l]))
                        lattice.specie[count] = np.int32(specInd)
                        lattice.pos[count*3] = np.float64(rx_tmp)
                        lattice.pos[count*3+1] = np.float64(ry_tmp)
                        lattice.pos[count*3+2] = np.float64(rz_tmp)
                        lattice.charge[count] = np.float64(q_uc)

                        totalQ += q_uc
                        count += 1

        lattice.NAtoms = count


        # cut trailing zero's if reqired
        if(count != len(lattice.specie) ):
            lattice.specie = lattice.specie[0:count]
            lattice.charge = lattice.charge[0:count]
            lattice.pos = lattice.pos[0:count*3]


        # min/max pos
        for i in range(3):
            lattice.minPos[i] = np.min(lattice.pos[i::3])
            lattice.maxPos[i] = np.max(lattice.pos[i::3])

        # atom ID
        lattice.atomID = np.arange(1, lattice.NAtoms + 1, dtype=np.int32)

        # periodic boundaries
        lattice.PBC[0] = int(args.pbcx)
        lattice.PBC[1] = int(args.pbcy)
        lattice.PBC[2] = int(args.pbcz)

        logger.info("  Number of atoms: %d", lattice.NAtoms)
        logger.info("  Dimensions: %s", str(dims))
        logger.info("  Total charge: %f", totalQ)

        # sort out charges with fixed boundaries
        if not args.pbcx and not args.pbcy and not args.pbcz:
            if args.charge1 != 0.0:
                logger.info("Fixing charges on fixed boundaries")

                totalQ = lattice_gen_utils.fixChargesOnFixedBoundaries(lattice)

                logger.info("  Total charge after modification: %f", totalQ)

        return 0, lattice
