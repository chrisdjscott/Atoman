
"""
Generate a Graphite lattice

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
    NCells: 3-tuple containing number of unit cells in each direction (default=(7,12,5))
    charge is defaulted to zero, and should not be changed for pure carbon systems
    a0: lattice 'a' constant (default=2.4175) (AIREBO)
    c0: lattice 'c' constant (default=3.358) (AIREBO)
    pbcx,pbcy,pbcz: PBCs in each direction (default=True)
    
    """
    def __init__(self, sym1="C_", charge1=0.0, NCells=[7,12,5], a0=2.4175, c0=3.358, 
                 stacking='ab',
                 pbcx=True, pbcy=True, pbcz=True):
        self.sym1 = sym1
        self.charge1 = charge1
        self.NCells = NCells
        self.a0 = a0
        self.c0 = c0
        self.stacking=stacking
        self.pbcx = pbcx
        self.pbcy = pbcy
        self.pbcz = pbcz

################################################################################

class GraphiteLatticeGenerator(object):
    """
    Graphite lattice generator.
    
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
        logger.info("Generating the Graphite lattice")
        
        # lattice constants
        a0 = args.a0
        c0 = args.c0
        stacking = args.stacking
        
        # atom symbol (this should be carbon)
        sym_uc = args.sym1
        
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
        dims = [1.73205080756888*a0*args.NCells[0], a0*args.NCells[1], c0*len(stacking)*args.NCells[2]]
        
        # lattice structure
        lattice = Lattice()
        
        # set dimensions
        lattice.setDims(dims)
        
        # create specie list
        lattice.addSpecie(args.sym1)
        
        # Create arrays with initial zero data
        tot_atoms = int(4 * len(stacking) * iStop * jStop * kStop)
        lattice.specie = np.zeros(tot_atoms, dtype=np.int32)                   
        lattice.charge = np.zeros(tot_atoms, dtype=np.float64)      
        lattice.pos = np.zeros((tot_atoms*3), dtype=np.float64)       
        
        # generate lattice
        count = 0
        for x in range(0,iStop):
            Xshift = x * 1.73205080756888 * a0
            for y in range(0,jStop):
                Yshift = y * a0
                for z in range(0,kStop):
                    for l in range(0,len(stacking)):
                        Zshift = (len(stacking)*z+l)*c0
                        
                        # stacking position a
                        if( stacking[l] == 'a' or stacking[l] == 'A'):
                            # Pos 1
                            rx_tmp = Xshift
                            ry_tmp = Yshift
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                            
                            # Pos 2
                            rx_tmp = Xshift + a0 * 1.15470053837925   # 2.0/math.sqrt(3)
                            ry_tmp = Yshift
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                                
                            # Pos 3
                            rx_tmp = Xshift + a0 * 0.288675134594813  # math.sqrt(3)/6.0  
                            ry_tmp = Yshift + a0 * 0.5
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                            
                            # Pos 4
                            rx_tmp = Xshift + a0 * 0.866025403784439  # math.sqrt(3)/2.0
                            ry_tmp = Yshift + a0 * 0.5
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                                
                                
                        # stacking position b
                        elif(stacking[l] == 'b' or stacking[l] == 'B'):   
                            # Pos 1
                            rx_tmp = Xshift
                            ry_tmp = Yshift
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                            
                            # Pos 2
                            rx_tmp = Xshift + a0 * 0.577350269189626  # / math.sqrt(3)
                            ry_tmp = Yshift 
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                                
                            # Pos 3
                            rx_tmp = Xshift + a0 * 0.866025403784439     # math.sqrt(3)/2.0)
                            ry_tmp = Yshift + a0 * 0.5
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                                
                            # Pos 4
                            rx_tmp = Xshift + a0 * 1.44337567297406 # math.sqrt(3)*5.0/6.0
                            ry_tmp = Yshift + a0 * 0.5
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1  
                                
                        # stacking position c
                        elif (stacking[l] == 'c' or stacking[l] == 'C'):
                            # Pos 1
                            rx_tmp = Xshift + a0 * 0.577350269189626  # / math.sqrt(3)
                            ry_tmp = Yshift
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                            
                            # Pos 2
                            rx_tmp = Xshift + a0 * 1.15470053837925  # 2.0/math.sqrt(3)
                            ry_tmp = Yshift
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                                
                            # Pos 3
                            rx_tmp = Xshift + a0 * 0.288675134594813      # math.sqrt(3)/6.0
                            ry_tmp = Yshift + a0 * 0.5
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                                
                            # Pos 4
                            rx_tmp = Xshift + a0 * 1.44337567297406    # math.sqrt(3)*5.0/6.0
                            ry_tmp = Yshift + a0 * 0.5
                            rz_tmp = Zshift
                            # If within the lattice  (ie skip atoms outside when making extra cell to get surface for non-periodic boundaries)
                            if (rx_tmp < dims[0]+0.0001) and (ry_tmp < dims[1]+0.0001) and (rz_tmp < dims[2]+0.0001):
                                specInd = lattice.getSpecieIndex(sym_uc)
                                lattice.specieCount[specInd] += 1
                                
                                lattice.specie[count] = np.int32(specInd)
                                lattice.pos[count*3] = np.float64(rx_tmp)    
                                lattice.pos[count*3+1] = np.float64(ry_tmp)    
                                lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                                lattice.charge[count] = np.float64(q_uc)
                        
                                count += 1
                        
        totalQ = count * q_uc
        
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
