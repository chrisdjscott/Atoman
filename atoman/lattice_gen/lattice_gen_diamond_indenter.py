
"""
Generate Carbon Diamond Indenter Lattice

@author: Kenny Jolley
Derived from similar form by Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

import numpy as np
import math as math

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
    def __init__(self, sym1="C_", charge1=0.0, AtomLayers=15, a0=3.556717,
                 TipCutLayers=1,CornerSliceLayers=3,
                 pbcx=False, pbcy=False, pbcz=False):
        self.sym1 = sym1
        self.charge1 = charge1
        self.AtomLayers = AtomLayers
        self.CornerSliceLayers = CornerSliceLayers
        self.a0 = a0
        self.pbcx = pbcx
        self.pbcy = pbcy
        self.pbcz = pbcz
        self.TipCutLayers=TipCutLayers

################################################################################

class DiamondIndenterGenerator(object):
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
            
    # Rotation about given axis by given angle (Using the Euler-Rodrigues formula)
    def rotation_matrix(self, axis, theta):
        """
        Using the Euler-Rodrigues formula:
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        Usage:
        v = [3, 5, 0]
        axis = [4, 4, 1]
        theta = 1.2
        
        print(np.dot(rotation_matrix(axis,theta), v))
        """
        axis = np.asarray(axis)
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2.0)
        b, c, d = -axis*math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    
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
        
        # Fill initial cube for the Indenter
        iStop = args.AtomLayers + 1
        jStop = args.AtomLayers + 1
        kStop = args.AtomLayers + 1
        
        
        # lattice dimensions
        dims = [2.0*a0*args.AtomLayers, 2.0*a0*args.AtomLayers, 2.0*a0*args.AtomLayers]
        
        # Center tip position in x and y
        tipposx = a0*args.AtomLayers
        tipposy = a0*args.AtomLayers
        # place at top of the box
        #tipposz = a0*args.AtomLayers*(2.0-0.577350269189626)   # height is a0*args.AtomLayers/math.sqrt(3)
        tipposz = 2.0*a0*args.AtomLayers - 0.577350269189626*a0*(args.AtomLayers+1)   # height is a0*args.AtomLayers/math.sqrt(3)
        
        
        
        # lattice structure
        lattice = Lattice()
        
        # set dimensions
        lattice.setDims(dims)
        
        # create specie list
        lattice.addSpecie(args.sym1)
        
        # slice plane x coord
        slice_x0 = np.float64(a0*args.AtomLayers + 1.0)
        # slice plane x coord tip cut plane
        slice_x0_tip = np.float64(a0*args.TipCutLayers + 1.0)
        
        # corner slices
        corner_slice_x0 = a0*(args.AtomLayers-args.CornerSliceLayers) - 0.5
        
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
                        if (rx_tmp > dims[0]/2.0+0.0001) or (ry_tmp > dims[1]/2.0+0.0001) or (rz_tmp > dims[2]/2.0+0.0001):
                            continue
                        
                        # rotation 90 degrees around y axis so dimer rows are paralel to surface
                        v = [rx_tmp,ry_tmp,rz_tmp]
                        axis = [0,1,0]
                        theta = -1.570796327
                        v = np.dot(self.rotation_matrix(axis,theta), v)
                        
                        rx_tmp = v[0]
                        ry_tmp = v[1]
                        rz_tmp = v[2]
                        
                        # Translate since the rotation was about origin 
                        rx_tmp = rx_tmp + a0*args.AtomLayers
                        
                        
                        # skip if above (111) slice plane
                        slice_z = slice_x0 - rx_tmp - ry_tmp
                        if (rz_tmp > slice_z):
                            continue
                        
                        # skip if below tip cut layers
                        if (rz_tmp < (slice_x0_tip - rx_tmp - ry_tmp) ):
                            continue
                        
                        # x corner slice, skip if above slice plane
                        if (rx_tmp > (corner_slice_x0 + ry_tmp + rz_tmp) ):
                            continue
                        
                        # y corner slice, skip if above slice plane
                        if (ry_tmp > (corner_slice_x0 + rx_tmp + rz_tmp) ):
                            continue
                        
                        # z corner slice, skip if above slice plane
                        if (rz_tmp > (corner_slice_x0 + rx_tmp + ry_tmp) ):
                            continue
                        
                        # skip edge atoms
                        if( (rx_tmp - 0.001 < 0) and (ry_tmp - 0.001 < 0) ):
                            c = int( rz_tmp/a0 + 0.5)
                            if(c%2!=0):
                                continue
                        if( (rx_tmp - 0.001 < 0) and (rz_tmp - 0.001 < 0) ):
                            c = int( ry_tmp/a0 + 0.5)
                            if(c%2!=0):
                                continue    
                        if( (ry_tmp - 0.001 < 0) and (rz_tmp - 0.001 < 0) ):
                            c = int( rx_tmp/a0 + 0.5)
                            if(c%2!=0):
                                continue     
                        
                        
                        # displace atoms on surface to form dimer rows.
                        if(rx_tmp - 0.001 < 0):
                            c = int( (ry_tmp + rz_tmp)/a0 + 0.5)
                            if(c%2==0):
                                ry_tmp = ry_tmp + 0.4
                                rz_tmp = rz_tmp + 0.4
                            else:
                                ry_tmp = ry_tmp - 0.4
                                rz_tmp = rz_tmp - 0.4
                        
                        if(ry_tmp - 0.001 < 0):
                            c = int( (rx_tmp + rz_tmp)/a0 + 0.5)
                            if(c%2==0):
                                rx_tmp = rx_tmp + 0.4
                                rz_tmp = rz_tmp + 0.4
                            else:
                                rx_tmp = rx_tmp - 0.4
                                rz_tmp = rz_tmp - 0.4
                        
                        if(rz_tmp - 0.001 < 0):
                            c = int( (rx_tmp + ry_tmp)/a0 + 0.5)
                            if(c%2==0):
                                rx_tmp = rx_tmp + 0.4
                                ry_tmp = ry_tmp + 0.4
                            else:
                                rx_tmp = rx_tmp - 0.4
                                ry_tmp = ry_tmp - 0.4
                        
                        
                        # Increment specie counter
                        specInd = lattice.getSpecieIndex(sym_uc)
                        lattice.specieCount[specInd] += 1
                        lattice.specie[count] = np.int32(specInd)
                        
                        
                        # Rotate lattice
                        v = [rx_tmp,ry_tmp,rz_tmp]
                        #axis = [-1,1,0]
                        #theta = -0.955316618124509 # -math.acos(1 / math.sqrt(3))
                        #v = np.dot(self.rotation_matrix(axis,theta), v)
                        
                        axis = [1,0,0]
                        theta = 0.785398163397448
                        v = np.dot(self.rotation_matrix(axis,theta), v)
                        
                        axis = [0,1,0]
                        theta = -0.615479709
                        v = np.dot(self.rotation_matrix(axis,theta), v)
                        
                        rx_tmp = v[0]
                        ry_tmp = v[1]
                        rz_tmp = v[2]
                        
                        # Translate to given tip position
                        rx_tmp = rx_tmp + tipposx
                        ry_tmp = ry_tmp + tipposy
                        rz_tmp = rz_tmp + tipposz
                        
                        
                        # Save position to lattice structure
                        lattice.pos[count*3] = np.float64(rx_tmp)    
                        lattice.pos[count*3+1] = np.float64(ry_tmp)    
                        lattice.pos[count*3+2] = np.float64(rz_tmp)                    
                        lattice.charge[count] = np.float64(q_uc)
                        
                        # Increment counters
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
