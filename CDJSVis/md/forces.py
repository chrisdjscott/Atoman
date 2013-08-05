
"""
This module handles force evaluations.

@author: Chris Scott

"""
import os
import sys

import numpy as np

from . import LBOMDInterface as lbomd_interface
from ..visutils.utilities import log_error



################################################################################

class ForceConfig(object):
    """
    Holds settings for force calculation.
    
    """
    def __init__(self):
        self.md_dir = None
        self.md_type = None
        self.log = None
        
################################################################################

def _calc_force_lbomd(lattice, force_config, image):
    """
    Calculate force using LBOMD interface.
    
    """
    log = force_config.log
    md_dir = force_config.md_dir
    
    if log is not None:
        log("LBOMD force calculation")
    
    # we work in the MD code dir, then change back to wherever we were originally
    # so store current working dir
    OWD = os.getcwd()
    
    # and change to MD dir
    try:
        os.chdir(md_dir)
    except OSError:
        log_error("ERROR: could not change to md_dir (%s)\n" % md_dir)
        return 33
        
    # which image
    if image is None:
        imageFor = -1
    
    else:
        imageFor = image
    
    # create specie string containing specie list to pass to fortran
    specieList = lattice.specieList
    specieString = "".join(specieList)
    
    try:
        lattice.totalEnergy, lattice.maxForce, lattice.maxForceAtomNo, status = lbomd_interface.calcforce(imageFor, specieString, lattice.specie, lattice.pos, lattice.charge, lattice.force, lattice.KE, lattice.PE, lattice.cellDims[0], lattice.cellDims[1], lattice.cellDims[2], np.empty(0, np.int32), np.empty(0, np.int32))
    except:
        log_error("ERROR: LBOMD calc_force failed (caught exception)")
        return 44
    
    # check status
    if status:
        log_error("ERROR: LBOMD calc_force failed with status %d\n" % status)
        return status
    
    if log is not None:
        log(__name__, "total energy is %f eV" % (lattice.totalEnergy), 3, 2)
#         log(__name__, "max force is %f eV/A (atom %d)" % (lattice.maxForce, lattice.maxForceAtomNo), 3, 2)
    
    # finally, change back to original working directory
    try:
        os.chdir(OWD)
    except OSError:
        sys.exit(__name__+": ERROR: could not change to dir: "+OWD)
    
    return status

################################################################################

def calc_force(lattice, force_config, image=None):
    """
    Calculate force on given lattice (or image within lattice).
    
    """
    if not os.path.exists(force_config.md_dir):
        log_error("ERROR: md_dir (%s) does not exist\n" % force_config.md_dir)
        return 11
    
    if force_config.md_type == "LBOMD":
        status = _calc_force_lbomd(lattice, force_config, image)
    
    else:
        log_error("ERROR: unrecognised md_type (%s)\n" % force_config.md_type)
        return 22
    
    return status
