
"""
Filters.

@author: Chris Scott

"""
import os
import copy
import time

import numpy as np
import vtk

from ..visclibs import filtering as filtering_c
from ..visclibs import defects as defects_c
from ..visclibs import clusters as clusters_c
from ..visclibs import bonds as bonds_c
from ..rendering import renderer
from ..rendering import renderBonds
from ..visutils import vectors
from . import clusters
from ..state.atoms import elements


################################################################################

class GenericFilter(object):
    """
    Generic filter.
    
    """
    def __init__(self):
        pass
    
    def runFilter(self, inputState, filterSettings, visibleAtoms=None, refState=None):
        """
        Run the filter.
        
        """
        return self.runFilterMain(inputState, filterSettings, visibleAtoms, refState)
    
    def runFilterMain(self, inputState, filterSettings, visibleAtoms, refState):
        """
        Run filter main (should be subclassed.
        
        """
        return visibleAtoms

################################################################################

class SpecieFilter(GenericFilter):
    """
    Specie filter.
    
    """
    def __init__(self):
        super(SpecieFilter, self).__init__()
    
    def runFilterMain(self, inputState, filterSettings, visibleAtoms, refState):
        """
        Run filter main.
        
        """
        



