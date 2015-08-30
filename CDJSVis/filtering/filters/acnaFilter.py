
"""
ACNA filter

"""
import numpy as np

from ..filterer import Filterer
from . import base


class AcnaFilterSettings(base.BaseSettings):
    """
    Settings for the ACNA filter
    
    """
    def __init__(self):
        super(AcnaFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("maxBondDistance", default=5.0)
        self.registerSetting("structureVisibility", default=np.ones(len(Filterer.knownStructures), dtype=np.int32))
