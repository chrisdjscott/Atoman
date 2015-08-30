
"""
Generic scalar filter

"""
from . import base


class GenericScalarFilterSettings(base.BaseSettings):
    """
    Settings for the generic scalar filter
    
    """
    def __init__(self):
        super(GenericScalarFilterSettings, self).__init__()
        
        self.registerSetting("minVal", default=-999.0)
        self.registerSetting("maxVal", default=999.0)
