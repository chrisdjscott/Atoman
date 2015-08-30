
"""
Crop sphere filter

"""
from . import base


class CropSphereFilterSettings(base.BaseSettings):
    """
    Settings for the crop sphere filter
    
    """
    def __init__(self):
        super(CropSphereFilterSettings, self).__init__()
        
        self.registerSetting("invertSelection", default=False)
        self.registerSetting("xCentre", default=0.0)
        self.registerSetting("yCentre", default=0.0)
        self.registerSetting("zCentre", default=0.0)
        self.registerSetting("radius", default=1.0)
