
"""
Crop box filter

"""
from . import base


class CropBoxFilterSettings(base.BaseSettings):
    """
    Settings for the crop box filter
    
    """
    def __init__(self):
        super(CropBoxFilterSettings, self).__init__()
        
        self.registerSetting("invertSelection", default=False)
        self.registerSetting("xEnabled", default=False)
        self.registerSetting("yEnabled", default=False)
        self.registerSetting("zEnabled", default=False)
        self.registerSetting("xmin", default=5.0)
        self.registerSetting("xmax", default=5.0)
        self.registerSetting("ymin", default=5.0)
        self.registerSetting("ymax", default=5.0)
        self.registerSetting("zmin", default=5.0)
        self.registerSetting("zmax", default=5.0)
