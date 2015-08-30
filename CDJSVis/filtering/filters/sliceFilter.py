
"""
Slice filter

"""
from . import base


class SliceFilterSettings(base.BaseSettings):
    """
    Settings for the slice filter
    
    """
    def __init__(self):
        super(SliceFilterSettings, self).__init__()
        
        self.registerSetting("showSlicePlaneChecked", default=False)
        self.registerSetting("x0", default=0.0)
        self.registerSetting("y0", default=0.0)
        self.registerSetting("z0", default=0.0)
        self.registerSetting("xn", default=1.0)
        self.registerSetting("yn", default=0.0)
        self.registerSetting("zn", default=0.0)
        self.registerSetting("invert", default=False)
