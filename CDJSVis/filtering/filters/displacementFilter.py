
"""
Displacement filter

"""
from . import base


class DisplacementFilterSettings(base.BaseSettings):
    """
    Settings for the displacement filter
    
    """
    def __init__(self):
        super(DisplacementFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("drawDisplacementVectors", default=False)
        self.registerSetting("minDisplacement", default=1.2)
        self.registerSetting("maxDisplacement", default=1000.0)
        self.registerSetting("bondThicknessVTK", default=0.4)
        self.registerSetting("bondThicknessPOV", default=0.4)
        self.registerSetting("bondNumSides", default=5)
