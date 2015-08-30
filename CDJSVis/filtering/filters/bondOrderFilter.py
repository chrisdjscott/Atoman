
"""
Bond order filter

"""
from . import base


class BondOrderFilterSettings(base.BaseSettings):
    """
    Settings for the bond order filter
    
    """
    def __init__(self):
        super(BondOrderFilterSettings, self).__init__()
        
        self.registerSetting("filterQ4Enabled", default=False)
        self.registerSetting("filterQ6Enabled", default=False)
        self.registerSetting("maxBondDistance", default=4.0)
        self.registerSetting("minQ4", default=0.0)
        self.registerSetting("maxQ4", default=99.0)
        self.registerSetting("minQ6", default=0.0)
        self.registerSetting("maxQ6", default=99.0)
