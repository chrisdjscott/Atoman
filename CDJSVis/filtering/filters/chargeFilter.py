
"""
Charge filter

"""
from . import base


class ChargeFilterSettings(base.BaseSettings):
    """
    Settings for the charge filter
    
    """
    def __init__(self):
        super(ChargeFilterSettings, self).__init__()
        
        self.registerSetting("minCharge", default=-100.0)
        self.registerSetting("maxCharge", default=100.0)
