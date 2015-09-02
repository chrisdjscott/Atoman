
"""
Contains GUI forms for a generic scalar filter.

"""
from . import base
from ...filtering.filters import genericScalarFilter


################################################################################

class GenericScalarSettingsDialog(base.GenericSettingsDialog):
    """
    Settings for generic scalar value filterer
    
    """
    def __init__(self, mainWindow, filterType, title, parent=None):
        super(GenericScalarSettingsDialog, self).__init__(title, parent)
        
        self.filterType = filterType
        
        # settings
        self._settings = genericScalarFilter.GenericScalarFilterSettings()
        
        # spin boxes
        self.addDoubleSpinBox("minVal", minVal=-99999, maxVal=99999, step=0.1, toolTip="Minimum visible value", label="Minimum")
        self.addDoubleSpinBox("maxVal", minVal=-99999, maxVal=99999, step=0.1, toolTip="Maximum visible value", label="Maximum")
