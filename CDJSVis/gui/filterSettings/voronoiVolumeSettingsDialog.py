
"""
Contains GUI forms for the Voronoi volume filter.

"""
from . import base
from ...filtering.filters import voronoiVolumeFilter


################################################################################

class VoronoiVolumeSettingsDialog(base.GenericSettingsDialog):
    """
    Settings for Voronoi volume filter
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(VoronoiVolumeSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Voronoi volume"
        self.addProvidedScalar("Voronoi volume")
        
        # settings
        self._settings = voronoiVolumeFilter.VoronoiVolumeFilterSettings()
        
        # filtering options
        self.addCheckBox("filteringEnabled", toolTip="Filter atoms by slip", label="<b>Enable filtering</b>", extraSlot=self.filteringToggled)
        
        self.minVoroVolSpin = self.addDoubleSpinBox("minVoroVol", minVal=0, maxVal=9999, step=0.1, toolTip="Minimum visible Voronoi volume",
                                                    label="Minimum", settingEnabled="filteringEnabled")
        
        self.maxVoroVolSpin = self.addDoubleSpinBox("maxVoroVol", minVal=0, maxVal=9999, step=0.1, toolTip="Maximum visible Voronoi volume",
                                                    label="Maximum", settingEnabled="filteringEnabled")
    
    def filteringToggled(self, enabled):
        """Filtering toggled."""
        self.minVoroVolSpin.setEnabled(enabled)
        self.maxVoroVolSpin.setEnabled(enabled)
