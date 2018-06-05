
"""
Contains GUI forms for the Voronoi volume filter.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from . import base
from ...filtering.filters import voronoiVolumeFilter


################################################################################

class VoronoiVolumeSettingsDialog(base.GenericSettingsDialog):
    """
    Settings for Voronoi volume filter
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(VoronoiVolumeSettingsDialog, self).__init__(title, parent, "Voronoi volume")
        self.setMinimumWidth(350)
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
