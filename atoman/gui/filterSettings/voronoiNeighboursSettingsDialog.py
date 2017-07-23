
"""
Contains GUI forms for the Voronoi neighbours filter.

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from . import base
from ...filtering.filters import voronoiNeighboursFilter


################################################################################
class VoronoiNeighboursSettingsDialog(base.GenericSettingsDialog):
    """
    Voronoi neighbours filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(VoronoiNeighboursSettingsDialog, self).__init__(title, parent, "Voronoi neighbours")
        self.setMinimumWidth(350)
        self.addProvidedScalar("Voronoi neighbours")
        
        # settings
        self._settings = voronoiNeighboursFilter.VoronoiNeighboursFilterSettings()
        
        # filtering options
        self.addCheckBox("filteringEnabled", toolTip="Filter atoms by Voronoi neighbours", label="<b>Enable filtering</b>",
                         extraSlot=self.filteringToggled)
        
        self.minVoroNebsSpin = self.addSpinBox("minVoroNebs", minVal=0, maxVal=999, step=1, toolTip="Minimum number of Voronoi neighbours",
                                               label="Minimum", settingEnabled="filteringEnabled")
        
        self.maxVoroNebsSpin = self.addSpinBox("maxVoroNebs", minVal=0, maxVal=999, step=1, toolTip="Maximum number of Voronoi neighbours",
                                               label="Maximum", settingEnabled="filteringEnabled")
    
    def filteringToggled(self, enabled):
        """Filtering toggled."""
        print("ENABLED", enabled)
        self.minVoroNebsSpin.setEnabled(enabled)
        self.maxVoroNebsSpin.setEnabled(enabled)
