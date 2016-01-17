
"""
Contains GUI forms for the ACNA filter.

"""
import functools

import numpy as np
from PySide import QtGui, QtCore

from . import base
from ...filtering.filters import acnaFilter


class AcnaSettingsDialog(base.GenericSettingsDialog):
    """
    Settings for adaptive common neighbour analysis
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(AcnaSettingsDialog, self).__init__(title, parent, "ACNA")
        
        self.addProvidedScalar("ACNA")
        self._settings = acnaFilter.AcnaFilterSettings()
        
        # max bond distance spin box
        toolTip = "<p>This is used for spatially decomposing the system. "
        toolTip += "This should be set large enough that the required neighbours will be included.</p>"
        self.addDoubleSpinBox("maxBondDistance", label="Max bond distance", minVal=2.0, maxVal=9.99, step=0.1,
                              toolTip=toolTip)
        
        self.addHorizontalDivider()
        
        # filter check
        filterByStructureCheck = QtGui.QCheckBox()
        filterByStructureCheck.setChecked(self._settings.getSetting("filteringEnabled"))
        filterByStructureCheck.setToolTip("Filter atoms by structure type")
        filterByStructureCheck.stateChanged.connect(self.filteringToggled)
        self.contentLayout.addRow("<b>Filter by structure</b>", filterByStructureCheck)
        
        # filter options group
        filterer = self.parent.filterer
        self.structureChecks = {}
        for i, structure in enumerate(filterer.knownStructures):
            cb = QtGui.QCheckBox()
            cb.setChecked(True)
            cb.stateChanged.connect(functools.partial(self.visToggled, i))
            self.contentLayout.addRow(structure, cb)
            self.structureChecks[structure] = cb
            
            if not self._settings.getSetting("filteringEnabled"):
                cb.setEnabled(False)
        
        self.structureVisibility = np.ones(len(filterer.knownStructures), dtype=np.int32)
        
        self.addLinkToHelpPage("usage/analysis/filters/acna.html")
    
    def visToggled(self, index, checkState):
        """
        Visibility toggled for one structure type
        
        """
        visible = 0 if checkState == QtCore.Qt.Unchecked else 1
        self._settings.updateSettingArray("structureVisibility", index, visible)
    
    def filteringToggled(self, state):
        """
        Filtering toggled
        
        """
        enabled = False if state == QtCore.Qt.Unchecked else True
        self._settings.updateSetting("filteringEnabled", enabled)
        
        # disable structure checks
        for key in self.structureChecks:
            cb = self.structureChecks[key]
            cb.setEnabled(enabled)
