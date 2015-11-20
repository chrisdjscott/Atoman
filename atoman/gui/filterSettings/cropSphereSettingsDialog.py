
"""
Contains GUI forms for the crop sphere filter.

"""
import functools

from PySide import QtGui, QtCore

from . import base
from ...filtering.filters import cropSphereFilter


################################################################################

class CropSphereSettingsDialog(base.GenericSettingsDialog):
    """
    Crop sphere filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(CropSphereSettingsDialog, self).__init__(title, parent, "Crop sphere")
        
        self._settings = cropSphereFilter.CropSphereFilterSettings()
        
        self.xCentreSpinBox = QtGui.QDoubleSpinBox()
        self.xCentreSpinBox.setSingleStep(0.01)
        self.xCentreSpinBox.setMinimum(-9999.0)
        self.xCentreSpinBox.setMaximum( 9999.0)
        self.xCentreSpinBox.setValue(self._settings.getSetting("xCentre"))
        self.xCentreSpinBox.setToolTip("Centre of crop region (x)")
        self.xCentreSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "xCentre"))
        
        self.yCentreSpinBox = QtGui.QDoubleSpinBox()
        self.yCentreSpinBox.setSingleStep(0.01)
        self.yCentreSpinBox.setMinimum(-9999.0)
        self.yCentreSpinBox.setMaximum( 9999.0)
        self.yCentreSpinBox.setValue(self._settings.getSetting("yCentre"))
        self.yCentreSpinBox.setToolTip("Centre of crop region (y)")
        self.yCentreSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "yCentre"))
        
        self.zCentreSpinBox = QtGui.QDoubleSpinBox()
        self.zCentreSpinBox.setSingleStep(0.01)
        self.zCentreSpinBox.setMinimum(-9999.0)
        self.zCentreSpinBox.setMaximum( 9999.0)
        self.zCentreSpinBox.setValue(self._settings.getSetting("zCentre"))
        self.zCentreSpinBox.setToolTip("Centre of crop region (z)")
        self.zCentreSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "zCentre"))
        
        self.contentLayout.addRow("Centre (x)", self.xCentreSpinBox)
        self.contentLayout.addRow("Centre (y)", self.yCentreSpinBox)
        self.contentLayout.addRow("Centre (z)", self.zCentreSpinBox)
        
        # radius
        self.radiusSpinBox = QtGui.QDoubleSpinBox()
        self.radiusSpinBox.setSingleStep(1)
        self.radiusSpinBox.setMinimum(0.0)
        self.radiusSpinBox.setMaximum(9999.0)
        self.radiusSpinBox.setValue(self._settings.getSetting("radius"))
        self.radiusSpinBox.setToolTip("Radius of sphere")
        self.radiusSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "radius"))
        self.contentLayout.addRow("Radius", self.radiusSpinBox)
        
        # invert selection
        self.invertCheckBox = QtGui.QCheckBox()
        self.invertCheckBox.setChecked(self._settings.getSetting("invertSelection"))
        self.invertCheckBox.setToolTip("Invert selection")
        self.invertCheckBox.stateChanged.connect(self.invertChanged)
        self.contentLayout.addRow("Invert selection", self.invertCheckBox)
        
        # set to centre
        self.setToLatticeButton = QtGui.QPushButton('Set to lattice centre')
        self.setToLatticeButton.setAutoDefault(0)
        self.setToLatticeButton.setToolTip('Set to lattice centre')
        self.setToLatticeButton.clicked.connect(self.setToLattice)
        self.contentLayout.addRow(self.setToLatticeButton)
    
    def invertChanged(self, state):
        """Invert setting changed."""
        invert = False if state == QtCore.Qt.Unchecked else True
        self._settings.updateSetting("invertSelection", invert)
    
    def setToLattice(self):
        """Set centre to lattice centre."""
        self.xCentreSpinBox.setValue(self.pipelinePage.inputState.cellDims[0] / 2.0)
        self.yCentreSpinBox.setValue(self.pipelinePage.inputState.cellDims[1] / 2.0)
        self.zCentreSpinBox.setValue(self.pipelinePage.inputState.cellDims[2] / 2.0)
