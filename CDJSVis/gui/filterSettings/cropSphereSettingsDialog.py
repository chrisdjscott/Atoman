
"""
Contains GUI forms for the crop sphere filter.

"""
from PySide import QtGui

from . import base


################################################################################

class CropSphereSettingsDialog(base.GenericSettingsDialog):
    """
    Crop sphere filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(CropSphereSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Crop sphere"
        
        self.xCentre = 0.0
        self.yCentre = 0.0
        self.zCentre = 0.0
        self.radius = 1.0
        self.invertSelection = 0
        
        self.xCentreSpinBox = QtGui.QDoubleSpinBox()
        self.xCentreSpinBox.setSingleStep(0.01)
        self.xCentreSpinBox.setMinimum(-9999.0)
        self.xCentreSpinBox.setMaximum( 9999.0)
        self.xCentreSpinBox.setValue(self.xCentre)
        self.xCentreSpinBox.setToolTip("Centre of crop region (x)")
        self.xCentreSpinBox.valueChanged.connect(self.xCentreChanged)
        
        self.yCentreSpinBox = QtGui.QDoubleSpinBox()
        self.yCentreSpinBox.setSingleStep(0.01)
        self.yCentreSpinBox.setMinimum(-9999.0)
        self.yCentreSpinBox.setMaximum( 9999.0)
        self.yCentreSpinBox.setValue(self.yCentre)
        self.yCentreSpinBox.setToolTip("Centre of crop region (y)")
        self.yCentreSpinBox.valueChanged.connect(self.yCentreChanged)
        
        self.zCentreSpinBox = QtGui.QDoubleSpinBox()
        self.zCentreSpinBox.setSingleStep(0.01)
        self.zCentreSpinBox.setMinimum(-9999.0)
        self.zCentreSpinBox.setMaximum( 9999.0)
        self.zCentreSpinBox.setValue(self.zCentre)
        self.zCentreSpinBox.setToolTip("Centre of crop region (z)")
        self.zCentreSpinBox.valueChanged.connect(self.zCentreChanged)
        
        self.contentLayout.addRow("Centre (x)", self.xCentreSpinBox)
        self.contentLayout.addRow("Centre (y)", self.yCentreSpinBox)
        self.contentLayout.addRow("Centre (z)", self.zCentreSpinBox)
        
        # radius
        self.radiusSpinBox = QtGui.QDoubleSpinBox()
        self.radiusSpinBox.setSingleStep(1)
        self.radiusSpinBox.setMinimum(0.0)
        self.radiusSpinBox.setMaximum(9999.0)
        self.radiusSpinBox.setValue(self.radius)
        self.radiusSpinBox.setToolTip("Radius of sphere")
        self.radiusSpinBox.valueChanged.connect(self.radiusChanged)
        self.contentLayout.addRow("Radius", self.radiusSpinBox)
        
        # invert selection
        self.invertCheckBox = QtGui.QCheckBox()
        self.invertCheckBox.setChecked(0)
        self.invertCheckBox.setToolTip("Invert selection")
        self.invertCheckBox.stateChanged.connect(self.invertChanged)
        self.contentLayout.addRow("Invert selection", self.invertCheckBox)
        
        # set to centre
        self.setToLatticeButton = QtGui.QPushButton('Set to lattice centre')
        self.setToLatticeButton.setAutoDefault(0)
        self.setToLatticeButton.setToolTip('Set to lattice centre')
        self.setToLatticeButton.clicked.connect(self.setToLattice)
        self.contentLayout.addRow(self.setToLatticeButton)
    
    def invertChanged(self, index):
        """
        Invert setting changed.
        
        """
        if self.invertCheckBox.isChecked():
            self.invertSelection = 1
        
        else:
            self.invertSelection = 0
    
    def setToLattice(self):
        """
        Set centre to lattice centre.
        
        """
        self.xCentreSpinBox.setValue(self.pipelinePage.inputState.cellDims[0] / 2.0)
        self.yCentreSpinBox.setValue(self.pipelinePage.inputState.cellDims[1] / 2.0)
        self.zCentreSpinBox.setValue(self.pipelinePage.inputState.cellDims[2] / 2.0)
    
    def radiusChanged(self, val):
        """
        Radius changed.
        
        """
        self.radius = val
    
    def xCentreChanged(self, val):
        """
        X centre changed.
        
        """
        self.xCentre = val
    
    def yCentreChanged(self, val):
        """
        Y centre changed.
        
        """
        self.yCentre = val
    
    def zCentreChanged(self, val):
        """
        Z centre changed.
        
        """
        self.zCentre = val
