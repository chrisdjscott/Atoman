
"""
Contains GUI forms for the crop box filter.

"""
from PySide import QtGui

from . import base


################################################################################

class CropBoxSettingsDialog(base.GenericSettingsDialog):
    """
    GUI for the Crop Box filter settings.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(CropBoxSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Crop box"
        
        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0
        self.zmin = 0.0
        self.zmax = 0.0
        self.xEnabled = 0
        self.yEnabled = 0
        self.zEnabled = 0
        self.invertSelection = 0
        
        # x
        self.xCropCheckBox = QtGui.QCheckBox()
        self.xCropCheckBox.setChecked(0)
        self.xCropCheckBox.setToolTip("Enable cropping in the x direction")
        self.xCropCheckBox.stateChanged[int].connect(self.changedXEnabled)
        self.xMinRangeSpinBox = QtGui.QDoubleSpinBox()
        self.xMinRangeSpinBox.setSingleStep(1)
        self.xMinRangeSpinBox.setMinimum(-9999.0)
        self.xMinRangeSpinBox.setMaximum(9999.0)
        self.xMinRangeSpinBox.setValue(self.xmin)
        self.xMinRangeSpinBox.setToolTip("Minimum x value")
        self.xMinRangeSpinBox.valueChanged.connect(self.setXMin)
        self.xMaxRangeSpinBox = QtGui.QDoubleSpinBox()
        self.xMaxRangeSpinBox.setSingleStep(1)
        self.xMaxRangeSpinBox.setMinimum(-9999.0)
        self.xMaxRangeSpinBox.setMaximum(9999.0)
        self.xMaxRangeSpinBox.setValue(self.xmax)
        self.xMaxRangeSpinBox.setToolTip("Maximum x value")
        self.xMaxRangeSpinBox.valueChanged.connect(self.setXMax)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.xMinRangeSpinBox)
        hbox.addWidget(QtGui.QLabel("-"))
        hbox.addWidget(self.xMaxRangeSpinBox)
        
        self.contentLayout.addRow("X Crop Enabled", self.xCropCheckBox)
        self.contentLayout.addRow("Range", hbox)
        
        # divider
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        self.contentLayout.addRow(line)
        
        # y
        self.yCropCheckBox = QtGui.QCheckBox()
        self.yCropCheckBox.setChecked(0)
        self.yCropCheckBox.setToolTip("Enable cropping in the y direction")
        self.yCropCheckBox.stateChanged[int].connect(self.changedYEnabled)
        self.yMinRangeSpinBox = QtGui.QDoubleSpinBox()
        self.yMinRangeSpinBox.setSingleStep(1)
        self.yMinRangeSpinBox.setMinimum(-9999.0)
        self.yMinRangeSpinBox.setMaximum(9999.0)
        self.yMinRangeSpinBox.setValue(self.ymin)
        self.yMinRangeSpinBox.setToolTip("Minimum y value")
        self.yMinRangeSpinBox.valueChanged.connect(self.setYMin)
        self.yMaxRangeSpinBox = QtGui.QDoubleSpinBox()
        self.yMaxRangeSpinBox.setSingleStep(1)
        self.yMaxRangeSpinBox.setMinimum(-9999.0)
        self.yMaxRangeSpinBox.setMaximum(9999.0)
        self.yMaxRangeSpinBox.setValue(self.ymax)
        self.yMaxRangeSpinBox.setToolTip("Maximum y value")
        self.yMaxRangeSpinBox.valueChanged.connect(self.setYMax)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.yMinRangeSpinBox)
        hbox.addWidget(QtGui.QLabel("-"))
        hbox.addWidget(self.yMaxRangeSpinBox)
        
        self.contentLayout.addRow("Y Crop Enabled", self.yCropCheckBox)
        self.contentLayout.addRow("Range", hbox)
        
        # divider
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        self.contentLayout.addRow(line)
        
        # z
        self.zCropCheckBox = QtGui.QCheckBox()
        self.zCropCheckBox.setChecked(0)
        self.zCropCheckBox.setToolTip("Enable cropping in the z direction")
        self.zCropCheckBox.stateChanged[int].connect(self.changedZEnabled)
        self.zMinRangeSpinBox = QtGui.QDoubleSpinBox()
        self.zMinRangeSpinBox.setSingleStep(1)
        self.zMinRangeSpinBox.setMinimum(-9999.0)
        self.zMinRangeSpinBox.setMaximum(9999.0)
        self.zMinRangeSpinBox.setValue(self.zmin)
        self.zMinRangeSpinBox.setToolTip("Minimum z value")
        self.zMinRangeSpinBox.valueChanged.connect(self.setZMin)
        self.zMaxRangeSpinBox = QtGui.QDoubleSpinBox()
        self.zMaxRangeSpinBox.setSingleStep(1)
        self.zMaxRangeSpinBox.setMinimum(-9999.0)
        self.zMaxRangeSpinBox.setMaximum(9999.0)
        self.zMaxRangeSpinBox.setValue(self.zmax)
        self.zMaxRangeSpinBox.setToolTip("Maximum z value")
        self.zMaxRangeSpinBox.valueChanged.connect(self.setZMax)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.zMinRangeSpinBox)
        hbox.addWidget(QtGui.QLabel("-"))
        hbox.addWidget(self.zMaxRangeSpinBox)
        
        self.contentLayout.addRow("Z Crop Enabled", self.zCropCheckBox)
        self.contentLayout.addRow("Range", hbox)
        
        # divider
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        self.contentLayout.addRow(line)
        
        # invert selection
        self.invertCheckBox = QtGui.QCheckBox()
        self.invertCheckBox.setChecked(0)
        self.invertCheckBox.stateChanged.connect(self.invertChanged)
        self.invertCheckBox.setToolTip("Invert selection")
        self.contentLayout.addRow("Invert selection", self.invertCheckBox)
        
        # reset
        self.setToLatticeButton = QtGui.QPushButton('Set to lattice')
        self.setToLatticeButton.setAutoDefault(0)
        self.setToLatticeButton.setToolTip('Set crop to lattice dimensions')
        self.setToLatticeButton.clicked.connect(self.setCropToLattice)
        self.contentLayout.addRow(self.setToLatticeButton)
    
    def invertChanged(self, index):
        """
        Invert setting changed.
        
        """
        if self.invertCheckBox.isChecked():
            self.invertSelection = 1
        
        else:
            self.invertSelection = 0
    
    def setCropToLattice(self):
        self.xMinRangeSpinBox.setValue(0.0)
        self.xMaxRangeSpinBox.setValue(self.pipelinePage.inputState.cellDims[0])
        self.yMinRangeSpinBox.setValue(0.0)
        self.yMaxRangeSpinBox.setValue(self.pipelinePage.inputState.cellDims[1])
        self.zMinRangeSpinBox.setValue(0.0)
        self.zMaxRangeSpinBox.setValue(self.pipelinePage.inputState.cellDims[2])
    
    def changedXEnabled(self):
        if self.xCropCheckBox.isChecked():
            self.xEnabled = 1
        else:
            self.xEnabled = 0
    
    def changedYEnabled(self):
        if self.yCropCheckBox.isChecked():
            self.yEnabled = 1
        else:
            self.yEnabled = 0
    
    def changedZEnabled(self):
        if self.zCropCheckBox.isChecked():
            self.zEnabled = 1
        else:
            self.zEnabled = 0
    
    def setXMin(self, val):
        self.xmin = val
    
    def setXMax(self, val):
        self.xmax = val
    
    def setYMin(self, val):
        self.ymin = val
    
    def setYMax(self, val):
        self.ymax = val
    
    def setZMin(self, val):
        self.zmin = val
    
    def setZMax(self, val):
        self.zmax = val
    
    def refresh(self):
        self.xMinRangeSpinBox.setValue(self.xmin)
        self.xMaxRangeSpinBox.setValue(self.xmax)
        self.yMinRangeSpinBox.setValue(self.ymin)
        self.yMaxRangeSpinBox.setValue(self.ymax)
        self.zMinRangeSpinBox.setValue(self.zmin)
        self.zMaxRangeSpinBox.setValue(self.zmax)
        self.xCropCheckBox.setChecked( self.xEnabled )
        self.yCropCheckBox.setChecked( self.yEnabled )
        self.zCropCheckBox.setChecked( self.zEnabled )
    
    def reset(self):
        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0
        self.zmin = 0.0
        self.zmax = 0.0
        self.xEnabled = 0
        self.yEnabled = 0
        self.zEnabled = 0
        self.refresh()
