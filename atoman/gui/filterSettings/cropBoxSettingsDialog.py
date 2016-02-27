
"""
Contains GUI forms for the crop box filter.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import functools

from PyQt5 import QtCore, QtWidgets


from . import base
from ...filtering.filters import cropBoxFilter

################################################################################

class CropBoxSettingsDialog(base.GenericSettingsDialog):
    """
    GUI for the Crop Box filter settings.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(CropBoxSettingsDialog, self).__init__(title, parent, "Crop box")
        
        self._settings = cropBoxFilter.CropBoxFilterSettings()
        
        # x
        self.xCropCheckBox = QtWidgets.QCheckBox()
        self.xCropCheckBox.setChecked(self._settings.getSetting("xEnabled"))
        self.xCropCheckBox.setToolTip("Enable cropping in the x direction")
        self.xCropCheckBox.stateChanged.connect(self.changedXEnabled)
        self.xMinRangeSpinBox = QtWidgets.QDoubleSpinBox()
        self.xMinRangeSpinBox.setSingleStep(1)
        self.xMinRangeSpinBox.setMinimum(-9999.0)
        self.xMinRangeSpinBox.setMaximum(9999.0)
        self.xMinRangeSpinBox.setValue(self._settings.getSetting("xmin"))
        self.xMinRangeSpinBox.setToolTip("Minimum x value")
        self.xMinRangeSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "xmin"))
        self.xMaxRangeSpinBox = QtWidgets.QDoubleSpinBox()
        self.xMaxRangeSpinBox.setSingleStep(1)
        self.xMaxRangeSpinBox.setMinimum(-9999.0)
        self.xMaxRangeSpinBox.setMaximum(9999.0)
        self.xMaxRangeSpinBox.setValue(self._settings.getSetting("xmax"))
        self.xMaxRangeSpinBox.setToolTip("Maximum x value")
        self.xMaxRangeSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "xmax"))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.xMinRangeSpinBox)
        hbox.addWidget(QtWidgets.QLabel("-"))
        hbox.addWidget(self.xMaxRangeSpinBox)
        
        self.contentLayout.addRow("X Crop Enabled", self.xCropCheckBox)
        self.contentLayout.addRow("Range", hbox)
        
        # divider
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.contentLayout.addRow(line)
        
        # y
        self.yCropCheckBox = QtWidgets.QCheckBox()
        self.yCropCheckBox.setChecked(self._settings.getSetting("yEnabled"))
        self.yCropCheckBox.setToolTip("Enable cropping in the y direction")
        self.yCropCheckBox.stateChanged.connect(self.changedYEnabled)
        self.yMinRangeSpinBox = QtWidgets.QDoubleSpinBox()
        self.yMinRangeSpinBox.setSingleStep(1)
        self.yMinRangeSpinBox.setMinimum(-9999.0)
        self.yMinRangeSpinBox.setMaximum(9999.0)
        self.yMinRangeSpinBox.setValue(self._settings.getSetting("ymin"))
        self.yMinRangeSpinBox.setToolTip("Minimum y value")
        self.yMinRangeSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "ymin"))
        self.yMaxRangeSpinBox = QtWidgets.QDoubleSpinBox()
        self.yMaxRangeSpinBox.setSingleStep(1)
        self.yMaxRangeSpinBox.setMinimum(-9999.0)
        self.yMaxRangeSpinBox.setMaximum(9999.0)
        self.yMaxRangeSpinBox.setValue(self._settings.getSetting("ymax"))
        self.yMaxRangeSpinBox.setToolTip("Maximum y value")
        self.yMaxRangeSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "ymax"))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.yMinRangeSpinBox)
        hbox.addWidget(QtWidgets.QLabel("-"))
        hbox.addWidget(self.yMaxRangeSpinBox)
        
        self.contentLayout.addRow("Y Crop Enabled", self.yCropCheckBox)
        self.contentLayout.addRow("Range", hbox)
        
        # divider
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.contentLayout.addRow(line)
        
        # z
        self.zCropCheckBox = QtWidgets.QCheckBox()
        self.zCropCheckBox.setChecked(self._settings.getSetting("zEnabled"))
        self.zCropCheckBox.setToolTip("Enable cropping in the z direction")
        self.zCropCheckBox.stateChanged.connect(self.changedZEnabled)
        self.zMinRangeSpinBox = QtWidgets.QDoubleSpinBox()
        self.zMinRangeSpinBox.setSingleStep(1)
        self.zMinRangeSpinBox.setMinimum(-9999.0)
        self.zMinRangeSpinBox.setMaximum(9999.0)
        self.zMinRangeSpinBox.setValue(self._settings.getSetting("zmin"))
        self.zMinRangeSpinBox.setToolTip("Minimum z value")
        self.zMinRangeSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "zmin"))
        self.zMaxRangeSpinBox = QtWidgets.QDoubleSpinBox()
        self.zMaxRangeSpinBox.setSingleStep(1)
        self.zMaxRangeSpinBox.setMinimum(-9999.0)
        self.zMaxRangeSpinBox.setMaximum(9999.0)
        self.zMaxRangeSpinBox.setValue(self._settings.getSetting("zmax"))
        self.zMaxRangeSpinBox.setToolTip("Maximum z value")
        self.zMaxRangeSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "zmax"))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.zMinRangeSpinBox)
        hbox.addWidget(QtWidgets.QLabel("-"))
        hbox.addWidget(self.zMaxRangeSpinBox)
        
        self.contentLayout.addRow("Z Crop Enabled", self.zCropCheckBox)
        self.contentLayout.addRow("Range", hbox)
        
        # divider
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.contentLayout.addRow(line)
        
        # invert selection
        self.invertCheckBox = QtWidgets.QCheckBox()
        self.invertCheckBox.setChecked(self._settings.getSetting("invertSelection"))
        self.invertCheckBox.stateChanged.connect(self.invertChanged)
        self.invertCheckBox.setToolTip("Invert selection")
        self.contentLayout.addRow("Invert selection", self.invertCheckBox)
        
        # reset
        self.setToLatticeButton = QtWidgets.QPushButton('Set to lattice')
        self.setToLatticeButton.setAutoDefault(0)
        self.setToLatticeButton.setToolTip('Set crop to lattice dimensions')
        self.setToLatticeButton.clicked.connect(self.setCropToLattice)
        self.contentLayout.addRow(self.setToLatticeButton)
    
    def invertChanged(self, state):
        """Invert setting changed."""
        invert = False if state == QtCore.Qt.Unchecked else True
        self._settings.updateSetting("invertSelection", invert)
    
    def setCropToLattice(self):
        self.xMinRangeSpinBox.setValue(0.0)
        self.xMaxRangeSpinBox.setValue(self.pipelinePage.inputState.cellDims[0])
        self.yMinRangeSpinBox.setValue(0.0)
        self.yMaxRangeSpinBox.setValue(self.pipelinePage.inputState.cellDims[1])
        self.zMinRangeSpinBox.setValue(0.0)
        self.zMaxRangeSpinBox.setValue(self.pipelinePage.inputState.cellDims[2])
    
    def changedXEnabled(self, state):
        """Toggle crop in x direction."""
        enabled = False if state == QtCore.Qt.Unchecked else True
        self._settings.updateSetting("xEnabled", enabled)
    
    def changedYEnabled(self, state):
        """Toggle crop in y direction."""
        enabled = False if state == QtCore.Qt.Unchecked else True
        self._settings.updateSetting("yEnabled", enabled)
    
    def changedZEnabled(self, state):
        """Toggle crop in z direction."""
        enabled = False if state == QtCore.Qt.Unchecked else True
        self._settings.updateSetting("zEnabled", enabled)
    
    def refresh(self):
        """Refresh GUI."""
        self.xMinRangeSpinBox.setValue(self._settings.getSetting("xmin"))
        self.xMaxRangeSpinBox.setValue(self._settings.getSetting("xmax"))
        self.yMinRangeSpinBox.setValue(self._settings.getSetting("ymin"))
        self.yMaxRangeSpinBox.setValue(self._settings.getSetting("ymax"))
        self.zMinRangeSpinBox.setValue(self._settings.getSetting("zmin"))
        self.zMaxRangeSpinBox.setValue(self._settings.getSetting("zmax"))
        self.xCropCheckBox.setChecked(self._settings.getSetting("xEnabled"))
        self.yCropCheckBox.setChecked(self._settings.getSetting("yEnabled"))
        self.zCropCheckBox.setChecked(self._settings.getSetting("zEnabled"))
    
    def reset(self):
        """Reset crop settings."""
        self._settings.updateSetting("xmin", 0.0)
        self._settings.updateSetting("xmax", 0.0)
        self._settings.updateSetting("ymin", 0.0)
        self._settings.updateSetting("ymax", 0.0)
        self._settings.updateSetting("zmin", 0.0)
        self._settings.updateSetting("zmax", 0.0)
        self._settings.updateSetting("xEnabled", False)
        self._settings.updateSetting("yEnabled", False)
        self._settings.updateSetting("zEnabled", False)
        self.refresh()
