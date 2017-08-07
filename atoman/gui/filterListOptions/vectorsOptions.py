
"""
Vectors options
---------------

Vectors display options for a filter list.

* Vectors that have been loaded onto the current input state are shown in
  the list. If one of the options is checked the vectors will be displayed
  by drawing arrows for each of the visible atoms. The size of the arrows
  is calculated from the magnitude of that atoms component.
* Vectors will be scaled by "Scale vector" before being rendered.
* "Vector resolution" sets the resolution of the arrows cone and shaft.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

from PySide2 import QtCore, QtWidgets

from six.moves import range


################################################################################

class VectorsListItem(QtWidgets.QListWidgetItem):
    """
    Item in the vectors list widget.
    
    """
    def __init__(self, name):
        super(VectorsListItem, self).__init__()
        
        # add check box
        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable)
        
        # don't allow it to be selected
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsSelectable)
        
        # set unchecked initially
        self.setCheckState(QtCore.Qt.Unchecked)
        
        # store vectors name
        self.vectorsName = name
        
        # set text
        self.setText(self.vectorsName)

################################################################################

class VectorsOptionsWindow(QtWidgets.QDialog):
    """
    Vectors display options dialog.
    
    """
    modified = QtCore.Signal(str)
    
    def __init__(self, mainWindow, parent=None):
        super(VectorsOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        
        self.setWindowTitle("Display vectors options")
#         self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        self.mainWindow = mainWindow
        
        # logger
        self.logger = logging.getLogger(__name__+".VectorsOptionsWindow")
        
        # options
        self.selectedVectorsName = None
        self.vectorRadiusPOV = 0.03
        self.vectorRadiusVTK = 0.03
        self.vectorResolution = 6
        self.vectorScaleFactor = 1.0
        self.vectorNormalise = False
        
        # layout
        layout = QtWidgets.QFormLayout(self)
        self.setLayout(layout)
        
        # draw vectors list widget
        self.vectorsList = QtWidgets.QListWidget(self)
        self.vectorsList.setFixedHeight(100)
        self.vectorsList.setFixedWidth(180)
        self.vectorsList.itemChanged.connect(self.listItemChanged)
        layout.addRow(self.vectorsList)
        
        # normalise vectors
        normaliseVectorsCheck = QtWidgets.QCheckBox()
        normaliseVectorsCheck.setChecked(self.vectorNormalise)
        normaliseVectorsCheck.setToolTip("Normalise the vector before applying the scaling")
        normaliseVectorsCheck.stateChanged.connect(self.normaliseChanged)
        layout.addRow("Normalise vector", normaliseVectorsCheck)
        
        # scale vectors
        scaleVectorsCheck = QtWidgets.QDoubleSpinBox()
        scaleVectorsCheck.setMinimum(0.1)
        scaleVectorsCheck.setMaximum(100)
        scaleVectorsCheck.setSingleStep(0.1)
        scaleVectorsCheck.setValue(self.vectorScaleFactor)
        scaleVectorsCheck.valueChanged.connect(self.vectorScaleFactorChanged)
        scaleVectorsCheck.setToolTip("Scale the vector by this amount")
        layout.addRow("Scale vector", scaleVectorsCheck)
        
        # vtk radius
#         vtkRadiusSpin = QtGui.QDoubleSpinBox()
#         vtkRadiusSpin.setMinimum(0.01)
#         vtkRadiusSpin.setMaximum(2)
#         vtkRadiusSpin.setSingleStep(0.1)
#         vtkRadiusSpin.setValue(self.vectorRadiusVTK)
#         vtkRadiusSpin.valueChanged.connect(self.vtkRadiusChanged)
#         vtkRadiusSpin.setToolTip("Set the radius of the vectors (in the VTK window)")
#         layout.addRow("Vector radius (VTK)", vtkRadiusSpin)
#         
#         # pov
#         povRadiusSpin = QtGui.QDoubleSpinBox()
#         povRadiusSpin.setMinimum(0.01)
#         povRadiusSpin.setMaximum(2)
#         povRadiusSpin.setSingleStep(0.1)
#         povRadiusSpin.setValue(self.vectorRadiusPOV)
#         povRadiusSpin.valueChanged.connect(self.povRadiusChanged)
#         povRadiusSpin.setToolTip("Set the radius of the vectors (when using POV-Ray)")
#         layout.addRow("Vector radius (POV)", povRadiusSpin)
        
        # resolution
        resSpin = QtWidgets.QSpinBox()
        resSpin.setMinimum(3)
        resSpin.setMaximum(100)
        resSpin.setSingleStep(1)
        resSpin.setValue(self.vectorResolution)
        resSpin.valueChanged.connect(self.vectorResolutionChanged)
        resSpin.setToolTip("Set the resolution of the vectors")
        layout.addRow("Vector resolution", resSpin)
        
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)
        
        # always refresh
        self.refresh()
    
    def normaliseChanged(self, state):
        """
        Normalise check changed
        
        """
        if state == QtCore.Qt.Unchecked:
            self.vectorNormalise = False
        
        else:
            self.vectorNormalise = True
    
    def vectorScaleFactorChanged(self, val):
        """
        Vector scale factor has changed
        
        """
        self.vectorScaleFactor = val
    
    def vectorResolutionChanged(self, val):
        """
        Vector resolution changed
        
        """
        self.vectorResolution = val
    
#     def vtkRadiusChanged(self, val):
#         """
#         VTK radius changed.
#         
#         """
#         self.vectorRadiusVTK = val
#     
#     def povRadiusChanged(self, val):
#         """
#         POV radius changed.
#         
#         """
#         self.vectorRadiusPOV = val
    
    def listItemChanged(self, changedItem):
        """
        Item has changed.
        
        """
        index = self.vectorsList.indexFromItem(changedItem).row()

        if changedItem.checkState() == QtCore.Qt.Unchecked:
            if changedItem.vectorsName == self.selectedVectorsName:
                self.logger.debug("Deselecting vectors: '%s'", self.selectedVectorsName)
                self.selectedVectorsName = None
                self.modified.emit("Vectors options: None")
        
        else:
            self.selectedVectorsName = changedItem.vectorsName
            self.modified.emit("Vectors options: '{0}'".format(self.selectedVectorsName))
            
            # deselect others
            for i in range(self.vectorsList.count()):
                item = self.vectorsList.item(i)
                
                if i == index:
                    continue
                
                if item.checkState() == QtCore.Qt.Checked:
                    item.setCheckState(QtCore.Qt.Unchecked)
            
            self.logger.debug("Selected vectors: '%s'", self.selectedVectorsName)
    
    def refresh(self):
        """
        Refresh available vectors.
        
        Should be called whenever a new input or vector data is loaded.
        
        """
        inputState = self.parent.filterTab.inputState
        if inputState is None:
            return
        
        self.logger.debug("Refreshing vectors options (%d - %d)", self.parent.pipelinePage.pipelineIndex, self.parent.tab)
        
        # set of added pairs
        currentVectors = set()
        
        # remove vectors that no longer exist
        num = self.vectorsList.count()
        for i in range(num - 1, -1, -1):
            item = self.vectorsList.item(i)
            
            # make this 'and' so that if a lattice is missing one specie we still
            # keep the pair in case it comes back later... 
            if item.vectorsName not in inputState.vectorsDict:
                self.logger.debug("  Removing vectors option: '%s'", item.vectorsName)
                item = self.vectorsList.takeItem(i)
                if self.selectedVectorsName == item.vectorsName:
                    self.selectedVectorsName = None
             
            else:
                currentVectors.add(item.vectorsName)
         
        # add vectors that aren't already added
        for vectorsName in inputState.vectorsDict:
            if vectorsName in currentVectors:
                self.logger.debug("  Keeping vectors option: '%s'", vectorsName)
             
            else:
                self.logger.debug("  Adding vectors option: '%s'", vectorsName)
                item = VectorsListItem(vectorsName)
                self.vectorsList.addItem(item)
