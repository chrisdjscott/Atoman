
"""
Contains GUI forms for the clusters filter.

"""
from PySide import QtGui, QtCore

from . import base


################################################################################
class ClusterSettingsDialog(base.GenericSettingsDialog):
    """
    Cluster filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(ClusterSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Clusters"
        
        self.minClusterSize = 8
        self.maxClusterSize = -1
        self.drawConvexHulls = 0
        self.neighbourRadius = 5.0
        self.calculateVolumes = False
        self.calculateVolumesVoro = True
        self.calculateVolumesHull = False
        self.hullCol = [0]*3
        self.hullCol[2] = 1
        self.hullOpacity = 0.5
        self.hideAtoms = 0
        
        # neighbour rad spin box
        self.nebRadSpinBox = QtGui.QDoubleSpinBox()
        self.nebRadSpinBox.setSingleStep(0.01)
        self.nebRadSpinBox.setMinimum(0.01)
        self.nebRadSpinBox.setMaximum(100.0)
        self.nebRadSpinBox.setValue(self.neighbourRadius)
        self.nebRadSpinBox.valueChanged.connect(self.nebRadChanged)
        self.nebRadSpinBox.setToolTip("Clusters are constructed using a recursive algorithm where "
                                      "two atoms are said to be neighbours if their separation "
                                      "is less than this value.")
        self.contentLayout.addRow("Neighbour radius", self.nebRadSpinBox)
        
        # minimum size spin box
        self.minNumSpinBox = QtGui.QSpinBox()
        self.minNumSpinBox.setMinimum(1)
        self.minNumSpinBox.setMaximum(1000)
        self.minNumSpinBox.setValue(self.minClusterSize)
        self.minNumSpinBox.valueChanged.connect(self.minNumChanged)
        self.minNumSpinBox.setToolTip("Only show clusters that contain more than this number of atoms.")
        self.contentLayout.addRow("Minimum cluster size", self.minNumSpinBox)
        
        # maximum size spin box
        self.maxNumSpinBox = QtGui.QSpinBox()
        self.maxNumSpinBox.setMinimum(-1)
        self.maxNumSpinBox.setMaximum(999999)
        self.maxNumSpinBox.setValue(self.maxClusterSize)
        self.maxNumSpinBox.valueChanged.connect(self.maxNumChanged)
        self.maxNumSpinBox.setToolTip("Only show clusters that contain less than this number of atoms. Set to "
                                      "'-1' to disable this condition.")
        self.contentLayout.addRow("Maximum cluster size", self.maxNumSpinBox)
        
        self.addHorizontalDivider()
        
        # calculate volumes options
        self.calcVolsCheck = QtGui.QCheckBox()
        self.calcVolsCheck.setToolTip("Calculate volumes of clusters of atoms.")
        self.calcVolsCheck.stateChanged.connect(self.calcVolsChanged)
        self.calcVolsCheck.setChecked(self.calculateVolumes)
        self.contentLayout.addRow("<b>Calculate volumes</b>", self.calcVolsCheck)
        
        # radio buttons
        self.convHullVolRadio = QtGui.QRadioButton(parent=self.calcVolsCheck)
        self.convHullVolRadio.toggled.connect(self.calcVolsMethodChanged)
        self.convHullVolRadio.setToolTip("Volume is determined from the convex hull of the atom positions.")
        self.voroVolRadio = QtGui.QRadioButton(parent=self.calcVolsCheck)
        self.voroVolRadio.setToolTip("Volume is determined by summing the Voronoi volumes of the atoms in "
                                     "the cluster.")
        self.voroVolRadio.setChecked(True)
        self.contentLayout.addRow("Convex hull volumes", self.convHullVolRadio)
        self.contentLayout.addRow("Sum Voronoi volumes", self.voroVolRadio)
        
        # make sure setup properly
        self.calcVolsChanged(QtCore.Qt.Unchecked)
        
        # draw hulls options
        self.drawHullsCheck = QtGui.QCheckBox()
        self.drawHullsCheck.setChecked(False)
        self.drawHullsCheck.setToolTip("Draw convex hulls of atom clusters")
        self.drawHullsCheck.stateChanged.connect(self.drawHullsChanged)
        self.displaySettingsLayout.addRow("<b>Draw convex hulls</b>", self.drawHullsCheck)
        
        # hull colour
        col = QtGui.QColor(self.hullCol[0]*255.0, self.hullCol[1]*255.0, self.hullCol[2]*255.0)
        self.hullColourButton = QtGui.QPushButton("")
        self.hullColourButton.setFixedWidth(50)
        self.hullColourButton.setFixedHeight(30)
        self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
        self.hullColourButton.clicked.connect(self.showColourDialog)
        self.hullColourButton.setToolTip("The colour of the hull.")
        self.displaySettingsLayout.addRow("Hull colour", self.hullColourButton)
        
        # hull opacity
        self.hullOpacitySpinBox = QtGui.QDoubleSpinBox()
        self.hullOpacitySpinBox.setSingleStep(0.01)
        self.hullOpacitySpinBox.setMinimum(0.01)
        self.hullOpacitySpinBox.setMaximum(1.0)
        self.hullOpacitySpinBox.setValue(self.hullOpacity)
        self.hullOpacitySpinBox.setToolTip("The opacity of the convex hulls")
        self.hullOpacitySpinBox.valueChanged[float].connect(self.hullOpacityChanged)
        self.displaySettingsLayout.addRow("Hull opacity", self.hullOpacitySpinBox)
        
        # hide atoms
        self.hideAtomsCheckBox = QtGui.QCheckBox()
        self.hideAtomsCheckBox.stateChanged.connect(self.hideAtomsChanged)
        self.hideAtomsCheckBox.setToolTip("Don't show the atoms when rendering the convex hulls")
        self.displaySettingsLayout.addRow("Hide atoms", self.hideAtomsCheckBox)
        
        self.drawHullsChanged(QtCore.Qt.Unchecked)
    
    def hideAtomsChanged(self, val):
        """
        Hide atoms check changed.
        
        """
        if self.hideAtomsCheckBox.isChecked():
            self.hideAtoms = 1
        
        else:
            self.hideAtoms = 0
    
    def hullOpacityChanged(self, val):
        """
        Change hull opacity setting.
        
        """
        self.hullOpacity = val
    
    def showColourDialog(self):
        """
        Show hull colour dialog.
        
        """
        col = QtGui.QColorDialog.getColor()
        
        if col.isValid():
            self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
            
            self.hullCol[0] = float(col.red()) / 255.0
            self.hullCol[1] = float(col.green()) / 255.0
            self.hullCol[2] = float(col.blue()) / 255.0
    
    def calcVolsMethodChanged(self, val=None):
        """
        Calc vols method changed
        
        """
        if self.convHullVolRadio.isChecked():
            self.calculateVolumesHull = True
        else:
            self.calculateVolumesHull = False
        
        if self.voroVolRadio.isChecked():
            self.calculateVolumesVoro = True
        else:
            self.calculateVolumesVoro = False
    
    def calcVolsChanged(self, state):
        """
        Changed calc vols.
        
        """
        if state == QtCore.Qt.Unchecked:
            self.calculateVolumes = False
            
            # disable buttons
            self.convHullVolRadio.setEnabled(False)
            self.voroVolRadio.setEnabled(False)
        
        else:
            self.calculateVolumes = True
            
            # enable buttons
            self.convHullVolRadio.setEnabled(True)
            self.voroVolRadio.setEnabled(True)
        
        self.calcVolsMethodChanged()
    
    def minNumChanged(self, val):
        """
        Change min cluster size.
        
        """
        self.minClusterSize = val
    
    def maxNumChanged(self, val):
        """
        Change max cluster size.
        
        """
        self.maxClusterSize = val
    
    def nebRadChanged(self, val):
        """
        Change neighbour radius.
        
        """
        self.neighbourRadius = val
    
    def drawHullsChanged(self, state):
        """
        Change draw hulls setting.
        
        """
        if state == QtCore.Qt.Unchecked:
            self.drawConvexHulls = 0
            
            self.hullColourButton.setEnabled(False)
            self.hullOpacitySpinBox.setEnabled(False)
            self.hideAtomsCheckBox.setEnabled(False)
        
        else:
            self.drawConvexHulls = 1
            
            self.hullColourButton.setEnabled(True)
            self.hullOpacitySpinBox.setEnabled(True)
            self.hideAtomsCheckBox.setEnabled(True)
