
"""
Contains GUI forms for the clusters filter.

"""
from PySide import QtGui

from . import base
from ...filtering.filters import clusterFilter


################################################################################
class ClusterSettingsDialog(base.GenericSettingsDialog):
    """
    Cluster filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(ClusterSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Clusters"
        
        self._settings = clusterFilter.ClusterFilterSettings()
        
        # neighbour rad spin box
        toolTip = "Clusters are constructed using a recursive algorithm where two atoms "
        toolTip += "are said to be neighbours if their separation is less than this value."
        self.addDoubleSpinBox("neighbourRadius", minVal=0.01, maxVal=99.99, step=0.1, toolTip=toolTip,
                              label="Neighbour radius")
        
        # minimum size spin box
        toolTip = "Only show clusters that contain more than this number of atoms."
        self.addSpinBox("minClusterSize", minVal=1, maxVal=999999, toolTip=toolTip, label="Minimum cluster size")
        
        # maximum size spin box
        toolTip = "Only show clusters that contain less than this number of atoms. Set to -1 to disable this condition."
        self.addSpinBox("maxClusterSize", minVal=-1, maxVal=999999, toolTip=toolTip, label="Maximum cluster size")
        
        self.addHorizontalDivider()
        
        # calculate volumes options
        calcVolsCheck = self.addCheckBox("calculateVolumes", toolTip="Calculate the volumes of the clusters of atoms",
                                         label="<b>Calculate volumes</b>", extraSlot=self.calcVolsChanged)
        
        # radio buttons
        self.convHullVolRadio = QtGui.QRadioButton(parent=calcVolsCheck)
        self.convHullVolRadio.toggled.connect(self.calcVolsMethodChanged)
        self.convHullVolRadio.setToolTip("Volume is determined from the convex hull of the atom positions.")
        self.convHullVolRadio.setEnabled(self._settings.getSetting("calculateVolumes"))
        self.voroVolRadio = QtGui.QRadioButton(parent=calcVolsCheck)
        self.voroVolRadio.setToolTip("Volume is determined by summing the Voronoi volumes of the atoms in "
                                     "the cluster.")
        self.voroVolRadio.setChecked(True)
        self.voroVolRadio.setEnabled(self._settings.getSetting("calculateVolumes"))
        self.contentLayout.addRow("Convex hull volumes", self.convHullVolRadio)
        self.contentLayout.addRow("Sum Voronoi volumes", self.voroVolRadio)
        
        # draw hulls options
        self.addCheckBox("drawConvexHulls", toolTip="Draw convex hulls around atom clusters", label="<b>Draw convex hulls</b>",
                         extraSlot=self.drawHullsChanged, displayLayout=True)
        
        # hull colour
        hullCol = self._settings.getSetting("hullCol")
        col = QtGui.QColor(hullCol[0]*255.0, hullCol[1]*255.0, hullCol[2]*255.0)
        self.hullColourButton = QtGui.QPushButton("")
        self.hullColourButton.setFixedWidth(50)
        self.hullColourButton.setFixedHeight(30)
        self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
        self.hullColourButton.clicked.connect(self.showColourDialog)
        self.hullColourButton.setToolTip("The colour of the hull.")
        self.hullColourButton.setEnabled(self._settings.getSetting("drawConvexHulls"))
        self.displaySettingsLayout.addRow("Hull colour", self.hullColourButton)
        
        # hull opacity
        self.hullOpacitySpinBox = self.addDoubleSpinBox("hullOpacity", minVal=0.01, maxVal=1.0, step=0.1,
                                                        toolTip="The opacity of the convex hull", label="Hull opacity",
                                                        settingEnabled="drawConvexHulls", displayLayout=True)
        
        # hide atoms
        self.hideAtomsCheckBox = self.addCheckBox("hideAtoms", toolTip="Do not show atoms when rendering convex hulls",
                                                  label="Hide atoms", displayLayout=True, settingEnabled="drawConvexHulls")
    
    def showColourDialog(self):
        """
        Show hull colour dialog.
        
        """
        col = QtGui.QColorDialog.getColor()
        
        if col.isValid():
            self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
            
            self._settings.updateSettingArray("hullCol", 0, float(col.red()) / 255.0)
            self._settings.updateSettingArray("hullCol", 1, float(col.green()) / 255.0)
            self._settings.updateSettingArray("hullCol", 2, float(col.blue()) / 255.0)
    
    def calcVolsMethodChanged(self, val=None):
        """
        Calc vols method changed
        
        """
        if self.convHullVolRadio.isChecked():
            self._settings.updateSetting("calculateVolumesHull", True)
        else:
            self._settings.updateSetting("calculateVolumesHull", False)
        
        if self.voroVolRadio.isChecked():
            self._settings.updateSetting("calculateVolumesVoro", True)
        else:
            self._settings.updateSetting("calculateVolumesVoro", False)
    
    def calcVolsChanged(self, enabled):
        """
        Changed calc vols.
        
        """
        # enable buttons
        self.convHullVolRadio.setEnabled(enabled)
        self.voroVolRadio.setEnabled(enabled)
        
        self.calcVolsMethodChanged()
    
    def drawHullsChanged(self, drawHulls):
        """
        Change draw hulls setting.
        
        """
        self.hullColourButton.setEnabled(drawHulls)
        self.hullOpacitySpinBox.setEnabled(drawHulls)
        self.hideAtomsCheckBox.setEnabled(drawHulls)
