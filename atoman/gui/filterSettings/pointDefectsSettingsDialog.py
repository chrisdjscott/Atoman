
"""
Contains GUI forms for the point defects filter.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

from PyQt5 import QtGui, QtCore, QtWidgets


from . import base
from .speciesSettingsDialog import SpeciesListItem
from ...filtering.filters import pointDefectsFilter
from ...filtering import filterer
from six.moves import range


class PointDefectsSettingsDialog(base.GenericSettingsDialog):
    """
    Point defects filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(PointDefectsSettingsDialog, self).__init__(title, parent, "Point defects")
        
        self.logger = logging.getLogger(__name__)
        
        # settings
        self._settings = pointDefectsFilter.PointDefectsFilterSettings()
        
        # vacancy radius option
        tip = "<p>The vacancy radius is used to determine if an input atom is associated with a reference site</p>"
        self.addDoubleSpinBox("vacancyRadius", minVal=0.01, maxVal=10, step=0.1, label="Vacancy radius", toolTip=tip)
        
        self.addHorizontalDivider()
        
        # defect type options
        self.intTypeCheckBox = QtWidgets.QCheckBox("Interstitials")
        self.intTypeCheckBox.setChecked(1)
        self.intTypeCheckBox.stateChanged[int].connect(self.intVisChanged)
        self.intTypeCheckBox.setToolTip("Show interstitials")
        
        self.vacTypeCheckBox = QtWidgets.QCheckBox("Vacancies")
        self.vacTypeCheckBox.setChecked(1)
        self.vacTypeCheckBox.stateChanged[int].connect(self.vacVisChanged)
        self.vacTypeCheckBox.setToolTip("Show vacancies")
        
        self.antTypeCheckBox = QtWidgets.QCheckBox("Antisites")
        self.antTypeCheckBox.setChecked(1)
        self.antTypeCheckBox.stateChanged[int].connect(self.antVisChanged)
        self.antTypeCheckBox.setToolTip("Show antisites")
        
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.intTypeCheckBox)
        vbox.addWidget(self.vacTypeCheckBox)
        vbox.addWidget(self.antTypeCheckBox)
        self.contentLayout.addRow("Defect visibility", vbox)
        
        self.addHorizontalDivider()
        
        # identify split ints check box
        self.addCheckBox("identifySplitInts", toolTip="Turn split interstitial detection on",
                         label="Identify split interstitials")
        
        self.addHorizontalDivider()
        
        # use acna options
        toolTip = "<p>Use Adaptive Common Neighbour Analysis to complement the comparison "
        toolTip += "to a reference system. Interstitials with the selected structure are "
        toolTip += "not classified as defects if they have a neighbouring vacancy.</p>"
        self.addCheckBox("useAcna", toolTip=toolTip, label="<b>Use ACNA</b>", extraSlot=self.useAcnaToggled)
        
        # acna max bond distance
        toolTip = "<p>This value is used for spatially decomposing the system. It should be set large enough to "
        toolTip += "include all required atoms. If unsure just set it to something big, eg. 10.0</p>"
        self.maxBondDistanceSpin = self.addDoubleSpinBox("acnaMaxBondDistance", minVal=2, maxVal=9.99, step=0.1,
                                                         toolTip=toolTip, label="Max bond distance",
                                                         settingEnabled="useAcna")
        
        # acna ideal structure
        knownStructures = filterer.Filterer.knownStructures
        self.acnaStructureCombo = self.addComboBox("acnaStructureType", items=knownStructures, label="Structure",
                                                   toolTip="<p>Select the ideal structure for the given lattice</p>",
                                                   settingEnabled="useAcna")
        
        self.addHorizontalDivider()
        
        # find clusters settings
        self.addCheckBox("findClusters", toolTip="<p>Identify clusters of defects in the system.</p>",
                         label="<b>Identify clusters</b>", extraSlot=self.findClustersChanged)
        
        # neighbour rad spin box
        toolTip = "<p>Clusters are constructed using a recursive algorithm where two defects are said to be neighbours "
        toolTip += "if their separation is less than this value.</p>"
        self.nebRadSpinBox = self.addDoubleSpinBox("neighbourRadius", minVal=0.01, maxVal=99, step=0.1, toolTip=toolTip,
                                                   label="Neighbour radius", settingEnabled="findClusters")
        
        # minimum size spin box
        tip = "<p>Only show clusters that contain more than this number of defects.</p>"
        self.minNumSpinBox = self.addSpinBox("minClusterSize", minVal=1, maxVal=1000, label="Minimum cluster size",
                                             toolTip=tip, settingEnabled="findClusters")
        
        # maximum size spin box
        toolTip = "<p>Only show clusters that contain less than this number of defects."
        toolTip += "Set to -1 to disable this condition.</p>"
        self.maxNumSpinBox = self.addSpinBox("maxClusterSize", minVal=-11, maxVal=1000, label="Maximum cluster size",
                                             toolTip=toolTip, settingEnabled="findClusters")
        
        # calculate volumes options
        self.calcVolsCheck = self.addCheckBox("calculateVolumes", toolTip="Calculate volumes of defect clusters.",
                                              label="<b>Calculate volumes</b>", extraSlot=self.calcVolsChanged,
                                              settingEnabled="findClusters")
        
        # radio buttons
        self.convHullVolRadio = QtWidgets.QRadioButton(parent=self.calcVolsCheck)
        self.convHullVolRadio.toggled.connect(self.calcVolsMethodChanged)
        self.convHullVolRadio.setToolTip("Volume is determined from the convex hull of the defect positions.")
        self.voroVolRadio = QtWidgets.QRadioButton(parent=self.calcVolsCheck)
        self.voroVolRadio.setToolTip("<p>Volume is determined by summing the Voronoi volumes of the defects in "
                                     "the cluster. Ghost atoms are added for vacancies when computing the "
                                     "individual Voronoi volumes.</p>")
        self.voroVolRadio.setChecked(True)
        self.contentLayout.addRow("Convex hull volumes", self.convHullVolRadio)
        self.contentLayout.addRow("Sum Voronoi volumes", self.voroVolRadio)
        findClusters = self._settings.getSetting("findClusters")
        calcVols = self._settings.getSetting("calculateVolumes")
        enabled = True if findClusters and calcVols else False
        self.convHullVolRadio.setEnabled(enabled)
        self.voroVolRadio.setEnabled(enabled)
        
        self.addHorizontalDivider()
        
        # filter species group
        self.filterSpeciesCheck = QtWidgets.QCheckBox()
        self.filterSpeciesCheck.setChecked(self._settings.getSetting("filterSpecies"))
        self.filterSpeciesCheck.setToolTip("Filter visible defects by species")
        self.filterSpeciesCheck.stateChanged.connect(self.filterSpeciesToggled)
        
        self.specieList = QtWidgets.QListWidget(self)
        self.specieList.setFixedHeight(80)
        self.specieList.setFixedWidth(100)
        self.specieList.setEnabled(self._settings.getSetting("filterSpecies"))
        self.specieList.itemChanged.connect(self.speciesListChanged)
        
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.filterSpeciesCheck)
        vbox.addWidget(self.specieList)
        self.contentLayout.addRow("Filter species", vbox)
        
        # draw hulls options
        self.drawHullsCheck = self.addCheckBox("drawConvexHulls", toolTip="Draw convex hulls around defect clusters",
                                               label="<b>Draw convex hulls</b>", extraSlot=self.drawHullsChanged,
                                               displayLayout=True, settingEnabled="findClusters")
        
        # hull colour
        hullCol = self._settings.getSetting("hullCol")
        col = QtGui.QColor(hullCol[0] * 255.0, hullCol[1] * 255.0, hullCol[2] * 255.0)
        self.hullColourButton = QtWidgets.QPushButton("")
        self.hullColourButton.setFixedWidth(50)
        self.hullColourButton.setFixedHeight(30)
        self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
        self.hullColourButton.clicked.connect(self.showColourDialog)
        self.hullColourButton.setToolTip("The colour of the hull.")
        self.displaySettingsLayout.addRow("Hull colour", self.hullColourButton)
        
        # hull opacity
        self.hullOpacitySpinBox = self.addDoubleSpinBox("hullOpacity", minVal=0.01, maxVal=1, step=0.01,
                                                        displayLayout=True, toolTip="The opacity of the convex hulls",
                                                        label="Hull opacity")
        
        # hide atoms
        tip = "<p>Do not show the defects when rendering the convex hulls</p>"
        self.hideAtomsCheckBox = self.addCheckBox("hideDefects", toolTip=tip, label="Hide defects", displayLayout=True)
        
        findClusters = self._settings.getSetting("findClusters")
        drawHulls = self._settings.getSetting("drawConvexHulls")
        enabled = True if findClusters and drawHulls else False
        self.hullColourButton.setEnabled(enabled)
        self.hullOpacitySpinBox.setEnabled(enabled)
        self.hideAtomsCheckBox.setEnabled(enabled)
        
        self.addHorizontalDivider(displaySettings=True)
        
        # vac display settings
        # scale size
        toolTip = "<p>When rendering vacancies scale the atomic radius by this amount (usually < 1)</p>"
        self.addDoubleSpinBox("vacScaleSize", minVal=0.1, maxVal=2, step=0.1, label="Vacancy scale size",
                              displayLayout=True, toolTip=toolTip)
        
        # opacity
        self.addDoubleSpinBox("vacOpacity", minVal=0.01, maxVal=1, step=0.1, label="Vacancy opacity",
                              displayLayout=True, toolTip="<p>The opacity value for vacancies.</p>")
        
        # specular
        self.addDoubleSpinBox("vacSpecular", minVal=0.01, maxVal=1, step=0.01, label="Vacancy specular",
                              displayLayout=True, toolTip="<p>The specular value for vacancies.</p>")
        
        # specular power
        self.addDoubleSpinBox("vacSpecularPower", minVal=0, maxVal=100, step=0.1, label="Vacancy specular power",
                              displayLayout=True, toolTip="<p>The specular power value for vacancies.</p>")
        
        self.addHorizontalDivider(displaySettings=True)
        
        # draw displacement vector settings
        toolTip = "<p>Draw displacement vectors (movement) of defects</p>"
        self.drawVectorsCheck = self.addCheckBox("drawDisplacementVectors", toolTip=toolTip, displayLayout=True,
                                                 label="<b>Draw displacement vectors</b>",
                                                 extraSlot=self.drawVectorsChanged)
        
        # vtk thickness
        self.vtkThickSpin = self.addDoubleSpinBox("bondThicknessVTK", minVal=0.01, maxVal=10, step=0.1,
                                                  displayLayout=True, label="Bond thickness (VTK)",
                                                  toolTip="<p>Thickness of lines showing defect movement (VTK)</p>")
        
        # pov thickness
        self.povThickSpin = self.addDoubleSpinBox("bondThicknessPOV", minVal=0.01, maxVal=10, step=0.1,
                                                  displayLayout=True, label="Bond thickness (POV)",
                                                  toolTip="<p>Thickness of lines showing defect movement (POV-Ray)</p>")
        
        # num sides
        toolTip = "<p>Number of sides when rendering displacement vectors (more looks better but is slower)</p>"
        self.numSidesSpin = self.addSpinBox("bondNumSides", minVal=3, maxVal=99, step=1, displayLayout=True,
                                            label="Bond number of sides", toolTip=toolTip)
        
        self.disableDrawVectorsCheck()
        self.refresh()
    
    def useAcnaToggled(self, enabled):
        """
        Use ACNA toggled
        
        """
        self.maxBondDistanceSpin.setEnabled(enabled)
        self.acnaStructureCombo.setEnabled(enabled)
    
    def enableDrawVectorsCheck(self):
        """
        Enable
        
        """
        self.drawVectorsCheck.setEnabled(True)
        if self._settings.getSetting("drawDisplacementVectors"):
            self.vtkThickSpin.setEnabled(True)
            self.povThickSpin.setEnabled(True)
            self.numSidesSpin.setEnabled(True)
    
    def disableDrawVectorsCheck(self):
        """
        Disable
        
        """
        self.drawVectorsCheck.setEnabled(False)
        self.vtkThickSpin.setEnabled(False)
        self.povThickSpin.setEnabled(False)
        self.numSidesSpin.setEnabled(False)
    
    def drawVectorsChanged(self, enabled):
        """
        Draw displacement vectors toggled
        
        """
        self.logger.debug("Draw defect displacement vectors: %r", enabled)
        self.vtkThickSpin.setEnabled(enabled)
        self.povThickSpin.setEnabled(enabled)
        self.numSidesSpin.setEnabled(enabled)
        
    def toggleCalcVolsCheck(self, enabled):
        """
        Enable calc vols check box
        
        """
        self.calcVolsCheck.setEnabled(enabled)
        
        enabled = enabled and self._settings.getSetting("calculateVolumes")
        self.convHullVolRadio.setEnabled(enabled)
        self.voroVolRadio.setEnabled(enabled)
    
    def findClustersChanged(self, enabled):
        """
        Change find volumes setting.
        
        """
        # disable associated settings
        self.nebRadSpinBox.setEnabled(enabled)
        self.minNumSpinBox.setEnabled(enabled)
        self.maxNumSpinBox.setEnabled(enabled)
        self.toggleCalcVolsCheck(enabled)
        self.toggleDrawHullsCheck(enabled)
    
    def intVisChanged(self, val):
        """
        Change visibility of interstitials
        
        """
        if self.intTypeCheckBox.isChecked():
            self._settings.updateSetting("showInterstitials", True)
        else:
            self._settings.updateSetting("showInterstitials", False)
    
    def vacVisChanged(self, val):
        """
        Change visibility of vacancies
        
        """
        if self.vacTypeCheckBox.isChecked():
            self._settings.updateSetting("showVacancies", True)
        else:
            self._settings.updateSetting("showVacancies", False)
    
    def antVisChanged(self, val):
        """
        Change visibility of antisites
        
        """
        if self.antTypeCheckBox.isChecked():
            self._settings.updateSetting("showAntisites", True)
        else:
            self._settings.updateSetting("showAntisites", False)
    
    def filterSpeciesToggled(self, state):
        """
        Filter species toggled
        
        """
        filterSpecies = False if state == QtCore.Qt.Unchecked else True
        self._settings.updateSetting("filterSpecies", filterSpecies)
        self.specieList.setEnabled(filterSpecies)
    
    def speciesListChanged(self, *args):
        """Species selection has changed."""
        visibleSpeciesList = []
        if self._settings.getSetting("filterSpecies"):
            for i in range(self.specieList.count()):
                item = self.specieList.item(i)
                if item.checkState() == QtCore.Qt.Checked:
                    visibleSpeciesList.append(item.symbol)
        
        else:
            for i in range(self.specieList.count()):
                item = self.specieList.item(i)
                visibleSpeciesList.append(item.symbol)
        
        self.logger.debug("Species selection has changed: %r", visibleSpeciesList)
        
        self._settings.updateSetting("visibleSpeciesList", visibleSpeciesList)
    
    def refresh(self):
        """
        Refresh the specie list
        
        """
        self.logger.debug("Refreshing point defect filter options")
        
        refState = self.pipelinePage.refState
        inputState = self.pipelinePage.inputState
        refSpecieList = refState.specieList
        inputSpecieList = inputState.specieList
        
        # set of added species
        currentSpecies = set()
        
        # remove species that don't exist
        num = self.specieList.count()
        for i in range(num - 1, -1, -1):
            item = self.specieList.item(i)
            
            # remove if doesn't exist both ref and input
            if item.symbol not in inputSpecieList and item.symbol not in refSpecieList:
                self.logger.debug("  Removing specie option: %s", item.symbol)
                self.specieList.takeItem(i)  # does this delete it?
            
            else:
                currentSpecies.add(item.symbol)
        
        # unique species from ref/input
        combinedSpecieList = list(inputSpecieList) + list(refSpecieList)
        uniqueCurrentSpecies = set(combinedSpecieList)
        
        # add species that aren't already added
        for sym in uniqueCurrentSpecies:
            if sym in currentSpecies:
                self.logger.debug("  Keeping specie option: %s", sym)
            
            else:
                self.logger.debug("  Adding specie option: %s", sym)
                item = SpeciesListItem(sym)
                self.specieList.addItem(item)
        
        # update the visible species list
        self.speciesListChanged()
        
        # enable/disable draw vectors
        if inputState.NAtoms == refState.NAtoms:
            self.enableDrawVectorsCheck()
        else:
            self.disableDrawVectorsCheck()
    
    def toggleDrawHullsCheck(self, enabled):
        """
        Toggle the check box and associated widgets
        
        """
        self.drawHullsCheck.setEnabled(enabled)
        
        enabled = self._settings.getSetting("drawConvexHulls")
        self.hullColourButton.setEnabled(enabled)
        self.hullOpacitySpinBox.setEnabled(enabled)
        self.hideAtomsCheckBox.setEnabled(enabled)
    
    def drawHullsChanged(self, enabled):
        """
        Change draw hulls setting.
        
        """
        self.logger.debug("Drawing hulls changed (%s)", enabled)
        self.hullColourButton.setEnabled(enabled)
        self.hullOpacitySpinBox.setEnabled(enabled)
        self.hideAtomsCheckBox.setEnabled(enabled)
    
    def showColourDialog(self):
        """
        Show hull colour dialog.
        
        """
        col = QtWidgets.QColorDialog.getColor()
        
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
