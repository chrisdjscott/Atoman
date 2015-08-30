
"""
Contains GUI forms for the point defects filter.

"""
from PySide import QtGui, QtCore

from . import base
from .speciesSettingsDialog import SpeciesListItem


################################################################################

class PointDefectsSettingsDialog(base.GenericSettingsDialog):
    """
    Point defects filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(PointDefectsSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Point defects"
        
        # settings
        self.vacancyRadius = 1.3
        self.showInterstitials = 1
        self.showAntisites = 1
        self.showVacancies = 1
        self.findClusters = 0
        self.neighbourRadius = 3.5
        self.minClusterSize = 3
        self.maxClusterSize = -1
        self.hullCol = [0]*3
        self.hullCol[2] = 1
        self.hullOpacity = 0.5
        self.calculateVolumes = False
        self.calculateVolumesVoro = True
        self.calculateVolumesHull = False
        self.drawConvexHulls = 0
        self.hideDefects = 0
        self.identifySplitInts = 1
        self.vacScaleSize = 0.75
        self.vacOpacity = 0.8
        self.vacSpecular = 0.4
        self.vacSpecularPower = 10
        
        # vacancy radius option
        self.vacRadSpinBox = QtGui.QDoubleSpinBox()
        self.vacRadSpinBox.setSingleStep(0.1)
        self.vacRadSpinBox.setMinimum(0.01)
        self.vacRadSpinBox.setMaximum(10.0)
        self.vacRadSpinBox.setValue(self.vacancyRadius)
        self.vacRadSpinBox.setToolTip("The vacancy radius is used to determine if an input atom "
                                      "is associated with a reference site.")
        self.vacRadSpinBox.valueChanged[float].connect(self.vacRadChanged)
        self.contentLayout.addRow("Vacancy radius", self.vacRadSpinBox)
        
        self.addHorizontalDivider()
        
        # defect type options
        self.intTypeCheckBox = QtGui.QCheckBox("Interstitials")
        self.intTypeCheckBox.setChecked(1)
        self.intTypeCheckBox.stateChanged[int].connect(self.intVisChanged)
        self.intTypeCheckBox.setToolTip("Show interstitials")
        
        self.vacTypeCheckBox = QtGui.QCheckBox("Vacancies")
        self.vacTypeCheckBox.setChecked(1)
        self.vacTypeCheckBox.stateChanged[int].connect(self.vacVisChanged)
        self.vacTypeCheckBox.setToolTip("Show vacancies")
        
        self.antTypeCheckBox = QtGui.QCheckBox("Antisites")
        self.antTypeCheckBox.setChecked(1)
        self.antTypeCheckBox.stateChanged[int].connect(self.antVisChanged)
        self.antTypeCheckBox.setToolTip("Show antisites")
        
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.intTypeCheckBox)
        vbox.addWidget(self.vacTypeCheckBox)
        vbox.addWidget(self.antTypeCheckBox)
        self.contentLayout.addRow("Defect visibility", vbox)
        
        self.addHorizontalDivider()
        
        # identify split ints check box
        self.identifySplitsCheck = QtGui.QCheckBox()
        self.identifySplitsCheck.setChecked(1)
        self.identifySplitsCheck.stateChanged.connect(self.identifySplitsChanged)
        self.identifySplitsCheck.setToolTip("Attempt to identify split interstitials (2 interstitials and 1 vacancy). "
                                            "Note: a split interstitial is counted as 1 defect, not 3.")
        self.contentLayout.addRow("Identify split interstitials", self.identifySplitsCheck)
        
        self.addHorizontalDivider()
        
        # use acna options
        self.useAcna = False
        useAcnaCheck = QtGui.QCheckBox()
        useAcnaCheck.setChecked(self.useAcna)
        useAcnaCheck.stateChanged.connect(self.useAcnaToggled)
        useAcnaCheck.setToolTip("Use Adaptive Common Neighbour Analysis to complement the comparison "
                                "to a reference system. Interstitials with the selected structure are "
                                "not classified as defects if they have a neighbouring vacancy.")
        self.contentLayout.addRow("<b>Use ACNA</b>", useAcnaCheck)
        
        # acna max bond distance
        self.acnaMaxBondDistance = 5.0
        self.maxBondDistanceSpin = QtGui.QDoubleSpinBox()
        self.maxBondDistanceSpin.setSingleStep(0.1)
        self.maxBondDistanceSpin.setMinimum(2.0)
        self.maxBondDistanceSpin.setMaximum(9.99)
        self.maxBondDistanceSpin.setValue(self.acnaMaxBondDistance)
        self.maxBondDistanceSpin.valueChanged[float].connect(self.setAcnaMaxBondDistance)
        self.maxBondDistanceSpin.setToolTip("This value is used for spatially decomposing the system. It "
                                            "should be set large enough to include all required atoms. "
                                            "If unsure just set it to something big, eg. 10.0")
        self.contentLayout.addRow("Max bond distance", self.maxBondDistanceSpin)
        
        # acna ideal structure
        self.acnaStructureType = 1
        self.acnaStructureCombo = QtGui.QComboBox()
        filterer = self.parent.filterer
        self.acnaStructureCombo.addItems(filterer.knownStructures)
        self.acnaStructureCombo.setCurrentIndex(self.acnaStructureType)
        self.acnaStructureCombo.currentIndexChanged.connect(self.acnaStructureTypeChanged)
        self.contentLayout.addRow("Structure", self.acnaStructureCombo)
        
        # make sure set up properly
        self.useAcnaToggled(QtCore.Qt.Unchecked)
        
        self.addHorizontalDivider()
        
        # find clusters settings
        findClustersCheck = QtGui.QCheckBox()
        findClustersCheck.setToolTip("Identify clusters of defects in the system.")
        findClustersCheck.stateChanged.connect(self.findClustersChanged)
        findClustersCheck.setChecked(self.findClusters)
        self.contentLayout.addRow("<b>Identify clusters</b>", findClustersCheck)
        
        # neighbour rad spin box
        self.nebRadSpinBox = QtGui.QDoubleSpinBox()
        self.nebRadSpinBox.setSingleStep(0.1)
        self.nebRadSpinBox.setMinimum(0.01)
        self.nebRadSpinBox.setMaximum(100.0)
        self.nebRadSpinBox.setValue(self.neighbourRadius)
        self.nebRadSpinBox.valueChanged[float].connect(self.nebRadChanged)
        self.nebRadSpinBox.setToolTip("Clusters are constructed using a recursive algorithm where "
                                      "two defects are said to be neighbours if their separation "
                                      "is less than this value.")
        self.contentLayout.addRow("Neighbour radius", self.nebRadSpinBox)
        
        # minimum size spin box
        self.minNumSpinBox = QtGui.QSpinBox()
        self.minNumSpinBox.setMinimum(1)
        self.minNumSpinBox.setMaximum(1000)
        self.minNumSpinBox.setValue(self.minClusterSize)
        self.minNumSpinBox.valueChanged[int].connect(self.minNumChanged)
        self.minNumSpinBox.setToolTip("Only show clusters that contain more than this number of defects.")
        self.contentLayout.addRow("Minimum cluster size", self.minNumSpinBox)
        
        # maximum size spin box
        self.maxNumSpinBox = QtGui.QSpinBox()
        self.maxNumSpinBox.setMinimum(-1)
        self.maxNumSpinBox.setMaximum(9999)
        self.maxNumSpinBox.setValue(self.maxClusterSize)
        self.maxNumSpinBox.valueChanged[int].connect(self.maxNumChanged)
        self.maxNumSpinBox.setToolTip("Only show clusters that contain less than this number of defects. "
                                      "Set to '-1' to disable this condition.")
        self.contentLayout.addRow("Maximum cluster size", self.maxNumSpinBox)
        
        # calculate volumes options
        self.calcVolsCheck = QtGui.QCheckBox()
        self.calcVolsCheck.setToolTip("Calculate volumes of defect clusters.")
        self.calcVolsCheck.stateChanged.connect(self.calcVolsChanged)
        self.calcVolsCheck.setChecked(self.calculateVolumes)
        self.contentLayout.addRow("<b>Calculate volumes</b>", self.calcVolsCheck)
        
        # radio buttons
        self.convHullVolRadio = QtGui.QRadioButton(parent=self.calcVolsCheck)
        self.convHullVolRadio.toggled.connect(self.calcVolsMethodChanged)
        self.convHullVolRadio.setToolTip("Volume is determined from the convex hull of the defect positions.")
        self.voroVolRadio = QtGui.QRadioButton(parent=self.calcVolsCheck)
        self.voroVolRadio.setToolTip("Volume is determined by summing the Voronoi volumes of the defects in "
                                     "the cluster. Ghost atoms are added for vacancies when computing the "
                                     "individual Voronoi volumes.")
        self.voroVolRadio.setChecked(True)
        self.contentLayout.addRow("Convex hull volumes", self.convHullVolRadio)
        self.contentLayout.addRow("Sum Voronoi volumes", self.voroVolRadio)
        
        # make sure setup properly
        self.calcVolsChanged(QtCore.Qt.Unchecked)
        
        self.addHorizontalDivider()
        
        # filter species group
        self.filterSpecies = False
        self.filterSpeciesCheck = QtGui.QCheckBox()
        self.filterSpeciesCheck.setChecked(self.filterSpecies)
        self.filterSpeciesCheck.setToolTip("Filter visible defects by species")
        self.filterSpeciesCheck.stateChanged.connect(self.filterSpeciesToggled)
        
        self.specieList = QtGui.QListWidget(self)
        self.specieList.setFixedHeight(80)
        self.specieList.setFixedWidth(100)
        self.specieList.setEnabled(self.filterSpecies)
        
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.filterSpeciesCheck)
        vbox.addWidget(self.specieList)
        self.contentLayout.addRow("Filter species", vbox)
        
        # draw hulls options
        self.drawHullsCheck = QtGui.QCheckBox()
        self.drawHullsCheck.setChecked(False)
        self.drawHullsCheck.setToolTip("Draw convex hulls of defect clusters")
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
        self.hideAtomsCheckBox.stateChanged.connect(self.hideDefectsChanged)
        self.hideAtomsCheckBox.setToolTip("Don't show the defects when rendering the convex hulls")
        self.displaySettingsLayout.addRow("Hide defects", self.hideAtomsCheckBox)
        
        self.drawHullsChanged(QtCore.Qt.Unchecked)
        
        self.addHorizontalDivider(displaySettings=True)
        
        # vac display settings
        # scale size
        vacScaleSizeSpin = QtGui.QDoubleSpinBox()
        vacScaleSizeSpin.setMinimum(0.1)
        vacScaleSizeSpin.setMaximum(2.0)
        vacScaleSizeSpin.setSingleStep(0.1)
        vacScaleSizeSpin.setValue(self.vacScaleSize)
        vacScaleSizeSpin.valueChanged.connect(self.vacScaleSizeChanged)
        vacScaleSizeSpin.setToolTip("When rendering vacancies scale the atomic radius by this amount (usually < 1)")
        self.displaySettingsLayout.addRow("Vacancy scale size", vacScaleSizeSpin)
        
        # opacity
        vacOpacitySpin = QtGui.QDoubleSpinBox()
        vacOpacitySpin.setMinimum(0.01)
        vacOpacitySpin.setMaximum(1.0)
        vacOpacitySpin.setSingleStep(0.1)
        vacOpacitySpin.setValue(self.vacOpacity)
        vacOpacitySpin.setToolTip("The opacity value for vacancies.")
        vacOpacitySpin.valueChanged.connect(self.vacOpacityChanged)
        self.displaySettingsLayout.addRow("Vacancy opacity", vacOpacitySpin)
        
        # specular
        vacSpecularSpin = QtGui.QDoubleSpinBox()
        vacSpecularSpin.setMinimum(0.01)
        vacSpecularSpin.setMaximum(1.0)
        vacSpecularSpin.setSingleStep(0.01)
        vacSpecularSpin.setValue(self.vacSpecular)
        vacSpecularSpin.setToolTip("Vacancy specular value")
        vacSpecularSpin.valueChanged.connect(self.vacSpecularChanged)
        self.displaySettingsLayout.addRow("Vacancy specular", vacSpecularSpin)
        
        # specular power
        vacSpecularPowerSpin = QtGui.QDoubleSpinBox()
        vacSpecularPowerSpin.setMinimum(0)
        vacSpecularPowerSpin.setMaximum(100)
        vacSpecularPowerSpin.setSingleStep(0.1)
        vacSpecularPowerSpin.setValue(self.vacSpecularPower)
        vacSpecularPowerSpin.setToolTip("Vacancy specu;ar power value")
        vacSpecularPowerSpin.valueChanged.connect(self.vacSpecularPowerChanged)
        self.displaySettingsLayout.addRow("Vacancy specular power", vacSpecularPowerSpin)
        
        self.addHorizontalDivider(displaySettings=True)
        
        self.drawDisplacementVectors = False
        self.bondThicknessVTK = 0.4
        self.bondThicknessPOV = 0.4
        self.bondNumSides = 5
        
        # draw displacement vector settings
        self.drawVectorsCheck = QtGui.QCheckBox()
        self.drawVectorsCheck.stateChanged.connect(self.drawVectorsChanged)
        self.drawVectorsCheck.setCheckState(QtCore.Qt.Unchecked)
        self.drawVectorsCheck.setToolTip("Draw displacement vectors (movement) of defects")
        self.disableDrawVectorsCheck()
        
        self.displaySettingsLayout.addRow("<b>Draw displacement vectors</b>", self.drawVectorsCheck)
        
        # vtk thickness
        self.vtkThickSpin = QtGui.QDoubleSpinBox()
        self.vtkThickSpin.setMinimum(0.01)
        self.vtkThickSpin.setMaximum(10)
        self.vtkThickSpin.setSingleStep(0.1)
        self.vtkThickSpin.setValue(self.bondThicknessVTK)
        self.vtkThickSpin.valueChanged.connect(self.vtkThickChanged)
        self.vtkThickSpin.setToolTip("Thickness of lines showing defect movement (VTK)")
        self.displaySettingsLayout.addRow("Bond thickness (VTK)", self.vtkThickSpin)
        
        # pov thickness
        self.povThickSpin = QtGui.QDoubleSpinBox()
        self.povThickSpin.setMinimum(0.01)
        self.povThickSpin.setMaximum(10)
        self.povThickSpin.setSingleStep(0.01)
        self.povThickSpin.setValue(self.bondThicknessPOV)
        self.povThickSpin.valueChanged.connect(self.povThickChanged)
        self.povThickSpin.setToolTip("Thickness of lines showing defect movement (POV-Ray)")
        self.displaySettingsLayout.addRow("Bond thickness (POV)", self.povThickSpin)
        
        # num sides
        self.numSidesSpin = QtGui.QSpinBox()
        self.numSidesSpin.setMinimum(3)
        self.numSidesSpin.setMaximum(999)
        self.numSidesSpin.setSingleStep(1)
        self.numSidesSpin.setValue(self.bondNumSides)
        self.numSidesSpin.valueChanged.connect(self.numSidesChanged)
        self.numSidesSpin.setToolTip("Number of sides when rendering displacement vectors (more looks better but is slower)")
        self.displaySettingsLayout.addRow("Bond number of sides", self.numSidesSpin)
        
        self.drawVectorsChanged(QtCore.Qt.Unchecked)
        self.findClustersChanged(QtCore.Qt.Unchecked)
        
        self.refresh()
    
    def acnaStructureTypeChanged(self, index):
        """
        ACNA structure type has changed
        
        """
        self.acnaStructureType = index
    
    def setAcnaMaxBondDistance(self, val):
        """
        Set max bond distance for ACNA
        
        """
        self.acnaMaxBondDistance = val
    
    def useAcnaToggled(self, state):
        """
        Use ACNA toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.useAcna = False
            
            # disable associated
            self.maxBondDistanceSpin.setEnabled(False)
            self.acnaStructureCombo.setEnabled(False)
        
        else:
            self.useAcna = True
            
            # disable associated
            self.maxBondDistanceSpin.setEnabled(True)
            self.acnaStructureCombo.setEnabled(True)
    
    def numSidesChanged(self, val):
        """
        Number of sides changed.
        
        """
        self.bondNumSides = val
    
    def vtkThickChanged(self, val):
        """
        VTK thickness changed.
        
        """
        self.bondThicknessVTK = val
    
    def povThickChanged(self, val):
        """
        POV thickness changed.
        
        """
        self.bondThicknessPOV = val
    
    def enableDrawVectorsCheck(self):
        """
        Enable
        
        """
        self.drawVectorsCheck.setEnabled(True)
        if self.drawDisplacementVectors:
            self.vtkThickSpin.setEnabled(True)
            self.povThickSpin.setEnabled(True)
    
    def disableDrawVectorsCheck(self):
        """
        Disable
        
        """
        self.drawVectorsCheck.setEnabled(False)
        if self.drawDisplacementVectors:
            self.vtkThickSpin.setEnabled(False)
            self.povThickSpin.setEnabled(False)
        
    
    def drawVectorsChanged(self, state):
        """
        Draw displacement vectors toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.drawDisplacementVectors = False
            
            self.vtkThickSpin.setEnabled(False)
            self.povThickSpin.setEnabled(False)
            self.numSidesSpin.setEnabled(False)
        
        else:
            self.drawDisplacementVectors = True
            
            self.vtkThickSpin.setEnabled(True)
            self.povThickSpin.setEnabled(True)
            self.numSidesSpin.setEnabled(True)
        
        self.logger.debug("Draw defect displacement vectors: %r", self.drawDisplacementVectors)
    
    def vacSpecularPowerChanged(self, val):
        """
        Vac specular power changed.
        
        """
        self.vacSpecularPower = val
    
    def vacSpecularChanged(self, val):
        """
        Vac specular changed.
        
        """
        self.vacSpecular = val
    
    def vacOpacityChanged(self, val):
        """
        Vac opacity changed.
        
        """
        self.vacOpacity = val
    
    def vacScaleSizeChanged(self, val):
        """
        Vac scale size changed.
        
        """
        self.vacScaleSize = val
    
    def identifySplitsChanged(self, state):
        """
        Identity splits changed.
        
        """
        if self.identifySplitsCheck.isChecked():
            self.identifySplitInts = 1
        
        else:
            self.identifySplitInts = 0
    
    def hideDefectsChanged(self, val):
        """
        Hide atoms check changed.
        
        """
        if self.hideAtomsCheckBox.isChecked():
            self.hideDefects = 1
        
        else:
            self.hideDefects = 0
    
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
    
    def disableCalcVolsCheck(self):
        """
        Disable calc vols check box
        
        """
        self.calcVolsCheck.setEnabled(False)
        if self.calculateVolumes:
            self.convHullVolRadio.setEnabled(False)
            self.voroVolRadio.setEnabled(False)
    
    def enableCalcVolsCheck(self):
        """
        Enable calc vols check box
        
        """
        self.calcVolsCheck.setEnabled(True)
        if self.calculateVolumes:
            self.convHullVolRadio.setEnabled(True)
            self.voroVolRadio.setEnabled(True)
    
    def findClustersChanged(self, state):
        """
        Change find volumes setting.
        
        """
        if state == QtCore.Qt.Unchecked:
            self.findClusters = 0
            
            # disable associated settings
            self.nebRadSpinBox.setEnabled(False)
            self.minNumSpinBox.setEnabled(False)
            self.maxNumSpinBox.setEnabled(False)
            self.disableCalcVolsCheck()
            self.disableDrawHullsCheck()
        
        else:
            self.findClusters = 1
            
            # enable associated settings
            self.nebRadSpinBox.setEnabled(True)
            self.minNumSpinBox.setEnabled(True)
            self.maxNumSpinBox.setEnabled(True)
            self.enableCalcVolsCheck()
            self.enableDrawHullsCheck()
    
    def vacRadChanged(self, val):
        """
        Update vacancy radius
        
        """
        self.vacancyRadius = val
    
    def intVisChanged(self, val):
        """
        Change visibility of interstitials
        
        """
        if self.intTypeCheckBox.isChecked():
            self.showInterstitials = 1
        else:
            self.showInterstitials = 0
    
    def vacVisChanged(self, val):
        """
        Change visibility of vacancies
        
        """
        if self.vacTypeCheckBox.isChecked():
            self.showVacancies = 1
        else:
            self.showVacancies = 0
    
    def antVisChanged(self, val):
        """
        Change visibility of antisites
        
        """
        if self.antTypeCheckBox.isChecked():
            self.showAntisites = 1
        else:
            self.showAntisites = 0
    
    def filterSpeciesToggled(self, state):
        """
        Filter species toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filterSpecies = False
            self.specieList.setEnabled(False)
        
        else:
            self.filterSpecies = True
            self.specieList.setEnabled(True)
    
    def getVisibleSpecieList(self):
        """
        Return list of visible species
        
        """
        visibleSpecieList = []
        if self.filterSpecies:
            for i in xrange(self.specieList.count()):
                item = self.specieList.item(i)
                if item.checkState() == QtCore.Qt.Checked:
                    visibleSpecieList.append(item.symbol)
        
        else:
            for i in xrange(self.specieList.count()):
                item = self.specieList.item(i)
                visibleSpecieList.append(item.symbol)
        
        return visibleSpecieList
    
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
        for i in xrange(num - 1, -1, -1):
            item = self.specieList.item(i)
            
            # remove if doesn't exist both ref and input
            if item.symbol not in inputSpecieList and item.symbol not in refSpecieList:
                self.logger.debug("  Removing specie option: %s", item.symbol)
                self.specieList.takeItem(i) # does this delete it?
            
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
        
        # enable/disable draw vectors
        if inputState.NAtoms == refState.NAtoms:
            self.enableDrawVectorsCheck()
        else:
            self.disableDrawVectorsCheck()
    
    def disableDrawHullsCheck(self):
        """
        Disable the check box and associated widgets
        
        """
        self.drawHullsCheck.setEnabled(False)
        if self.drawConvexHulls:
            self.hullColourButton.setEnabled(False)
            self.hullOpacitySpinBox.setEnabled(False)
            self.hideAtomsCheckBox.setEnabled(False)
    
    def enableDrawHullsCheck(self):
        """
        Enable the check box and associated widgets
        
        """
        self.drawHullsCheck.setEnabled(True)
        if self.drawConvexHulls:
            self.hullColourButton.setEnabled(True)
            self.hullOpacitySpinBox.setEnabled(True)
            self.hideAtomsCheckBox.setEnabled(True)
    
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
