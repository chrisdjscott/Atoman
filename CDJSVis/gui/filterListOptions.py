
"""
Additional options for filter lists.

"""
import sys
import functools
import logging
import math

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from . import genericForm
from __builtin__ import False

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class VoronoiOptionsWindow(QtGui.QDialog):
    """
    Voronoi tessellation computations are carried out using `Voro++ 
    <http://math.lbl.gov/voro++/>`_. A Python extension 
    was written to provide direct access to Voro++ from the Python code.
    
    * Ticking "Display Voronoi cells" will render the Voronoi cells around all visible
      atoms.  
    * Ticking "Use radii" will perform a radical Voronoi tessellation (or Laguerre 
      tessellation). More information can be found on the `Voro++ website 
      <http://math.lbl.gov/voro++/about.html>`_.
    * "Face area threshold" is used when determining the number of Voronoi 
      neighbours. This is done by counting the number of faces of the Voronoi
      cell. Faces with an area less than "Face area threshold" are ignored in
      this calculation. A value of 0.1 seems to work well for most systems.
    * There is also an option to save the volumes and number of neighbours to a file
      during the computation.
    
    """
    def __init__(self, mainWindow, parent=None):
        super(VoronoiOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.logger = logging.getLogger(__name__)
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Voronoi options") # filter list id should be in here
#        self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        # options
        self.dispersion = 10.0
        self.displayVoronoi = False
        self.useRadii = False
        self.opacity = 0.8
        self.outputToFile = False
        self.outputFilename = "voronoi.csv"
        self.faceAreaThreshold = 0.1
        
        # layout
        dialogLayout = QtGui.QFormLayout(self)
        
        # use radii
        self.useRadiiCheck = QtGui.QCheckBox()
        self.useRadiiCheck.stateChanged.connect(self.useRadiiChanged)
        self.useRadiiCheck.setToolTip("Positions are weighted by their radii")
        dialogLayout.addRow("Use radii", self.useRadiiCheck)
        
        # face area threshold
        faceThreshSpin = QtGui.QDoubleSpinBox()
        faceThreshSpin.setMinimum(0.0)
        faceThreshSpin.setMaximum(1.0)
        faceThreshSpin.setSingleStep(0.1)
        faceThreshSpin.setDecimals(1)
        faceThreshSpin.setValue(self.faceAreaThreshold)
        faceThreshSpin.valueChanged.connect(self.faceAreaThresholdChanged)
        faceThreshSpin.setToolTip("When counting the number of neighbouring cells, faces with area lower than this value are ignored")
        dialogLayout.addRow("Face area threshold", faceThreshSpin)
        
        # save to file
        saveToFileCheck = QtGui.QCheckBox()
        saveToFileCheck.stateChanged.connect(self.saveToFileChanged)
        saveToFileCheck.setToolTip("Save Voronoi volumes/number of neighbours to file")
        filenameEdit = QtGui.QLineEdit(self.outputFilename)
        filenameEdit.textChanged.connect(self.filenameChanged)
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(saveToFileCheck)
        vbox.addWidget(filenameEdit)
        dialogLayout.addRow("Save to file", vbox)
        
        # break
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        dialogLayout.addRow(line)
        
        # display voronoi cells
        self.displayVoronoiCheck = QtGui.QCheckBox()
        self.displayVoronoiCheck.stateChanged.connect(self.displayVoronoiToggled)
        self.displayVoronoiCheck.setToolTip("Display the Voronoi cells of the visible atoms")
        dialogLayout.addRow("Display Voronoi cells", self.displayVoronoiCheck)
        
        # opacity
        self.opacitySpin = QtGui.QDoubleSpinBox()
        self.opacitySpin.setMinimum(0.0)
        self.opacitySpin.setMaximum(1.0)
        self.opacitySpin.setSingleStep(0.01)
        self.opacitySpin.setValue(self.opacity)
        self.opacitySpin.valueChanged.connect(self.opacityChanged)
        self.opacitySpin.setToolTip("Opacity of displayed Voronoi cells")
        dialogLayout.addRow("Opacity", self.opacitySpin)
        
        # button box
        buttonBox = QtGui.QDialogButtonBox()
        dialogLayout.addRow(buttonBox)
        
        # help button
        helpButton = buttonBox.addButton(buttonBox.Help)
        helpButton.setAutoDefault(0)
        buttonBox.helpRequested.connect(self.loadHelpPage)
        self.helpPage = "usage/analysis/filterListOptions.html#voronoi-options"
        
        # close button
        closeButton = buttonBox.addButton(buttonBox.Close)
        buttonBox.rejected.connect(self.close)
        closeButton.setDefault(1)
    
    def loadHelpPage(self):
        """
        Load the help page
        
        """
        if self.helpPage is None:
            return
        
        self.mainWindow.helpWindow.loadPage(self.helpPage)
        self.mainWindow.showHelp()
    
    def faceAreaThresholdChanged(self, val):
        """
        Face area threshold has changed.
        
        """
        self.faceAreaThreshold = val
    
    def getVoronoiDictKey(self):
        """
        Return unique key based on current (calculate) settings
        
        The settings that matter are:
            dispersion
            useRadii
        
        """
        key = "%f_%d" % (self.dispersion, int(self.useRadii))
        
        self.logger.debug("Voronoi dict key: %s", key)
        
        return key
    
    def clearVoronoiResults(self):
        """
        Clear Voronoi results from lattices
        
        """
        pass
#         for state in self.mainWindow.systemsDialog.lattice_list:
#             state.voronoi = None
    
    def saveToFileChanged(self, state):
        """
        Save to file changed
        
        """
        if state == QtCore.Qt.Unchecked:
            self.outputToFile = False
        
        else:
            self.outputToFile = True
        
        self.clearVoronoiResults()
    
    def filenameChanged(self, text):
        """
        Filename changed
        
        """
        self.outputFilename = str(text)
        
        self.clearVoronoiResults()
    
    def opacityChanged(self, val):
        """
        Opacity changed
        
        """
        self.opacity = val
    
    def useRadiiChanged(self, val):
        """
        Use radii changed
        
        """
        self.useRadii = bool(val)
        
        self.clearVoronoiResults()
    
    def dispersionChanged(self, val):
        """
        Dispersion changed
        
        """
        self.dispersion = val
        
        self.clearVoronoiResults()
    
    def displayVoronoiToggled(self, val):
        """
        Display Voronoi toggled
        
        """
        self.displayVoronoi = bool(val)
    
    def newRow(self, align=None):
        """
        New row
        
        """
        row = genericForm.FormRow(align=align)
        self.contentLayout.addWidget(row)
        
        return row

################################################################################

class DisplayOptionsWindow(QtGui.QDialog):
    """
    Display options for a filter list.
    
    * "Atom size scale factor" scales the radii of the atoms by the selected amount
    * The "Sphere resolution" settings determine how the atoms (spheres) are drawn.
      There are three defaults: "low", "medium" and "high, or you can enter the 
      settings manually.  In the formula "N" is the number of visible spheres.
    
    """
    def __init__(self, mainWindow, parent=None):
        super(DisplayOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Filter list %d display options" % self.parent.tab)
#        self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        # settings
        settings = QtCore.QSettings()
        
        # default options (read from settings if appropriate)
        self.atomScaleFactor = 1.0
        self.resA = float(settings.value("display/resA", 250.0))
        self.resB = float(settings.value("display/resB", 0.36))
        
        self.resDefaults = {
            "medium": (250, 0.36),
            "high": (330, 0.36),
            "low": (170, 0.36),
        }
        
        # layout 
        layout = QtGui.QVBoxLayout(self)
        
        # group box
        scaleFactorGroup = genericForm.GenericForm(self, None, "Atom size scale factor")
        scaleFactorGroup.show()
        
        # scale factor
        self.atomScaleFactorSpin = QtGui.QDoubleSpinBox()
        self.atomScaleFactorSpin.setMinimum(0.1)
        self.atomScaleFactorSpin.setMaximum(2.0)
        self.atomScaleFactorSpin.setSingleStep(0.1)
        self.atomScaleFactorSpin.setValue(self.atomScaleFactor)
        
        row = scaleFactorGroup.newRow()
        row.addWidget(self.atomScaleFactorSpin)
        
        self.atomScaleFactorSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.atomScaleFactorSlider.setMinimum(1)
        self.atomScaleFactorSlider.setMaximum(20)
        self.atomScaleFactorSlider.setSingleStep(1)
        self.atomScaleFactorSlider.setValue(int(self.atomScaleFactor * 10))
        
        self.atomScaleFactorSpin.valueChanged.connect(self.atomScaleSpinChanged)
        self.atomScaleFactorSlider.valueChanged.connect(self.atomScaleSliderChanged)
        
        row = scaleFactorGroup.newRow()
        row.addWidget(self.atomScaleFactorSlider)
        
        layout.addWidget(scaleFactorGroup)
        
        # group box for resolution settings
        resGroupBox = genericForm.GenericForm(self, None, "Sphere resolution")
        resGroupBox.show()
        
        label = QtGui.QLabel("res = a.N^(-b)")
        row = resGroupBox.newRow()
        row.addWidget(label)
        
        label = QtGui.QLabel("a = ")
        self.resASpin = QtGui.QDoubleSpinBox()
        self.resASpin.setMinimum(1)
        self.resASpin.setMaximum(500)
        self.resASpin.setSingleStep(1)
        self.resASpin.valueChanged.connect(self.resAChanged)
        row = resGroupBox.newRow()
        row.addWidget(label)
        row.addWidget(self.resASpin)
        
        label = QtGui.QLabel("b = ")
        self.resBSpin = QtGui.QDoubleSpinBox()
        self.resBSpin.setMinimum(0.01)
        self.resBSpin.setMaximum(1)
        self.resBSpin.setSingleStep(0.01)
        self.resBSpin.valueChanged.connect(self.resBChanged)
        row = resGroupBox.newRow()
        row.addWidget(label)
        row.addWidget(self.resBSpin)
        
        # defaults buttons
        self.defaultButtonsDict = {}
        for setting in self.resDefaults:
            settingButton = QtGui.QPushButton(setting, parent=self)
            settingButton.setToolTip("Use default: %s" % setting)
            settingButton.clicked.connect(functools.partial(self.applyDefault, setting))
            settingButton.setAutoDefault(0)
            settingButton.setCheckable(1)
            settingButton.setChecked(0)
            row = resGroupBox.newRow()
            row.addWidget(settingButton)
            self.defaultButtonsDict[setting] = settingButton
        
        # set values 
        self.resASpin.setValue(self.resA)
        self.resBSpin.setValue(self.resB)
        
        # store as default
        storeDefaultButton = QtGui.QPushButton("Store as default", parent=self)
        storeDefaultButton.setToolTip("Store settings as default values")
        storeDefaultButton.setAutoDefault(0)
        storeDefaultButton.clicked.connect(self.storeResSettings)
        row = resGroupBox.newRow()
        row.addWidget(storeDefaultButton)
        
        layout.addWidget(resGroupBox)
        
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)
    
    def storeResSettings(self):
        """
        Store current settings as default
        
        """
        settings = QtCore.QSettings()
        settings.setValue("display/resA", self.resA)
        settings.setValue("display/resB", self.resB)
    
    def applyDefault(self, setting):
        """
        Use default resA
        
        """
        resA, resB = self.resDefaults[setting]
        self.resASpin.setValue(resA)
        self.resBSpin.setValue(resB)
        
        # make sure this one is checked
        self.defaultButtonsDict[setting].setChecked(1)
    
    def resAChanged(self, val):
        """
        a changed
        
        """
        self.resA = val
        
        for setting, values in self.resDefaults.iteritems():
            aval, bval = values
            if aval == self.resA and bval == self.resB:
                self.defaultButtonsDict[setting].setChecked(1)
            else:
                self.defaultButtonsDict[setting].setChecked(0)
    
    def resBChanged(self, val):
        """
        b changed
        
        """
        self.resB = val
    
    def atomScaleSpinChanged(self, val):
        """
        Atom scale factor spin box changed.
        
        """
        self.atomScaleFactor = val
        self.atomScaleFactorSlider.setValue(int(val * 10))
    
    def atomScaleSliderChanged(self, val):
        """
        Atom scale factor slider changed.
        
        """
        self.atomScaleFactor = float(val) / 10.0
        self.atomScaleFactorSpin.setValue(self.atomScaleFactor)

################################################################################

class BondsOptionsWindow(QtGui.QDialog):
    """
    Selecting "Draw bonds" will result in bonds being drawn between visible 
    atoms from this filter list. You must also select the bonds you want to 
    draw from the list (eg. "Pu-Pu" or "Pu-Ga"). 
    
    The "Bond thickness" settings determine the size of the bonds when they are
    rendered.  "VTK" is the onscreen rendering while "POV" is used during 
    POV-Ray rendering.
    
    The "Number of sides" settings determines how many sides make up the tube
    used to render the bond.  A higher setting will look better but will be 
    much slower to render and interact with.
    
    """
    def __init__(self, mainWindow, parent=None):
        super(BondsOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Bonds options")
        self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        self.mainWindow = mainWindow
        
        # logger
        self.logger = logging.getLogger(__name__+".BondsOptionsWindow")
        
        # options
        self.drawBonds = False
        self.bondThicknessPOV = 0.2
        self.bondThicknessVTK = 0.2
        self.bondNumSides = 5
        
        # layout
        layout = QtGui.QVBoxLayout(self)
#        layout.setSpacing(0)
#        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        
        # draw bonds group box
        self.drawBondsGroup = QtGui.QGroupBox("Draw bonds")
        self.drawBondsGroup.setCheckable(True)
        self.drawBondsGroup.setChecked(False)
#         self.drawBondsGroup.setAlignment(QtCore.Qt.AlignCenter)
        self.drawBondsGroup.toggled.connect(self.drawBondsToggled)
        layout.addWidget(self.drawBondsGroup)
        
        self.groupLayout = QtGui.QVBoxLayout()
#        self.groupLayout.setSpacing(0)
#        self.groupLayout.setContentsMargins(0, 0, 0, 0)
        self.groupLayout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        
        self.bondsList = QtGui.QListWidget(self)
        self.bondsList.setFixedHeight(100)
        self.bondsList.setFixedWidth(120)
        self.groupLayout.addWidget(self.bondsList)
        
        self.drawBondsGroup.setLayout(self.groupLayout)
        
        # thickness
        bondThicknessGroup = QtGui.QGroupBox("Bond thickness")
        bondThicknessGroup.setAlignment(QtCore.Qt.AlignCenter)
        bondThicknessLayout = QtGui.QVBoxLayout()
        bondThicknessGroup.setLayout(bondThicknessLayout)
        layout.addWidget(bondThicknessGroup)
        
        # vtk
        vtkThickSpin = QtGui.QDoubleSpinBox()
        vtkThickSpin.setMinimum(0.01)
        vtkThickSpin.setMaximum(10)
        vtkThickSpin.setSingleStep(0.01)
        vtkThickSpin.setValue(self.bondThicknessVTK)
        vtkThickSpin.valueChanged.connect(self.vtkThickChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("VTK:"))
        row.addWidget(vtkThickSpin)
        bondThicknessLayout.addLayout(row)
        
        # pov
        povThickSpin = QtGui.QDoubleSpinBox()
        povThickSpin.setMinimum(0.01)
        povThickSpin.setMaximum(10)
        povThickSpin.setSingleStep(0.01)
        povThickSpin.setValue(self.bondThicknessPOV)
        povThickSpin.valueChanged.connect(self.povThickChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("POV:"))
        row.addWidget(povThickSpin)
        bondThicknessLayout.addLayout(row)
        
        # thickness
        numSidesGroup = QtGui.QGroupBox("Number of sides")
        numSidesGroup.setAlignment(QtCore.Qt.AlignCenter)
        numSidesLayout = QtGui.QVBoxLayout()
        numSidesGroup.setLayout(numSidesLayout)
        layout.addWidget(numSidesGroup)
        
        # pov
        numSidesSpin = QtGui.QSpinBox()
        numSidesSpin.setMinimum(3)
        numSidesSpin.setMaximum(999)
        numSidesSpin.setSingleStep(1)
        numSidesSpin.setValue(self.bondNumSides)
        numSidesSpin.valueChanged.connect(self.numSidesChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(numSidesSpin)
        numSidesLayout.addLayout(row)
        
        # button box
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)
        
        # always refresh
        self.refresh()
    
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
    
    def drawBondsToggled(self, state):
        """
        Draw bonds changed.
        
        """
        self.drawBonds = state
        
        if self.drawBonds:
            self.parent.bondsOptionsButton.setText("Bonds options: On")
        
        else:
            self.parent.bondsOptionsButton.setText("Bonds options: Off")
    
    def refresh(self):
        """
        Refresh available bonds.
        
        Should be called whenever a new input is loaded!?
        If the species are the same don't change anything!?
        
        """
        inputState = self.parent.filterTab.inputState
        if inputState is None:
            return
        
        self.logger.debug("Refreshing bonds options (%d - %d)", self.parent.pipelinePage.pipelineIndex, self.parent.tab)
        
        specieList = inputState.specieList
        
        # set of added pairs
        currentPairs = set()
        
        # remove pairs that don't exist
        num = self.bondsList.count()
        for i in xrange(num - 1, -1, -1):
            item = self.bondsList.item(i)
            
            # make this 'and' so that if a lattice is missing one specie we still
            # keep the pair in case it comes back later... 
            if item.syma not in specieList and item.symb not in specieList:
                self.logger.debug("  Removing bond option: %s - %s", item.syma, item.symb)
                self.bondsList.takeItem(i) # does this delete it?
            
            else:
                currentPairs.add("%s - %s" % (item.syma, item.symb))
                currentPairs.add("%s - %s" % (item.symb, item.syma))
        
        # add pairs that aren't already added
        for i in xrange(len(inputState.specieList)):
            for j in xrange(i, len(inputState.specieList)):
                p1 = "%s - %s" % (specieList[i], specieList[j])
                p2 = "%s - %s" % (specieList[j], specieList[i])
                if p1 in currentPairs:
                    self.logger.debug("  Keeping bond option: %s", p1)
                
                elif p2 in currentPairs:
                    self.logger.debug("  Keeping bond option: %s", p2)
                
                else:
                    self.logger.debug("  Adding bond option: %s", p1)
                    item = BondListItem(specieList[i], specieList[j])
                    self.bondsList.addItem(item)

################################################################################

class BondListItem(QtGui.QListWidgetItem):
    """
    Item in the bonds list widget.
    
    """
    def __init__(self, syma, symb):
        super(BondListItem, self).__init__()
        
        # add check box
        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable)
        
        # don't allow it to be selected
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsSelectable)
        
        # set unchecked initially
        self.setCheckState(QtCore.Qt.Unchecked)
        
        # store bond pair
        self.syma = syma
        self.symb = symb
        
        # set text
        self.setText("%s - %s" % (syma, symb))

################################################################################

class ColouringOptionsWindow(QtGui.QDialog):
    """
    By default the following colouring options are available:
    
      * **Specie**: colour by atom specie
      * **Height**: colour by the x, y or z coordinate
      * **Solid colour**: colour all atoms in the filter list by the selected
        colour
      * **Atom property**: colour by on of the available atom properties: 
        "Kinetic energy", "Potential energy" or "Charge"
    
    If you add filters/calculators that calculate scalar properties of the 
    system, for example "Displacement" or "Bond order", then there will also
    be an option to colour by these scalar values (you must "Apply lists" 
    after adding these calculators before they appear in the combo box).
    
    There are options to set the min/max values for colouring; and option to
    set these min/max values to the range of the chosen scalar and and option
    to set the text that appears on the scalar bar.
    
    """
    def __init__(self, parent=None):
        super(ColouringOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.setModal(0)
#        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Filter list colouring options")
        self.setWindowIcon(QtGui.QIcon(iconPath("painticon.png")))
        
        # defaults
        self.colourBy = "Specie"
        self.heightAxis = 1
        self.minVal = 0.0
        self.maxVal = 1.0
        self.solidColour = QtGui.QColor(255, 0, 0)
        self.solidColourRGB = (float(self.solidColour.red()) / 255.0, 
                               float(self.solidColour.green()) / 255.0,
                               float(self.solidColour.blue()) / 255.0)
        self.scalarBarText = "Height in Y (A)"
        
        # layout
        windowLayout = QtGui.QVBoxLayout(self)
        
        # combo box
        self.colouringCombo = QtGui.QComboBox()
        self.colouringCombo.addItem("Specie")
        self.colouringCombo.addItem("Height")
        self.colouringCombo.addItem("Solid colour")
        self.colouringCombo.addItem("Charge")
        self.colouringCombo.currentIndexChanged.connect(self.colourByChanged)
        
        windowLayout.addWidget(self.colouringCombo)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget(self)
        
        # specie widget
        self.specieOptions = genericForm.GenericForm(self, 0, "Specie colouring options")
        
        self.stackedWidget.addWidget(self.specieOptions)
        
        # height widget
        heightOptions = genericForm.GenericForm(self, 0, "Height colouring options")
        
        # axis
        axisCombo = QtGui.QComboBox()
        axisCombo.addItem("Height in X")
        axisCombo.addItem("Height in Y")
        axisCombo.addItem("Height in Z")
        axisCombo.setCurrentIndex(1)
        axisCombo.currentIndexChanged.connect(self.axisChanged)
        
        row = heightOptions.newRow()
        row.addWidget(axisCombo)
        
        # min/max
        self.minValSpinBox = QtGui.QDoubleSpinBox()
        self.minValSpinBox.setSingleStep(0.1)
        self.minValSpinBox.setMinimum(-9999.0)
        self.minValSpinBox.setMaximum(9999.0)
        self.minValSpinBox.setValue(0)
        self.minValSpinBox.valueChanged.connect(self.minValChanged)
        
        self.maxValSpinBox = QtGui.QDoubleSpinBox()
        self.maxValSpinBox.setSingleStep(0.1)
        self.maxValSpinBox.setMinimum(-9999.0)
        self.maxValSpinBox.setMaximum(9999.0)
        self.maxValSpinBox.setValue(1)
        self.maxValSpinBox.valueChanged.connect(self.maxValChanged)
        
        label = QtGui.QLabel( " Min " )
        label2 = QtGui.QLabel( " Max " )
        
        row = heightOptions.newRow()
        row.addWidget(label)
        row.addWidget(self.minValSpinBox)
        
        row = heightOptions.newRow()
        row.addWidget(label2)
        row.addWidget(self.maxValSpinBox)
        
        # set to lattice
        setHeightToLatticeButton = QtGui.QPushButton("Set to lattice")
        setHeightToLatticeButton.setAutoDefault(0)
        setHeightToLatticeButton.clicked.connect(self.setHeightToLattice)
        
        row = heightOptions.newRow()
        row.addWidget(setHeightToLatticeButton)
        
        # scalar bar text
        self.scalarBarTextEdit = QtGui.QLineEdit("Height in Y (A)")
        self.scalarBarTextEdit.textChanged.connect(self.scalarBarTextChanged)
        
        label = QtGui.QLabel("Scalar bar title:")
        row = heightOptions.newRow()
        row.addWidget(label)
        
        row = heightOptions.newRow()
        row.addWidget(self.scalarBarTextEdit)
        
        self.stackedWidget.addWidget(heightOptions)
        
        # solid colour widget
        solidColourOptions = genericForm.GenericForm(self, 0, "Solid colour options")
        
        # solid colour button
        self.colourButton = QtGui.QPushButton("")
        self.colourButton.setFixedWidth(60)
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.solidColour.name())
        self.colourButton.clicked.connect(self.changeSolidColour)
        
        row = solidColourOptions.newRow()
        row.addWidget(self.colourButton)
        
        self.stackedWidget.addWidget(solidColourOptions)
        
        # atom property widget
        chargeOptions = genericForm.GenericForm(self, 0, "Charge colouring options")
        
        # min/max
        self.chargeMinSpin = QtGui.QDoubleSpinBox()
        self.chargeMinSpin.setSingleStep(0.1)
        self.chargeMinSpin.setMinimum(-9999.0)
        self.chargeMinSpin.setMaximum(9999.0)
        self.chargeMinSpin.setValue(0)
        
        self.chargeMaxSpin = QtGui.QDoubleSpinBox()
        self.chargeMaxSpin.setSingleStep(0.1)
        self.chargeMaxSpin.setMinimum(-9999.0)
        self.chargeMaxSpin.setMaximum(9999.0)
        self.chargeMaxSpin.setValue(1)
        
        label = QtGui.QLabel( " Min " )
        label2 = QtGui.QLabel( " Max " )
        
        row = chargeOptions.newRow()
        row.addWidget(label)
        row.addWidget(self.chargeMinSpin)
        
        row = chargeOptions.newRow()
        row.addWidget(label2)
        row.addWidget(self.chargeMaxSpin)
        
        # set to scalar range
        setToChargeRangeButton = QtGui.QPushButton("Set to charge range")
        setToChargeRangeButton.setAutoDefault(0)
        setToChargeRangeButton.clicked.connect(self.setToChargeRange)
        
        row = chargeOptions.newRow()
        row.addWidget(setToChargeRangeButton)
        
        # scalar bar text
        self.scalarBarTextEdit3 = QtGui.QLineEdit("Charge")
        
        label = QtGui.QLabel("Scalar bar title:")
        row = chargeOptions.newRow()
        row.addWidget(label)
        row = chargeOptions.newRow()
        row.addWidget(self.scalarBarTextEdit3)
        
        self.stackedWidget.addWidget(chargeOptions)
        
        # scalar widgets
        self.scalarWidgets = {}
        self.scalarMinSpins = {}
        self.scalarMaxSpins = {}
        self.scalarBarTexts = {}
        
        windowLayout.addWidget(self.stackedWidget)
        
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        windowLayout.addWidget(buttonBox)
    
    def setToChargeRange(self):
        """
        Set min/max to scalar range.
        
        """
        lattice = self.parent.filterTab.inputState
        
        minVal = min(lattice.charge)
        maxVal = max(lattice.charge)
        
        if minVal == maxVal:
            maxVal += 1
        
        self.chargeMinSpin.setValue(minVal)
        self.chargeMaxSpin.setValue(maxVal)
    
    def setToScalarRange(self, scalarType):
        """
        Set min/max to scalar range.
        
        """
        logger = logging.getLogger(__name__)
        logger.debug("Setting to scalar range (%s)", scalarType)
        
        if scalarType.startswith("Lattice: "):
            key = scalarType[9:]
            if key in self.parent.filterer.latticeScalarsDict:
                scalarsDict = self.parent.filterer.latticeScalarsDict
            else:
                scalarsDict = self.parent.pipelinePage.inputState.scalarsDict
            scalars = scalarsDict[key]
            
        else:
            scalarsDict = self.parent.filterer.scalarsDict
            scalars = scalarsDict[scalarType]
        
        minVal = min(scalars)
        maxVal = max(scalars)
        if math.fabs(minVal - maxVal) < 0.01:
            maxVal += 1
        
        self.scalarMinSpins[scalarType].setValue(minVal)
        self.scalarMaxSpins[scalarType].setValue(maxVal)
    
    def removeScalarWidget(self, name):
        """
        Remove scalar widget
        
        """
        widget = self.scalarWidgets[name]
        self.stackedWidget.removeWidget(widget)
        del widget
    
    def addScalarWidget(self, name):
        """
        Add scalar widget
        
        """
        scalarOptions = genericForm.GenericForm(self, 0, "Options")
         
        # min/max
        scalarMinSpin = QtGui.QDoubleSpinBox()
        scalarMinSpin.setSingleStep(0.1)
        scalarMinSpin.setMinimum(-9999.0)
        scalarMinSpin.setMaximum(9999.0)
        scalarMinSpin.setValue(0)
        self.scalarMinSpins[name] = scalarMinSpin
         
        scalarMaxSpin = QtGui.QDoubleSpinBox()
        scalarMaxSpin.setSingleStep(0.1)
        scalarMaxSpin.setMinimum(-9999.0)
        scalarMaxSpin.setMaximum(9999.0)
        scalarMaxSpin.setValue(1)
        self.scalarMaxSpins[name] = scalarMaxSpin
         
        label = QtGui.QLabel( " Min " )
        label2 = QtGui.QLabel( " Max " )
         
        row = scalarOptions.newRow()
        row.addWidget(label)
        row.addWidget(scalarMinSpin)
         
        row = scalarOptions.newRow()
        row.addWidget(label2)
        row.addWidget(scalarMaxSpin)
         
        # set to scalar range
        setToScalarRangeButton = QtGui.QPushButton("Set to scalar range")
        setToScalarRangeButton.setAutoDefault(0)
        setToScalarRangeButton.clicked.connect(functools.partial(self.setToScalarRange, name))
         
        row = scalarOptions.newRow()
        row.addWidget(setToScalarRangeButton)
         
        # scalar bar text
        if name.startswith("Lattice: "):
            scalarBarName = name[9:]
        else:
            scalarBarName = name
        
        scalarBarTextEdit = QtGui.QLineEdit("%s" % scalarBarName)
        self.scalarBarTexts[name] = scalarBarTextEdit
         
        label = QtGui.QLabel("Scalar bar title:")
        row = scalarOptions.newRow()
        row.addWidget(label)
        row = scalarOptions.newRow()
        row.addWidget(scalarBarTextEdit)
        
        self.scalarWidgets[name] = scalarOptions
        self.stackedWidget.addWidget(scalarOptions)
    
    def refreshScalarColourOption(self):
        """
        Refresh colour by scalar options.
        
        """
        logger = logging.getLogger(__name__)
        logger.debug("Refreshing scalar colour options")
        
        # store current item
        iniText = str(self.colouringCombo.currentText())
        logger.debug("Initially selected: '%s'", iniText)
        
        # ref to scalarsDict
        scalarsDict = self.parent.filterer.scalarsDict
        
        # lattice scalars dict
        inputState = self.parent.pipelinePage.inputState
        latticeScalarsDict = inputState.scalarsDict
        latticeScalarsNames = ["Lattice: {0}".format(key) for key in latticeScalarsDict.keys()]
        
        # list of previous scalar types
        previousScalarTypes = []
        for i in xrange(4, self.colouringCombo.count()):
            previousScalarTypes.append(str(self.colouringCombo.itemText(i)))
        
        logger.debug("New scalars: %r", scalarsDict.keys())
        logger.debug("New scalars (L): %r", latticeScalarsNames)
        logger.debug("Old scalars: %r", previousScalarTypes)
        
        # check if need to remove any scalar types
        for i, name in enumerate(previousScalarTypes):
            if name not in scalarsDict and name not in latticeScalarsNames:
                logger.debug("Removing '%s'", name)
                
                # if selected set zero
                if str(self.colouringCombo.currentText()) == name:
                    self.colouringCombo.setCurrentIndex(0)
                
                # remove (inefficient...)
                for j in xrange(4, self.colouringCombo.count()):
                    if str(self.colouringCombo.itemText(j)) == name:
                        self.colouringCombo.removeItem(j)
                        self.removeScalarWidget(name)
        
        # add new
        for scalarType in scalarsDict:
            # already in?
            if scalarType in previousScalarTypes:
                logger.debug("Skipping '%s'; already exists", scalarType)
            
            else:
                logger.debug("Adding: '%s'", scalarType)
                self.colouringCombo.addItem(scalarType)
                self.addScalarWidget(scalarType)
        
        for scalarType in latticeScalarsNames:
            # already in?
            if scalarType in previousScalarTypes:
                logger.debug("Skipping '%s'; already exists", scalarType)
            
            else:
                logger.debug("Adding: '%s'", scalarType)
                self.colouringCombo.addItem(scalarType)
                self.addScalarWidget(scalarType)
    
    def scalarBarTextChanged(self, text):
        """
        Scalar bar text changed.
        
        """
        self.scalarBarText = str(text)
    
    def changeSolidColour(self):
        """
        Change solid colour.
        
        """
        col = QtGui.QColorDialog.getColor(initial=self.solidColour, title="Set solid colour")
        
        if col.isValid():
            self.solidColour = col
            self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.solidColour.name())
            
            self.solidColourRGB = (float(self.solidColour.red()) / 255.0, 
                                   float(self.solidColour.green()) / 255.0,
                                   float(self.solidColour.blue()) / 255.0)
    
    def setHeightToLattice(self):
        """
        Set height to lattice.
        
        """
        self.minValSpinBox.setValue(0.0)
        self.maxValSpinBox.setValue(self.parent.filterTab.refState.cellDims[self.heightAxis])
    
    def maxValChanged(self, val):
        """
        Max height changed.
        
        """
        self.maxVal = val
    
    def minValChanged(self, val):
        """
        Min height changed.
        
        """
        self.minVal = val
    
    def axisChanged(self, index):
        """
        Changed axis.
        
        """
        self.heightAxis = index
        
        axis = ["X", "Y", "Z"]
        self.scalarBarTextEdit.setText("Height in %s (A)" % axis[index])
    
    def colourByChanged(self, index):
        """
        Colour by changed.
        
        """
        self.colourBy = str(self.colouringCombo.currentText())
        
        if self.colourBy.startswith("Lattice: "):
            cbtext = self.colourBy[9:] + "(L)"
        else:
            cbtext = self.colourBy
        
        self.parent.colouringOptionsButton.setText("Colouring: %s" % cbtext)
        
        self.stackedWidget.setCurrentIndex(index)
    
    def closeEvent(self, event):
        """
        Close event.
        
        """
        self.parent.colouringOptionsOpen = False
        self.hide()

################################################################################

class TraceOptionsWindow(QtGui.QDialog):
    """
    This feature is currently in development...
    
    The trace options settings allow you to track the movement of the visible 
    atoms within the filter list.  This will calculate the displacement of all
    visible atoms and draw a tube/bond from their initial locations to their
    current positions. 
    
    * The "Bond thickness" option specifies the thickness of the bond during 
      onscreen rendering ("VTK") or offline/POV-Ray rendering ("POV")
    * The "Number of sides" option determines how many sides make up the tube
      used to render the bond.  A higher setting will look better but will be 
      much slower to render and interact with.
    
    """
    def __init__(self, mainWindow, parent=None):
        super(TraceOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.logger = logging.getLogger(__name__)
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Trace options") # filter list id should be in here
#        self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        # defaults
        self.bondThicknessVTK = 0.4
        self.bondThicknessPOV = 0.4
        self.bondNumSides = 5
        self.drawTraceVectors = False
        
        # layout
        windowLayout = QtGui.QVBoxLayout(self)
        
        # draw trace vector settings
        self.drawVectorsGroup = QtGui.QGroupBox("Draw trace vectors")
        self.drawVectorsGroup.setCheckable(True)
        self.drawVectorsGroup.setChecked(False)
        self.drawVectorsGroup.toggled.connect(self.drawTraceVectorsChanged)
        
        grpLayout = QtGui.QVBoxLayout(self.drawVectorsGroup)
        grpLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # thickness
        bondThicknessGroup = QtGui.QGroupBox("Bond thickness")
        bondThicknessGroup.setAlignment(QtCore.Qt.AlignCenter)
        bondThicknessLayout = QtGui.QVBoxLayout()
        bondThicknessGroup.setLayout(bondThicknessLayout)
        grpLayout.addWidget(bondThicknessGroup)
        
        # vtk
        vtkThickSpin = QtGui.QDoubleSpinBox()
        vtkThickSpin.setMinimum(0.01)
        vtkThickSpin.setMaximum(10)
        vtkThickSpin.setSingleStep(0.01)
        vtkThickSpin.setValue(self.bondThicknessVTK)
        vtkThickSpin.valueChanged.connect(self.vtkThickChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("VTK:"))
        row.addWidget(vtkThickSpin)
        bondThicknessLayout.addLayout(row)
        
        # pov
        povThickSpin = QtGui.QDoubleSpinBox()
        povThickSpin.setMinimum(0.01)
        povThickSpin.setMaximum(10)
        povThickSpin.setSingleStep(0.01)
        povThickSpin.setValue(self.bondThicknessPOV)
        povThickSpin.valueChanged.connect(self.povThickChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("POV:"))
        row.addWidget(povThickSpin)
        bondThicknessLayout.addLayout(row)
        
        # num sides group
        numSidesGroup = QtGui.QGroupBox("Number of sides")
        numSidesGroup.setAlignment(QtCore.Qt.AlignCenter)
        numSidesLayout = QtGui.QVBoxLayout()
        numSidesGroup.setLayout(numSidesLayout)
        grpLayout.addWidget(numSidesGroup)
        
        # num sides
        numSidesSpin = QtGui.QSpinBox()
        numSidesSpin.setMinimum(3)
        numSidesSpin.setMaximum(999)
        numSidesSpin.setSingleStep(1)
        numSidesSpin.setValue(self.bondNumSides)
        numSidesSpin.valueChanged.connect(self.numSidesChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(numSidesSpin)
        numSidesLayout.addLayout(row)
        
        windowLayout.addWidget(self.drawVectorsGroup)
        
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        windowLayout.addWidget(buttonBox)
    
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
    
    def drawTraceVectorsChanged(self, drawVectors):
        """
        Draw trace vectors toggled
        
        """
        self.drawTraceVectors = drawVectors

################################################################################

class VectorsOptionsWindow(QtGui.QDialog):
    """
    Vectors display options for a filter list.
    
    * Vectors that have been loaded onto the current input state are shown in
      the list. If one of the options is checked the vectors will be displayed
      by drawing arrows for each of the visible atoms. The size of the arrows
      is calculated from the magnitude of that atoms component.
    * Vectors will be scaled by "Scale vector" before being rendered.
    * "Vector resolution" sets the resolution of the arrows cone and shaft.
    
    """
    def __init__(self, mainWindow, parent=None):
        super(VectorsOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
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
        layout = QtGui.QFormLayout(self)
        self.setLayout(layout)
        
        # draw vectors list widget
        self.vectorsList = QtGui.QListWidget(self)
        self.vectorsList.setFixedHeight(100)
        self.vectorsList.setFixedWidth(180)
        self.vectorsList.itemChanged.connect(self.listItemChanged)
        layout.addRow(self.vectorsList)
        
        # normalise vectors
        normaliseVectorsCheck = QtGui.QCheckBox()
        normaliseVectorsCheck.setChecked(self.vectorNormalise)
        normaliseVectorsCheck.setToolTip("Normalise the vector before applying the scaling")
        normaliseVectorsCheck.stateChanged.connect(self.normaliseChanged)
        layout.addRow("Normalise vector", normaliseVectorsCheck)
        
        # scale vectors
        scaleVectorsCheck = QtGui.QDoubleSpinBox()
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
        resSpin = QtGui.QSpinBox()
        resSpin.setMinimum(3)
        resSpin.setMaximum(100)
        resSpin.setSingleStep(1)
        resSpin.setValue(self.vectorResolution)
        resSpin.valueChanged.connect(self.vectorResolutionChanged)
        resSpin.setToolTip("Set the resolution of the vectors")
        layout.addRow("Vector resolution", resSpin)
        
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
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
                self.parent.vectorsOptionsButton.setText("Vectors options: None")
        
        else:
            self.selectedVectorsName = changedItem.vectorsName
            self.parent.vectorsOptionsButton.setText("Vectors options: '{0}'".format(self.selectedVectorsName))
            
            # deselect others
            for i in xrange(self.vectorsList.count()):
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
        for i in xrange(num - 1, -1, -1):
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

################################################################################

class VectorsListItem(QtGui.QListWidgetItem):
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

class ActorsOptionsWindow(QtGui.QDialog):
    """
    Actors options options
    
    """
    def __init__(self, mainWindow, parent=None):
        super(ActorsOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Actors options")
#         self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        self.mainWindow = mainWindow
        
        # logger
        self.logger = logging.getLogger(__name__+".ActorsOptionsWindow")
        
        # defaults
        self.refreshing = False
        
        # layout
        layout = QtGui.QFormLayout(self)
        self.setLayout(layout)
        
        # draw vectors list widget
        self.tree = QtGui.QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.itemChanged.connect(self.itemChanged)
        self.tree.setHeaderLabel("Visibility")
        layout.addRow(self.tree)
        
        # button box
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)
    
    def itemChanged(self, item, column):
        """
        Item has changed.
        
        """
        if self.refreshing:
            return
        
        #TODO: if child is unchecked, parent should be too
        #TODO: when all children are checked, parent should be checked too
        
        if item.checkState(0) == QtCore.Qt.Unchecked:
            if item.childCount():
                # uncheck all children that are checked
                for i in xrange(item.childCount()):
                    child = item.child(i)
                    if child.checkState(0) == QtCore.Qt.Checked:
                        child.setCheckState(0, QtCore.Qt.Unchecked)
            
            else:
                # hide actor
                parentName = None
                parent = item.parent()
                if parent is not None:
                    parentName = parent.text(0)
                self.parent.filterer.hideActor(item.text(0), parentName=parentName)
                
                # also uncheck parent
                if parent is not None and parent.checkState(0) == QtCore.Qt.Checked:
                    self.refreshing = True
                    parent.setCheckState(0, QtCore.Qt.Unchecked)
                    self.refreshing = False
        
        else:
            if item.childCount():
                # check all children that aren't checked
                for i in xrange(item.childCount()):
                    child = item.child(i)
                    if child.checkState(0) == QtCore.Qt.Unchecked:
                        child.setCheckState(0, QtCore.Qt.Checked)
            
            else:
                # show actor
                parentName = None
                parent = item.parent()
                if parent is not None:
                    parentName = parent.text(0)
                self.parent.filterer.addActor(item.text(0), parentName=parentName)
                
                # if all parents children are checked, make sure parent is too
                if parent is not None and parent.checkState(0) == QtCore.Qt.Unchecked:
                    # count children
                    allChecked = True
                    for i in xrange(parent.childCount()):
                        child = parent.child(i)
                        if child.checkState(0) == QtCore.Qt.Unchecked:
                            allChecked = False
                            break
                    
                    if allChecked:
                        self.refreshing = True
                        parent.setCheckState(0, QtCore.Qt.Checked)
                        self.refreshing = False
    
    def addCheckedActors(self):
        """
        Add all actors that are checked (but not already added)
        
        """
        it = QtGui.QTreeWidgetItemIterator(self.tree)
        
        globalChanges = False
        while it.value():
            item = it.value()
            
            if item.childCount() == 0:
                if item.checkState(0) == QtCore.Qt.Checked:
                    
                    parent = item.parent()
                    parentName = None
                    if parent is not None:
                        parentName = parent.text(0)
                    
                    changes = self.parent.filterer.addActor(item.text(0), parentName=parentName, reinit=False)
                    
                    if changes:
                        globalChanges = True
            
            it += 1
        
        if globalChanges:
            self.parent.filterer.reinitialiseRendererWindows()
    
    def refresh(self, actorsDict):
        """
        Refresh actor visibility options
        
        Should be called whenever the filters are run
        
        """
        self.refreshing = True
        
        try:
            inputState = self.parent.filterTab.inputState
            if inputState is None:
                return
            
            self.logger.debug("Refreshing actor visibility options")
            
            # clear the tree
            self.tree.clear()
            
            # populate
            for key in sorted(actorsDict.keys()):
                val = actorsDict[key]
                
                if isinstance(val, dict):
                    parent = QtGui.QTreeWidgetItem(self.tree)
                    parent.setText(0, key)
                    parent.setFlags(parent.flags() | QtCore.Qt.ItemIsUserCheckable)
                    parent.setFlags(parent.flags() & ~QtCore.Qt.ItemIsSelectable)
                    parent.setCheckState(0, QtCore.Qt.Checked)
                    
                    for actorName in sorted(val.keys()):
                        item = QtGui.QTreeWidgetItem(parent)
                        item.setText(0, actorName)
                        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsSelectable)
                        item.setCheckState(0, QtCore.Qt.Checked)
                
                else:
                    actorName = key
                    
                    item = QtGui.QTreeWidgetItem(self.tree)
                    item.setText(0, actorName)
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsSelectable)
                    item.setCheckState(0, QtCore.Qt.Checked)
        
        finally:
            self.refreshing = False
