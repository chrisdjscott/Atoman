
"""
Options for filter lists.

@author: Chris Scott

"""
import sys
import functools
import logging
import math

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from . import genericForm

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class VoronoiOptionsWindow(QtGui.QDialog):
    """
    Options for Voronoi tesselation
    
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
        
        # layout
        self.contentLayout = QtGui.QVBoxLayout(self)
#        layout.setSpacing(0)
#        layout.setContentsMargins(0, 0, 0, 0)
        self.contentLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # display voronoi cells
        self.displayVoronoiCheck = QtGui.QCheckBox("Display Voronoi cells")
        self.displayVoronoiCheck.stateChanged.connect(self.displayVoronoiToggled)
        
        row = self.newRow()
        row.addWidget(self.displayVoronoiCheck)
        
        # dispersion
        label = QtGui.QLabel("Dispersion:")
        
        self.dispersionSpin = QtGui.QDoubleSpinBox()
        self.dispersionSpin.setMinimum(0.1)
        self.dispersionSpin.setMaximum(99.9)
        self.dispersionSpin.setSingleStep(0.1)
        self.dispersionSpin.setValue(self.dispersion)
        self.dispersionSpin.valueChanged.connect(self.dispersionChanged)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.dispersionSpin)
        
        # use radii
        self.useRadiiCheck = QtGui.QCheckBox("Use radii")
        self.useRadiiCheck.stateChanged.connect(self.useRadiiChanged)
        
        row = self.newRow()
        row.addWidget(self.useRadiiCheck)
        
        # opacity
        label = QtGui.QLabel("Opacity:")
        
        self.opacitySpin = QtGui.QDoubleSpinBox()
        self.opacitySpin.setMinimum(0.0)
        self.opacitySpin.setMaximum(1.0)
        self.opacitySpin.setSingleStep(0.01)
        self.opacitySpin.setValue(self.opacity)
        self.opacitySpin.valueChanged.connect(self.opacityChanged)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.opacitySpin)
        
        # save to file
        saveToFileGroup = QtGui.QGroupBox("Save to file")
        saveToFileGroup.setCheckable(True)
        saveToFileGroup.setChecked(False)
#         saveToFileGroup.setAlignment(QtCore.Qt.AlignCenter)
        saveToFileGroup.toggled.connect(self.saveToFileChanged)
        
        layout = QtGui.QVBoxLayout(saveToFileGroup)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
                
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        
        filenameEdit = QtGui.QLineEdit(self.outputFilename)
        filenameEdit.setFixedWidth(130)
        filenameEdit.textChanged.connect(self.filenameChanged)
        
        rowLayout.addWidget(filenameEdit)
        
        layout.addWidget(row)
        
        row = self.newRow()
        row.addWidget(saveToFileGroup)
        
        label = QtGui.QLabel("""<qt>See <a href="https://github.com/joe-jordan/pyvoro/blob/master/README.md">here</a> and <a href="http://math.lbl.gov/voro++/about.html">here</a> for info</qt>""")
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        row = self.newRow()
        row.addWidget(label)
    
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
    
    def saveToFileChanged(self, val):
        """
        Save to file changed
        
        """
        self.outputToFile = val
        
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
    Display options for filter list.
    
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
    Bond options for filter list.
    
    """
    def __init__(self, mainWindow, parent=None):
        super(BondsOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Filter list colouring options")
        self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        self.mainWindow = mainWindow
        
        # current species set
        self.currentSpecies = set()
        self.bondChecksList = []
        self.bondPairDrawStatus = []
        self.bondPairsList = []
        self.NBondPairs = 0
        
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
        self.drawBondsGroup.setAlignment(QtCore.Qt.AlignCenter)
        self.drawBondsGroup.toggled.connect(self.drawBondsToggled)
        
        layout.addWidget(self.drawBondsGroup)
        
        self.groupLayout = QtGui.QVBoxLayout()
#        self.groupLayout.setSpacing(0)
#        self.groupLayout.setContentsMargins(0, 0, 0, 0)
        self.groupLayout.setAlignment(QtCore.Qt.AlignTop)
        
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
    
    def addSpecie(self, sym):
        """
        Add specie to bonding options.
        
        """
        # first test if already added
        if sym in self.currentSpecies:
            return
        
        # add to set
        self.currentSpecies.add(sym)
        
        # now add line(s) to group box
        for symb in self.currentSpecies:
            # check box
            check = QtGui.QCheckBox("  %s - %s" % (sym, symb))
            check.stateChanged.connect(functools.partial(self.checkStateChanged, len(self.bondPairDrawStatus)))
            self.bondChecksList.append(check)
            
            # draw status
            self.bondPairDrawStatus.append(False)
            
            # label
#            label = QtGui.QLabel("%s - %s" % (sym, symb))
            
            # pair
            pair = (sym, symb)
            self.bondPairsList.append(pair)
            
            # row
            row = QtGui.QHBoxLayout()
            row.addWidget(check)
#            row.addWidget(label)
            
            self.groupLayout.addLayout(row)
            
            self.NBondPairs += 1
    
    def checkStateChanged(self, index, state):
        """
        Check state changed.
        
        """
        if state == 2:
            self.bondPairDrawStatus[index] = True
        
        else:
            self.bondPairDrawStatus[index] = False
    
    def refresh(self):
        """
        Refresh available bonds.
        
        Should be called whenever a new input is loaded!?
        If the species are the same don't change anything!?
        
        """
        inputState = self.parent.filterTab.inputState
        
        if inputState is None:
            return
        
        for specie in inputState.specieList:
            self.addSpecie(specie)



################################################################################

class ColouringOptionsWindow(QtGui.QDialog):
    """
    Window for displaying colouring options for filter list
    
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
        self.atomPropertyType = "Kinetic energy"
        
        # layout
        windowLayout = QtGui.QVBoxLayout(self)
        
        # combo box
        self.colouringCombo = QtGui.QComboBox()
        self.colouringCombo.addItem("Specie")
        self.colouringCombo.addItem("Height")
        self.colouringCombo.addItem("Solid colour")
        self.colouringCombo.addItem("Atom property")
#        self.colouringCombo.addItem("Scalar")
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
        atomPropertyOptions = genericForm.GenericForm(self, 0, "Atom property options")
        
        # type
        self.propertyTypeCombo = QtGui.QComboBox()
        self.propertyTypeCombo.addItems(("Kinetic energy", "Potential energy", "Charge"))
        self.propertyTypeCombo.currentIndexChanged.connect(self.propertyTypeChanged)
        row = atomPropertyOptions.newRow()
        row.addWidget(self.propertyTypeCombo)
        
        # min/max
        self.propertyMinSpin = QtGui.QDoubleSpinBox()
        self.propertyMinSpin.setSingleStep(0.1)
        self.propertyMinSpin.setMinimum(-9999.0)
        self.propertyMinSpin.setMaximum(9999.0)
        self.propertyMinSpin.setValue(0)
        
        self.propertyMaxSpin = QtGui.QDoubleSpinBox()
        self.propertyMaxSpin.setSingleStep(0.1)
        self.propertyMaxSpin.setMinimum(-9999.0)
        self.propertyMaxSpin.setMaximum(9999.0)
        self.propertyMaxSpin.setValue(1)
        
        label = QtGui.QLabel( " Min " )
        label2 = QtGui.QLabel( " Max " )
        
        row = atomPropertyOptions.newRow()
        row.addWidget(label)
        row.addWidget(self.propertyMinSpin)
        
        row = atomPropertyOptions.newRow()
        row.addWidget(label2)
        row.addWidget(self.propertyMaxSpin)
        
        # set to scalar range
        setToPropertyRangeButton = QtGui.QPushButton("Set to scalar range")
        setToPropertyRangeButton.setAutoDefault(0)
        setToPropertyRangeButton.clicked.connect(self.setToPropertyRange)
        
        row = atomPropertyOptions.newRow()
        row.addWidget(setToPropertyRangeButton)
        
        # scalar bar text
        self.scalarBarTextEdit3 = QtGui.QLineEdit("<insert title>")
        
        label = QtGui.QLabel("Scalar bar title:")
        row = atomPropertyOptions.newRow()
        row.addWidget(label)
        row = atomPropertyOptions.newRow()
        row.addWidget(self.scalarBarTextEdit3)
        
        self.stackedWidget.addWidget(atomPropertyOptions)
        
        # scalar widgets
        self.scalarWidgets = {}
        self.scalarMinSpins = {}
        self.scalarMaxSpins = {}
        self.scalarBarTexts = {}
        
        windowLayout.addWidget(self.stackedWidget)
    
    def propertyTypeChanged(self, val):
        """
        Property type changed.
        
        """
        self.atomPropertyType = str(self.propertyTypeCombo.currentText())
        
        self.parent.colouringOptionsButton.setText("Colouring options: %s" % self.atomPropertyType)
        self.scalarBarTextEdit3.setText(self.atomPropertyType)
    
    def setToPropertyRange(self):
        """
        Set min/max to scalar range.
        
        """
        lattice = self.parent.filterTab.inputState
        
        if self.atomPropertyType == "Kinetic energy":
            minVal = min(lattice.KE)
            maxVal = max(lattice.KE)
        
        elif self.atomPropertyType == "Potential energy":
            minVal = min(lattice.PE)
            maxVal = max(lattice.PE)
        
        else:
            minVal = min(lattice.charge)
            maxVal = max(lattice.charge)
        
        if minVal == maxVal:
            maxVal += 1
        
        self.propertyMinSpin.setValue(minVal)
        self.propertyMaxSpin.setValue(maxVal)
    
    def setToScalarRange(self, scalarType):
        """
        Set min/max to scalar range.
        
        """
        logger = logging.getLogger(__name__)
        logger.debug("Setting to scalar range (%s)", scalarType)
        
        scalarsDict = self.parent.filterer.scalarsDict
        
        minVal = min(scalarsDict[scalarType])
        maxVal = max(scalarsDict[scalarType])
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
        scalarBarTextEdit = QtGui.QLineEdit("%s" % name)
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
        
        # list of previous scalar types
        previousScalarTypes = []
        for i in xrange(4, self.colouringCombo.count()):
            previousScalarTypes.append(str(self.colouringCombo.itemText(i)))
        
        logger.debug("New scalars: %r", scalarsDict.keys())
        logger.debug("Old scalars: %r", previousScalarTypes)
        
        # check if need to remove any scalar types
        for i, name in enumerate(previousScalarTypes):
            if name not in scalarsDict:
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
    
    def scalarBarTextChanged(self, text):
        """
        Scalar bar text changed.
        
        """
        self.scalarBarText = str(text)
    
    def changeSolidColour(self):
        """
        Change solid colour.
        
        """
        col = QtGui.QColorDialog.getColor()
        
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
        
        if self.colourBy == "Atom property":
            colourByText = str(self.propertyTypeCombo.currentText())
            self.scalarBarTextEdit3.setText(self.atomPropertyType)
        else:
            colourByText = self.colourBy
        
        self.parent.colouringOptionsButton.setText("Colouring options: %s" % colourByText)
        
        self.stackedWidget.setCurrentIndex(index)
    
    def closeEvent(self, event):
        """
        Close event.
        
        """
        self.parent.colouringOptionsOpen = False
        self.hide()

