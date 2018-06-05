
"""
Colouring options
-----------------

By default the following colouring options are available:

  * **Species**: colour by atom species
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
from __future__ import absolute_import
from __future__ import unicode_literals

import functools
import logging
import math

from PySide import QtGui, QtCore

from ...visutils.utilities import iconPath
from .. import genericForm
from six.moves import range


################################################################################

class ColouringOptionsWindow(QtGui.QDialog):
    """
    Dialog for colouring options.
    
    """
    modified = QtCore.Signal(str)
    
    def __init__(self, parent=None):
        super(ColouringOptionsWindow, self).__init__(parent)
        self.setMinimumWidth(300)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.setModal(0)
#        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Filter list colouring options")
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/applications-graphics.png")))
        
        # defaults
        self.colourBy = "Species"
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
        self.colouringCombo.addItem("Species")
        self.colouringCombo.addItem("Height")
        self.colouringCombo.addItem("Solid colour")
        self.colouringCombo.addItem("Charge")
        self.colouringCombo.currentIndexChanged.connect(self.colourByChanged)
        
        windowLayout.addWidget(self.colouringCombo)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget(self)
        
        # specie widget
        self.specieOptions = genericForm.GenericForm(self, 0, "Species colouring options")
        
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
        self.minValSpinBox.setDecimals(5)
        self.minValSpinBox.setMinimum(-9999.0)
        self.minValSpinBox.setMaximum(9999.0)
        self.minValSpinBox.setValue(0)
        self.minValSpinBox.valueChanged.connect(self.minValChanged)
        
        self.maxValSpinBox = QtGui.QDoubleSpinBox()
        self.maxValSpinBox.setSingleStep(0.1)
        self.maxValSpinBox.setDecimals(5)
        self.maxValSpinBox.setMinimum(-9999.0)
        self.maxValSpinBox.setMaximum(9999.0)
        self.maxValSpinBox.setValue(1)
        self.maxValSpinBox.valueChanged.connect(self.maxValChanged)
        
        label = QtGui.QLabel(" Min ")
        label2 = QtGui.QLabel(" Max ")
        
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
        self.chargeMinSpin.setDecimals(5)
        self.chargeMinSpin.setMinimum(-9999.0)
        self.chargeMinSpin.setMaximum(9999.0)
        self.chargeMinSpin.setValue(0)
        
        self.chargeMaxSpin = QtGui.QDoubleSpinBox()
        self.chargeMaxSpin.setSingleStep(0.1)
        self.chargeMaxSpin.setDecimals(5)
        self.chargeMaxSpin.setMinimum(-9999.0)
        self.chargeMaxSpin.setMaximum(9999.0)
        self.chargeMaxSpin.setValue(1)
        
        label = QtGui.QLabel(" Min ")
        label2 = QtGui.QLabel(" Max ")
        
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
            try:
                scalars = scalarsDict[scalarType]
            except KeyError:
                logger.warning("You must run the filters before setting to scalar range")
                return
        
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
        scalarMinSpin.setDecimals(5)
        scalarMinSpin.setMinimum(-9999.0)
        scalarMinSpin.setMaximum(9999.0)
        scalarMinSpin.setValue(0)
        self.scalarMinSpins[name] = scalarMinSpin
         
        scalarMaxSpin = QtGui.QDoubleSpinBox()
        scalarMaxSpin.setSingleStep(0.1)
        scalarMaxSpin.setDecimals(5)
        scalarMaxSpin.setMinimum(-9999.0)
        scalarMaxSpin.setMaximum(9999.0)
        scalarMaxSpin.setValue(1)
        self.scalarMaxSpins[name] = scalarMaxSpin
         
        label = QtGui.QLabel(" Min ")
        label2 = QtGui.QLabel(" Max ")
         
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
        
        # scalars provided by current filters
        currentFiltersScalars = self.parent.getCurrentFilterScalars()
        
        # lattice scalars dict
        inputState = self.parent.pipelinePage.inputState
        latticeScalarsDict = inputState.scalarsDict
        latticeScalarsNames = ["Lattice: {0}".format(key) for key in list(latticeScalarsDict.keys())]
        
        # list of previous scalar types
        previousScalarTypes = []
        for i in range(4, self.colouringCombo.count()):
            previousScalarTypes.append(str(self.colouringCombo.itemText(i)))
        
        logger.debug("New scalars (Lattice): %r", latticeScalarsNames)
        logger.debug("New scalars (Filters): %r", currentFiltersScalars)
        logger.debug("Old scalars: %r", previousScalarTypes)
        
        # check if need to remove any scalar types
        for i, name in enumerate(previousScalarTypes):
            if name not in currentFiltersScalars and name not in latticeScalarsNames:
                logger.debug("Removing '%s'", name)
                
                # if selected set zero
                if str(self.colouringCombo.currentText()) == name:
                    self.colouringCombo.setCurrentIndex(0)
                
                # remove (inefficient...)
                for j in range(4, self.colouringCombo.count()):
                    if str(self.colouringCombo.itemText(j)) == name:
                        self.colouringCombo.removeItem(j)
                        self.removeScalarWidget(name)
        
        # add new
        for scalarType in currentFiltersScalars:
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
        
        self.modified.emit("Colouring: %s" % cbtext)
        
        self.stackedWidget.setCurrentIndex(index)
    
    def closeEvent(self, event):
        """
        Close event.
        
        """
        self.parent.colouringOptionsOpen = False
        self.hide()
