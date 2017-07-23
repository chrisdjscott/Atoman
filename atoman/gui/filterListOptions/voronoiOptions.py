
"""
Voronoi options
---------------

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
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

from PySide import QtGui, QtCore


################################################################################

class VoronoiOptionsWindow(QtGui.QDialog):
    """
    Options dialog for Voronoi tessellation.

    """
    def __init__(self, mainWindow, parent=None):
        super(VoronoiOptionsWindow, self).__init__(parent)

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.mainWindow = mainWindow

        self.logger = logging.getLogger(__name__)

        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        self.setWindowTitle("Voronoi options")  # filter list id should be in here
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
        helpButton.setToolTip("<p>Show help (opens in browser)</p>")
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

        self.mainWindow.showHelp(relativeUrl=self.helpPage)

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
