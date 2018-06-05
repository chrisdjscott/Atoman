
"""
Preferences dialog.

@author: Chris Scott

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import logging
import datetime
import multiprocessing as mp
import functools

from PySide import QtGui, QtCore
import numpy as np

from ..visutils.utilities import iconPath, resourcePath
from ..visutils import utilities
from . import _preferences
from six.moves import range
# from ..md import forces

################################################################################

class GenericPreferencesSettingsForm(QtGui.QWidget):
    """
    Tab for preference dialog.

    """
    def __init__(self, parent=None):
        super(GenericPreferencesSettingsForm, self).__init__(parent)

        # tab layout
        self.layout = QtGui.QFormLayout()
#         self.layout = QtGui.QVBoxLayout()
#         self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        self.parent = parent

        self.show()

    def newRow(self):
        """
        Create new row.

        """
        row = QtGui.QWidget()

        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)

        self.layout.addWidget(row)

        return rowLayout

    def addHorizontalDivide(self):
        """Add a horizontal divider to the layout."""
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        self.layout.addWidget(line)

    def init(self):
#         self.layout.addStretch(1)
        pass

################################################################################

class LogFileSettingsForm(GenericPreferencesSettingsForm):
    """
    Settings for creating a log file (for debugging).  Options that can be set
    on this form are:

    **Create log file**
        Create a log file. This setting persists when you close and reopen the
        application, so you can get a log file with everything from startup.

    **Log dir**
        Directory to place log files into.

    **Level**
        The logging level (should be self-explanatory).

    """
    def __init__(self, parent):
        super(LogFileSettingsForm, self).__init__(parent)

        self.logger = logging.getLogger(__name__+".LogFileSettingsForm")

        # settings object
        settings = QtCore.QSettings()

        # default settings
        self.createLogFile = bool(int(settings.value("logfile/create_log_file", 0)))
        self.logDirectory = settings.value("logfile/directory", "/tmp")
        self.logLevel = int(settings.value("logfile/level", logging.DEBUG))

        # create log file
        createCheck = QtGui.QCheckBox("Create log file")
        createCheck.setChecked(self.createLogFile)
        createCheck.stateChanged.connect(self.createToggled)
        self.layout.addRow("Create log file", createCheck)

        # directory button
        self.logDirButton = QtGui.QPushButton(self.logDirectory)
        self.logDirButton.clicked.connect(self.showLogDirectoryDialog)
        self.logDirButton.setAutoDefault(False)
        self.logDirButton.setDefault(False)
        self.logDirButton.setFixedWidth(160)
        self.layout.addRow("Log directory", self.logDirButton)

        # logging levels
        self.loggingLevels = {"CRITICAL": logging.CRITICAL,
                              "ERROR": logging.ERROR,
                              "WARNING": logging.WARNING,
                              "INFO": logging.INFO,
                              "DEBUG": logging.DEBUG}

        self.loggingLevelsSorted = ["CRITICAL",
                                    "ERROR",
                                    "WARNING",
                                    "INFO",
                                    "DEBUG"]

        # log file level
        levelCombo = QtGui.QComboBox()
        levelCombo.addItems(self.loggingLevelsSorted)
        levelIndex = self.getLevelIndex(self.logLevel)
        levelCombo.setCurrentIndex(levelIndex)
        levelCombo.currentIndexChanged[str].connect(self.logLevelChanged)
        self.layout.addRow("Level", levelCombo)

        # create handler... (should be on main window!)
        if self.createLogFile:
            logfile = os.path.join(self.logDirectory, "atoman-%s.log" % datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
            self.logger.info("Logging to file: '%s'", logfile)

            # handler
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)

            # formatter
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(funcName)s():%(lineno)d: %(message)s")
            fh.setFormatter(formatter)

            # add to root logger
            logging.getLogger().addHandler(fh)

            # write version first


    def logLevelChanged(self, levelString):
        """
        Log level has changed

        """
        level = self.loggingLevels[str(levelString)]
        self.logger.debug("Log level (file) changed: %s (%d)", levelString, level)

        # change level on handler


        # settings object
        settings = QtCore.QSettings()

        # store
        settings.setValue("logfile/level", level)




    def getLevelIndex(self, level):
        """
        Get the level index in the list

        """
        levelIndex = None
        count = 0
        for key in self.loggingLevelsSorted:
            if self.loggingLevels[key] == level:
                levelIndex = count
                break

            count += 1

        return levelIndex

    def showLogDirectoryDialog(self):
        """
        Show dialog to choose log dir

        """
        new_dir = QtGui.QFileDialog.getExistingDirectory(self, "Log directory", self.logDirectory)

        if new_dir and os.path.isdir(new_dir):
            self.logDirectory = new_dir
            self.logDirButton.setText(new_dir)

            self.logger.debug("Log file directory changed: '%s'", new_dir)

            # settings object
            settings = QtCore.QSettings()

            # store
            settings.setValue("logfile/directory", self.logDirectory)

            self.logger.warning("Changes to log file directory will only take effect when the application is restarted")

    def createToggled(self, state):
        """
        Create check box toggled

        """
        if state == QtCore.Qt.Unchecked:
            self.createLogFile = False
        else:
            self.createLogFile = True

        self.logger.debug("Create log file toggled: %s", not state == QtCore.Qt.Unchecked)

        # settings object
        settings = QtCore.QSettings()

        # store
        settings.setValue("logfile/create_log_file", int(self.createLogFile))

        # start writing log file...
        self.logger.warning("Changes to 'Create log file' will only take effect when the application is restarted")

################################################################################

class RenderingSettingsForm(GenericPreferencesSettingsForm):
    """
    Rendering settings form for the preferences dialog. Options that can be set on this form
    are:

    **Max atoms auto run**
        If the number of atoms in the selected input file is less that this value then the
        filter/calculator lists run automatically after the system is loaded.

    **Number of scalar bar labels**
        The number of labels to display above the scalar bar (defaults to 5).

    **Enable custom scalar bar labels**
        Enable the custom format for the scalar bar labels.

    **Custom scalar bar label format**
        The custom format to use for the scalar bar labels. Must match the regular expression:
        "%[+- 0#]*[0-9]*([.]?[0-9]+)?[adefgADEFG]".

    """
    def __init__(self, parent):
        super(RenderingSettingsForm, self).__init__(parent)

        # settings object
        settings = QtCore.QSettings()
        self.logger = logging.getLogger(__name__ + ".RenderingSettingsForm")

        # default settings
        self.maxAtomsAutoRun = int(settings.value("rendering/maxAtomsAutoRun", 10000))
        self.numScalarBarLabels = int(settings.value("rendering/numScalarBarLabels", 5))
        self.enableFmtScalarBarLabels = bool(int(settings.value("rendering/enableFmtScalarBarLabels", 0)))
        self.fmtScalarBarLabels = settings.value("rendering/fmtScalarBarLabels", "%+#6.2e")

        # max atoms auto run
        maxAtomsSpin = QtGui.QSpinBox()
        maxAtomsSpin.setMinimum(1)
        maxAtomsSpin.setMaximum(99999)
        maxAtomsSpin.setValue(self.maxAtomsAutoRun)
        maxAtomsSpin.valueChanged.connect(self.maxAtomsChanged)
        maxAtomsSpin.setToolTip("<p>Automatically render a system after loading if fewer atoms than this value.</p>")
        self.layout.addRow("Max atoms auto run", maxAtomsSpin)

        self.addHorizontalDivide()

        # scalar bar options
        numScalarBarLabelsSpin = QtGui.QSpinBox()
        numScalarBarLabelsSpin.setMinimum(2)
        numScalarBarLabelsSpin.setMaximum(9)
        numScalarBarLabelsSpin.setValue(self.numScalarBarLabels)
        numScalarBarLabelsSpin.setToolTip("<p>Set the number of labels on the scalar bar.</p>")
        numScalarBarLabelsSpin.valueChanged.connect(self.numScalarBarLabelsChanged)
        self.layout.addRow("Number of scalar bar labels", numScalarBarLabelsSpin)

        self.fmtScalarBarLabelsEdit = QtGui.QLineEdit(self.fmtScalarBarLabels)
        self.fmtScalarBarLabelsEdit.setToolTip("<p>Custom format for scalar bar labels</p>")
        self.fmtScalarBarLabelsEdit.editingFinished.connect(self.fmtEdited)
        regexp = QtCore.QRegExp("%[+- 0#]*[0-9]*([.]?[0-9]+)?[adefgADEFG]")
        validator = QtGui.QRegExpValidator(regexp, self)
        self.fmtScalarBarLabelsEdit.setValidator(validator)

        enableFmtCheck = QtGui.QCheckBox()
        if self.enableFmtScalarBarLabels:
            enableFmtCheck.setCheckState(QtCore.Qt.Checked)
            self.fmtScalarBarLabelsEdit.setEnabled(True)
        else:
            enableFmtCheck.setCheckState(QtCore.Qt.Unchecked)
            self.fmtScalarBarLabelsEdit.setEnabled(False)
        enableFmtCheck.setToolTip("<p>Enable custom format for scalar bar labels (below)</p>")
        enableFmtCheck.stateChanged.connect(self.enableFmtCheckChanged)

        self.layout.addRow("Enable custom scalar bar labels", enableFmtCheck)
        self.layout.addRow("Custom scalar bar label format", self.fmtScalarBarLabelsEdit)

        self.init()

    def fmtEdited(self):
        """Custom format has been edited."""
        text = str(self.fmtScalarBarLabelsEdit.text())
        self.logger.debug("Custom scalar bar format changed: '%s'", text)
        self.fmtScalarBarLabels = text

    def enableFmtCheckChanged(self, state):
        """Enable custom format changed."""
        self.enableFmtScalarBarLabels = False if state == QtCore.Qt.Unchecked else True
        self.fmtScalarBarLabelsEdit.setEnabled(self.enableFmtScalarBarLabels)

        # store in settings
        settings = QtCore.QSettings()
        settings.setValue("rendering/maxAtomsAutoRun", int(self.enableFmtScalarBarLabels))

    def maxAtomsChanged(self, val):
        """
        maxAtomsAutoRun changed.

        """
        self.maxAtomsAutoRun = val

        # store in settings
        settings = QtCore.QSettings()
        settings.setValue("rendering/maxAtomsAutoRun", val)

    def numScalarBarLabelsChanged(self, val):
        """Number of scalar bar labels changed."""
        self.numScalarBarLabels = val

        # store in settings
        settings = QtCore.QSettings()
        settings.setValue("rendering/numScalarBarLabels", val)

################################################################################

class FfmpegSettingsForm(GenericPreferencesSettingsForm):
    """
    FFmpeg settings form for the preferences dialog. Options that can be set on this form
    are:

    **Path to FFmpeg**
        Specify the path to the "ffmpeg" executable.  If FFmpeg is installed in a default location,
        for example "/usr/bin", then you can just write "ffmpeg" in the box (or whatever you FFmpeg
        executable is called). Otherwise, you should specify the full path, eg
        "/opt/somewhere/bin/ffmpeg".

    **Bitrate**
        Here you can specify the bitrate to use when creating movies using FFmpeg.

    """
    def __init__(self, parent):
        super(FfmpegSettingsForm, self).__init__(parent)

        # settings object
        settings = QtCore.QSettings()

        # default settings
        self.bitrate = 10000

        self.pathToFFmpeg = str(settings.value("ffmpeg/pathToFFmpeg", "ffmpeg"))
        if not os.path.exists(self.pathToFFmpeg):
            self.pathToFFmpeg = "ffmpeg"

        # path to povray
        pathToFFmpegLineEdit = QtGui.QLineEdit(self.pathToFFmpeg)
        pathToFFmpegLineEdit.textChanged.connect(self.pathToFFmpegChanged)
        pathToFFmpegLineEdit.editingFinished.connect(self.pathToFFmpegEdited)
        self.layout.addRow("FFmpeg path", pathToFFmpegLineEdit)

        # bitrate
        bitrateSpin = QtGui.QSpinBox()
        bitrateSpin.setMinimum(1)
        bitrateSpin.setMaximum(1e8)
        bitrateSpin.setValue(self.bitrate)
        bitrateSpin.valueChanged.connect(self.bitrateChanged)
        self.layout.addRow("Bitrate (kbits/s)", bitrateSpin)

        self.init()

    def pathToFFmpegEdited(self):
        """
        Path to FFmpeg finished being edited.

        """
        exe = utilities.checkForExe(self.pathToFFmpeg)

        if exe:
            print("STORING FFMPEG PATH IN SETTINGS", exe, self.pathToFFmpeg)
            settings = QtCore.QSettings()
            settings.setValue("ffmpeg/pathToFFmpeg", exe)

    def pathToFFmpegChanged(self, text):
        """
        Path to FFmpeg changed.

        """
        self.pathToFFmpeg = str(text)

    def bitrateChanged(self, val):
        """
        Bitrate changed.

        """
        self.bitrate = val

################################################################################

class PovraySettingsForm(GenericPreferencesSettingsForm):
    """
    POV-Ray settings form for the preferences dialog. Options that can be set on this form
    are:

    **Path to POV-Ray**
        Specify the path to the "povray" executable.  If POV-Ray is installed in a default location,
        for example "/usr/bin", then you can just write "povray" in the box (or whatever you POV-Ray
        executable is called). Otherwise, you should specify the full path, eg
        "/opt/somewhere/bin/povray".

    **Overlay image**
        Here you can specify whether or not to overlay on screen information onto the generated POV-Ray
        image.  Examples of things that will be overlayed onto the POV-Ray image are on-screen text and
        the scalar bar.

    **Shadowless**
        Use the shadowless option when rendering POV-Ray images.

    **Dimensions**
        The dimensions of the POV-Ray image

    **View angle**
        The view angle to use when creating the POV-Ray image

    **Cell frame radius**
        The radius of the lattice cell frame to use in POV-Ray images

    """
    def __init__(self, parent):
        super(PovraySettingsForm, self).__init__(parent)

        # settings object
        settings = QtCore.QSettings()

        # default settings
        self.overlayImage = True
        self.shadowless = False
        self.HRes = 800
        self.VRes = 600
        self.viewAngle = 45
        self.cellFrameRadius = 0.15

        self.pathToPovray = str(settings.value("povray/pathToPovray", "povray"))
        if not os.path.exists(self.pathToPovray):
            self.pathToPovray = "povray"

        # path to povray
        pathToPovrayLineEdit = QtGui.QLineEdit(self.pathToPovray)
        pathToPovrayLineEdit.textChanged.connect(self.pathToPovrayChanged)
        pathToPovrayLineEdit.editingFinished.connect(self.pathToPovrayEdited)
        self.layout.addRow("POV-Ray path", pathToPovrayLineEdit)

        # overlay check box
        self.overlayImageCheck = QtGui.QCheckBox()
        self.overlayImageCheck.setChecked(1)
        self.overlayImageCheck.stateChanged.connect(self.overlayImageChanged)
        self.layout.addRow("Overlay image", self.overlayImageCheck)

        # shadowless check box
        self.shadowlessCheck = QtGui.QCheckBox()
        self.shadowlessCheck.stateChanged.connect(self.shadowlessChanged)
        self.layout.addRow("Shadowless", self.shadowlessCheck)

        # dimensions
        HResSpinBox = QtGui.QSpinBox()
        HResSpinBox.setMinimum(1)
        HResSpinBox.setMaximum(10000)
        HResSpinBox.setValue(self.HRes)
        HResSpinBox.valueChanged.connect(self.HResChanged)

        VResSpinBox = QtGui.QSpinBox()
        VResSpinBox.setMinimum(1)
        VResSpinBox.setMaximum(10000)
        VResSpinBox.setValue(self.VRes)
        VResSpinBox.valueChanged.connect(self.VResChanged)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(HResSpinBox)
        hbox.addWidget(QtGui.QLabel("x"))
        hbox.addWidget(VResSpinBox)
        self.layout.addRow("Dimensions", hbox)

        # view angle
        angleSpinBox = QtGui.QDoubleSpinBox()
        angleSpinBox.setSingleStep(0.1)
        angleSpinBox.setMinimum(0.1)
        angleSpinBox.setMaximum(360.0)
        angleSpinBox.setValue(self.viewAngle)
        angleSpinBox.valueChanged.connect(self.viewAngleChanged)
        self.layout.addRow("View angle (degrees)", angleSpinBox)

        # cell frame radius
        cellFrameSpinBox = QtGui.QDoubleSpinBox()
        cellFrameSpinBox.setSingleStep(0.01)
        cellFrameSpinBox.setMinimum(0.01)
        cellFrameSpinBox.setMaximum(5.0)
        cellFrameSpinBox.setValue(self.cellFrameRadius)
        cellFrameSpinBox.valueChanged.connect(self.cellFrameRadiusChanged)
        self.layout.addRow("Cell frame radius", cellFrameSpinBox)

        self.init()

    def pathToPovrayEdited(self):
        """
        Path to povray finished being edited.

        """
        exe = utilities.checkForExe(self.pathToPovray)

        if exe:
            settings = QtCore.QSettings()
            settings.setValue("povray/pathToPovray", exe)

    def pathToPovrayChanged(self, text):
        """
        Path to POV-Ray changed.

        """
        self.pathToPovray = str(text)

    def viewAngleChanged(self, val):
        """
        View angle changed.

        """
        self.viewAngle = val

    def cellFrameRadiusChanged(self, val):
        """
        Cell frame radius changed

        """
        self.cellFrameRadius = val

    def VResChanged(self, val):
        """
        Horizontal resolution changed.

        """
        self.VRes = val

    def HResChanged(self, val):
        """
        Horizontal resolution changed.

        """
        self.HRes = val

    def overlayImageChanged(self, state):
        """
        Overlay image changed.

        """
        if self.overlayImageCheck.isChecked():
            self.overlayImage = True

        else:
            self.overlayImage = False

    def shadowlessChanged(self, state):
        """
        Overlay image changed.

        """
        if self.shadowlessCheck.isChecked():
            self.shadowless = True

        else:
            self.shadowless = False

################################################################################

class MatplotlibSettingsForm(GenericPreferencesSettingsForm):
    """
    Matplotlib settings for the preferences dialog. Options that can be set on this
    form are:

    **Fig size**
        The size of the Matplotlib figure

    **Dpi**
        The dpi value to use when creating the Matplotlib figure

    **Show grid**
        Whether or not to show a grid on the Matplotlib figures

    **Font size**
        Set the font sizes for different labels on the Matplotlib figure

    """
    def __init__(self, parent):
        super(MatplotlibSettingsForm, self).__init__(parent)

        # settings
        self.figWidth = 8
        self.figHeight = 6
        self.figDpi = 100
        self.showGrid = True
        self.fontsize = 18
        self.tickFontsize = 16
        self.legendFontsize = 16

        # dimensions
        widthSpinBox = QtGui.QDoubleSpinBox()
        widthSpinBox.setMinimum(1)
        widthSpinBox.setMaximum(50)
        widthSpinBox.setSingleStep(0.1)
        widthSpinBox.setValue(self.figWidth)
        widthSpinBox.valueChanged.connect(self.widthChanged)

        heightSpinBox = QtGui.QDoubleSpinBox()
        heightSpinBox.setMinimum(1)
        heightSpinBox.setMaximum(50)
        heightSpinBox.setSingleStep(0.1)
        heightSpinBox.setValue(self.figHeight)
        heightSpinBox.valueChanged.connect(self.heightChanged)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(widthSpinBox)
        hbox.addWidget(QtGui.QLabel("x"))
        hbox.addWidget(heightSpinBox)
        self.layout.addRow("Fig size (inches)", hbox)

        # dpi
        dpiSpinBox = QtGui.QSpinBox()
        dpiSpinBox.setMinimum(1)
        dpiSpinBox.setMaximum(1000)
        dpiSpinBox.setValue(self.figDpi)
        dpiSpinBox.valueChanged.connect(self.dpiChanged)
        self.layout.addRow("Dpi", dpiSpinBox)

        # show grid
        self.showGridCheck = QtGui.QCheckBox()
        self.showGridCheck.setChecked(self.showGrid)
        self.showGridCheck.stateChanged.connect(self.showGridChanged)
        self.layout.addRow("Show grid", self.showGridCheck)

        # general font size
        generalFontSizeSpin = QtGui.QSpinBox()
        generalFontSizeSpin.setMinimum(1)
        generalFontSizeSpin.setMaximum(100)
        generalFontSizeSpin.setValue(self.fontsize)
        generalFontSizeSpin.valueChanged.connect(self.generalFontSizeChanged)
        self.layout.addRow("Font size (general)", generalFontSizeSpin)

        # ticks
        legendFontSizeSpin = QtGui.QSpinBox()
        legendFontSizeSpin.setMinimum(1)
        legendFontSizeSpin.setMaximum(100)
        legendFontSizeSpin.setValue(self.legendFontsize)
        legendFontSizeSpin.valueChanged.connect(self.legendFontSizeChanged)
        self.layout.addRow("Font size (legend)", legendFontSizeSpin)

        # ticks
        tickFontSizeSpin = QtGui.QSpinBox()
        tickFontSizeSpin.setMinimum(1)
        tickFontSizeSpin.setMaximum(100)
        tickFontSizeSpin.setValue(self.tickFontsize)
        tickFontSizeSpin.valueChanged.connect(self.tickFontSizeChanged)
        self.layout.addRow("Font size (ticks)", tickFontSizeSpin)

        self.init()

    def legendFontSizeChanged(self, val):
        """
        Legend font size changed changed.

        """
        self.legendFontsize = val

    def tickFontSizeChanged(self, val):
        """
        Tick font size changed changed.

        """
        self.tickFontsize = val

    def generalFontSizeChanged(self, val):
        """
        General font size changed changed.

        """
        self.fontsize = val

    def showGridChanged(self, state):
        """
        Show grid changed.

        """
        if self.showGridCheck.isChecked():
            self.showGrid = True

        else:
            self.showGrid = False

    def dpiChanged(self, val):
        """
        dpi changed.

        """
        self.figDpi = val

    def widthChanged(self, val):
        """
        Width changed.

        """
        self.figWidth = val

    def heightChanged(self, val):
        """
        Height changed.

        """
        self.figHeight = val

################################################################################

class ForcesSettingsForm(GenericPreferencesSettingsForm):
    """
    Forces settings form for preferences dialog.

    *This is not implemented currently!*

    """
    def __init__(self, parent):
        super(ForcesSettingsForm, self).__init__(parent)

        # logging
        logger = logging.getLogger(__name__)

        # settings object
#         settings = QtCore.QSettings()
#
#         # default settings
#         self.forcesConfig = forces.ForceConfig()
#         self.forcesConfig.log = self.parent.parent.console.write
#
#         # path to md dir
#         self.pathToMDDir = str(settings.value("forces/pathToMDDir", ""))
#         if len(self.pathToMDDir) and not os.path.isdir(self.pathToMDDir):
#             self.pathToMDDir = ""
#         logger.debug("Path to MD dir ini: '%s'", self.pathToMDDir)
#
#         if not len(self.pathToMDDir):
#             self.forcesConfig.md_dir = resourcePath("", dirname="md")
#
#         else:
#             self.forcesConfig.md_dir = self.pathToMDDir
#
#         logger.debug("Config path to MD dir ini: '%s'", self.forcesConfig.md_dir)
#
#         # interface type
#         self.interfaceType = str(settings.value("forces/interface", "LBOMD"))
#         logger.debug("Force interface ini: '%s'", self.interfaceType)
#
#         self.forcesConfig.md_type = self.interfaceType

        # path to povray
        self.pathToMDDir = "<not implemented>"
        pathToMDDirLineEdit = QtGui.QLineEdit(self.pathToMDDir)
#         pathToMDDirLineEdit.textChanged.connect(self.pathToMDDirChanged)
#         pathToMDDirLineEdit.editingFinished.connect(self.pathToMDDirEdited)

        rowLayout = self.newRow()
        rowLayout.addWidget(QtGui.QLabel("Path to MD dir:"))
        rowLayout.addWidget(pathToMDDirLineEdit)

        # file suffix
        rowLayout = self.newRow()

        label = QtGui.QLabel("Interface:")
        rowLayout.addWidget(label)

        interfaceCombo = QtGui.QComboBox()
        interfaceCombo.addItem("LBOMD")
#         interfaceCombo.currentIndexChanged[str].connect(self.interfaceChanged)
        rowLayout.addWidget(interfaceCombo)

        self.init()

    def pathToMDDirEdited(self):
        """
        Path to MD dir finished being edited.

        """
        if len(self.pathToMDDir) == 0 or os.path.isdir(self.pathToMDDir):
            print("STORING MD DIR PATH IN SETTINGS", self.pathToMDDir)
            settings = QtCore.QSettings()
            settings.setValue("forces/pathToMDDir", self.pathToMDDir)

            if not len(self.pathToMDDir):
                self.forcesConfig.md_dir = resourcePath("", dirname="md")

            else:
                self.forcesConfig.md_dir = self.pathToMDDir

            print("CONFIG PATH TO MD DIR", self.forcesConfig.md_dir)

    def pathToMDDirChanged(self, text):
        """
        Path to MD dir changed.

        """
        self.pathToMDDir = str(text)

    def interfaceChanged(self, text):
        """
        Interface changed

        """
        self.interfaceType = str(text)

        settings = QtCore.QSettings()
        settings.setValue("forces/interface", self.interfaceType)

        self.forcesConfig.md_type = self.interfaceType

################################################################################

class GeneralSettingsForm(GenericPreferencesSettingsForm):
    """
    General settings form for preferences dialog. Options that can be set on this form
    are:

    **OMP_NUM_THREADS**
        This sets the number of threads that will be used in parts of the C extensions
        that are parallelised using OpenMP. At present this is only relevant in a few
        places, for example the "Bond order" and "ACNA" filters and RDF plotting. The
        default is "0" which will run on all available threads.

    **DISABLE_MOUSE_WHEEL**
        Setting this option disables the use of the mouse wheel for zooming in/out of
        the VTK window. This was added because it is easy to accidentally touch the
        Apple Magic mouse and zoom in/out really far.

    **DEFAULT_PBCS**
        Set the default periodic boundary conditions for all subsequently loaded
        systems. This does not effect systems that are currently loaded or systems
        that are generated.

    """
    def __init__(self, parent):
        super(GeneralSettingsForm, self).__init__(parent)

        # logging
        self.logger = logging.getLogger(__name__+".ForcesSettingsForm")

        # settings object
        self.settings = QtCore.QSettings()

        # initial value
        maxthread = mp.cpu_count()
        ini = int(self.settings.value("omp/numThreads", 0))
        if ini > maxthread:
            ini = maxthread
        self.maxNumThreads = maxthread
        self.ompNumThreadsChanged(ini)
        self.logger.debug("Initial value of OMP_NUM_THREADS = %d", self.openmpNumThreads)

        # omp number of threads to use
        ompNumThreadsSpin = QtGui.QSpinBox()
        ompNumThreadsSpin.setMinimum(0)
        ompNumThreadsSpin.setMaximum(maxthread)
        ompNumThreadsSpin.setValue(ini)
        ompNumThreadsSpin.valueChanged.connect(self.ompNumThreadsChanged)
        ompNumThreadsSpin.setToolTip('<p>The number of threads that can be used by OpenMP. "0" means use all available processors.</p>')
        self.layout.addRow("OpenMP threads", ompNumThreadsSpin)

        # disable mouse wheel
        disableMouseWheel = int(self.settings.value("mouse/disableWheel", 0))
        self.disableMouseWheel = bool(disableMouseWheel)
        self.logger.debug("Disable mouse wheel (initial value): %s", self.disableMouseWheel)
        disableMouseWheelCheck = QtGui.QCheckBox()
        tip = "<p>Disables the mouse wheel in the VTK window. The "
        tip += "mouse wheel can be used to zoom in and out. This is "
        tip += "most useful with the wireless Apple Magic mouse.</p>"
        disableMouseWheelCheck.setToolTip(tip)
        if self.disableMouseWheel:
            disableMouseWheelCheck.setCheckState(QtCore.Qt.Checked)
        else:
            disableMouseWheelCheck.setCheckState(QtCore.Qt.Unchecked)
        disableMouseWheelCheck.stateChanged.connect(self.disableMouseWheelChanged)
        self.layout.addRow("Disable mouse wheel (VTK)", disableMouseWheelCheck)

        # default PBCs
        xpbc = int(self.settings.value("defaultPBC/x", 1))
        ypbc = int(self.settings.value("defaultPBC/y", 1))
        zpbc = int(self.settings.value("defaultPBC/z", 1))
        self.defaultPBC = np.array([xpbc, ypbc, zpbc], dtype=np.int32)
        self.logger.debug("Default PBCs (initial value): %r", list(self.defaultPBC))
        row = QtGui.QHBoxLayout()
        xyz = ["x", "y", "z"]
        for i in range(3):
            check = QtGui.QCheckBox(xyz[i])
            if self.defaultPBC[i]:
                check.setCheckState(QtCore.Qt.Checked)
            else:
                check.setCheckState(QtCore.Qt.Unchecked)
            check.stateChanged.connect(functools.partial(self.defaultPBCChanged, i))
            check.setToolTip('<p>Set the default PBC value in the "%s" direction for subsequently loaded systems</p>' % xyz[i])
            row.addWidget(check)
        self.layout.addRow("Default PBCs", row)

        self.init()

    def defaultPBCChanged(self, axis, state):
        """
        Default PBC has changed

        """
        if state == QtCore.Qt.Unchecked:
            self.defaultPBC[axis] = 0
        else:
            self.defaultPBC[axis] = 1

        if axis == 0:
            self.settings.setValue("defaultPBC/x", int(self.defaultPBC[0]))
        elif axis == 1:
            self.settings.setValue("defaultPBC/y", int(self.defaultPBC[1]))
        else:
            self.settings.setValue("defaultPBC/z", int(self.defaultPBC[2]))

        self.logger.debug("Updated default PBCs: %r", list(self.defaultPBC))

    def disableMouseWheelChanged(self, state):
        """
        Disable mouse wheel changed

        """
        if state == QtCore.Qt.Unchecked:
            self.disableMouseWheel = False

        else:
            self.disableMouseWheel = True

        self.settings.setValue("mouse/disableWheel", int(self.disableMouseWheel))

        # broadcast
        for rw in self.parent.mainWindow.rendererWindows:
            rw.vtkRenWinInteract.changeDisableMouseWheel(self.disableMouseWheel)

    def ompNumThreadsChanged(self, n):
        """
        Number of OpenMP threads has been changed

        """
        if n == 0:
            self.logger.debug("OMP_NUM_THREADS CHANGED TO: %d (%d)", n, self.maxNumThreads)
            self.openmpNumThreads = self.maxNumThreads

        else:
            self.logger.debug("OMP_NUM_THREADS CHANGED TO: %d", n)
            self.openmpNumThreads = n

        self.settings.setValue("omp/numThreads", n)

        # set environment variable
        os.environ["OMP_NUM_THREADS"] = "{0}".format(self.openmpNumThreads)

        # set C value
        _preferences.setNumThreads(self.openmpNumThreads)

################################################################################

class PreferencesDialog(QtGui.QDialog):
    """
    A number of global application settings can be configured on the Preferences dialog.

    """
    def __init__(self, mainWindow, parent=None):
        super(PreferencesDialog, self).__init__(parent)

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.parent = parent
        self.mainWindow = mainWindow
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        self.setWindowTitle("Preferences")
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/configure.png")))
        self.resize(360, 540)

        self.buttonCount = 0

        # layout
        dlgLayout = QtGui.QVBoxLayout(self)

        # toolbox
        self.toolbox = QtGui.QToolBox()

        # add toolbox to layout
        dlgLayout.addWidget(self.toolbox)

        # general settings
        self.generalForm = GeneralSettingsForm(self)
        self.toolbox.addItem(self.generalForm, QtGui.QIcon(iconPath("oxygen/applications-system.png")), "General")

        # log file settings
        self.logFileForm = LogFileSettingsForm(self)
        self.toolbox.addItem(self.logFileForm, QtGui.QIcon(iconPath("oxygen/accessories-text-editor.png")), "Log file")

        # rendering tab
        self.renderingForm = RenderingSettingsForm(self)
        self.toolbox.addItem(self.renderingForm, QtGui.QIcon(iconPath("oxygen/applications-graphics.png")), "Rendering")

        # povray tab
        self.povrayForm = PovraySettingsForm(self)
        self.toolbox.addItem(self.povrayForm, QtGui.QIcon(iconPath("other/pov-icon.svg")), "POV-Ray")

        # ffmpeg tab
        self.ffmpegForm = FfmpegSettingsForm(self)
        self.toolbox.addItem(self.ffmpegForm, QtGui.QIcon(iconPath("other/ffmpeg.png")), "FFmpeg")

        # matplotlib tab
        self.matplotlibForm = MatplotlibSettingsForm(self)
        self.toolbox.addItem(self.matplotlibForm, QtGui.QIcon(iconPath("oxygen/office-chart-bar.png")), "Matplotlib")

        # forces tab
        self.forcesForm = ForcesSettingsForm(self)
        self.toolbox.addItem(self.forcesForm, QtGui.QIcon(iconPath("capital_f.gif")), "Forces")

        # help button (links to help page on preferences dialog)
        helpButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/system-help.png")), "Show help")
        helpButton.setToolTip("<p>Show help page (opens in browser)</p>")
        helpButton.setFixedWidth(150)
        helpButton.setAutoDefault(0)
        helpButton.clicked.connect(self.loadHelpPage)
        self.helpPage = "usage/preferences.html"

        row = QtGui.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.addWidget(helpButton)
        dlgLayout.addLayout(row)


    def loadHelpPage(self):
        """
        Load the help page

        """
        if self.helpPage is None:
            return

        self.parent.showHelp(relativeUrl=self.helpPage)
