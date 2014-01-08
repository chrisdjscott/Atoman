
"""
Preferences dialog

@author: Chris Scott

"""
import os
import sys
import logging

from PySide import QtGui, QtCore

from . import genericForm
from ..visutils.utilities import iconPath, resourcePath
from ..visutils import utilities
from ..md import forces

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)





################################################################################

class GenericPreferencesSettingsForm(QtGui.QWidget):
    """
    Tab for preference dialog.
    
    """
    def __init__(self, parent=None):
        super(GenericPreferencesSettingsForm, self).__init__(parent)
        
        # tab layout
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
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
    
    def init(self):
        self.layout.addStretch(1)

################################################################################

class FfmpegSettingsForm(GenericPreferencesSettingsForm):
    """
    FFMPEG settings form for preferences dialog.
    
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
        
        rowLayout = self.newRow()
        rowLayout.addWidget(QtGui.QLabel("Path to FFmpeg:"))
        rowLayout.addWidget(pathToFFmpegLineEdit)
        
        # bitrate
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("Bitrate:")
        rowLayout.addWidget(label)
        
        bitrateSpin = QtGui.QSpinBox()
        bitrateSpin.setMinimum(1)
        bitrateSpin.setMaximum(1e8)
        bitrateSpin.setValue(self.bitrate)
        bitrateSpin.valueChanged.connect(self.bitrateChanged)
        rowLayout.addWidget(bitrateSpin)
        
        label = QtGui.QLabel("kbits/s")
        rowLayout.addWidget(label)
        
        self.init()
    
    def pathToFFmpegEdited(self):
        """
        Path to FFmpeg finished being edited.
        
        """
        exe = utilities.checkForExe(self.pathToFFmpeg)
        
        if exe:
            print "STORING FFMPEG PATH IN SETTINGS", exe, self.pathToFFmpeg
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
    POV-Ray settings form for preferences dialog.
    
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
        
        self.pathToPovray = str(settings.value("povray/pathToPovray", "povray"))
        if not os.path.exists(self.pathToPovray):
            self.pathToPovray = "povray"
        
        # path to povray
        pathToPovrayLineEdit = QtGui.QLineEdit(self.pathToPovray)
        pathToPovrayLineEdit.textChanged.connect(self.pathToPovrayChanged)
        pathToPovrayLineEdit.editingFinished.connect(self.pathToPovrayEdited)
        
        rowLayout = self.newRow()
        rowLayout.addWidget(QtGui.QLabel("Path to POV-Ray:"))
        rowLayout.addWidget(pathToPovrayLineEdit)
        
        # overlay check box
        self.overlayImageCheck = QtGui.QCheckBox("Overlay image")
        self.overlayImageCheck.setChecked(1)
        self.overlayImageCheck.stateChanged.connect(self.overlayImageChanged)
        
        rowLayout = self.newRow()
        rowLayout.addWidget(self.overlayImageCheck)
        
        # shadowless check box
        self.shadowlessCheck = QtGui.QCheckBox("Shadowless")
        self.shadowlessCheck.stateChanged.connect(self.shadowlessChanged)
        
        rowLayout = self.newRow()
        rowLayout.addWidget(self.shadowlessCheck)
        
        # dimensions
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("Dimensions: ")
        rowLayout.addWidget(label)
        
        HResSpinBox = QtGui.QSpinBox()
        HResSpinBox.setMinimum(1)
        HResSpinBox.setMaximum(10000)
        HResSpinBox.setValue(self.HRes)
        HResSpinBox.valueChanged.connect(self.HResChanged)
        rowLayout.addWidget(HResSpinBox)
        
        label = QtGui.QLabel(" x ")
        rowLayout.addWidget(label)
        
        VResSpinBox = QtGui.QSpinBox()
        VResSpinBox.setMinimum(1)
        VResSpinBox.setMaximum(10000)
        VResSpinBox.setValue(self.VRes)
        VResSpinBox.valueChanged.connect(self.VResChanged)
        rowLayout.addWidget(VResSpinBox)
        
        # view angle
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("View angle: ")
        rowLayout.addWidget(label)
        
        angleSpinBox = QtGui.QDoubleSpinBox()
        angleSpinBox.setSingleStep(0.1)
        angleSpinBox.setMinimum(0.1)
        angleSpinBox.setMaximum(360.0)
        angleSpinBox.setValue(self.viewAngle)
        angleSpinBox.valueChanged.connect(self.viewAngleChanged)
        rowLayout.addWidget(angleSpinBox)
        
        label = QtGui.QLabel(" degrees")
        rowLayout.addWidget(label)
        
        self.init()
    
    def pathToPovrayEdited(self):
        """
        Path to povray finished being edited.
        
        """
        exe = utilities.checkForExe(self.pathToPovray)
        
        if exe:
            print "STORING POV PATH IN SETTINGS", exe, self.pathToPovray
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
    Matplotlib settings.
    
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
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("Fig size:")
        rowLayout.addWidget(label)
        
        widthSpinBox = QtGui.QDoubleSpinBox()
        widthSpinBox.setMinimum(1)
        widthSpinBox.setMaximum(50)
        widthSpinBox.setSingleStep(0.1)
        widthSpinBox.setValue(self.figWidth)
        widthSpinBox.valueChanged.connect(self.widthChanged)
        rowLayout.addWidget(widthSpinBox)
        
        label = QtGui.QLabel("x")
        rowLayout.addWidget(label)
        
        heightSpinBox = QtGui.QDoubleSpinBox()
        heightSpinBox.setMinimum(1)
        heightSpinBox.setMaximum(50)
        heightSpinBox.setSingleStep(0.1)
        heightSpinBox.setValue(self.figHeight)
        heightSpinBox.valueChanged.connect(self.heightChanged)
        rowLayout.addWidget(heightSpinBox)
        
        label = QtGui.QLabel("inches")
        rowLayout.addWidget(label)
        
        # dpi
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("Dpi:")
        rowLayout.addWidget(label)
        
        dpiSpinBox = QtGui.QSpinBox()
        dpiSpinBox.setMinimum(1)
        dpiSpinBox.setMaximum(1000)
        dpiSpinBox.setValue(self.figDpi)
        dpiSpinBox.valueChanged.connect(self.dpiChanged)
        rowLayout.addWidget(dpiSpinBox)
        
        # show grid
        self.showGridCheck = QtGui.QCheckBox("Show grid")
        self.showGridCheck.setChecked(self.showGrid)
        self.showGridCheck.stateChanged.connect(self.showGridChanged)
        
        rowLayout = self.newRow()
        rowLayout.addWidget(self.showGridCheck)
        
        # font size group
        fontSizeGroup = genericForm.GenericForm(self, 0, "Font size")
        fontSizeGroup.show()
        
        rowLayout = self.newRow()
        rowLayout.addWidget(fontSizeGroup)
        
        # general
        label = QtGui.QLabel("General:")
        generalFontSizeSpin = QtGui.QSpinBox()
        generalFontSizeSpin.setMinimum(1)
        generalFontSizeSpin.setMaximum(100)
        generalFontSizeSpin.setValue(self.fontsize)
        generalFontSizeSpin.valueChanged.connect(self.generalFontSizeChanged)
        
        row = fontSizeGroup.newRow()
        row.addWidget(label)
        row.addWidget(generalFontSizeSpin)
        
        # ticks
        label = QtGui.QLabel("Legend:")
        legendFontSizeSpin = QtGui.QSpinBox()
        legendFontSizeSpin.setMinimum(1)
        legendFontSizeSpin.setMaximum(100)
        legendFontSizeSpin.setValue(self.legendFontsize)
        legendFontSizeSpin.valueChanged.connect(self.legendFontSizeChanged)
        
        row = fontSizeGroup.newRow()
        row.addWidget(label)
        row.addWidget(legendFontSizeSpin)
        
        # ticks
        label = QtGui.QLabel("Ticks:")
        tickFontSizeSpin = QtGui.QSpinBox()
        tickFontSizeSpin.setMinimum(1)
        tickFontSizeSpin.setMaximum(100)
        tickFontSizeSpin.setValue(self.tickFontsize)
        tickFontSizeSpin.valueChanged.connect(self.tickFontSizeChanged)
        
        row = fontSizeGroup.newRow()
        row.addWidget(label)
        row.addWidget(tickFontSizeSpin)
        
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
    
    """
    def __init__(self, parent):
        super(ForcesSettingsForm, self).__init__(parent)
        
        # logging
        logger = logging.getLogger(__name__)
        
        # settings object
        settings = QtCore.QSettings()
        
        # default settings
        self.forcesConfig = forces.ForceConfig()
        self.forcesConfig.log = self.parent.parent.console.write
        
        # path to md dir
        self.pathToMDDir = str(settings.value("forces/pathToMDDir", ""))
        if len(self.pathToMDDir) and not os.path.isdir(self.pathToMDDir):
            self.pathToMDDir = ""
        logger.debug("Path to MD dir ini: '%s'", self.pathToMDDir)
        
        if not len(self.pathToMDDir):
            self.forcesConfig.md_dir = resourcePath("", dirname="md")
        
        else:
            self.forcesConfig.md_dir = self.pathToMDDir
        
        logger.debug("Config path to MD dir ini: '%s'", self.forcesConfig.md_dir)
        
        # interface type
        self.interfaceType = str(settings.value("forces/interface", "LBOMD"))
        logger.debug("Force interface ini: '%s'", self.interfaceType)
        
        self.forcesConfig.md_type = self.interfaceType
        
        # path to povray
        pathToMDDirLineEdit = QtGui.QLineEdit(self.pathToMDDir)
        pathToMDDirLineEdit.textChanged.connect(self.pathToMDDirChanged)
        pathToMDDirLineEdit.editingFinished.connect(self.pathToMDDirEdited)
        
        rowLayout = self.newRow()
        rowLayout.addWidget(QtGui.QLabel("Path to MD dir:"))
        rowLayout.addWidget(pathToMDDirLineEdit)
        
        # file suffix
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("Interface:")
        rowLayout.addWidget(label)
        
        interfaceCombo = QtGui.QComboBox()
        interfaceCombo.addItem("LBOMD")
        interfaceCombo.currentIndexChanged[str].connect(self.interfaceChanged)
        rowLayout.addWidget(interfaceCombo)
        
        self.init()
    
    def pathToMDDirEdited(self):
        """
        Path to MD dir finished being edited.
        
        """
        if len(self.pathToMDDir) == 0 or os.path.isdir(self.pathToMDDir):
            print "STORING MD DIR PATH IN SETTINGS", self.pathToMDDir
            settings = QtCore.QSettings()
            settings.setValue("forces/pathToMDDir", self.pathToMDDir)
            
            if not len(self.pathToMDDir):
                self.forcesConfig.md_dir = resourcePath("", dirname="md")
            
            else:
                self.forcesConfig.md_dir = self.pathToMDDir
            
            print "CONFIG PATH TO MD DIR", self.forcesConfig.md_dir
    
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

class LoggingSettingsForm(GenericPreferencesSettingsForm):
    """
    Logging preferences
    
    """
    def __init__(self, parent):
        super(LoggingSettingsForm, self).__init__(parent)
        
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
        
        settings = QtCore.QSettings()
        
        # should get these from settings
        consoleLevel = settings.value("logging/console", logging.INFO)
        consoleLevelIndex = self.getLevelIndex(consoleLevel)
        
        # console window settings
        self.consoleLevelCombo = QtGui.QComboBox()
        self.consoleLevelCombo.addItems(self.loggingLevelsSorted)
        self.consoleLevelCombo.currentIndexChanged[str].connect(self.consoleLevelChanged)
        self.consoleLevelCombo.setCurrentIndex(consoleLevelIndex)
        
        label = QtGui.QLabel("Console window level:")
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.consoleLevelCombo)
        
        self.init()
    
    def getLevelIndex(self, level):
        """
        Return index of level
        
        """
        levelKey = None
        for key, val in self.loggingLevels.iteritems():
            if val == level:
                levelKey = key
                break
        
        if levelKey is None:
            logger = logging.getLogger(__name__)
            logger.critical("No match for log level: %s", str(level))
            return 2
        
        return self.loggingLevelsSorted.index(levelKey)
    
    def consoleLevelChanged(self, levelKey):
        """
        Console window logging level has changed
        
        """
        levelKey = str(levelKey)
        level = self.loggingLevels[levelKey]
        
        # get handler (console window is second)
        handler = logging.getLogger().handlers[1]
        
        # set level
        handler.setLevel(level)
        
        # update settings
        settings = QtCore.QSettings()
        settings.setValue("logging/console", level)

################################################################################

class PreferencesDialog(QtGui.QDialog):
    """
    Preferences dialog.
    
    """
    def __init__(self, parent=None):
        super(PreferencesDialog, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Preferences")
        self.setWindowIcon(QtGui.QIcon(iconPath("applications-system.svg")))
        self.resize(320, 380)
        
        self.buttonCount = 0
        
        # layout
        dlgLayout = QtGui.QVBoxLayout(self)
        
        # toolbox
        self.toolbox = QtGui.QToolBox()
        
        # add toolbox to layout
        dlgLayout.addWidget(self.toolbox)
        
        # logging tab
        self.loggingForm = LoggingSettingsForm(self)
        self.toolbox.addItem(self.loggingForm, QtGui.QIcon(iconPath("logging-icon.png")), "Logging")
        
        # povray tab
        self.povrayForm = PovraySettingsForm(self)
        self.toolbox.addItem(self.povrayForm, QtGui.QIcon(iconPath("pov-icon.svg")), "POV-Ray")
        
        # ffmpeg tab
        self.ffmpegForm = FfmpegSettingsForm(self)
        self.toolbox.addItem(self.ffmpegForm, QtGui.QIcon(iconPath("ffmpeg.png")), "FFmpeg")
        
        # matplotlib tab
        self.matplotlibForm = MatplotlibSettingsForm(self)
        self.toolbox.addItem(self.matplotlibForm, QtGui.QIcon(iconPath("Plotter.png")), "Matplotlib")
        
        # forces tab
        self.forcesForm = ForcesSettingsForm(self)
        self.toolbox.addItem(self.forcesForm, QtGui.QIcon(iconPath("capital_f.gif")), "Forces")
