
"""
Preferences dialog.

@author: Chris Scott

"""
import os
import sys
import logging
import datetime
import multiprocessing as mp

from PySide import QtGui, QtCore

from . import genericForm
from ..visutils.utilities import iconPath, resourcePath
from ..visutils import utilities
# from ..md import forces

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
        rowLayout = self.newRow()
        rowLayout.addWidget(createCheck)
        
        # directory
        self.logDirLabel = QtGui.QLabel("Log dir: '%s'" % self.logDirectory)
        row = self.newRow()
        row.addWidget(self.logDirLabel)
        
        # directory dialog button
        button = QtGui.QPushButton("Choose directory")
        button.clicked.connect(self.showLogDirectoryDialog)
        row = self.newRow()
        row.addWidget(button)
        
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
        label = QtGui.QLabel("Level:")
        levelCombo = QtGui.QComboBox()
        levelCombo.addItems(self.loggingLevelsSorted)
        levelIndex = self.getLevelIndex(self.logLevel)
        levelCombo.setCurrentIndex(levelIndex)
        levelCombo.currentIndexChanged[str].connect(self.logLevelChanged)
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(levelCombo)
        
        # create handler... (should be on main window!)
        if self.createLogFile:
            logfile = os.path.join(self.logDirectory, "CDJSVis-%s.log" % datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
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
            self.logDirLabel.setText("Log dir: '%s" % new_dir)
            
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
    
    """
    def __init__(self, parent):
        super(RenderingSettingsForm, self).__init__(parent)
        
        # settings object
        settings = QtCore.QSettings()
        
        # default settings
        self.maxAtomsAutoRun = int(settings.value("rendering/maxAtomsAutoRun", 10000))
        
        # max atoms auto run
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("Max atoms auto run:")
        rowLayout.addWidget(label)
        
        maxAtomsSpin = QtGui.QSpinBox()
        maxAtomsSpin.setMinimum(1)
        maxAtomsSpin.setMaximum(99999)
        maxAtomsSpin.setValue(self.maxAtomsAutoRun)
        maxAtomsSpin.valueChanged.connect(self.maxAtomsChanged)
        rowLayout.addWidget(maxAtomsSpin)
        
        self.init()
    
    def maxAtomsChanged(self, val):
        """
        maxAtomsAutoRun changed.
        
        """
        self.maxAtomsAutoRun = val
        
        # store in settings
        settings = QtCore.QSettings()
        settings.setValue("rendering/maxAtomsAutoRun", val)

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
        
        # cell frame radius
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("Cell frame radius: ")
        rowLayout.addWidget(label)
        
        cellFrameSpinBox = QtGui.QDoubleSpinBox()
        cellFrameSpinBox.setSingleStep(0.01)
        cellFrameSpinBox.setMinimum(0.01)
        cellFrameSpinBox.setMaximum(5.0)
        cellFrameSpinBox.setValue(self.cellFrameRadius)
        cellFrameSpinBox.valueChanged.connect(self.cellFrameRadiusChanged)
        rowLayout.addWidget(cellFrameSpinBox)
        
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

class GeneralSettingsForm(GenericPreferencesSettingsForm):
    """
    General settings form for preferences dialog. Options that can be set on this form
    are:
    
    **OMP_NUM_THREADS**
        This sets the number of threads that will be used in parts of the C extensions
        that are parallelised using OpenMP. At present this is only relevant in a few
        places, for example the "Bond order" and "ACNA" filters and RDF plotting. The 
        default is "0" which will run on all available threads.
    
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
        
        rowLayout = self.newRow()
        rowLayout.addWidget(QtGui.QLabel("OMP_NUM_THREADS: "))
        rowLayout.addWidget(ompNumThreadsSpin)
        
        self.init()
    
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

################################################################################

class PreferencesDialog(QtGui.QDialog):
    """
    A number of global application settings can be configured on the Preferences dialog.
    
    """
    def __init__(self, parent=None):
        super(PreferencesDialog, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Preferences")
        self.setWindowIcon(QtGui.QIcon(iconPath("applications-system.svg")))
        self.resize(320, 520)
        
        self.buttonCount = 0
        
        # layout
        dlgLayout = QtGui.QVBoxLayout(self)
        
        # toolbox
        self.toolbox = QtGui.QToolBox()
        
        # add toolbox to layout
        dlgLayout.addWidget(self.toolbox)
        
        # general settings
        self.generalForm = GeneralSettingsForm(self)
        self.toolbox.addItem(self.generalForm, QtGui.QIcon(iconPath("applications-system.svg")), "General")
        
        # log file settings
        self.logFileForm = LogFileSettingsForm(self)
        self.toolbox.addItem(self.logFileForm, QtGui.QIcon(iconPath("accessories-text-editor.svg")), "Log file")
        
        # rendering tab
        self.renderingForm = RenderingSettingsForm(self)
        self.toolbox.addItem(self.renderingForm, QtGui.QIcon(iconPath("applications-graphics.svg")), "Rendering")
        
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
        
        # help button (links to help page on preferences dialog)
        helpButton = QtGui.QPushButton(QtGui.QIcon(iconPath("Help-icon.png")), "Show help")
        helpButton.setToolTip("Show help page")
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
        
        self.parent.helpWindow.loadPage(self.helpPage)
        self.parent.showHelp()
