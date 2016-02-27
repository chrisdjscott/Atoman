
"""
Window for displaying output from the program to the user. The level of the output can be set,
using the standard logging module levels (DEBUG, INFO, ...).

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

from PyQt5 import QtGui, QtCore, QtWidgets


from ...visutils.utilities import iconPath
from ...visutils import utilities
import six
from six.moves import range


################################################################################

class ConsoleWindow(QtWidgets.QDialog):
    """
    Console window for displaying output to the user.
    
    """
    def __init__(self, parent=None):
        super(ConsoleWindow, self).__init__(parent)
        
        self.iniWinFlags = self.windowFlags()
        self.setWindowFlags(self.iniWinFlags | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.setModal(0)
#        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Console")
        self.setWindowIcon(QtGui.QIcon(iconPath("console-icon.png")))
        self.resize(800,400)
        
        consoleLayout = QtWidgets.QVBoxLayout(self)
        consoleLayout.setAlignment(QtCore.Qt.AlignTop)
        consoleLayout.setContentsMargins(0, 0, 0, 0)
        consoleLayout.setSpacing(0)
        
        self.textWidget = QtWidgets.QTextEdit()
        self.textWidget.setReadOnly(1)
        
        consoleLayout.addWidget(self.textWidget)
        
        #TODO: add save text.
        
        self.clearButton = QtWidgets.QPushButton("Clear")
        self.clearButton.setAutoDefault(0)
        self.clearButton.clicked.connect(self.clearText)
        
        self.saveButton = QtWidgets.QPushButton("Save")
        self.saveButton.setAutoDefault(0)
        self.saveButton.clicked.connect(self.saveText)
        
        self.closeButton = QtWidgets.QPushButton("Hide")
        self.closeButton.setAutoDefault(1)
        self.closeButton.clicked.connect(self.close)
        
        # logging handler
        handler = utilities.TextEditHandler(self.textWidget)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        handler.setFormatter(formatter)
        
        # add to root logger
        logging.getLogger().addHandler(handler)
        
        # set level, try settings first or fallback to INFO
        settings = QtCore.QSettings()
        level = int(settings.value("logging/console", logging.INFO))
        
        logger = logging.getLogger(__name__)
        handler.setLevel(int(level))
        logger.debug("Initial console window logging level: %s", logging.getLevelName(level))
        
        self.logger = logger
        
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
        
        # should get these from settings
        consoleLevel = level
        consoleLevelIndex = self.getLevelIndex(consoleLevel)
        
        self.consoleLevelCombo = QtWidgets.QComboBox()
        self.consoleLevelCombo.addItems(self.loggingLevelsSorted)
        self.consoleLevelCombo.currentIndexChanged[str].connect(self.consoleLevelChanged)
        self.consoleLevelCombo.setCurrentIndex(consoleLevelIndex)
        label = QtWidgets.QLabel("Level:")
        
        buttonWidget = QtWidgets.QWidget()
        buttonLayout = QtWidgets.QHBoxLayout(buttonWidget)
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.saveButton)
        buttonLayout.addStretch()
        buttonLayout.addWidget(label)
        buttonLayout.addWidget(self.consoleLevelCombo)
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.closeButton)
        
        consoleLayout.addWidget(buttonWidget)
    
    def getLevelIndex(self, level):
        """
        Return index of level
        
        """
        levelKey = None
        for key, val in six.iteritems(self.loggingLevels):
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
    
    def saveText(self):
        """
        Save text to file
        
        """
        self.setWindowFlags(self.iniWinFlags)
        
        # get file name
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Console Output', '.', "HTML files (*.html)")[0][0]
        
        self.setWindowFlags(self.iniWinFlags | QtCore.Qt.WindowStaysOnTopHint)
        self.show()
        
        if len(filename):
            if not filename.endswith(".html"):
                filename += ".html"
            
            self.logger.debug("Saving console output to file: '%s'", filename)
            
            # write to file
            f = open(filename, "w")
            f.write(self.textWidget.toHtml())
            f.close()
    
    def clearText(self):
        """
        Clear all text.
        
        """
        self.textWidget.clear()
    
    def write(self, string, level=0, indent=0):
        """
        Write to the console window
        
        """
        #TODO: change colour depending on level
        if level < self.parent.verboseLevel:
            ind = ""
            for _ in range(indent):
                ind += "  "
            self.textWidget.append("%s %s%s" % (">", ind, string))
        
    def closeEvent(self, event):
        self.hide()
        self.parent.consoleOpen = 0
