
"""
Settings for filters.

Dialogs must be named like: FilterNameSettingsDialog
where FilterName is the (capitalised) name of the
filter with no spaces. Eg "Point defects" becomes
"PointDefectsSettingsDialog".

@author: Chris Scott

"""
import logging

from PySide import QtGui, QtCore

from ...visutils.utilities import iconPath
from .. import genericForm


################################################################################
class GenericSettingsDialog(QtGui.QDialog):
    def __init__(self, title, parent):
        super(GenericSettingsDialog, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.mainWindow = self.parent.mainWindow
        self.pipelinePage = self.parent.filterTab
        
        self.logger = logging.getLogger(__name__)
        
        # get tab and filter id's
        array = title.split("(")[1].split(")")[0].split()
        self.listID = int(array[1])
        self.filterID = int(array[3])
        
        self.setModal(0)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/configure.png")))
#        self.resize(500,300)
        
        dialogLayout = QtGui.QVBoxLayout()
        dialogLayout.setAlignment(QtCore.Qt.AlignTop)
        
        tabWidget = QtGui.QTabWidget()
        
        # layout/widget
        self.contentLayout = QtGui.QFormLayout()
        contentWidget = QtGui.QWidget()
        contentWidget.setLayout(self.contentLayout)
        
        tabWidget.addTab(contentWidget, "Calculate")
        
        # display settings
        self.displaySettingsLayout = QtGui.QFormLayout()
        displaySettingsWidget = QtGui.QWidget()
        displaySettingsWidget.setLayout(self.displaySettingsLayout)
        
        tabWidget.addTab(displaySettingsWidget, "Display")
        
        dialogLayout.addWidget(tabWidget)
        self.setLayout(dialogLayout)
        
        # button box
        self.buttonBox = QtGui.QDialogButtonBox()
        
        # add close button
        closeButton = self.buttonBox.addButton(self.buttonBox.Close)
        closeButton.setDefault(True)
        self.buttonBox.rejected.connect(self.close)
        
        dialogLayout.addWidget(self.buttonBox)
        
        # filtering enabled by default
        self.filteringEnabled = True
        
        # help page
        self.helpPage = None
        
        # does this filter provide scalars
        self._providedScalars = []
    
    def addProvidedScalar(self, name):
        """Add scalar option."""
        self._providedScalars.append(name)
    
    def getProvidedScalars(self):
        """Return a list of scalars provided by this filter."""
        return self._providedScalars
    
    def addHorizontalDivider(self, displaySettings=False):
        """Add horizontal divider (QFrame)."""
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        if displaySettings:
            self.displaySettingsLayout.addRow(line)
        else:
            self.contentLayout.addRow(line)
    
    def addLinkToHelpPage(self, page):
        """Add button with link to help page."""
        helpButton = self.buttonBox.addButton(self.buttonBox.Help)
        helpButton.setAutoDefault(False)
        helpButton.setToolTip("Show help page")
        self.buttonBox.helpRequested.connect(self.loadHelpPage)
        
        self.helpPage = page
    
    def loadHelpPage(self):
        """Load the help page."""
        self.logger.debug("Help requested: '%s'", self.helpPage)
        
        if self.helpPage is None:
            return
        
        self.mainWindow.helpWindow.loadPage(self.helpPage)
        self.mainWindow.showHelp()
    
    def addFilteringGroupBox(self, title="Enable filtering", slot=None, checked=False):
        """Add a group box that contains filtering options."""
        # widget
        grp = QtGui.QGroupBox(title)
        grp.setCheckable(True)
        
        # layout
        grpLayout = QtGui.QVBoxLayout()
        grpLayout.setAlignment(QtCore.Qt.AlignTop)
        grpLayout.setContentsMargins(0,0,0,0)
        grpLayout.setSpacing(0)
        grp.setLayout(grpLayout)
        
        # connect toggled signal
        if slot is not None:
            grp.toggled.connect(slot)
        
        # initial check status
        grp.setChecked(checked)
        
        # add to form layout
        row = self.newRow()
        row.addWidget(grp)
        
        return grpLayout
    
    def newRow(self, align=None):
        """New filter settings row."""
        row = genericForm.FormRow(align=align)
        self.contentLayout.addWidget(row)
        
        return row
    
    def removeRow(self,row):
        """Remove filter settings row."""
        self.contentLayout.removeWidget(row)  
    
    def closeEvent(self, event):
        self.hide()
    
    def refresh(self):
        """
        Called whenever a new input is loaded.
        
        Should be overridden if required.
        
        """
        pass
