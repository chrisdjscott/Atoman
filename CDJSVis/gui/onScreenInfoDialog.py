
"""
On screen info dialog

@author: Chris Scott

"""
import logging

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath


################################################################################

class TextSettingsDialog(QtGui.QDialog):
    """
    Dialog for setting text options.
    
    """
    def __init__(self, item, parent=None):
        super(TextSettingsDialog, self).__init__(parent)
        
        self.resize(240, 210)
        
        self.parent = parent
        self.item = item
        
#         self.setModal(1)
        
        titleText = "%s settings" % item.title
        self.setWindowTitle(titleText)
        
        dialogLayout = QtGui.QVBoxLayout(self)
        
        # defaults
        textPosition = item.position
        
        # location of text
        form = QtGui.QGroupBox("Text location")
        formLayout = QtGui.QVBoxLayout()
        form.setLayout(formLayout)
        dialogLayout.addWidget(form)
        
        self.positionRadios = {}
        self.positionRadios["Top left"] = QtGui.QRadioButton("Top &left", parent=form)
        self.positionRadios["Top right"] = QtGui.QRadioButton("Top &right", parent=form)
        self.positionRadios[textPosition].setChecked(True)
        self.positionRadios["Top left"].toggled.connect(self.positionChanged)
        
        formLayout.addWidget(self.positionRadios["Top left"])
        formLayout.addWidget(self.positionRadios["Top right"])
        formLayout.addStretch(1)
        
        # text format
        form = QtGui.QGroupBox("Text format")
        formLayout = QtGui.QVBoxLayout()
        form.setLayout(formLayout)
        dialogLayout.addWidget(form)
        
        self.textFormatLine = QtGui.QLineEdit(item.format)
        self.textFormatLine.editingFinished.connect(self.textFormatEdited)
        formLayout.addWidget(self.textFormatLine)
        
        for k, v in item.formatInfo.iteritems():
            formLayout.addWidget(QtGui.QLabel("'%s' = %s" % (k, v)))
        
        # add close button
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        
        dialogLayout.addWidget(buttonBox)
    
    def textFormatEdited(self):
        """
        Text format has been edited.
        
        """
        self.item.format = str(self.textFormatLine.text())
        
        logger = logging.getLogger(__name__+".TextSettingsDialog")
        logger.debug("Text format changed for '%s' to '%s'", self.item.title, self.item.format)
    
    def positionChanged(self):
        """
        Position changed.
        
        """
        for pos in self.positionRadios.keys():
            rb = self.positionRadios[pos]
            if rb.isChecked():
                self.item.positionChanged(pos)
                break
        
        logger = logging.getLogger(__name__+".TextSettingsDialog")
        logger.debug("Text position changed for '%s' to '%s'", self.item.title, self.item.position)

################################################################################

class TextListWidgetItem(QtGui.QListWidgetItem):
    """
    Item for going in the list widget
    
    """
    def __init__(self, title, defaultFormat, formatInfo, defaultPosition, defaultChecked, multiLine=False):
        super(TextListWidgetItem, self).__init__()
        
        # add check box
        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable)
        
        self.title = title
        self.defaultFormat = defaultFormat
        self.defaultPosition = defaultPosition
        self.multiLine = multiLine
        self.formatInfo = formatInfo
        
        # format and position (read from settings?)
        self.format = defaultFormat
        self.position = defaultPosition
        
        # checked (read from settings?)
        if defaultChecked:
            self.setCheckState(QtCore.Qt.Checked)
        
        else:
            self.setCheckState(QtCore.Qt.Unchecked)
        
        self.setText("%s (%s)" % (title, self.position))
        
        self.logger = logging.getLevelName(__name__+".TextListWidgetItem")
    
    def positionChanged(self, position):
        """
        Position has changed
        
        """
        self.position = position
        self.setText("%s (%s)" % (self.title, self.position))
    
    def makeText(self, args):
        """
        Attempt to format the string with the given args.
        Fall back to default if it fails.
        
        """
        try:
            text = self.format.format(*args)
        
        except:
            logging.error("Could not apply format: '%s'; '%s'; %r", self.title, self.format, args)
            text = self.defaultFormat.format(*args)
        
        return text

################################################################################

class OnScreenInfoDialog(QtGui.QDialog):
    """
    On screen info selector.
    
    """
    def __init__(self, mainWindow, index, parent=None):
        super(OnScreenInfoDialog, self).__init__(parent)
        
#         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.logger = logging.getLogger(__name__)
        self.resize(340, 380)
        
        self.parent = parent
        self.rendererWindow = parent
        self.mainWindow = mainWindow
        
        self.setWindowTitle("On screen info - Render window %d" % index)
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/preferences-desktop-font.png")))
        
        dialogLayout = QtGui.QVBoxLayout()
        self.setLayout(dialogLayout)
        
        # list containing selected text
        self.textList = QtGui.QListWidget()
        self.textList.setDragEnabled(True)
        self.textList.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        dialogLayout.addWidget(self.textList)
        
        # refresh button
        refreshButton = QtGui.QPushButton("Refresh")
        refreshButton.setAutoDefault(0)
        refreshButton.setStatusTip("Refresh on-screen information")
        refreshButton.clicked.connect(self.refresh)
        
        buttonWidget = QtGui.QWidget()
        buttonLayout = QtGui.QHBoxLayout(buttonWidget)
        buttonLayout.addStretch()
        buttonLayout.addWidget(refreshButton)
        buttonLayout.addStretch()
        
        dialogLayout.addWidget(buttonWidget)
        
        # add default options
        self.textList.addItem(TextListWidgetItem("Atom count", "{0} atoms", {"{0}": "Atom count"}, "Top left", True))
        self.textList.addItem(TextListWidgetItem("Visible count", "{0} visible", {"{0}": "Visible count"}, "Top left", True))
        self.textList.addItem(TextListWidgetItem("Visible species count", "{0} {1}", {"{0}": "Count", "{1}": "Species"}, 
                                                 "Top left", True, multiLine=True))
        self.textList.addItem(TextListWidgetItem("Defect count", "{0} {1}", {"{0}": "Count", "{1}": "Defect type"}, "Top left", 
                                                 True, multiLine=True))
        self.textList.addItem(TextListWidgetItem("Defect species count", "{0} {1} {2}", {"{0}": "Count", "{1}": "Species", "{2}": "Defect type"}, 
                                                 "Top left", True, multiLine=True))
        self.textList.addItem(TextListWidgetItem("ACNA structure count", "{0} {1}", {"{0}": "Count", "{1}": "Structure"}, 
                                                 "Top left", True, multiLine=True))
        self.textList.addItem(TextListWidgetItem("Cluster count", "{0} clusters", {"{0}": "Cluster count"}, "Top left", True))
        self.textList.addItem(TextListWidgetItem("Time", "{0} {1}", {"{0}": "Time", "{1}": "Units"}, "Top right", True))
        self.textList.addItem(TextListWidgetItem("KMC step", "Step {0}", {"{0}": "Step number"}, "Top right", True))
        self.textList.addItem(TextListWidgetItem("Barrier", "{0} eV", {"{0}": "Barrier"}, "Top right", True))
        self.textList.addItem(TextListWidgetItem("Temperature", "{0} K", {"{0}": "Temperature"}, "Top right", True))
        
        # refresh additional available options
        self.refreshLists()
        
        # connect
        self.textList.itemDoubleClicked.connect(self.showTextSettingsDialog)
    
    def selectedText(self):
        """
        Return selected text
        
        """
        selectedText = []
        for i in xrange(self.textList.count()):
            item = self.textList.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                selectedText.append(item)
        
        return selectedText
    
    def showTextSettingsDialog(self, item):
        """
        Show text settings dialog.
        
        """
        TextSettingsDialog(item, parent=self).exec_()
    
    def refresh(self):
        """
        Refresh on screen info.
        
        """
        self.parent.refreshOnScreenInfo()
    
    def refreshLists(self):
        """
        Refresh lists.
        
        Remove options that are no longer available.
        Add options that are now available.
        
        """
        self.logger.debug("Refreshing on-screen text options")
        
        #TODO: automatically add stuff from Lattice.attributes
        #TODO: automatically add magnitude off Lattice.vectorsData
        #TODO: automatically add sum of Lattice.scalarData
