
"""
Data input is handled by the systems dialog. 
From here you can load or generate a lattice; view loaded lattices; set input/ref system, etc.
Once loaded systems will be added to the "Loaded systems" list.  
Systems can be removed from the list by selecting them (multiple selection is possible) and clicking the minus sign. 
Note that systems that are currently selected on an analysis pipeline, as either a ref or input, cannot be removed.

"""
import os
import sys
import logging
import functools
import copy

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from .genericForm import GenericForm
from . import latticeReaderForms
from . import latticeGeneratorForms
from . import sftpDialog

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class GenerateInputForm(GenericForm):
    """
    Form for generating a new system
    
    """
    def __init__(self, parent, mainWindow, mainToolbar):
        super(GenerateInputForm, self).__init__(parent, None, "Generate new system")
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.mainToolbar = mainToolbar
        
        # system type combo
        self.inputTypeCombo = QtGui.QComboBox()
        self.inputTypeCombo.addItem("FCC")
        self.inputTypeCombo.addItem("Pu-Ga (L12 method)")
        self.inputTypeCombo.addItem("BCC")
        self.inputTypeCombo.addItem("Fluorite (CaF2)")
        self.inputTypeCombo.addItem("Rock salt (NaCl)")
#         self.inputTypeCombo.addItem("Pyrochlore")
#         self.inputTypeCombo.addItem("6H")
        self.inputTypeCombo.currentIndexChanged.connect(self.setWidgetStack)
        
        row = self.newRow()
        row.addWidget(self.inputTypeCombo)
        
        self.show()
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget()
        
        row = self.newRow()
        row.addWidget(self.stackedWidget)
        
        # FCC generator
        self.fcc_generator = latticeGeneratorForms.FCCLatticeGeneratorForm(self, self.mainWindow)
        self.stackedWidget.addWidget(self.fcc_generator)
        
        # Pu3Ga generator
        self.pu3ga_generator = latticeGeneratorForms.Pu3GaLatticeGeneratorForm(self, self.mainWindow)
        self.stackedWidget.addWidget(self.pu3ga_generator)
        
        # BCC generator
        self.bcc_generator = latticeGeneratorForms.BCCLatticeGeneratorForm(self, self.mainWindow)
        self.stackedWidget.addWidget(self.bcc_generator)
        
        # Fluorite generator
        self.fluorite_generator = latticeGeneratorForms.FluoriteLatticeGeneratorForm(self, self.mainWindow)
        self.stackedWidget.addWidget(self.fluorite_generator)
        
        # Rock Salt generator
        self.rockSalt_generator = latticeGeneratorForms.RockSaltLatticeGeneratorForm(self, self.mainWindow)
        self.stackedWidget.addWidget(self.rockSalt_generator)
        
        # help icon
        row = self.newRow()
        row.RowLayout.addStretch(1)
        
        helpButton = QtGui.QPushButton(QtGui.QIcon(iconPath("Help-icon.png")), "")
        helpButton.setFixedWidth(20)
        helpButton.setFixedHeight(20)
        helpButton.setToolTip("Show help page")
        helpButton.clicked.connect(self.loadHelpPage)
        row.addWidget(helpButton)
        
        self.show()
    
    def loadHelpPage(self):
        """
        Load the help page for this form
        
        """
        self.mainWindow.helpWindow.loadPage("usage/input/lattice_generation.html")
        self.mainWindow.showHelp()
    
    def file_generated(self, lattice, filename):
        """
        File generated
        
        """
        self.parent.file_generated(lattice, filename)
    
    def setWidgetStack(self, index):
        """
        Set stack
        
        """
        self.stackedWidget.setCurrentIndex(index)

################################################################################

class LoadSystemForm(GenericForm):
    """
    Form for reading new lattice.
    
    """
    def __init__(self, parent, mainWindow, mainToolbar):
        super(LoadSystemForm, self).__init__(parent, None, "Load new system")
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.mainToolbar = mainToolbar
        
        # sftp browser
        self.sftp_browser = sftpDialog.SFTPBrowserDialog(self.mainWindow, parent=self)
        
        # ordered list of keys
        self.readerFormsKeys = [
            "LBOMD DAT",
            "LBOMD REF",
            "LBOMD XYZ",
            "AUTO DETECT"
        ]
        
        # reader forms
        self.readerForms = {
            "LBOMD DAT": latticeReaderForms.LbomdDatReaderForm(self, self.mainToolbar, self.mainWindow, "LBOMD DAT"),
            "LBOMD REF": latticeReaderForms.LbomdRefReaderForm(self, self.mainToolbar, self.mainWindow, "LBOMD REF"),
            "LBOMD XYZ": latticeReaderForms.LbomdXYZReaderForm(self, self.mainToolbar, self.mainWindow, "LBOMD XYZ"),
            "AUTO DETECT": latticeReaderForms.AutoDetectReaderForm(self, self.mainToolbar, self.mainWindow, "AUTO DETECT"),
        }
        
        # file type combo
        self.inputTypeCombo = QtGui.QComboBox()
        self.inputTypeCombo.addItems(self.readerFormsKeys)
        self.inputTypeCombo.currentIndexChanged.connect(self.setWidgetStack)
        
        row = self.newRow()
        row.addWidget(self.inputTypeCombo)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget()
        
        # add reader forms
        for key in self.readerFormsKeys:
            self.stackedWidget.addWidget(self.readerForms[key])
        
        row = self.newRow()
        row.addWidget(self.stackedWidget)
        
        # select auto by default
        self.inputTypeCombo.setCurrentIndex(self.readerFormsKeys.index("AUTO DETECT"))
        
        # help icon
        row = self.newRow()
        row.RowLayout.addStretch(1)
        
        helpButton = QtGui.QPushButton(QtGui.QIcon(iconPath("Help-icon.png")), "")
        helpButton.setFixedWidth(20)
        helpButton.setFixedHeight(20)
        helpButton.setToolTip("Show help page")
        helpButton.clicked.connect(self.loadHelpPage)
        row.addWidget(helpButton)
        
        self.show()
    
    def openSFTPBrowser(self):
        """
        Open SFTP browser
        
        """
        
        
    
    def loadHelpPage(self):
        """
        Load the help page for this form
        
        """
        self.mainWindow.helpWindow.loadPage("usage/input/file_input.html")
        self.mainWindow.showHelp()
    
    def setWidgetStack(self, index):
        """
        Change load ref stack.
        
        """
        self.stackedWidget.setCurrentIndex(index)
    
    def fileLoaded(self, fileType, state, filename, extension, readerStackIndex):
        """
        Called when a file is loaded
        
        """
        self.parent.file_loaded(state, filename, extension, readerStackIndex)

################################################################################

class SystemsListWidgetItem(QtGui.QListWidgetItem):
    """
    Item that goes in the systems list
    
    """
    def __init__(self, lattice, filename, displayName, stackIndex, abspath, extension):
        super(SystemsListWidgetItem, self).__init__()
        
        self.lattice = lattice
        self.filename = filename
        self.displayName = displayName
        self.stackIndex = stackIndex
        self.abspath = abspath
        self.extension = extension
        
        self.setText("%s (%d atoms)" % (displayName, lattice.NAtoms))
        self.setToolTip(abspath)
    
    def changeDisplayName(self, displayName):
        """
        Change the display name
        
        """
        self.displayName = displayName
        
        self.setText("%s (%d atoms)" % (displayName, self.lattice.NAtoms))
    

################################################################################

class SystemsDialog(QtGui.QDialog):
    """
    Systems dialog
    
    """
    def __init__(self, parent, mainWindow):
        super(SystemsDialog, self).__init__(parent)
        
        self.setWindowTitle("Systems dialog")
        self.setModal(False)
        
        self.iniWinFlags = self.windowFlags()
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.mainToolbar = mainWindow
        self.mainWindow = mainWindow
        
        self.logger = logging.getLogger(__name__)
        
        self.resize(80, 120)
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        # dialog layout
        dialog_layout = QtGui.QVBoxLayout(self)
        
        # box for list of loaded systems
        list_holder = GenericForm(self, None, "Loaded systems")
        list_holder.show()
        dialog_layout.addWidget(list_holder)
        
        # add list widget
        self.systems_list_widget = QtGui.QListWidget(self)
#         self.systems_list_widget.setFixedHeight(60)
        self.systems_list_widget.setSelectionMode(self.systems_list_widget.ExtendedSelection)
        self.systems_list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.systems_list_widget.customContextMenuRequested.connect(self.showListWidgetContextMenu)
        
        row = list_holder.newRow()
        row.addWidget(self.systems_list_widget)
        
        # remove system button
        remove_system = QtGui.QPushButton(QtGui.QIcon(iconPath("list-remove.svg")), "")
        remove_system.setAutoDefault(False)
        remove_system.setToolTip("Remove system")
        remove_system.clicked.connect(self.remove_system)
        
        row = list_holder.newRow()
        row.addWidget(remove_system)
        
        # box for new system stuff
        new_holder = GenericForm(self, None, "New system")
        new_holder.show()
        dialog_layout.addWidget(new_holder)
        
        # load or generate combo
        self.new_type_combo = QtGui.QComboBox()
        self.new_type_combo.addItem("Load system")
        self.new_type_combo.addItem("Generate system")
        self.new_type_combo.currentIndexChanged.connect(self.set_new_system_stack)
        
        row = new_holder.newRow()
        row.addWidget(self.new_type_combo)
        
        # stacked widget
        self.new_system_stack = QtGui.QStackedWidget()
        
        row = new_holder.newRow()
        row.addWidget(self.new_system_stack)
        
        # load input form
        self.load_system_form = LoadSystemForm(self, self.mainWindow, self.mainToolbar)
        self.new_system_stack.addWidget(self.load_system_form)
        
        # generate input form
        self.generate_system_form = GenerateInputForm(self, self.mainWindow, self.mainToolbar)
        self.new_system_stack.addWidget(self.generate_system_form)
        
        # help icon
        helpButton = QtGui.QPushButton(QtGui.QIcon(iconPath("Help-icon.png")), "")
        helpButton.setFixedWidth(20)
        helpButton.setFixedHeight(20)
        helpButton.setToolTip("Show help page")
        helpButton.clicked.connect(self.load_help_page)
        
        # hide button
        hideButton = QtGui.QPushButton("&Hide")
        hideButton.clicked.connect(self.close)
        row = QtGui.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(hideButton)
        row.addStretch(1)
        row.addWidget(helpButton)
        dialog_layout.addLayout(row)
    
    def tmpHide(self):
        """
        Temporarily remove staysOnTop hint
        
        """
        self.setWindowFlags(self.iniWinFlags)
    
    def showAgain(self):
        """
        Readd stayOnTop hint and show
        
        """
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.show()
    
    def showListWidgetContextMenu(self, point):
        """
        Show context menu for listWidgetItem
        
        """
        logger = logging.getLogger(__name__)
        logger.debug("Show list widget context menu")
        
        # position to open menu at
        globalPos = self.systems_list_widget.mapToGlobal(point)
        
        # index of item
        t = self.systems_list_widget.indexAt(point)
        index = t.row()
        
        # item
        item = self.systems_list_widget.item(index)
        
        if item is not None:
            logger.debug("Showing context menu for item at row: %d", index)
            
            # context menu
            menu = QtGui.QMenu(self)
            
            # make actions
            # change display name action
            dnAction = QtGui.QAction("Set display name", self)
            dnAction.setToolTip("Change display name")
            dnAction.setStatusTip("Change display name")
            dnAction.triggered.connect(functools.partial(self.changeDisplayName, index))
            
            # duplicate action
            duplicateAction = QtGui.QAction("Duplicate system", self)
            duplicateAction.setToolTip("Duplicate selected system")
            duplicateAction.setStatusTip("Duplicate selected system")
            duplicateAction.triggered.connect(functools.partial(self.duplicate_system, index))
            
            # remove action
            removeAction = QtGui.QAction("Remove system(s)", self)
            removeAction.setToolTip("Remove selected system(s)")
            removeAction.setStatusTip("Remove selected system(s)")
            removeAction.triggered.connect(self.remove_system)
            
            # reload action
            reloadAction = QtGui.QAction("Reload system(s)", self)
            reloadAction.setToolTip("Reload selected system(s)")
            reloadAction.setStatusTip("Reload selected system(s)")
            reloadAction.triggered.connect(self.reload_system)
            
            # add action
            menu.addAction(dnAction)
            menu.addAction(duplicateAction)
            menu.addAction(reloadAction)
            menu.addAction(removeAction)
            
            # show menu
            menu.exec_(globalPos)
    
    def duplicate_system(self, index):
        """
        Duplicate selected system
        
        """
        self.logger.debug("Duplicating system (%d)", index)
        
        # item
        item = self.systems_list_widget.item(index)
        
        # first we get a new display name
        text, ok = QtGui.QInputDialog.getText(self, 'Display name', "Enter new display name:", text=item.displayName)
        
        if ok:
            text = text.strip()
            
            if not len(text.strip()):
                text = item.displayName
        
            # then we copy and add the new system
            newState = copy.deepcopy(item.lattice)
            
            # stack indexes
            ida, idb = item.stackIndex
            
            self.add_lattice(newState, item.filename, item.extension, ida=ida, idb=idb, displayName=text, allowDuplicate=True)
    
    def changeDisplayName(self, index):
        """
        Change display name
        
        """
        self.logger.debug("Changing display name (%d)", index)
        
        # item
        item = self.systems_list_widget.item(index)
        
        # show dialog
        text, ok = QtGui.QInputDialog.getText(self, 'Display name', "Enter new display name:", text=item.displayName)
        
        if ok and len(text.strip()):
            text = text.strip()
            
            self.logger.debug("  Changing display name to: '%s'", text)
            
            item.changeDisplayName(text)
            self.mainWindow.mainToolbar.changeStateDisplayName(index, text)
    
    def reload_system(self):
        """
        Reload selected systems
        
        """
        self.logger.debug("Reloading selected systems")
        
        if not self.systems_list_widget.count():
            return
        
        items = self.systems_list_widget.selectedItems()
        
        # loops over selected items
        for item in items:
            if item is None:
                continue
            
            # stack index
            ida, idb = item.stackIndex
            
            # check this is not a generated lattice
            if ida != 0:
                self.logger.debug("Cannot reload a generated lattice")
                return
            
            self.logger.info("  Reloading: '%s' (%s)", item.displayName, item.filename)
            self.logger.debug("  Input stack index: %d, %d", ida, idb)
        
            systemsDialog = self.mainWindow.systemsDialog
            
            systemsDialog.new_system_stack.setCurrentIndex(ida)
            in_page = systemsDialog.new_system_stack.currentWidget()
            
            in_page.stackedWidget.setCurrentIndex(idb)
            readerForm = in_page.stackedWidget.currentWidget()
            reader = readerForm.latticeReader
            
            self.logger.debug("  Reader: %s %s", str(readerForm), str(reader))
            
            # read in state
            if reader.requiresRef:
                status, state = reader.readFile(item.abspath, readerForm.currentRefState)
            
            else:
                status, state = reader.readFile(item.abspath)
            
            if status:
                self.logger.error("Reload read file failed with status: %d" % status)
                continue
            
            # set on item
            item.lattice = state
            item.changeDisplayName(item.displayName)
            
            # need index of this item
            t = self.systems_list_widget.indexFromItem(item)
            index = t.row()
            self.logger.debug("  Item index: %d (%s)", index, item.displayName)
            
            # set on pipeline pages
            for pp in self.mainWindow.mainToolbar.pipelineList:
                refIndex, inputIndex = pp.getCurrentStateIndexes()
                
                changed = False
                if refIndex == index:
                    pp.refState = state
                    changed = True
                
                if inputIndex == index:
                    pp.inputState = state
                    changed = True
                
                if changed:
                    pp.runAllFilterLists()
    
    def load_help_page(self):
        """
        Load the help page for this form
        
        """
        self.mainWindow.helpWindow.loadPage("usage/input/index.html")
        self.mainWindow.showHelp()
    
    def file_generated(self, lattice, filename):
        """
        File generated
        
        """
        self.add_lattice(lattice, filename, "dat", allowDuplicate=True)
    
    def file_loaded(self, lattice, filename, extension, readerStackIndex):
        """
        Called after a file had been loaded (or generated too?)
        
        """
        self.add_lattice(lattice, filename, extension, idb=readerStackIndex, ida=0)
    
    def add_lattice(self, lattice, filename, extension, ida=None, idb=None, displayName=None, allowDuplicate=False):
        """
        Add lattice
        
        """
        index = self.systems_list_widget.count()
        
        abspath = os.path.abspath(filename)
        
        if not allowDuplicate:
            abspathList = self.getAbspathList()
            
            if abspath in abspathList:
                self.logger.info("This file has already been loaded (%s): not adding it again", filename)
                
                index = abspathList.index(abspath)
                
                # select this one
                for row in xrange(self.systems_list_widget.count()):
                    self.systems_list_widget.item(row).setSelected(False)
                self.systems_list_widget.item(index).setSelected(True)
                
                return
        
        # stack index
        if ida is None:
            ida = self.new_system_stack.currentIndex()
            page = self.new_system_stack.currentWidget()
        
        if idb is None:
            idb = page.stackedWidget.currentIndex()
        
        stackIndex = (ida, idb)
        
        self.logger.debug("Adding new lattice to systemsList (%d): %s; %d,%d", index, filename, ida, idb)
        
        if displayName is None:
            displayName = os.path.basename(filename)
        
        # item for list
        list_item = SystemsListWidgetItem(lattice, filename, displayName, stackIndex, abspath, extension)
        
        # add to list
        self.systems_list_widget.addItem(list_item)
        
        # select last one added only
        for row in xrange(index):
            self.systems_list_widget.item(row).setSelected(False)
        self.systems_list_widget.item(index).setSelected(True)
        
        # also add lattice to pipeline forms
        self.mainWindow.mainToolbar.addStateOptionToPipelines(displayName)
        
        return index
    
    def getAbspathList(self):
        """
        Return ordered list of lattices
        
        """
        abspathList = []
        for i in xrange(self.systems_list_widget.count()):
            item = self.systems_list_widget.item(i)
            
            abspathList.append(item.abspath)
        
        return abspathList
    
    def getLatticeList(self):
        """
        Return ordered list of lattices
        
        """
        latticeList = []
        for i in xrange(self.systems_list_widget.count()):
            item = self.systems_list_widget.item(i)
            
            latticeList.append(item.lattice)
        
        return latticeList
    
    def getDisplayNames(self):
        """
        Return ordered list of display names
        
        """
        displayNames = []
        for i in xrange(self.systems_list_widget.count()):
            item = self.systems_list_widget.item(i)
            
            displayNames.append(item.displayName)
        
        return displayNames
    
    def set_new_system_stack(self, index):
        """
        Set new system stack
        
        """
        self.new_system_stack.setCurrentIndex(index)
    
    def remove_system(self):
        """
        Remove system(s) from list
        
        """
        if not self.systems_list_widget.count():
            return
        
        items = self.systems_list_widget.selectedItems()
        
        # loops over selected items
        for item in items:
            # get the index
            modelIndex = self.systems_list_widget.indexFromItem(item)
            index = modelIndex.row()
            
            if index < 0:
                continue
            
            # do not allow removal of system that is tagged with ref or input
            # first find out which ones are refs and inputs
            currentStateIndexes = self.mainWindow.mainToolbar.getSelectedStatesFromPipelines()
            
            if index in currentStateIndexes:
                self.mainWindow.displayWarning("Cannot remove state that is currently selected")
                continue
            
            # remove item and delete (not sure if required)
            itemWidget = self.systems_list_widget.takeItem(index)
            del itemWidget
            
            self.mainWindow.mainToolbar.removeStateFromPipelines(index)
            
        # select last one in list
        for row in xrange(self.systems_list_widget.count() - 1):
            self.systems_list_widget.item(row).setSelected(False)
        self.systems_list_widget.item(self.systems_list_widget.count() - 1).setSelected(True)

