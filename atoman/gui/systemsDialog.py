
"""
Data input is handled by the *Input* tab of the main toolbar.
From here you can load or generate a lattice, view loaded lattices, etc.
Once loaded, systems will be added to the "Loaded systems" list.
Various actions can be performed on the loaded systems by right clicking them, such as
removing them from the list.
Note that systems that are currently selected on an analysis pipeline, as either a ref
or input, cannot be removed.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import logging
import functools
import copy

from PySide import QtGui, QtCore
import numpy as np

from ..visutils.utilities import iconPath
from .genericForm import GenericForm
from . import generalReaderForm
from . import latticeGeneratorForms
from . import sftpDialog
from .dialogs import infoDialogs
from .filterList import FilterList
from six.moves import range


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
        self.inputTypeCombo.addItem("SiC 4H (diamond)")
        self.inputTypeCombo.addItem("C_ Graphite")
        self.inputTypeCombo.addItem("C_ Diamond")
        self.inputTypeCombo.addItem("Diamond Indenter")
#         self.inputTypeCombo.addItem("Pyrochlore")
#         self.inputTypeCombo.addItem("6H")
        self.inputTypeCombo.currentIndexChanged.connect(self.setWidgetStack)
        self.inputTypeCombo.setToolTip("Select the type of lattice to generate")

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

        # SiC 4H generator
        self.sic4h_generator = latticeGeneratorForms.SiC4HLatticeGeneratorForm(self, self.mainWindow)
        self.stackedWidget.addWidget(self.sic4h_generator)

        # Carbon Graphite generator
        self.graphite_generator = latticeGeneratorForms.GraphiteLatticeGeneratorForm(self, self.mainWindow)
        self.stackedWidget.addWidget(self.graphite_generator)

        # Carbon Diamond generator
        self.diamond_generator = latticeGeneratorForms.DiamondLatticeGeneratorForm(self, self.mainWindow)
        self.stackedWidget.addWidget(self.diamond_generator)

        # Diamond Indenter generator
        self.DiamondIndenter_generator = latticeGeneratorForms.DiamondIndenterGeneratorForm(self, self.mainWindow)
        self.stackedWidget.addWidget(self.DiamondIndenter_generator)

        self.show()

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

        self.systemsDialog = parent
        self.mainWindow = mainWindow
        self.mainToolbar = mainToolbar

        # sftp browser
        if sftpDialog.PARAMIKO_LOADED:
            self.sftp_browser = sftpDialog.SFTPBrowserDialog(self.mainWindow, parent=self)
        else:
            self.sftp_browser = None

        # reader form
        self.readerForm = generalReaderForm.GeneralLatticeReaderForm(self, self.mainToolbar, self.mainWindow)
        row = self.newRow()
        row.addWidget(self.readerForm)

        self.show()

    def fileLoaded(self, state, filename, fileFormat, sftpPath, linked):
        """
        Called when a file is loaded

        """
        # default PBC
        state.PBC[:] = self.mainWindow.preferences.generalForm.defaultPBC[:]

        # load file
        self.systemsDialog.file_loaded(state, filename, fileFormat, sftpPath, linked)

################################################################################

class SystemsListWidgetItem(QtGui.QListWidgetItem):
    """
    Item that goes in the systems list

    """
    def __init__(self, lattice, filename, displayName, abspath, fileFormat, linkedLattice, fromSFTP, generated=False):
        super(SystemsListWidgetItem, self).__init__()

        self.lattice = lattice
        self.filename = filename
        self.displayName = displayName
        self.fileFormat = fileFormat
        self.abspath = abspath
        self.linkedLattice = linkedLattice
        self.infoDialog = None
        self.generated = generated
        self.fromSFTP = fromSFTP

        zip_exts = ('.bz2', '.gz')
        root, ext = os.path.splitext(filename)
        if ext in zip_exts:
            ext = os.path.splitext(root)[1]
        self.extension = ext

        self.setText("%s (%d atoms)" % (displayName, lattice.NAtoms))
        self.setToolTip(abspath)

        # files holding data for this lattice
        # key is the name to go in the scalars/vectors dict,
        # value is the filename
        self.vectorDataFiles = {}
        self.scalarDataFiles = {}

    def changeDisplayName(self, displayName):
        """
        Change the display name

        """
        self.displayName = displayName

        self.setText("%s (%d atoms)" % (displayName, self.lattice.NAtoms))


################################################################################

class SystemsDialog(QtGui.QWidget):
    """
    Systems dialog

    """
    def __init__(self, parent, mainWindow):
        super(SystemsDialog, self).__init__(parent)

        self.mainToolbar = mainWindow
        self.mainWindow = mainWindow

        self.logger = logging.getLogger(__name__)

        # dialog layout
        dialog_layout = QtGui.QVBoxLayout()
        dialog_layout.setContentsMargins(0,0,0,0)
        self.setLayout(dialog_layout)

        # box for list of loaded systems
        list_holder = QtGui.QGroupBox("Loaded systems")
        list_holder.setAlignment(QtCore.Qt.AlignHCenter)
        vbox = QtGui.QVBoxLayout()
        vbox.setAlignment(QtCore.Qt.AlignHCenter)
        vbox.setSpacing(0)
        vbox.setContentsMargins(0,0,0,0)
        list_holder.setLayout(vbox)
        dialog_layout.addWidget(list_holder)

        # add list widget
        self.systems_list_widget = QtGui.QListWidget(self)
        self.systems_list_widget.setSelectionMode(self.systems_list_widget.ExtendedSelection)
        self.systems_list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.systems_list_widget.customContextMenuRequested.connect(self.showListWidgetContextMenu)
        self.systems_list_widget.setFixedHeight(120)
        vbox.addWidget(self.systems_list_widget)

        # remove system button
        removeSystemButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/list-remove.png")), "")
        removeSystemButton.setAutoDefault(False)
        removeSystemButton.setToolTip("Remove system")
        removeSystemButton.clicked.connect(self.removeSystem)
        removeSystemButton.setFixedWidth(60)
        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(removeSystemButton)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        # load input form
        self.load_system_form = LoadSystemForm(self, self.mainWindow, self.mainToolbar)
        dialog_layout.addWidget(self.load_system_form)

        # generate input form
        self.generate_system_form = GenerateInputForm(self, self.mainWindow, self.mainToolbar)
        dialog_layout.addWidget(self.generate_system_form)

        dialog_layout.addStretch(1)

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

            # show system info
            showInfoAction = QtGui.QAction("Show information", self)
            showInfoAction.setToolTip("Show system information")
            showInfoAction.setStatusTip("Show system information")
            showInfoAction.triggered.connect(functools.partial(self.showSystemInformation, index))

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
            removeAction.triggered.connect(self.removeSystem)

            # reload action
            reloadAction = QtGui.QAction("Reload system(s)", self)
            reloadAction.setToolTip("Reload selected system(s)")
            reloadAction.setStatusTip("Reload selected system(s)")
            reloadAction.triggered.connect(self.reload_system)

            # load scalar data action
            loadScalarAction = QtGui.QAction("Load scalar data", self)
            loadScalarAction.setToolTip("Load scalar data from a file")
            loadScalarAction.triggered.connect(functools.partial(self.loadScalarData, index))

            # load vector data action
            loadVectorAction = QtGui.QAction("Load vector data", self)
            loadVectorAction.setToolTip("Load vector data from a file")
            loadVectorAction.triggered.connect(functools.partial(self.loadVectorData, index))

            # add action
            menu.addAction(showInfoAction)
            menu.addAction(dnAction)
            menu.addAction(loadScalarAction)
            menu.addAction(loadVectorAction)
            menu.addAction(duplicateAction)
            menu.addAction(reloadAction)
            menu.addAction(removeAction)

            # show menu
            menu.exec_(globalPos)

    def loadScalarData(self, index):
        """
        Load scalar data for associated Lattice

        """
        self.logger.debug("Loading scalar data from file")

        # item
        item = self.systems_list_widget.item(index)

        # lattice
        lattice = item.lattice

        # open a dialog to get the name of scalar data and the filename
        inputdiag = QtGui.QInputDialog(self)
        inputdiag.setOkButtonText("Select file")
        inputdiag.setLabelText("Name:")
        inputdiag.setWindowTitle("Load scalar data")
        inputdiag.setInputMode(QtGui.QInputDialog.TextInput)

        scalarName = None
        while scalarName is None:
            # open dialog
            retcode = inputdiag.exec_()

            if retcode == QtGui.QDialog.Rejected:
                break

            # get text
            text = inputdiag.textValue()
            text = text.strip()

            if not len(text):
                break

            # check name is not the same as a filter list item
            if text in FilterList.defaultFilters:
                # show warning
                self.mainWindow.displayWarning("The chosen name is reserved ({0})".format(text))

            # check the name is not already used
            elif text in lattice.scalarsDict:
                # show warning
                self.mainWindow.displayWarning("The chosen name already exists ({0})".format(text))

            else:
                scalarName = text

        if scalarName is not None:
            self.logger.debug("Got name for scalar data: '%s'", scalarName)

            # get filename
            filename = QtGui.QFileDialog.getOpenFileName(self, "Select file containing scalar data: '{0}'".format(scalarName), os.getcwd())[0]
            filename = str(filename)

            if len(filename):
                # read file
                self.logger.debug("Got file name containing scalar data: '%s'", filename)

                # load file...
                with open(filename) as f:
                    scalars = []
                    try:
                        for line in f:
                            scalars.append(float(line))

                    except:
                        self.logger.error("Error reading scalar file")
                        self.mainWindow.displayError("Could not read scalar file.")
                        return

                if len(scalars) != lattice.NAtoms:
                    self.logger.error("The scalar data is the wrong length")
                    self.mainWindow.displayError("The scalar data is the wrong length")
                    return

                # convert to numpy array
                scalars = np.asarray(scalars, dtype=np.float64)

                # store on lattice
                lattice.scalarsDict[scalarName] = scalars
                lattice.scalarsFiles[scalarName] = filename

                self.logger.info("Added '%s' scalars to '%s'", scalarName, item.displayName)

                # refresh vectors settings (should just do it on pipelines that have this system as the input state)
                for pp in self.mainWindow.mainToolbar.pipelineList:
                    for filterList in pp.filterLists:
                        filterList.refreshAvailableFilters()

    def loadVectorData(self, index):
        """
        Load vector data for associated Lattice

        """
        self.logger.debug("Loading vector data from file")

        # item
        item = self.systems_list_widget.item(index)

        # lattice
        lattice = item.lattice

        # open a dialog to get the name of scalar data and the filename
        inputdiag = QtGui.QInputDialog(self)
        inputdiag.setOkButtonText("Select file")
        inputdiag.setLabelText("Name:")
        inputdiag.setWindowTitle("Load vector data")
        inputdiag.setInputMode(QtGui.QInputDialog.TextInput)

        vectorName = None
        while vectorName is None:
            # open dialog
            retcode = inputdiag.exec_()

            if retcode == QtGui.QDialog.Rejected:
                break

            # get text
            text = inputdiag.textValue()
            text = text.strip()

            if not len(text):
                break

            # check name is not the same as a filter list item
            if text in FilterList.defaultFilters:
                # show warning
                self.mainWindow.displayWarning("The chosen name is reserved ({0})".format(text))

            # check the name is not already used
            elif text in lattice.vectorsDict:
                # show warning
                self.mainWindow.displayWarning("The chosen name already exists ({0})".format(text))

            else:
                vectorName = str(text)

        if vectorName is not None:
            self.logger.debug("Got name for vector data: '%s'", vectorName)

            # get filename
            filename = QtGui.QFileDialog.getOpenFileName(self, "Select file containing vector data: '{0}'".format(vectorName), os.getcwd())[0]
            filename = str(filename)

            if len(filename):
                # read file
                self.logger.debug("Got file name containing vector data: '%s'", filename)

                # load file...
                with open(filename) as f:
                    vectors = []
                    try:
                        for line in f:
                            array = line.split()
                            array[0] = float(array[0])
                            array[1] = float(array[1])
                            array[2] = float(array[2])

                            vectors.append(array)

                    except:
                        self.logger.error("Error reading vector file")
                        self.mainWindow.displayError("Could not read vector file.")
                        return

                if len(vectors) != lattice.NAtoms:
                    self.logger.error("The vector data is the wrong length")
                    self.mainWindow.displayError("The vector data is the wrong length")
                    return

                # convert to numpy array
                vectors = np.asarray(vectors, dtype=np.float64)
                assert vectors.shape[0] == lattice.NAtoms and vectors.shape[1] == 3

                # store on lattice
                lattice.vectorsDict[vectorName] = vectors
                lattice.vectorsFiles[vectorName] = filename

                self.logger.info("Added '%s' vectors to '%s'", vectorName, item.displayName)

                # refresh vectors settings (should just do it on pipelines that have this system as the input state)
                for pp in self.mainWindow.mainToolbar.pipelineList:
                    for filterList in pp.filterLists:
                        filterList.vectorsOptions.refresh()

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

    def showSystemInformation(self, index):
        """
        Show info window about a system

        """
        # item
        item = self.systems_list_widget.item(index)

        # create dialog
        if item.infoDialog is None:
            dlg = infoDialogs.SystemInfoWindow(item, parent=self)
            item.infoDialog = dlg
        else:
            dlg = item.infoDialog

        # show
        dlg.hide()
        dlg.show()

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

            # check this is not a generated lattice
            if item.generated:
                self.logger.warning("Cannot reload a generated lattice")
                self.mainWindow.displayWarning("Cannot reload a generated lattice")
                return

            self.logger.info("  Reloading: '%s' (%s)", item.displayName, item.filename)

            # reader
            systemsDialog = self.mainWindow.systemsDialog
            loadPage = systemsDialog.load_system_form
            readerForm = loadPage.readerForm
            reader = readerForm.latticeReader

            # read in state
            status, state = reader.readFile(item.abspath, item.fileFormat, linkedLattice=item.linkedLattice)
            if status:
                self.logger.error("Reload read file failed with status: %d" % status)
                continue

            # set on item
            item.lattice = state
            item.changeDisplayName(item.displayName)

            # remove info window
            if item.infoDialog is not None:
                item.infoDialog.accept()
                item.infoDialog.close()
                item.infoDialog = None

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
#                     pp.runAllFilterLists()
                    pp.postInputLoaded()

    def load_help_page(self):
        """
        Load the help page for this form

        """
        self.mainWindow.showHelp(relativeUrl="usage/input/index.html")

    def file_generated(self, lattice, filename):
        """
        File generated

        """
        self.add_lattice(lattice, filename, allowDuplicate=True, generated=True)

    def file_loaded(self, lattice, filename, fileFormat, sftpPath, linked):
        """
        Called after a file had been loaded (or generated too?)

        """
        self.add_lattice(lattice, filename, fileFormat=fileFormat, sftpPath=sftpPath, linkedLattice=linked)

    def add_lattice(self, lattice, filename, fileFormat=None, displayName=None, allowDuplicate=False, sftpPath=None, linkedLattice=None, generated=False):
        """
        Add lattice

        """
        index = self.systems_list_widget.count()

        if sftpPath is None:
            abspath = os.path.abspath(filename)
            fromSFTP = False
        else:
            abspath = sftpPath
            fromSFTP = True

        if not allowDuplicate:
            abspathList = self.getAbspathList()

            if abspath in abspathList:
                self.logger.warning("This file has already been loaded (%s)", abspath)

                index = abspathList.index(abspath)

                # select this one
                for row in range(self.systems_list_widget.count()):
                    self.systems_list_widget.item(row).setSelected(False)
                self.systems_list_widget.item(index).setSelected(True)

                return

        if displayName is None:
            zip_exts = ('.gz', '.bz2')
            displayName = os.path.basename(filename)
            if os.path.splitext(displayName)[1] in zip_exts:
                displayName = os.path.splitext(displayName)[0]

        self.logger.debug("Adding new lattice to systemsList (%d): %s", index, filename)
        self.logger.debug("Abspath is: '%s'", abspath)
        self.logger.debug("Display name is: '%s'", displayName)

        # item for list
        list_item = SystemsListWidgetItem(lattice, filename, displayName, abspath, fileFormat, linkedLattice, fromSFTP, generated=generated)

        # add to list
        self.systems_list_widget.addItem(list_item)

        # select last one added only
        for row in range(index):
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
        for i in range(self.systems_list_widget.count()):
            item = self.systems_list_widget.item(i)

            abspathList.append(item.abspath)

        return abspathList

    def getLatticeList(self):
        """
        Return ordered list of lattices

        """
        latticeList = []
        for i in range(self.systems_list_widget.count()):
            item = self.systems_list_widget.item(i)

            latticeList.append(item.lattice)

        return latticeList

    def getLatticesByFormat(self, formatName):
        """
        Return a list of loaded systems with the given format

        """
        lattices = []
        for i in range(self.systems_list_widget.count()):
            item = self.systems_list_widget.item(i)
            if item.fileFormat is not None and item.fileFormat.name == formatName:
                lattices.append((item.displayName, item.lattice))

        return lattices

    def getDisplayNames(self):
        """
        Return ordered list of display names

        """
        displayNames = []
        for i in range(self.systems_list_widget.count()):
            item = self.systems_list_widget.item(i)

            displayNames.append(item.displayName)

        return displayNames

    def set_new_system_stack(self, index):
        """
        Set new system stack

        """
        self.new_system_stack.setCurrentIndex(index)

    def removeSystem(self):
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
        for row in range(self.systems_list_widget.count() - 1):
            self.systems_list_widget.item(row).setSelected(False)
        self.systems_list_widget.item(self.systems_list_widget.count() - 1).setSelected(True)

