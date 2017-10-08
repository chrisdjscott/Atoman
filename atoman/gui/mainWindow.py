# -*- coding: utf-8 -*-

"""
The main window class

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
import shutil
import platform
import tempfile
import traceback
import logging
import datetime

from PyQt5 import QtGui, QtCore, QtWidgets

import vtk
import numpy as np
import matplotlib
import scipy

from ..visutils.utilities import iconPath, resourcePath, dataPath
from ..system import atoms
from ..system.atoms import elements
from . import toolbar as toolbarModule
from . import preferences
from . import rendererSubWindow
from . import systemsDialog
from . import viewPorts
from .dialogs import simpleDialogs
from .dialogs import bondEditor
from .dialogs import elementEditor
from .dialogs import consoleWindow
from .. import _version


################################################################################
class MainWindow(QtWidgets.QMainWindow):
    """
    The main window.

    """
    configDir = os.path.join(os.environ["HOME"], ".atoman")

    def __init__(self, desktop, parent=None, testing=False):
        super(MainWindow, self).__init__(parent)

        # logger
        self.logger = logging.getLogger(__name__)

        # first time show called
        self.testingFlag = testing
        self.firstShow = True

        # QDesktopWidget: gives access to screen geometry, which screen we're displayed on, etc...
        self.desktop = desktop

        # initialise user interface
        self.initUI()

        # start threadpool
        self.threadPool = QtCore.QThreadPool(self)
        if self.threadPool.maxThreadCount() < 2:
            self.threadPool.setMaxThreadCount(2)

        # set focus
        self.setFocus()

    def initUI(self):
        """Initialise the interface."""
        logger = self.logger
        logger.debug("Initialising user interface")

        # defaults (TODO: remove this)
        self.refLoaded = False

        # MD code resource (not currently used!?)
        logger.debug("MD resource path: %s (exists %s)", resourcePath("lbomd.IN", dirname="md_input"),
                     os.path.exists(resourcePath("lbomd.IN", dirname="md_input")))

        # get settings object
        settings = QtCore.QSettings()

        # initial directory
        currentDir = str(settings.value("mainWindow/currentDirectory", ""))
        logger.debug("Settings dir: '%s'", currentDir)

        if hasattr(sys, "_MEIPASS"):
            if not len(currentDir) or not os.path.exists(currentDir):
                # change to home directory if running from pyinstaller bundle
                currentDir = os.environ.get("HOME")
                logger.debug("Change dir $HOME: '%s'", currentDir)
        else:
            currentDir = os.getcwd()
            logger.debug("Use CWD: '%s'", currentDir)

        os.chdir(currentDir)

        # toolbar size (fixed)
        self.mainToolbarWidth = 350
        self.mainToolbarHeight = 460

        # default window widget size
        self.renderWindowWidth = 760 * 1.2
        self.renderWindowHeight = 715 * 1.2

        # default size
        windowWidth = self.renderWindowWidth + self.mainToolbarWidth
        windowHeight = self.renderWindowHeight

        self.defaultWindowWidth = windowWidth
        self.defaultWindowHeight = windowHeight

        # resize
        self.resize(settings.value("mainWindow/size", QtCore.QSize(windowWidth, windowHeight)))

        # location
        self.centre()

        self.setWindowTitle("Atoman")

        # create temporary directory for working in (needs to force tmp on mac so POV-Ray can run in it)
        self.tmpDirectory = tempfile.mkdtemp(prefix="atoman-", dir="/tmp")

        # console window for logging output to
        self.console = consoleWindow.ConsoleWindow(self)

        # image viewer
        self.imageViewer = simpleDialogs.ImageViewer(self, parent=self)

        # preferences dialog
        self.preferences = preferences.PreferencesDialog(self, parent=self)

        # bonds editor
        self.bondsEditor = bondEditor.BondEditorDialog(parent=self)

        # add file actions
        exitAction = self.createAction("Exit", self.close, "Ctrl-Q", "oxygen/application-exit.png", "Exit application")
        openFileAction = self.createAction("Open file", slot=self.showOpenFileDialog, icon="oxygen/document-open.png",
                                           tip="Open file")
        openRemoteFileAction = self.createAction("Open remote file", slot=self.showOpenRemoteFileDialog,
                                                 icon="oxygen/document-open-remote.png", tip="Open remote file")
        openCWDAction = self.createAction("Open CWD", slot=self.openCWD, icon="oxygen/folder-open.png",
                                          tip="Open current working directory")
        exportElementsAction = self.createAction("Export elements", slot=self.exportElements,
                                                 icon="oxygen/document-export", tip="Export element properties")
        importElementsAction = self.createAction("Import elements", slot=self.importElements,
                                                 icon="oxygen/document-import.png", tip="Import element properties")
        resetElementsAction = self.createAction("Reset elements", slot=self.resetElements, icon="oxygen/edit-undo.png",
                                                tip="Reset elements settings to default values")
        exportBondsAction = self.createAction("Export bonds", slot=self.exportBonds,
                                              icon="oxygen/document-export.png", tip="Export bonds file")
        importBondsAction = self.createAction("Import bonds", slot=self.importBonds,
                                              icon="oxygen/document-import.png", tip="Import bonds file")
        resetBondsAction = self.createAction("Reset bonds", slot=self.resetBonds, icon="oxygen/edit-undo.png",
                                             tip="Reset bonds settings to default values")
        showImageViewerAction = self.createAction("Image viewer", slot=self.showImageViewer,
                                                  icon="oxygen/applications-graphics.png", tip="Show image viewer")
        showPreferencesAction = self.createAction("Preferences", slot=self.showPreferences,
                                                  icon="oxygen/configure.png", tip="Show preferences window")
        changeCWDAction = self.createAction("Change CWD", slot=self.changeCWD, icon="oxygen/folder-new.png",
                                            tip="Change current working directory")

        # add file menu
        fileMenu = self.menuBar().addMenu("&File")
        self.addActions(fileMenu, (openFileAction, openRemoteFileAction,
                                   openCWDAction, changeCWDAction, None, exitAction))

        # settings menu
        settingsMenu = self.menuBar().addMenu("&Settings")
        self.addActions(settingsMenu, (importElementsAction, exportElementsAction, resetElementsAction,
                                       importBondsAction, exportBondsAction, resetBondsAction))

        # button to show console window
        openConsoleAction = self.createAction("Console", self.showConsole, None, "oxygen/utilities-log-viewer.png",
                                              "Show console window")

        # element editor action
        openElementEditorAction = self.createAction("Element editor", slot=self.openElementEditor,
                                                    icon="other/periodic-table-icon.png", tip="Show element editor")

        # open bonds editor action
        openBondsEditorAction = self.createAction("Bonds editor", slot=self.openBondsEditor, icon="other/molecule1.png",
                                                  tip="Show bonds editor")

        # default window size action
        defaultWindowSizeAction = self.createAction("Default size", slot=self.defaultWindowSize,
                                                    icon="oxygen/view-restore.png", tip="Resize window to default size")

        # add view menu
        viewMenu = self.menuBar().addMenu("&View")
        self.addActions(viewMenu, (openConsoleAction, showImageViewerAction, openElementEditorAction,
                        openBondsEditorAction, showPreferencesAction))

        # add window menu
        windowMenu = self.menuBar().addMenu("&Window")
        self.addActions(windowMenu, (defaultWindowSizeAction,))

        # add file toolbar
        fileToolbar = self.addToolBar("File")
        fileToolbar.addAction(exitAction)
        fileToolbar.addSeparator()
        fileToolbar.addAction(openFileAction)
        fileToolbar.addAction(openRemoteFileAction)
        fileToolbar.addAction(openCWDAction)
        fileToolbar.addAction(changeCWDAction)
        fileToolbar.addSeparator()

        # util tool bar
        viewToolbar = self.addToolBar("Utilities")
        viewToolbar.addAction(openConsoleAction)
        viewToolbar.addAction(showImageViewerAction)
        viewToolbar.addAction(openElementEditorAction)
        viewToolbar.addAction(openBondsEditorAction)
        viewToolbar.addAction(showPreferencesAction)
        viewToolbar.addSeparator()

        # vis tool bar
        visToolbar = self.addToolBar("Visualisation")
        numViewPortsCombo = QtWidgets.QComboBox()
        numViewPortsCombo.addItem("1")
        numViewPortsCombo.addItem("2")
        numViewPortsCombo.addItem("4")
        numViewPortsCombo.currentIndexChanged[str].connect(self.numViewPortsChanged)
        visToolbar.addWidget(QtWidgets.QLabel("View ports:"))
        visToolbar.addWidget(numViewPortsCombo)
        visToolbar.addSeparator()

        # add about action
        aboutAction = self.createAction("About Atoman", slot=self.aboutMe, icon="oxygen/help-about.png",
                                        tip="About Atoman")

        helpAction = self.createAction("Atoman Help", slot=self.showHelp, icon="oxygen/help-browser.png",
                                       tip="Show help window (opens in external browser)")

        # add help toolbar
        helpToolbar = self.addToolBar("Help")
        helpToolbar.addAction(aboutAction)
        helpToolbar.addAction(helpAction)

        helpMenu = self.menuBar().addMenu("&Help")
        self.addActions(helpMenu, (aboutAction, helpAction))

        # add cwd to status bar
        self.currentDirectoryLabel = QtWidgets.QLabel("")
        self.updateCWD()
        sb = QtWidgets.QStatusBar()
        self.setStatusBar(sb)
        self.progressBar = QtWidgets.QProgressBar(self.statusBar())
        self.statusBar().addPermanentWidget(self.progressBar)
        self.statusBar().addPermanentWidget(self.currentDirectoryLabel)
        self.hideProgressBar()

        # dict of currently loaded systems
        self.loaded_systems = {}

        # systems dialog
        self.systemsDialog = systemsDialog.SystemsDialog(self, self)

        # element editor
        self.elementEditor = elementEditor.ElementEditor(parent=self)

        # view ports / renderer windows
        self.rendererWindows = []  # TODO: remove
        self.viewPorts = viewPorts.ViewPortsWidget(parent=self)
        self.numViewPortsChanged(int(numViewPortsCombo.currentText()))
        self.setCentralWidget(self.viewPorts)

        # add the main tool bar
        self.mainToolbar = toolbarModule.MainToolbar(self, self.mainToolbarWidth, self.mainToolbarHeight)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.mainToolbar)

        self.setStatus('Ready')

    def showOpenRemoteFileDialog(self):
        """
        Open remote file

        """
        self.systemsDialog.load_system_form.readerForm.openSFTPBrowser()

    def showOpenFileDialog(self):
        """
        Open file

        """
        self.systemsDialog.load_system_form.readerForm.openFileDialog()

    def defaultWindowSize(self):
        """
        Resize window to default size

        """
        self.resize(self.defaultWindowWidth, self.defaultWindowHeight)

    def changeCWD(self):
        """
        Change current working directory...

        """
        new_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "New working directory", os.getcwd())

        logging.debug("Changing directory: '%s'", new_dir)

        if new_dir and os.path.isdir(new_dir):
            os.chdir(new_dir)
            self.updateCWD()

    def rendererWindowActivated(self, sw):
        """
        Sub window activated. (TEMPORARY)

        """
        pass

    def numViewPortsChanged(self, num_str):
        """Update the number of view ports."""
        self.viewPorts.numViewPortsChanged(int(num_str))

        self.rendererWindows = self.viewPorts.getViewPorts()
        for rw in self.rendererWindows:
            rw.outputDialog.imageTab.imageSequenceTab.refreshLinkedRenderers()

    def showPreferences(self):
        """
        Show preferences window.

        """
        self.preferences.hide()
        self.preferences.show()

    def showImageViewer(self):
        """
        Show the image viewer.

        """
        self.imageViewer.hide()
        self.imageViewer.show()

    def openBondsEditor(self):
        """
        Open bonds editor

        """
        self.bondsEditor.show()

    def openElementEditor(self):
        """
        Open element editor.

        """
        self.elementEditor.show()

    def importElements(self):
        """
        Import element properties file.

        """
        msg = "This will overwrite the current element properties file. You should create a backup first!\n\n"
        msg += "Do you wish to continue?"
        reply = QtWidgets.QMessageBox.question(self, "Message", msg,
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            # open file dialog
            title = "Atoman - Import element properties"
            fname = QtWidgets.QFileDialog.getOpenFileName(self, title, ".", "IN files (*.IN)")[0]

            if fname:
                self.logger.info("Importing elements settings from '%s'", fname)

                # read in new file
                elements.read(fname)

                # overwrite current file
                elements.write(dataPath("atoms.IN"))

                # set on Lattice objects too
                self.inputState.refreshElementProperties()
                self.refState.refreshElementProperties()

    def exportElements(self):
        """
        Export element properties to file.

        """
        fname = os.path.join(".", "atoms-exported.IN")
        fname = QtWidgets.QFileDialog.getSaveFileName(self, "Atoman - Export element properties", fname,
                                                      "IN files (*.IN)",
                                                      options=QtWidgets.QFileDialog.DontUseNativeDialog)[0]

        if fname:
            if "." not in fname or fname[-3:] != ".IN":
                fname += ".IN"

            self.logger.info("Exporting elements settings to '%s'", fname)
            elements.write(fname)

    def resetElements(self):
        """Reset elements settings."""
        msg = "This will overwrite the current element properties file. You should create a backup first!\n\n"
        msg += "Do you wish to continue?"
        reply = QtWidgets.QMessageBox.question(self, "Message", msg,
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            atoms.resetAtoms()

    def resetBonds(self):
        """Reset bonds settings."""
        msg = "This will overwrite the current bonds file. You should create a backup first!\n\n"
        msg += "Do you wish to continue?"
        reply = QtWidgets.QMessageBox.question(self, "Message", msg,
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            atoms.resetBonds()

    def importBonds(self):
        """
        Import bonds file.

        """
        msg = "This will overwrite the current bonds file. You should create a backup first!\n\n"
        msg += "Do you wish to continue?"
        reply = QtWidgets.QMessageBox.question(self, "Message", msg,
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            # open file dialog
            fname = QtWidgets.QFileDialog.getOpenFileName(self, "Atoman - Import bonds file", ".", "IN files (*.IN)",
                                                          options=QtWidgets.QFileDialog.DontUseNativeDialog)[0]

            if fname:
                self.logger.info("Import bonds settings from '%s'", fname)

                # read in new file
                elements.readBonds(fname)

                # overwrite current file
                elements.writeBonds(dataPath("bonds.IN"))

                self.setStatus("Imported bonds file")

    def exportBonds(self):
        """
        Export bonds file.

        """
        fname = os.path.join(".", "bonds-exported.IN")

        fname = QtWidgets.QFileDialog.getSaveFileName(self, "Atoman - Export bonds file", fname, "IN files (*.IN)",
                                                      options=QtWidgets.QFileDialog.DontUseNativeDialog)[0]

        if fname:
            if "." not in fname or fname[-3:] != ".IN":
                fname += ".IN"

            self.logger.info("Exporting bonds settings to '%s'", fname)
            elements.writeBonds(fname)

            self.setStatus("Bonds file exported")

    def openCWD(self):
        """
        Open current working directory.

        """
        dirname = os.getcwd()
        osname = platform.system()
        if osname == "Linux":
            os.system("xdg-open '%s'" % dirname)
        elif osname == "Darwin":
            os.system("open '%s'" % dirname)
        elif osname == "Windows":
            os.startfile(dirname)

    def centre(self):
        """
        Centre the window.

        """
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def showConsole(self):
        """
        Open the console window

        """
        self.console.show()

    def showHelp(self, relativeUrl=None):
        """
        Show the help window.

        """
        baseUrl = 'https://chrisdjscott.github.io/Atoman/'
        if relativeUrl is not None and relativeUrl:
            url = QtCore.QUrl(os.path.join(baseUrl, relativeUrl))
        else:
            url = QtCore.QUrl(baseUrl)
        self.logger.debug("Opening help url: {0}".format(url.toString()))
        QtGui.QDesktopServices.openUrl(url)

    def renderWindowClosed(self):
        """
        A render window has been closed.

        """
        i = 0
        while i < len(self.rendererWindows):
            rw = self.rendererWindows[i]

            if rw.closed:
                self.rendererWindows.pop(i)
                self.rendererWindowsSubWin.pop(i)

            else:
                i += 1

        for rw in self.rendererWindows:
            rw.outputDialog.imageTab.imageSequenceTab.refreshLinkedRenderers()

    def confirmCloseEvent(self):
        """
        Show a dialog to confirm closeEvent.

        """
        dlg = simpleDialogs.ConfirmCloseDialog(self)

        close = False
        clearSettings = False

        reply = dlg.exec_()

        if reply:
            close = True

            if dlg.clearSettingsCheck.isChecked():
                clearSettings = True

        return close, clearSettings

    def closeEvent(self, event):
        """
        Catch attempt to close

        """
        if self.testingFlag:
            event.accept()

        else:
            close, clearSettings = self.confirmCloseEvent()

            if close:
                self.tidyUp()

                if clearSettings:
                    self.clearSettings()

                else:
                    self.saveSettings()

                event.accept()

            else:
                event.ignore()

    def clearSettings(self):
        """
        Clear settings.

        """
        # settings object
        settings = QtCore.QSettings()

        settings.clear()

    def saveSettings(self):
        """
        Save settings before exit.

        """
        # settings object
        settings = QtCore.QSettings()

        # store current working directory
        settings.setValue("mainWindow/currentDirectory", os.getcwd())

        # window size
        settings.setValue("mainWindow/size", self.size())

    def tidyUp(self):
        """
        Tidy up before close application

        """
        shutil.rmtree(self.tmpDirectory)
        self.console.accept()
        self.threadPool.waitForDone()

    def hideProgressBar(self):
        """
        Hide the progress bar

        """
        self.progressBar.hide()
        self.progressBar.reset()
        self.setStatus("Finished")

    def updateProgress(self, n, nmax, message):
        """
        Update progress bar

        """
        self.progressBar.show()
        self.progressBar.setRange(0, nmax)
        self.progressBar.setValue(n)
        self.setStatus(message)
        QtWidgets.QApplication.processEvents()

    def setStatus(self, message):
        """
        Set temporary status in status bar

        """
        self.statusBar().showMessage(self.tr(message))

    def updateCWD(self):
        """
        Updates the CWD label in the status bar.

        """
        dirname = os.getcwd()

        self.currentDirectoryLabel.setText("CWD: '%s'" % dirname)
        self.imageViewer.changeDir(dirname)

    def readLBOMDIN(self):
        """
        Try to read sim identity and PBCs from lbomd.IN

        """
        logger = logging.getLogger(__name__)

        if os.path.exists("lbomd.IN"):
            f = open("lbomd.IN")

            try:
                f.readline()
                f.readline()
                f.readline()
                f.readline()

                line = f.readline().strip()
                array = line.split()
                try:
                    PBC = [0] * 3
                    PBC[0] = int(array[0])
                    PBC[1] = int(array[1])
                    PBC[2] = int(array[2])

                except IndexError:
                    logger.warning("Index error 2 (check lbomd.IN format)")

            except Exception:
                err = "Read lbomd.IN failed with error:\n\n%s" % "".join(traceback.format_exception(*sys.exc_info()))
                self.displayError(err)

            finally:
                f.close()

    def displayWarning(self, message):
        """
        Display warning message.

        """
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setText(message)
        msgBox.setWindowFlags(msgBox.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.exec_()

    def displayError(self, message):
        """
        Display error message

        """
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setText(message)
        msgBox.setWindowFlags(msgBox.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.setIcon(QtWidgets.QMessageBox.Critical)
        msgBox.exec_()

    def aboutMe(self):
        """
        Display about message.

        """
        msgBox = QtWidgets.QMessageBox(self)

        # get the version right
        version = _version.get_versions()['version']

        # construct paragraph with software versions
        softline = "Python %s - Qt %s - PyQt5 %s - VTK %s" % (platform.python_version(), QtCore.QT_VERSION_STR,
                                                              QtCore.PYQT_VERSION_STR, vtk.vtkVersion.GetVTKVersion())
        softline += " - NumPy %s - SciPy %s - Matplotlib %s" % (np.__version__, scipy.__version__,
                                                                matplotlib.__version__)

        # add paramiko if available
        try:
            import paramiko

        except ImportError:
            pass

        else:
            softline += " - paramiko %s" % paramiko.__version__

        softline += " on %s" % platform.system()

        msgBox.setText("""<p><b>Atoman</b> %s</p>
                          <p>Copyright &copy; %d Chris Scott</p>
                          <p>This application can be used to visualise atomistic simulations.</p>
                          <p>GUI based on <a href="http://sourceforge.net/projects/avas/">AVAS</a>
                             by Marc Robinson.</p>
                          <p>%s</p>""" % (version, datetime.date.today().year, softline))

        msgBox.setWindowFlags(msgBox.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
        msgBox.exec_()

#         dlg = dialogs.AboutMeDialog(parent=self)
#         dlg.exec_()

    def createAction(self, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False):
        """
        Create an action

        """
        action = QtWidgets.QAction(text, self)

        if icon is not None:
            action.setIcon(QtGui.QIcon(iconPath(icon)))

        if shortcut is not None:
            action.setShortcut(shortcut)

        if tip is not None:
            action.setToolTip("<p>{0}</p>".format(tip))
            action.setStatusTip(tip)

        if callable(slot):
            action.triggered.connect(slot)

        if checkable:
            action.setCheckable(True)

        return action

    def addActions(self, target, actions):
        """
        Add a tuple of actions to the target.

        """
        for action in actions:
            if action is None:
                target.addSeparator()

            else:
                target.addAction(action)
