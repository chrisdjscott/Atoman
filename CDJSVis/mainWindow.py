"""
The main window class

@author: Chris Scott

"""
import os
import sys
import shutil
import platform
import tempfile
import traceback

from PySide import QtGui, QtCore
import PySide
import vtk
import numpy as np
import matplotlib
import scipy

from .visutils.utilities import iconPath, resourcePath
from .atoms import elements
from .gui import toolbar as toolbarModule
from . import lattice
from .rendering import renderer
from .gui import helpForm
from .gui import dialogs
from .gui import rendererSubWindow
from .gui import inputDialog
from .gui import systemsDialog
try:
    from . import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


from .visutils import version
__version__ = version.getVersion()


################################################################################
class MainWindow(QtGui.QMainWindow):
    """
    The main window.
    
    """
    
    Instances = set()
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # multiple instances
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        MainWindow.Instances.add(self)
        
        # initialise user interface
        self.initUI()
        
        # set focue
        self.setFocus()
    
    def initUI(self):
        """
        Initialise the interface.
        
        """
        # defaults
        self.refFile = ""
        self.inputFile = ""
        self.fileType = ""
        self.fileExtension = ""
        self.refLoaded = 0
        self.inputLoaded = 0
        self.consoleOpen = 0
        self.helpOpen = 0
        self.verboseLevel = 3
        self.PBC = np.zeros(3, np.int32)
        self.PBC[0] = 1
        self.PBC[1] = 1
        self.PBC[2] = 1
        self.mouseMotion = 0
        
        print "*"*40 + "DEBUG" + "*"*40
        print "RESOURCE PATH", resourcePath("lbomd.IN", dirname="md_input"), os.path.exists(resourcePath("lbomd.IN", dirname="md_input"))
        print "*"*38 + "END DEBUG" + "*"*38
        
        # initiate lattice objects for storing reference and input states
        self.inputState = lattice.Lattice()
        self.refState = lattice.Lattice()  
        
        # get settings object
        settings = QtCore.QSettings()
        
        # initial directory
        currentDir = str(settings.value("mainWindow/currentDirectory", ""))
        
        if hasattr(sys, "_MEIPASS"):
            if not len(currentDir) or not os.path.exists(currentDir):
                # change to home directory if running from pyinstaller bundle
                currentDir = os.environ.get("HOME")
        
        else:
            currentDir = os.getcwd()
        
        os.chdir(currentDir)
        
        # toolbar size (fixed)
        self.mainToolbarWidth = 350
        self.mainToolbarHeight = 460
        
        # default window widget size
        self.renderWindowWidth = 760 * 1.2
        self.renderWindowHeight = 715 * 1.2
        
        # default size
        windowWidth = self.renderWindowWidth+self.mainToolbarWidth
        windowHeight = self.renderWindowHeight
        
        # resize
        self.resize(settings.value("mainWindow/size", QtCore.QSize(windowWidth, windowHeight)))
        
        # location
        self.centre()
                
        self.setWindowTitle("CDJSVis")
        
        # create temporary directory for working in
        self.tmpDirectory = tempfile.mkdtemp(prefix="VisTemp-", dir="/tmp")
        
        # console window for logging output to
        self.console = dialogs.ConsoleWindow(self)
        
        # help window for displaying help
        self.helpWindow = helpForm.HelpForm("index.html", parent=self)
        
        # image viewer
        self.imageViewer = dialogs.ImageViewer(self, parent=self)
        
        # preferences dialog
        self.preferences = dialogs.PreferencesDialog(parent=self)
        
        # add file actions
        exitAction = self.createAction("Exit", self.close, "Ctrl-Q", "system-log-out.svg", 
                                       "Exit application")
        newWindowAction = self.createAction("&New app window", self.openNewWindow, "Ctrl-N", 
                                            "CDJSVis.ico", "Open new application window")
        newRenWindowAction = self.createAction("New sub window", slot=self.addRendererWindow,
                                            icon="window-new.svg", tip="Open new render sub window")
#         loadInputAction = self.createAction("Load input", slot=self.showLoadInputDialog, icon="document-open.svg",
#                                             tip="Open load input dialog")
        systems_action = self.createAction("Systems dialog", slot=self.show_systems_dialog, icon="document-open.svg",
                                           tip="Show systems dialog")
        openCWDAction = self.createAction("Open CWD", slot=self.openCWD, icon="folder.svg", 
                                          tip="Open current working directory")
        exportElementsAction = self.createAction("Export elements", slot=self.exportElements,
                                                 icon="file-export-icon.png", tip="Export element properties")
        importElementsAction = self.createAction("Import elements", slot=self.importElements,
                                                 icon="file-import-icon.png", tip="Import element properties")
        exportBondsAction = self.createAction("Export bonds", slot=self.exportBonds,
                                                 icon="file-export-icon.png", tip="Export bonds file")
        importBondsAction = self.createAction("Import bonds", slot=self.importBonds,
                                                 icon="file-import-icon.png", tip="Import bonds file")
        showImageViewerAction = self.createAction("Image viewer", slot=self.showImageViewer, 
                                                  icon="applications-graphics.svg", tip="Show image viewer")
        showPreferencesAction = self.createAction("Preferences", slot=self.showPreferences, 
                                                  icon="applications-system.svg", tip="Show preferences window")
        
        # add file menu
        fileMenu = self.menuBar().addMenu("&File")
        self.addActions(fileMenu, (newWindowAction, newRenWindowAction, systems_action, openCWDAction, importElementsAction, 
                                   exportElementsAction, importBondsAction, exportBondsAction, None, exitAction))
        
        # button to show console window
        openConsoleAction = self.createAction("Console", self.showConsole, None, "console-icon.png", "Show console window")
        
        # element editor action
        openElementEditorAction = self.createAction("Element editor", slot=self.openElementEditor, icon="periodic-table-icon.png", 
                                                    tip="Show element editor")
        
        # add view menu
        viewMenu = self.menuBar().addMenu("&View")
        self.addActions(viewMenu, (openConsoleAction, showImageViewerAction, openElementEditorAction, showPreferencesAction))
        
        # add file toolbar
        fileToolbar = self.addToolBar("File")
        fileToolbar.addAction(exitAction)
        fileToolbar.addAction(newWindowAction)
        fileToolbar.addAction(newRenWindowAction)
        fileToolbar.addSeparator()
#         fileToolbar.addAction(loadInputAction)
        fileToolbar.addAction(systems_action)
        fileToolbar.addAction(openCWDAction)
        fileToolbar.addSeparator()
        
        # util tool bar
        viewToolbar = self.addToolBar("Utilities")
        viewToolbar.addAction(openConsoleAction)
        viewToolbar.addAction(showImageViewerAction)
        viewToolbar.addAction(openElementEditorAction)
        viewToolbar.addAction(showPreferencesAction)
        viewToolbar.addSeparator()
        
        
        # add about action
        aboutAction = self.createAction("About CDJSVis", slot=self.aboutMe, icon="Information-icon.png", 
                                           tip="About CDJSVis")
        
        helpAction = self.createAction("CDJSVis Help", slot=self.showHelp, icon="Help-icon.png", tip="Show help window")
        
        # add help toolbar
        helpToolbar = self.addToolBar("Help")
        helpToolbar.addAction(aboutAction)
        helpToolbar.addAction(helpAction)
        
        helpMenu = self.menuBar().addMenu("&Help")
        self.addActions(helpMenu, (aboutAction, helpAction))
        
        # add cwd to status bar
        self.currentDirectoryLabel = QtGui.QLabel(os.getcwd())
        self.statusBar = QtGui.QStatusBar()
        self.statusBar.addPermanentWidget(self.currentDirectoryLabel)
        self.setStatusBar(self.statusBar)
        
        # dict of currently loaded systems
        self.loaded_systems = {}
        
        # systems dialog
        self.systems_dialog = systemsDialog.SystemsDialog(self, self)
        
        # load input dialog
        self.loadInputDialog = inputDialog.InputDialog(self, self, None)
        
        self.mdiArea = QtGui.QMdiArea()
        self.mdiArea.subWindowActivated.connect(self.rendererWindowActivated)
        self.setCentralWidget(self.mdiArea)
        
        self.rendererWindows = []
        self.rendererWindowsSubWin = []
        self.subWinCount = 0
        
        self.addRendererWindow(ask=False)
        
        self.mdiArea.tileSubWindows()
#        self.mdiArea.cascadeSubWindows()
        
        # add the main tool bar
        self.mainToolbar = toolbarModule.MainToolbar(self, self.mainToolbarWidth, self.mainToolbarHeight)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.mainToolbar)
        
        # connect window destroyed to updateInstances
        self.connect(self, QtCore.SIGNAL("destroyed(QObject*)"), MainWindow.updateInstances)
        
        self.setStatus('Ready')
        
        self.show()
        
        # give focus
        self.raise_()
        
        # show system dialog
        self.show_systems_dialog()
        
        # show input dialog
#        self.showLoadInputDialog()
    
    def show_systems_dialog(self):
        """
        Show systems dialog.
        
        """
        self.systems_dialog.hide()
        self.systems_dialog.show()
    
    def showLoadInputDialog(self):
        """
        Show load input dialog.
        
        """
        self.loadInputDialog.hide()
        self.loadInputDialog.show()
    
    def rendererWindowActivated(self, sw):
        """
        Sub window activated. (TEMPORARY)
        
        """
        pass
    
    def addRendererWindow(self, ask=True):
        """
        Add renderer window to mdi area.
        
        """
#         if ask:
#             dlg = dialogs.NewRendererWindowDialog(parent=self)
#         
#         if not ask or dlg.exec_():
            # if ask, get num from dialog
            
        rendererWindow = rendererSubWindow.RendererWindow(self, self.subWinCount, parent=self)
        rendererWindow.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
        subwin = self.mdiArea.addSubWindow(rendererWindow)
        subwin.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        subwin.show()
        subwin.activateWindow()
        
        self.rendererWindows.append(rendererWindow)
        self.rendererWindowsSubWin.append(subwin)
        
        self.subWinCount += 1
        
        self.mdiArea.tileSubWindows()
    
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
    
    def openElementEditor(self):
        """
        Open element editor.
        
        """
        if not self.refLoaded:
            return
        
        elementEditor = dialogs.ElementEditor(parent=self)
        elementEditor.show()
    
    def importElements(self):
        """
        Import element properties file.
        
        """
        reply = QtGui.QMessageBox.question(self, "Message", 
                                           "This will overwrite the current element properties file. You should create a backup first!\n\nDo you wish to continue?",
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
        if reply == QtGui.QMessageBox.Yes:
            # open file dialog
            fname = QtGui.QFileDialog.getOpenFileName(self, "CDJSVis - Import element properties", ".", "IN files (*.IN)")[0]
            
            if fname:
                # read in new file
                elements.read(fname)
                
                # overwrite current file
                elements.write(resourcePath("atoms.IN"))
                
                # set on Lattice objects too
                self.inputState.refreshElementProperties()
                self.refState.refreshElementProperties()
                
                self.setStatus("Imported element properties")
    
    def exportElements(self):
        """
        Export element properties to file.
        
        """
        fname = os.path.join(".", "atoms-exported.IN")
        
        fname = QtGui.QFileDialog.getSaveFileName(self, "CDJSVis - Export element properties", fname, "IN files (*.IN)")[0]
        
        print "FNAME", fname
        
        if fname:
            if not "." in fname or fname[-3:] != ".IN":
                fname += ".IN"
            
            elements.write(fname)
            
            self.setStatus("Element properties exported")
    
    def importBonds(self):
        """
        Import bonds file.
        
        """
        reply = QtGui.QMessageBox.question(self, "Message", 
                                           "This will overwrite the current bonds file. You should create a backup first!\n\nDo you wish to continue?",
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
        if reply == QtGui.QMessageBox.Yes:
            # open file dialog
            fname = QtGui.QFileDialog.getOpenFileName(self, "CDJSVis - Import bonds file", ".", "IN files (*.IN)")[0]
            
            if fname:
                # read in new file
                elements.readBonds(fname)
                
                # overwrite current file
                elements.writeBonds(resourcePath("bonds.IN"))
                
                self.setStatus("Imported bonds file")
    
    def exportBonds(self):
        """
        Export bonds file.
        
        """
        fname = os.path.join(".", "bonds-exported.IN")
        
        fname = QtGui.QFileDialog.getSaveFileName(self, "CDJSVis - Export bonds file", fname, "IN files (*.IN)")[0]
        
        if fname:
            if not "." in fname or fname[-3:] != ".IN":
                fname += ".IN"
            
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
        
#        elif osname == "Windows":
#            os.startfile(dirname)
    
    def showFilterSummary(self):
        """
        Show the filter window.
        
        """
        self.mainToolbar.filterPage.showFilterSummary()
    
    def setCameraToCell(self):
        """
        Reset the camera to point at the cell
        
        """
        self.renderer.setCameraToCell()
    
    def toggleCellFrame(self):
        """
        Toggle lattice frame visibility
        
        """
        self.renderer.toggleLatticeFrame()
    
    def toggleAxes(self):
        """
        Toggle axes visibility
        
        """
        self.renderer.toggleAxes()
    
    def openNewWindow(self):
        """
        Open a new instance of the main window
        
        """
        mw = MainWindow()
        mw.setWindowIcon(QtGui.QIcon(iconPath("CDJSVis.ico")))
        mw.show()
    
    def centre(self):
        """
        Centre the window.
        
        """
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def showConsole(self):
        """
        Open the console window
        
        """
        if self.consoleOpen:
            self.console.closeEvent(1)
            self.console.show()
        
        else:
            self.console.show()
            self.consoleOpen = 1
    
    def showHelp(self):
        """
        Show the help window.
        
        """
        if self.helpOpen:
            self.helpWindow.closeEvent(1)
            self.helpWindow.show()
        
        else:
            self.helpWindow.show()
            self.helpOpen = 1
    
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
    
    def confirmCloseEvent(self):
        """
        Show a dialog to confirm closeEvent.
        
        """
        dlg = dialogs.ConfirmCloseDialog(self)
        
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
#        reply = QtGui.QMessageBox.question(self, 'Message', "Are you sure you want to quit", 
#                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
        close, clearSettings = self.confirmCloseEvent()
        
#        if reply == QtGui.QMessageBox.Yes:
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
    
    def setFileType(self, fileType):
        """
        Set the file type.
        
        """
        self.fileType = fileType
        
        if fileType == "DAT":
            self.fileExtension = "dat"
        
        elif fileType == "LBOMD":
            self.fileExtension = "xyz"
    
    def setCurrentRefFile(self, filename):
        """
        Set the current ref file in the main toolbar
        
        """
        self.mainToolbar.currentRefLabel.setText("Reference: " + filename)
        self.refFile = filename
    
    def setCurrentInputFile(self, filename):
        """
        Set the current input file in the main toolbar
        
        """
        self.mainToolbar.currentInputLabel.setText("Input: " + filename)
        self.inputFile = filename
    
    def setStatus(self, string):
        """
        Set temporary status in status bar
        
        """
        self.statusBar.showMessage(string)
    
    def updateCWD(self):
        """
        Updates the CWD label in the status bar.
        
        """
        dirname = os.getcwd()
        
        self.currentDirectoryLabel.setText(dirname)
        self.imageViewer.changeDir(dirname)
    
    def readLBOMDIN(self):
        """
        Try to read sim identity and PBCs from lbomd.IN
        
        """
        if os.path.exists("lbomd.IN"):
            f = open("lbomd.IN")
            
            try:
                f.readline()
                f.readline()
                f.readline()
                
                line = f.readline().strip()
                array = line.split()
                try:
                    # simulation identity
                    simIdentity = array[0]
                    
                    # update labels with simulation identity
                    self.loadInputDialog.lbomdXyzWidget_input.updateFileLabelCustom("%s%04d.xyz" % (simIdentity, 0), isRef=False)
                    for rw in self.rendererWindows:
                        rw.outputDialog.imageTab.imageSequenceTab.fileprefix.setText(simIdentity)
                
                except IndexError:
                    self.console.write("WARNING: INDEX ERROR 1 (check lbomd.IN format)")
                    pass
                
                line = f.readline().strip()
                array = line.split()
                try:
                    PBC = [0]*3
                    PBC[0] = int(array[0])
                    PBC[1] = int(array[1])
                    PBC[2] = int(array[2])
                    
                    if PBC[0]:
                        self.loadInputDialog.PBCXCheckBox.setCheckState(QtCore.Qt.Checked)
                    
                    else:
                        self.loadInputDialog.PBCXCheckBox.setCheckState(QtCore.Qt.Unchecked)
                    
                    if PBC[1]:
                        self.loadInputDialog.PBCYCheckBox.setCheckState(QtCore.Qt.Checked)
                    
                    else:
                        self.loadInputDialog.PBCYCheckBox.setCheckState(QtCore.Qt.Unchecked)
                    
                    if PBC[2]:
                        self.loadInputDialog.PBCZCheckBox.setCheckState(QtCore.Qt.Checked)
                    
                    else:
                        self.loadInputDialog.PBCZCheckBox.setCheckState(QtCore.Qt.Unchecked)
                
                except IndexError:
                    self.console.write("WARNING: INDEX ERROR 2 (check lbomd.IN format)")
            
            except Exception as e:
                self.displayError("Read lbomd.IN failed with error:\n\n%s" % "".join(traceback.format_exception(*sys.exc_info())))
            
            finally:
                f.close()
    
    def postFileLoaded(self, fileType, state, filename, extension):
        """
        Called when a new file has been loaded.
        
         - fileType should be "ref" or "input"
         - state is the new Lattice object
        
        """
        if fileType == "ref":
            # if a ref is already loaded, we need to 
            
            
            self.refState = state
            
            self.readLBOMDIN()
            
            self.postRefLoaded(filename)
        
        else:
            self.inputState = state
        
        self.postInputLoaded(filename)
        
        if self.fileExtension is not None:
            self.fileExtension = extension
    
    def postRefLoaded(self, filename):
        """
        Do stuff after the ref has been loaded.
        
        """
        if self.refLoaded:
            for filterPage in self.mainToolbar.pipelineList:
                filterPage.clearAllActors()
                filterPage.refreshAllFilters()
                
            for rw in self.rendererWindows:
                rw.textSelector.refresh()
            
            for rw in self.rendererWindows:
                rw.outputDialog.rdfTab.refresh()
        
        self.setCurrentRefFile(filename)
        self.refLoaded = 1
        
#         self.inputState.clone(self.refState)
        
        for rw in self.rendererWindows:
            rw.renderer.postRefRender()
        
#         self.loadInputDialog.loadRefBox.hide()
#         self.loadInputDialog.clearRefBox.show()
#         self.loadInputDialog.loadInputBox.show()
        
        for rw in self.rendererWindows:
            rw.textSelector.refresh()
    
    def postInputLoaded(self, filename):
        """
        Do stuff after the input has been loaded
        
        """
        self.setCurrentInputFile(filename)
        self.inputLoaded = 1
        
#         self.mainToolbar.loadInputForm.hide()
        self.mainToolbar.analysisPipelinesForm.show()
        
        for filterPage in self.mainToolbar.pipelineList:
            filterPage.refreshAllFilters()
        
        for rw in self.rendererWindows:
            rw.textSelector.refresh()
        
        for rw in self.rendererWindows:
            rw.outputDialog.rdfTab.refresh()
    
    def clearReference(self):
        """
        Clear the current reference file
        
        """
        self.refLoaded = 0
        self.inputLoaded = 0
        self.setCurrentRefFile("")
        self.setCurrentInputFile("")
        
        # close any open output dialogs!
        for rw in self.rendererWindows:
            rw.outputDialog.hide()
            rw.textSelector.hide()
        
        # close all render windows?
#        for rw in self.rendererWindows:
#            rw.close()
#        
#        # open new renderer window
#        self.addRendererWindow(ask=False)
        
        # should probably hide stuff like elements form too!
        
        
        # clear rdf tab
        
        
        # clear pipelines
        for pipeline in self.mainToolbar.pipelineList:
            pipeline.clearAllFilterLists()
        
        self.mainToolbar.analysisPipelinesForm.hide()
        self.mainToolbar.loadInputForm.show()
        
        self.loadInputDialog.loadInputBox.hide()
        self.loadInputDialog.clearRefBox.hide()
        self.loadInputDialog.loadRefBox.show()
        
        self.loadInputDialog.lbomdXyzWidget_ref.refLoaded = False
        self.loadInputDialog.lbomdXyzWidget_input.refLoaded = False
    
    def displayWarning(self, message):
        """
        Display warning message.
        
        """
        QtGui.QMessageBox.warning(self, "Warning", message)
    
    def displayError(self, message):
        """
        Display error message
        
        """
        QtGui.QMessageBox.critical(self, "Error", "A critical error has occurred.\n"+message)
    
    def aboutMe(self):
        """
        Display about message.
        
        """
        QtGui.QMessageBox.about(self, "About CDJSVis", 
                                """<b>CDJSVis</b> %s
                                <p>Copyright &copy; 2013 Chris Scott</p>
                                <p>This application can be used to visualise atomistic simulations.</p>
                                <p>GUI based on <a href="http://sourceforge.net/projects/avas/">AVAS</a> 
                                   by Marc Robinson.</p>
                                <p>Python %s - Qt %s - PySide %s - VTK %s - SciPy %s - Matplotlib %s on %s""" % (
                                __version__, platform.python_version(), QtCore.__version__, PySide.__version__,
                                vtk.vtkVersion.GetVTKVersion(), scipy.__version__, matplotlib.__version__, platform.system()))
    
    def createAction(self, text, slot=None, shortcut=None, icon=None,
                     tip=None, checkable=False, signal="triggered()"):
        """
        Create an action
        
        """
        action = QtGui.QAction(text, self)
        
        if icon is not None:
            action.setIcon(QtGui.QIcon(iconPath(icon)))
        
        if shortcut is not None:
            action.setShortcut(shortcut)
        
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        
        if slot is not None:
            self.connect(action, QtCore.SIGNAL(signal), slot)
        
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
    
    @staticmethod
    def updateInstances(qobj):
        """
        Make sure only alive windows appear in the set
        
        """
        MainWindow.Instances = set([window for window in MainWindow.Instances if isAlive(window)])   


################################################################################
def isAlive(qobj):
    """
    Check a window is alive
    
    """
    return True
    
    import sip
    try:
        sip.unwrapinstance(qobj)
    except RuntimeError:
        return False
    return True   
