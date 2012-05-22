
"""
The main window class

@author: Chris Scott

"""
import os
import sys
import shutil
import platform
import tempfile

from PyQt4 import QtGui, QtCore, Qt
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import numpy as np

from utilities import iconPath, resourcePath
from atoms import elements
import toolbar as toolbarModule
import lattice
import inputModule
import rendering
import helpForm
import dialogs

try:
    import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


__version__ = "0.0.1"


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
        
        
        self.initUI()
    
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
        
        # change to home directory if running from pyinstaller bundle
#        print os.environ.get("_MEIPASS2")
#        print os.environ.get("HOME")
#        if os.environ.get("_MEIPASS2") is not None:
#            os.chdir(os.environ.get("HOME"))
        
        # window size and location
        self.renderWindowWidth = 750 #650
        self.renderWindowHeight = 715 #650
        self.mainToolbarWidth = 350 #315
        self.mainToolbarHeight = 460 #420
        self.resize(self.renderWindowWidth+self.mainToolbarWidth, self.renderWindowHeight)
        self.centre()
        
        self.setWindowTitle("CDJSVis")
        
        # console window for logging output to
        self.console = dialogs.ConsoleWindow(self)
        
        # help window for displaying help
        self.helpWindow = helpForm.HelpForm("index.html", parent=self)
        
        # add the main tool bar
        self.mainToolbar = toolbarModule.MainToolbar(self, self.mainToolbarWidth, self.mainToolbarHeight)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.mainToolbar)
        
        # add file actions
        exitAction = self.createAction("Exit", self.close, "Ctrl-Q", "system-log-out.svg", 
                                       "Exit application")
        newWindowAction = self.createAction("&New window", self.openNewWindow, "Ctrl-N", 
                                            "document-new.svg", "Open new window")
        openCWDAction = self.createAction("Open CWD", slot=self.openCWD, icon="document-open.svg", 
                                          tip="Open current working directory")
        exportElementsAction = self.createAction("Export elements", slot=self.exportElements,
                                                 icon="file-export-icon.png", tip="Export element properties")
        importElementsAction = self.createAction("Import elements", slot=self.importElements,
                                                 icon="file-import-icon.png", tip="Import element properties")
        
        # add file menu
        fileMenu = self.menuBar().addMenu("&File")
        self.addActions(fileMenu, (newWindowAction, openCWDAction, importElementsAction, exportElementsAction, None, exitAction))
        
        # add file toolbar
        fileToolbar = self.addToolBar("File")
        fileToolbar.addAction(exitAction)
        fileToolbar.addAction(newWindowAction)
        fileToolbar.addAction(openCWDAction)
        fileToolbar.addSeparator()
        
        # button to show console window
        openConsoleAction = self.createAction("Console", self.showConsole, None, "console-icon.png", "Show console window")
        
        utilToolbar = self.addToolBar("Utilities")
        utilToolbar.addAction(openConsoleAction)
        utilToolbar.addSeparator()
        
        # button to displace lattice frame
        showCellAction = self.createAction("Toggle cell", slot=self.toggleCellFrame, icon="cell_icon.svg", 
                                           tip="Toggle cell frame visibility")
        
        # button to display axes
        showAxesAction = self.createAction("Toggle axes", slot=self.toggleAxes, icon="axis_icon2.svg", 
                                           tip="Toggle axes visiblity")
        
        openElementEditorAction = self.createAction("Element editor", slot=self.openElementEditor, icon="periodic-table-icon.png", 
                                                    tip="Open element editor")
        
        # reset camera to cell
        setCamToCellAction = self.createAction("Reset to cell", slot=self.setCameraToCell, icon="set_cam_cell.svg", 
                                           tip="Reset camera to cell")
        
        visualisationToolbar = self.addToolBar("Visualisation")
        visualisationToolbar.addAction(showCellAction)
        visualisationToolbar.addAction(showAxesAction)
        visualisationToolbar.addAction(setCamToCellAction)
        visualisationToolbar.addAction(openElementEditorAction)
        visualisationToolbar.addSeparator()
        
        renderingMenu = self.menuBar().addMenu("&Rendering")
        self.addActions(renderingMenu, (showCellAction, showAxesAction, openElementEditorAction))
        
        cameraMenu = self.menuBar().addMenu("&Camera")
        self.addActions(cameraMenu, (setCamToCellAction,))
        
        # add filtering actions
        showFilterSummaryAction = self.createAction("Show summary", self.showFilterSummary, 
                                                    icon="document-properties.svg", 
                                                    tip="Show filter summary")
        
        filteringToolbar = self.addToolBar("Filtering")
        filteringToolbar.addAction(showFilterSummaryAction)
        filteringToolbar.addSeparator()
        
        filteringMenu = self.menuBar().addMenu("&Filtering")
        self.addActions(filteringMenu, (showFilterSummaryAction,))
        
        # add about action
        aboutAction = self.createAction("About", slot=self.aboutMe, icon="Information-icon.png", 
                                           tip="About this application")
        
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
        
        # initialise the VTK container
        self.VTKContainer = QtGui.QWidget(self)
        VTKlayout = QtGui.QVBoxLayout(self.VTKContainer)
        self.VTKWidget = QVTKRenderWindowInteractor(self.VTKContainer)
        VTKlayout.addWidget(self.VTKWidget)
        VTKlayout.setContentsMargins(0,0,0,0)
        
#        self.VTKWidget.Initialize()
        self.VTKWidget.Start()
        
        self.VTKRen = vtk.vtkRenderer()
        self.VTKRen.SetBackground(1,1,1)
        self.VTKWidget.GetRenderWindow().AddRenderer(self.VTKRen)
        
        self.VTKWidget._Iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # add observers
        self.VTKWidget._Iren.AddObserver("LeftButtonPressEvent", self.leftButtonPressed)
        self.VTKWidget._Iren.AddObserver("MouseMoveEvent", self.mouseMoved)
        self.VTKWidget._Iren.AddObserver("LeftButtonReleaseEvent", self.leftButtonReleased)
        
#        self.VTKWidget.AddObserver("LeftButtonPressEvent", self.leftButtonPressed)
#        self.VTKWidget.AddObserver("MouseMoveEvent", self.mouseMoved)
#        self.VTKWidget.AddObserver("LeftButtonReleaseEvent", self.leftButtonReleased)
        
        # add picker
        self.VTKPicker = vtk.vtkCellPicker()
        self.VTKPicker.SetTolerance(0.000001)
        self.VTKPicker.AddObserver("EndPickEvent", self.endPickEvent)
        self.VTKWidget.SetPicker(self.VTKPicker)
        
        # distance representation
#        self.distanceWidget = vtk.vtkDistanceWidget()
#        self.distanceWidget.SetInteractor(self.VTKWidget)
#        self.distanceWidget.CreateDefaultRepresentation()
##        self.distanceWidget.GetRepresentation().SetLabelFormat()
#        self.distanceWidget.On()
        
        self.setCentralWidget(self.VTKContainer)
                
        self.renderer = rendering.Renderer(self)
        
        # connect window destroyed to updateInstances
        self.connect(self, QtCore.SIGNAL("destroyed(QObject*)"), MainWindow.updateInstances)
        
        # initiate lattice objects for storing reference and input states
        self.inputState = lattice.Lattice()
        self.refState = lattice.Lattice()        
        
        # create temporary directory for working in
        self.tmpDirectory = tempfile.mkdtemp(prefix="CDJSVisTemp-", dir="/tmp")
        
        self.setStatus('Ready')
        
        self.show()
    
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
            fname = QtGui.QFileDialog.getOpenFileName(self, "CDJSVis - Import element properties", ".", "IN files (*.IN)")
            
            if fname:
                # read in new file
                elements.read(fname)
                
                # overwrite current file
                elements.write(resourcePath("data/atoms.IN"))
                
                self.setStatus("Imported element properties")
            
            
    
    def exportElements(self):
        """
        Export element properties to file.
        
        """
        fname = os.path.join(".", "atoms-exported.IN")
        
        fname = QtGui.QFileDialog.getSaveFileName(self, "CDJSVis - Export element properties", fname, "IN files (*.IN)")
        
        if fname:
            if not "." in fname or fname[-3:] != ".IN":
                fname += ".IN"
            
            elements.write(fname)
            
            self.setStatus("Element properties exported")
    
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
    
    def endPickEvent(self, obj, event):
        """
        End of vtk pick event.
        
        """
        pass
    
    def leftButtonPressed(self, obj, event):
        """
        Left mouse button pressed
        
        """
        self.mouseMotion = 0
        
        # left release event isn't working so have to pick by double click
        if self.VTKWidget.GetRepeatCount() == 1:
            pos = self.VTKWidget.GetEventPosition()
            self.VTKPicker.Pick(pos[0], pos[1], 0, self.VTKRen)
    
    def mouseMoved(self, obj, event):
        """
        Mouse moved
        
        """
        self.mouseMotion = 1
    
    def leftButtonReleased(self, obj, event):
        """
        Left button released.
        
        """
        print "LEFT RELEASE", self.mouseMotion
    
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
        mw.setWindowIcon(QtGui.QIcon(iconPath("applications.ico")))
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
    
    def closeEvent(self, event):
        """
        Catch attempt to close
        
        """
        reply = QtGui.QMessageBox.question(self, 'Message', "Are you sure you want to quit", 
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
        if reply == QtGui.QMessageBox.Yes:
            self.tidyUp()
            event.accept()
        else:
            event.ignore()
        
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
        self.currentDirectoryLabel.setText(os.getcwd())
    
    def openFileDialog(self, state):
        """
        Open file dialog
        
        """
        fdiag = QtGui.QFileDialog()
        
        if self.fileType == "LBOMD":
            filesString = "LBOMD files (*.xyz *.xyz.bz2 *.xyz.gz)"
        elif self.fileType == "DAT":
            filesString = "Lattice files (*.dat *.dat.bz2 *.dat.gz)"
        else:
            self.displayError("openFileDialog: Unrecognised file type: "+self.fileType)
            return None
        
        filename = fdiag.getOpenFileName(self, "CDJSVis - Open file", os.getcwd(), filesString)
        filename = str(filename)
        
        if not len(filename):
            return None
        
        (nwd, filename) = os.path.split(filename)        
        
        # change to new working directory
        if nwd != os.getcwd():
            self.console.write("Changing to dir "+nwd)
            os.chdir(nwd)
            self.updateCWD()
        
        # open file
        result = self.openFile(filename, state)
        
        return result
        
    def openFile(self, filename, state):
        """
        Open file
        
        """
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        if state == "input" and not self.refLoaded:
            self.displayWarning("Must load reference before input")
            return None
        
        #TODO: split path to check in directory of file already
        
        self.setStatus("Reading " + filename)
        
        # need to handle different states differently depending on fileType.
        # eg LBOMD input does not have sym, may have charge, etc
        #    DAT input will have both
        if self.fileType == "LBOMD":
            if state == "ref":
                status = inputModule.readFile(filename, self.tmpDirectory, self.refState, self.fileType, state, self.console.write)
                
                self.readLBOMDIN()
                
            else:
                status = inputModule.readFile(filename, self.tmpDirectory, self.inputState, self.fileType, state, self.console.write, self.refState)
        
        elif self.fileType == "DAT":
            if state == "ref":
                status = inputModule.readFile(filename, self.tmpDirectory, self.refState, self.fileType, state, self.console.write)
                
            else:
                status = inputModule.readFile(filename, self.tmpDirectory, self.inputState, self.fileType, state, self.console.write)
        
        else:
            self.displayError("openFile: Unrecognised file type: "+self.fileType)
            return None
        
        if status:
            if status == -1:
                self.displayWarning("Could not find file: "+filename)
            
            elif status == -2:
                self.displayWarning("LBOMD XYZ input NAtoms does not match reference!")
            
            elif status == -3:
                self.displayWarning("Unrecognised format for LBOMD XYZ input file!")
            
            return None
        
        if state == "ref":
            self.postRefLoaded(filename)
            self.renderer.postRefRender()
            
            self.inputState.clone(self.refState)
        
        self.postInputLoaded(filename)
        
        self.setStatus("Ready")
        
        return filename
    
    def readLBOMDIN(self):
        """
        Try to read sim identity and PBCs from lbomd.IN
        
        """
        if os.path.exists("lbomd.IN"):
            f = open("lbomd.IN")
            
            f.readline()
            f.readline()
            f.readline()
            
            line = f.readline().strip()
            array = line.split()
            simIdentity = array[0]
            
            line = f.readline().strip()
            array = line.split()
            PBC = [0]*3
            PBC[0] = int(array[0])
            PBC[1] = int(array[1])
            PBC[2] = int(array[2])
            
            self.mainToolbar.inputTab.LBOMDPage.LBOMDInputLabel.setText("%s%04d.xyz" % (simIdentity, 0))
            self.mainToolbar.outputPage.imageTab.imageSequenceTab.fileprefix.setText(simIdentity)
            
            if PBC[0]:
                self.mainToolbar.inputTab.PBCXCheckBox.setCheckState(2)
            
            else:
                self.mainToolbar.inputTab.PBCXCheckBox.setCheckState(0)
            
            if PBC[1]:
                self.mainToolbar.inputTab.PBCYCheckBox.setCheckState(2)
            
            else:
                self.mainToolbar.inputTab.PBCYCheckBox.setCheckState(0)
            
            if PBC[2]:
                self.mainToolbar.inputTab.PBCZCheckBox.setCheckState(2)
            
            else:
                self.mainToolbar.inputTab.PBCZCheckBox.setCheckState(0)
    
    def postRefLoaded(self, filename):
        """
        Do stuff after the ref has been loaded.
        
        """
        self.setCurrentRefFile(filename)
        self.refLoaded = 1
#        self.mainToolbar.inputTab.clearRefButton.setCheckable(1)
    
    def postInputLoaded(self, filename):
        """
        Do stuff after the input has been loaded
        
        """
        self.setCurrentInputFile(filename)
        self.inputLoaded = 1
        
        self.mainToolbar.tabBar.setTabEnabled(1, True)
        self.mainToolbar.tabBar.setTabEnabled(2, True)
        
        # refresh filters eg specie filter
        self.mainToolbar.filterPage.refreshAllFilters()
    
    def clearReference(self):
        """
        Clear the current reference file
        
        """
        self.refLoaded = 0
        self.inputLoaded = 0
        self.setCurrentRefFile("")
        self.setCurrentInputFile("")
        self.mainToolbar.tabBar.setTabEnabled(1, False)
        self.mainToolbar.tabBar.setTabEnabled(2, False)
        self.mainToolbar.filterPage.clearAllFilterLists()
    
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
                                """<b>CDJSVis</b> v%s
                                <p>Copyright &copy; 2012 Chris Scott</p>
                                <p>This application can be used to visualise atomistic simulations.</p>
                                <p>GUI based on <a href="http://sourceforge.net/projects/avas/">AVAS</a> 
                                   by Marc Robinson.</p>
                                <p>Python %s - Qt %s - PyQt %s  - VTK %s on %s""" % (
                                __version__, platform.python_version(), Qt.QT_VERSION_STR, Qt.PYQT_VERSION_STR,
                                vtk.vtkVersion.GetVTKVersion(), platform.system()))
    
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
    import sip
    try:
        sip.unwrapinstance(qobj)
    except RuntimeError:
        return False
    return True   
