
"""
GUI classes

author: Chris Scott
last edited: February 2012
"""

import os
import sys
import shutil

try:
    from PyQt4 import QtGui, QtCore, Qt
except:
    print __name__+ ": ERROR: could not import PyQt4"
try:
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except:
    print __name__+ ": ERROR: could not import QVTKRenderWindowInteractor"
try:
    import vtk
except:
    print __name__+ ": ERROR: could not import vtk"

try:
    import utilities
    from utilities import iconPath
except:
    print __name__+ ": ERROR: utilities not found"
try:
    import toolbar as toolbarModule
except:
    print __name__+ ": ERROR: toolbar not found"
try:
    import lattice
except:
    print __name__+ ": ERROR: lattice not found"
try:
    import inputModule
except:
    print __name__+ ": ERROR: inputModule not found"
try:
    import resources
except:
    print __name__+ ": ERROR: could not import resources"



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
        self.applicationString = "Information about this application"
        self.refFile = ""
        self.inputFile = ""
        self.fileType = ""
        self.refLoaded = 0
        self.inputLoaded = 0
        self.consoleOpen = 0
        self.verboseLevel = 3
        
        # window size and location
        self.renderWindowWidth = 750 #650
        self.renderWindowHeight = 715 #650
        self.mainToolbarWidth = 350 #315
        self.mainToolbarHeight = 460 #420
        self.resize(self.renderWindowWidth+self.mainToolbarWidth, self.renderWindowHeight)
        self.centre()
        
        self.setWindowTitle("CDJSVis")
        
        # add the main tool bar
        self.mainToolbar = toolbarModule.MainToolbar(self, self.mainToolbarWidth, self.mainToolbarHeight)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.mainToolbar)
        
        # add exit action
        exitAction = QtGui.QAction(QtGui.QIcon(iconPath("system-log-out.svg")), "Exit", self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)
        
        # add new window action
        newWindowAction = QtGui.QAction(QtGui.QIcon(iconPath("document-new.svg")), "New", self)
        newWindowAction.setShortcut('Ctrl+N')
        newWindowAction.setStatusTip('New window')
        newWindowAction.triggered.connect(self.openNewWindow)
        
        # add file toolbar
        fileToolbar = self.addToolBar("File")
        fileToolbar.addAction(exitAction)
        fileToolbar.addAction(newWindowAction)
        fileToolbar.addSeparator()
                
        # button to show console window
        openConsoleAction = QtGui.QAction(QtGui.QIcon(iconPath("utilities-terminal.svg")), "Console", self)
        openConsoleAction.setStatusTip("Show console window")
        openConsoleAction.triggered.connect(self.showConsole)
        
        utilToolbar = self.addToolBar("Utilities")
        utilToolbar.addAction(openConsoleAction)
        utilToolbar.addSeparator()
        
        # add about action
        aboutAction = QtGui.QAction(QtGui.QIcon(iconPath("help-browser.svg")), "About", self)
        aboutAction.setStatusTip("About this application")
        aboutAction.triggered.connect(self.aboutMe)
        
        # add help toolbar
        helpToolbar = self.addToolBar("Help")
        helpToolbar.addAction(aboutAction)
        
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
        
        self.setCentralWidget(self.VTKContainer)
        
        # connect window destroyed to updateInstances
        self.connect(self, QtCore.SIGNAL("destroyed(QObject*)"), MainWindow.updateInstances)
        
        # initiate lattice objects for storing reference and input states
        self.inputState = lattice.Lattice()
        self.refState = lattice.Lattice()        
        
        # console window for logging output to
        self.console = ConsoleWindow(self)
        
        # create temporary directory for working in
        self.tmpDirectory = utilities.createTmpDirectory()
        
        self.setStatus('Ready')
        
        self.show()
    
    
    
    def openNewWindow(self):
        """
        Open a new instance of the main window
        
        """
        MainWindow().show()
    
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
    
    def setCurrentRefFile(self, filename):
        """
        Set the current ref file in the main toolbar
        
        """
        self.mainToolbar.currentRefLabel.setText("Reference: " + filename)
    
    def setCurrentInputFile(self, filename):
        """
        Set the current input file in the main toolbar
        
        """
        self.mainToolbar.currentInputLabel.setText("Input: " + filename)
    
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
            return
        
        filename = fdiag.getOpenFileName(self, "Open file", os.getcwd(), filesString)
        filename = str(filename)
        
        if not len(filename):
            return
        
        (nwd, filename) = os.path.split(filename)        
        
        # change to new working directory
        self.console.write("Changing to dir "+nwd)
        os.chdir(nwd)
        self.updateCWD()
        
        # open file
        self.openFile(filename, state)
        
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
            return
        
        #TODO: split path to check in directory of file already
        
        self.setStatus("Reading " + filename)
        
        # need to handle different states differently depending on fileType.
        # eg LBOMD input does not have sym, may have charge, etc
        #    DAT input will have both
        if self.fileType == "LBOMD":
            if state == "ref":
                status = inputModule.readFile(filename, self.tmpDirectory, self.refState, self.fileType, state, self.console.write)
            else:
                status = inputModule.readFile(filename, self.tmpDirectory, self.inputState, self.fileType, state, self.console.write)
        elif self.fileType == "DAT":
            if state == "ref":
                status = inputModule.readFile(filename, self.tmpDirectory, self.refState, self.fileType, state, self.console.write)
            else:
                status = inputModule.readFile(filename, self.tmpDirectory, self.inputState, self.fileType, state, self.console.write)
        else:
            self.displayError("openFile: Unrecognised file type: "+self.fileType)
            return
        
        if status:
            if status == -1:
                self.displayWarning("Could not find file: "+filename)
            return
        
        if state == "ref":
            self.setCurrentRefFile(filename)
            self.refLoaded = 1
        else:
            self.setCurrentInputFile(filename)
            self.inputLoaded = 1
        
        self.setStatus("Ready")
        
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
        QtGui.QMessageBox.about(self, "About me", self.applicationString)
    
     

    @staticmethod
    def updateInstances(qobj):
        """
        Make sure only alive windows appear in the set
        
        """
        MainWindow.Instances = set([window for window in MainWindow.Instances if isAlive(window)])   


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

################################################################################
class ConsoleWindow(QtGui.QDialog):
    def __init__(self, parent=None):
        super(ConsoleWindow, self).__init__(parent)
        
        self.parent = parent
        self.setModal(0)
#        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Console")
        self.resize(500,300)
        
        consoleLayout = QtGui.QVBoxLayout(self)
        consoleLayout.setAlignment(QtCore.Qt.AlignTop)
        consoleLayout.setContentsMargins(0, 0, 0, 0)
        consoleLayout.setSpacing(0)
        
        self.textWidget = QtGui.QTextEdit()
        self.textWidget.setReadOnly(1)
        
        consoleLayout.addWidget(self.textWidget)
        
        self.write("Hello there.")
        self.write("OCH")
    
        #TODO: add clear button and close button.
#        consoleLayout.addStretch()
        
        self.clearButton = QtGui.QPushButton("Clear")
        self.clearButton.setAutoDefault(0)
        self.connect(self.clearButton, QtCore.SIGNAL('clicked()'), self.clearText)
        
        self.closeButton = QtGui.QPushButton("Hide")
        self.closeButton.setAutoDefault(1)
        self.connect(self.closeButton, QtCore.SIGNAL('clicked()'), self.close)
        
        buttonWidget = QtGui.QWidget()
        buttonLayout = QtGui.QHBoxLayout(buttonWidget)
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.closeButton)
        
        consoleLayout.addWidget(buttonWidget)
        
        
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
            for i in xrange(indent):
                ind += "  "
            self.textWidget.append("%s %s%s" % (">", ind, string))
        
    def closeEvent(self, event):
        self.hide()
        self.parent.consoleOpen = 0
        
    
    


