
"""
GUI classes

author: Chris Scott
last edited: February 2012
"""

import os
import sys
import shutil

try:
    from PyQt4 import QtGui, QtCore
except:
    sys.exit(__name__, "ERROR: PyQt4 not found")
try:
    from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except:
    sys.exit(__name__, "ERROR: QVTKRenderWindowInteractor not found")
try:
    import vtk
except:
    sys.exit(__name__, "ERROR: vtk not found")

try:
    import utilities
    from utilities import iconPath
except:
    sys.exit(__name__, "ERROR: utilities not found")
try:
    import toolbar as toolbarModule
except:
    sys.exit(__name__, "ERROR: toolbar not found")
try:
    import lattice
except:
    sys.exit(__name__, "ERROR: lattice not found")
try:
    import inputModule
except:
    sys.exit(__name__, "ERROR: inputModule not found")




################################################################################
class MainWindow(QtGui.QMainWindow):
    """
    The main window.
    
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        
        self.initUI()
    
    def initUI(self):
        """
        Initialise the interface.
        
        """
        # defaults
        self.refFile = ""
        self.inputFile = ""
        self.fileType = ""
        
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
        
        # add actions
        exitAction = QtGui.QAction(QtGui.QIcon(iconPath("system-log-out.svg")), "Exit", self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)
        
        # add toolbar
        toolbar = self.addToolBar("Exit")
        toolbar.addAction(exitAction)
        
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
        
        self.VTKWidget.Initialize()
        self.VTKWidget.Start()
        
        self.VTKRen = vtk.vtkRenderer()
        self.VTKRen.SetBackground(1,1,1)
        self.VTKWidget.GetRenderWindow().AddRenderer(self.VTKRen)
        
        self.setCentralWidget(self.VTKContainer)
        
        # initiate lattice objects for storing reference and input states
        self.inputState = lattice.Lattice()
        self.refState = lattice.Lattice()
        
        # create temporary directory for working in
        self.tmpDirectory = utilities.createTmpDirectory()
        
        self.statusBar.showMessage('Ready')
        
        self.show()
    
    def centre(self):
        """
        Centre the window.
        
        """
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
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
    
    def setFileType(self, fileType):
        """
        Set the file type.
        
        """
        print "NEW FILE TYPE", fileType
        self.fileType = fileType
    
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
            print "WARNING: unknown file type: ", self.fileType
            return
        
        filename = fdiag.getOpenFileName(self, "Open file", os.getcwd(), filesString)
        filename = str(filename)
        
        (nwd, filename) = os.path.split(filename)        
        
        # change to new working directory
        os.chdir(nwd)
        self.currentDirectoryLabel.setText(os.getcwd())
        
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
        
        print "FILE", filename
        print "STATE", state
        
        # need to handle different states differently depending on fileType.
        # eg LBOMD input does not have sym, may have charge, etc
        #    DAT input will have both
        if self.fileType == "LBOMD":
            if state == "ref":
                inputModule.readFile(filename, self.tmpDirectory, self.refState, self.fileType, state)
            else:
                pass
        elif self.fileType == "DAT":
            pass
        else:
            print "WARNING: unknown file type: ", self.fileType
            return
        
        
        
        
        
