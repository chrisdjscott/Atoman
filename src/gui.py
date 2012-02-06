
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
    import utilities
    from utilities import iconPath
except:
    sys.exit(__name__, "ERROR: utilities not found")
try:
    import toolbar as toolbarModule
except:
    sys.exit(__name__, "ERROR: toolbar not found")






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
        
        
        
        # create temporary directory for working in
        self.tmpDirectory = utilities.createTmpDirectory()
        
        self.statusBar().showMessage('Ready')
        
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
        
