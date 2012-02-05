
"""
GUI classes

author: Chris Scott
last edited: February 2012
"""

import os
import sys

try:
    from PyQt4 import QtGui, QtCore, Qt
except:
    sys.exit(__name__, "ERROR: PyQt4 not found")




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
        self.renderWindowWidth = 650
        self.renderWindowHeight = 650
        self.mainToolBarWidth = 315
        self.mainToolBarHeight = 420
        self.resize(self.renderWindowWidth+self.mainToolBarWidth, self.renderWindowHeight)
        self.centre()
        
        self.setWindowTitle("CDJSVis")
        
        # add actions
        exitAction = QtGui.QAction(QtGui.QIcon(os.path.join("icons", "exit.png")), "Exit", self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)
        
        # add toolbar
        toolbar = self.addToolBar("Exit")
        toolbar.addAction(exitAction)
        
        
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
            event.accept()
        else:
            event.ignore()
        
        
        
