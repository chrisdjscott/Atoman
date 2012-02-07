
"""
The input tab for the main toolbar

author: Chris Scott
last edited: February 2012
"""

import os
import sys

try:
    from PyQt4 import QtGui, QtCore
except:
    sys.exit(__name__, "ERROR: PyQt4 not found")

try:
    from utilities import iconPath
except:
    sys.exit(__name__, "ERROR: utilities not found")
try:
    from genericForm import GenericForm
except:
    sys.exit(__name__, "ERROR: genericForm not found")
try:
    import resources
except:
    sys.exit(__name__+ ": ERROR: could not import resources")




################################################################################
class latticeTab(QtGui.QWidget):
    def __init__(self, parent, mainToolbar, mainWindow, width):
        super(latticeTab, self).__init__()
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        # layout
        latticeTabLayout = QtGui.QVBoxLayout(self)
        latticeTabLayout.setSpacing(0)
        latticeTabLayout.setContentsMargins(0, 0, 0, 0)
        latticeTabLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # add read lattice box
        self.latticeBox = GenericForm(self.inputTab, self.toolbarWidth, "Load reference file")
        self.latticeBox.show()
        
        # file name line
        row = self.latticeBox.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        self.latticeLabel = QtGui.QLineEdit("lattice.dat")
        self.latticeLabel.setFixedWidth(150)
        row.addWidget(self.latticeLabel)
        
        self.loadLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadLatticeButton.setStatusTip("Load reference")
        self.connect(self.loadLatticeButton, QtCore.SIGNAL('clicked()'), lambda who="ref": self.openFile(who))
        row.addWidget(self.loadLatticeButton)
        
        # open dialog
        row = self.latticeBox.newRow()
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open reference")
        self.openLatticeDialogButton.setStatusTip("Open reference")
        self.openLatticeDialogButton.setCheckable(0)
        self.connect(self.openLatticeDialogButton, QtCore.SIGNAL('clicked()'), lambda who="ref": self.openFileDialog(who))
        row.addWidget(self.openLatticeDialogButton)
        
        latticeTabLayout.addWidget(self.latticeBox)
        
        # add read input box
        self.inputBox = GenericForm(self.inputTab, self.toolbarWidth, "Load input file")
        self.inputBox.show()
        
        # file name line
        row = self.inputBox.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        self.inputLatticeLabel = QtGui.QLineEdit("lattice.dat")
        self.inputLatticeLabel.setFixedWidth(150)
        row.addWidget(self.inputLatticeLabel)
        
        self.loadLatticeInputButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadLatticeInputButton.setStatusTip("Load input")
        self.connect(self.loadLatticeInputButton, QtCore.SIGNAL('clicked()'), lambda who="input": self.openFile(who))
        row.addWidget(self.loadLatticeInputButton)
        
        # open dialog
        row = self.inputBox.newRow()
        self.openLatticeInputDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open input")
        self.openLatticeInputDialogButton.setStatusTip("Open input")
        self.openLatticeInputDialogButton.setCheckable(0)
        self.connect(self.openLatticeInputDialogButton, QtCore.SIGNAL('clicked()'), lambda who="input": self.openFileDialog(who))
        row.addWidget(self.openLatticeInputDialogButton)
        
        latticeTabLayout.addWidget(self.inputBox)
    
    def openFile(self, who):
        """
        Open the specified file
        
        """
        self.mainWindow.setFileType("DAT")
        
        if who == "ref":
            filename = self.latticeLabel.text()
        else:
            filename = self.inputLatticeLabel.text()
        
        self.mainWindow.openFile(str(filename), who)
        
        
    
    def openFileDialog(self, who):
        """
        Open the file dialog
        
        """
        # first set the file type
        self.mainWindow.setFileType("DAT")
        
        # then open the dialog
        self.mainWindow.openFileDialog(who)


################################################################################
class LBOMDTab(QtGui.QWidget):
    def __init__(self, parent, mainToolbar, mainWindow, width):
        super(LBOMDTab, self).__init__()
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        # layout
        LBOMDTabLayout = QtGui.QVBoxLayout(self)
        LBOMDTabLayout.setSpacing(0)
        LBOMDTabLayout.setContentsMargins(0, 0, 0, 0)
        LBOMDTabLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # add read reference box
        self.refBox = GenericForm(self.inputTab, self.toolbarWidth, "Load reference file")
        self.refBox.show()
        
        # file name line
        row = self.refBox.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        self.LBOMDRefLabel = QtGui.QLineEdit("animation-reference.xyz")
        self.LBOMDRefLabel.setFixedWidth(150)
        row.addWidget(self.LBOMDRefLabel)
        
        self.loadRefButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadRefButton.setStatusTip("Load reference")
        self.connect(self.loadRefButton, QtCore.SIGNAL('clicked()'), lambda who="ref": self.openFile(who))
        row.addWidget(self.loadRefButton)
        
        # open dialog
        row = self.refBox.newRow()
        self.openRefDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open reference")
        self.openRefDialogButton.setStatusTip("Open reference")
        self.openRefDialogButton.setCheckable(0)
        self.connect(self.openRefDialogButton, QtCore.SIGNAL('clicked()'), lambda who="ref": self.openFileDialog(who))
        row.addWidget(self.openRefDialogButton)
        
        LBOMDTabLayout.addWidget(self.refBox)
        
        # add read input box
        self.inputBox = GenericForm(self.inputTab, self.toolbarWidth, "Load input file")
        self.inputBox.show()
        
        # file name line
        row = self.inputBox.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        self.LBOMDInputLabel = QtGui.QLineEdit("track0000.xyz")
        self.LBOMDInputLabel.setFixedWidth(150)
        row.addWidget(self.LBOMDInputLabel)
        
        self.loadInputButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadInputButton.setStatusTip("Load input")
        self.connect(self.loadInputButton, QtCore.SIGNAL('clicked()'), lambda who="input": self.openFile(who))
        row.addWidget(self.loadInputButton)
        
        # open dialog
        row = self.inputBox.newRow()
        self.openInputDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open input")
        self.openInputDialogButton.setStatusTip("Open input")
        self.openInputDialogButton.setCheckable(0)
        self.connect(self.openInputDialogButton, QtCore.SIGNAL('clicked()'), lambda who="input": self.openFileDialog(who))
        row.addWidget(self.openInputDialogButton)
        
        LBOMDTabLayout.addWidget(self.inputBox)
            
    def openFile(self, who):
        """
        Open the specified file
        
        """
        self.mainWindow.setFileType("LBOMD")
        
        if who == "ref":
            filename = self.LBOMDRefLabel.text()
        else:
            filename = self.LBOMDInputLabel.text()
        
        self.mainWindow.openFile(str(filename), who)
        
    def openFileDialog(self, who):
        """
        Open the file dialog
        
        """
        # first set the file type
        self.mainWindow.setFileType("LBOMD")
        
        # then open the dialog
        self.mainWindow.openFileDialog(who)


################################################################################
class InputTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(InputTab, self).__init__()
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        # layout
        inputTabLayout = QtGui.QVBoxLayout(self)
        
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        
        self.tabBar = QtGui.QTabWidget(row)
        
        # add LBOMD page
        self.LBOMDTab = LBOMDTab(self, self.mainToolbar, self.mainWindow, self.toolbarWidth)
        self.tabBar.addTab(self.LBOMDTab, "LBOMD")
        
        # add DAT page
        self.latticeTab = latticeTab(self, self.mainToolbar, self.mainWindow, self.toolbarWidth)
        self.tabBar.addTab(self.latticeTab, "DAT")
        
        rowLayout.addWidget(self.tabBar)
        
        inputTabLayout.addWidget(row)
