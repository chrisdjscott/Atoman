
"""
The input tab for the main toolbar

@author: Chris Scott

"""

import os
import sys

from PyQt4 import QtGui, QtCore

from utilities import iconPath
from genericForm import GenericForm
import resources




################################################################################
class LatticePage(QtGui.QWidget):
    def __init__(self, parent, mainToolbar, mainWindow, width):
        super(LatticePage, self).__init__(parent)
        
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
        self.latticeBox = GenericForm(self.inputTab, self.toolbarWidth, "Load reference lattice")
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
        self.inputBox = GenericForm(self.inputTab, self.toolbarWidth, "Load input lattice")
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
class LBOMDPage(QtGui.QWidget):
    def __init__(self, parent, mainToolbar, mainWindow, width):
        super(LBOMDPage, self).__init__(parent)
        
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
        self.refBox = GenericForm(self.inputTab, self.toolbarWidth, "Load animation reference")
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
        self.inputBox = GenericForm(self.inputTab, self.toolbarWidth, "Load input xyz")
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
        super(InputTab, self).__init__(parent)
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        self.fileTypeCurrentIndex = None
        
        # layout
        inputTabLayout = QtGui.QVBoxLayout(self)
        
        selector = GenericForm(self, self.toolbarWidth, "File type")
        selector.show()
        row = selector.newRow()
        self.fileTypeComboBox = QtGui.QComboBox()
        self.fileTypeComboBox.addItem("LBOMD XYZ")
        self.fileTypeComboBox.addItem("LBOMD LATTICE")
        self.connect(self.fileTypeComboBox, QtCore.SIGNAL("currentIndexChanged(QString)"), self.setWidgetStack)
        row.addWidget(self.fileTypeComboBox)
        
        # clear reference button
        self.clearRefButton = QtGui.QPushButton(QtGui.QIcon(iconPath("edit-delete.svg")), "Clear reference")
        self.clearRefButton.setStatusTip("Clear current reference file")
        self.clearRefButton.setCheckable(0)
        self.clearRefButton.setChecked(1)
        self.connect(self.clearRefButton, QtCore.SIGNAL('clicked()'), self.mainWindow.clearReference)
        
        row = selector.newRow()
        row.addWidget(self.clearRefButton)
        
        inputTabLayout.addWidget(selector)
        
#        self.connect(trashButton, QtCore.SIGNAL('clicked()'), self.filterTab.removeFilterList)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget(self)
        
        self.LBOMDPage = LBOMDPage(self, self.mainToolbar, self.mainWindow, self.toolbarWidth)
        self.stackedWidget.addWidget(self.LBOMDPage)
        
        self.latticePage = LatticePage(self, self.mainToolbar, self.mainWindow, self.toolbarWidth)
        self.stackedWidget.addWidget(self.latticePage)
        
        inputTabLayout.addWidget(self.stackedWidget)
    
    def setWidgetStack(self, text):
        """
        Set which stacked widget is displayed
        
        """
        ok = self.okToChangeFileType()
        
        if ok:
            if text == "LBOMD XYZ":
                self.stackedWidget.setCurrentIndex(self.fileTypeComboBox.currentIndex())
                self.fileTypeCurrentIndex = 0
            
            elif text == "LBOMD LATTICE":
                self.stackedWidget.setCurrentIndex(self.fileTypeComboBox.currentIndex())
                self.fileTypeCurrentIndex = 1
                
        #TODO: when the stacked widget is changed I should change file type here,
        #      and warn that current data will be reset
        
    def okToChangeFileType(self):
        """
        Is it ok to change the filetype
        
        """
        if self.mainWindow.refLoaded:
            ok = 0
            
            if self.fileTypeComboBox.currentIndex() == self.fileTypeCurrentIndex:
                pass
            
            else:
                self.fileTypeComboBox.setCurrentIndex(self.fileTypeCurrentIndex)
                self.warnClearReference()
            
        else:
            ok = 1
            
        return ok
        
    def warnClearReference(self):
        """
        Warn user that they must clear the 
        reference file before changing file
        type
        
        """
        QtGui.QMessageBox.warning(self, "Warning", "Reference must be cleared before changing file type!")
