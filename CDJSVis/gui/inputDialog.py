
"""
The input tab for the main toolbar

@author: Chris Scott

"""
import sys

from PyQt4 import QtGui, QtCore

from ..visutils.utilities import iconPath
from .genericForm import GenericForm
from .. import latticeReaders
from . import latticeReaderForms

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class InputDialog(QtGui.QDialog):
    """
    Input dialog for toolbar.
    
    """
    def __init__(self, parent, mainWindow, width):
        super(InputDialog, self).__init__(parent)
        
        self.setWindowTitle("Load input")
        self.setModal(False)
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        self.refTypeCurrentIndex = 0
        self.inputTypeCurrentIndex = 0
        
        # layout
        inputTabLayout = QtGui.QVBoxLayout(self)
        
        # ref box
        self.loadRefBox = GenericForm(self, self.toolbarWidth, "Load reference state")
        self.loadRefBox.show()
        
        # file type combo
        self.refTypeCombo = QtGui.QComboBox()
        self.refTypeCombo.addItem("LBOMD DAT")
        self.refTypeCombo.addItem("LBOMD REF")
        self.refTypeCombo.addItem("LBOMD XYZ")
        self.refTypeCombo.currentIndexChanged.connect(self.setLoadRefStack)
        
        row = self.loadRefBox.newRow()
        row.addWidget(self.refTypeCombo)
        
        inputTabLayout.addWidget(self.loadRefBox)
        
        # stacked widget
        self.loadRefStack = QtGui.QStackedWidget()
        
        self.lbomdDatWidget_ref = latticeReaderForms.LbomdDatReaderForm(self, self.mainToolbar, self.mainWindow, self.toolbarWidth, "ref")
        self.loadRefStack.addWidget(self.lbomdDatWidget_ref)
        
        self.lbomdRefWidget_ref = latticeReaderForms.LbomdRefReaderForm(self, self.mainToolbar, self.mainWindow, self.toolbarWidth, "ref")
        self.loadRefStack.addWidget(self.lbomdRefWidget_ref)
        
        self.lbomdXyzWidget_ref = latticeReaderForms.LbomdXYZReaderForm(self, self.mainToolbar, self.mainWindow, self.toolbarWidth, "ref")
        self.loadRefStack.addWidget(self.lbomdXyzWidget_ref)
        
        row = self.loadRefBox.newRow()
        row.addWidget(self.loadRefStack)
        
        # clear ref box
        self.clearRefBox = GenericForm(self, self.toolbarWidth, "Load reference state")
        
        # clear ref button
        clearRefButton = QtGui.QPushButton(QtGui.QIcon(iconPath("edit-delete.svg")), "Clear reference")
        clearRefButton.setStatusTip("Clear current reference file")
        clearRefButton.setCheckable(0)
        clearRefButton.setChecked(1)
        clearRefButton.clicked.connect(self.mainWindow.clearReference)
        
        row = self.clearRefBox.newRow()
        row.addWidget(clearRefButton)
        
        inputTabLayout.addWidget(self.clearRefBox)
        
        # input box
        self.loadInputBox = GenericForm(self, self.toolbarWidth, "Load input state")
#        self.loadInputBox.show()
        
        # file type combo
        self.inputTypeCombo = QtGui.QComboBox()
        self.inputTypeCombo.addItem("LBOMD DAT")
        self.inputTypeCombo.addItem("LBOMD REF")
        self.inputTypeCombo.addItem("LBOMD XYZ")
        self.inputTypeCombo.currentIndexChanged.connect(self.setLoadInputStack)
        
        row = self.loadInputBox.newRow()
        row.addWidget(self.inputTypeCombo)
        
        inputTabLayout.addWidget(self.loadInputBox)
        
        # stacked widget
        self.loadInputStack = QtGui.QStackedWidget()
        
        self.lbomdDatWidget_input = latticeReaderForms.LbomdDatReaderForm(self, self.mainToolbar, self.mainWindow, self.toolbarWidth, "input")
        self.loadInputStack.addWidget(self.lbomdDatWidget_input)
        
        self.lbomdRefWidget_input = latticeReaderForms.LbomdRefReaderForm(self, self.mainToolbar, self.mainWindow, self.toolbarWidth, "input")
        self.loadInputStack.addWidget(self.lbomdRefWidget_input)
        
        self.lbomdXyzWidget_input = latticeReaderForms.LbomdXYZReaderForm(self, self.mainToolbar, self.mainWindow, self.toolbarWidth, "input")
        self.loadInputStack.addWidget(self.lbomdXyzWidget_input)
        
        row = self.loadInputBox.newRow()
        row.addWidget(self.loadInputStack)
        
        # periodic boundaries
        group = QtGui.QGroupBox("Periodic boundaries")
        group.setAlignment(QtCore.Qt.AlignHCenter)
        
        groupLayout = QtGui.QVBoxLayout(group)
        
        self.PBCXCheckBox = QtGui.QCheckBox("x")
        self.PBCXCheckBox.setChecked(1)
        self.PBCYCheckBox = QtGui.QCheckBox("y")
        self.PBCYCheckBox.setChecked(1)
        self.PBCZCheckBox = QtGui.QCheckBox("z")
        self.PBCZCheckBox.setChecked(1)
        
        self.connect(self.PBCXCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.PBCXChanged)
        self.connect(self.PBCYCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.PBCYChanged)
        self.connect(self.PBCZCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.PBCZChanged)
        
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.addWidget(self.PBCXCheckBox)
        rowLayout.addWidget(self.PBCYCheckBox)
        rowLayout.addWidget(self.PBCZCheckBox)
        
        groupLayout.addWidget(row)
        
        inputTabLayout.addWidget(group)
        
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.hide)
        
        row = QtGui.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(closeButton)
        inputTabLayout.addLayout(row)
        
        inputTabLayout.addStretch(1)
    
    def openFileDialog(self, fileFormatString):
        """
        Open file dialog.
        
        """
        
    
    def fileLoaded(self, fileType, state, filename, extension):
        """
        Called when a new file is loaded.
        
         - fileType should be "ref" or "input"
         - state is the new Lattice object
        
        """
        self.mainWindow.postFileLoaded(fileType, state, filename, extension)
    
    def PBCXChanged(self, val):
        """
        PBC changed.
        
        """
        if self.PBCXCheckBox.isChecked():
            self.mainWindow.PBC[0] = 1
        
        else:
            self.mainWindow.PBC[0] = 0
    
    def PBCYChanged(self, val):
        """
        PBC changed.
        
        """
        if self.PBCYCheckBox.isChecked():
            self.mainWindow.PBC[1] = 1
        
        else:
            self.mainWindow.PBC[1] = 0
    
    def PBCZChanged(self, val):
        """
        PBC changed.
        
        """
        if self.PBCZCheckBox.isChecked():
            self.mainWindow.PBC[2] = 1
        
        else:
            self.mainWindow.PBC[2] = 0
    
    def setLoadRefStack(self, index):
        """
        Change load ref stack.
        
        """
        ok = self.okToChangeFileType() 
        
        if ok:
            self.loadRefStack.setCurrentIndex(index)
            self.refTypeCurrentIndex = index
            
            if index == 0:
                self.inputTypeCombo.setCurrentIndex(0)
            
            elif index == 1:
                self.inputTypeCombo.setCurrentIndex(2)
    
    def setLoadInputStack(self, index):
        """
        Change load input stack.
        
        """
        self.loadInputStack.setCurrentIndex(index)
        self.inputTypeCurrentIndex = index
    
    def okToChangeFileType(self):
        """
        Is it ok to change the filetype
        
        """
        if self.mainWindow.refLoaded:
            ok = 0
            
            if self.refTypeCombo.currentIndex() == self.refTypeCurrentIndex:
                pass
            
            else:
                self.refTypeCombo.setCurrentIndex(self.refTypeCurrentIndex)
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
    
#    def closeEvent(self, event):
#        """
#        Close.
#        
#        """
#        self.hide()


################################################################################
#class LatticePage(QtGui.QWidget):
#    def __init__(self, parent, mainToolbar, mainWindow, width):
#        super(LatticePage, self).__init__(parent)
#        
#        self.inputTab = parent
#        self.mainToolbar = mainToolbar
#        self.mainWindow = mainWindow
#        self.toolbarWidth = width
#        
#        # layout
#        latticeTabLayout = QtGui.QVBoxLayout(self)
#        latticeTabLayout.setSpacing(0)
#        latticeTabLayout.setContentsMargins(0, 0, 0, 0)
#        latticeTabLayout.setAlignment(QtCore.Qt.AlignTop)
#        
#        # add read lattice box
#        self.latticeBox = GenericForm(self.inputTab, self.toolbarWidth, "Load reference lattice")
#        self.latticeBox.show()
#        
#        # file name line
#        row = self.latticeBox.newRow()
#        label = QtGui.QLabel("File name")
#        row.addWidget(label)
#        
#        self.latticeLabel = QtGui.QLineEdit("lattice.dat")
#        self.latticeLabel.setFixedWidth(150)
#        row.addWidget(self.latticeLabel)
#        
#        self.loadLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
#        self.loadLatticeButton.setStatusTip("Load reference")
#        self.connect(self.loadLatticeButton, QtCore.SIGNAL('clicked()'), lambda who="ref": self.openFile(who))
#        row.addWidget(self.loadLatticeButton)
#        
#        # open dialog
#        row = self.latticeBox.newRow()
#        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open reference")
#        self.openLatticeDialogButton.setStatusTip("Open reference")
#        self.openLatticeDialogButton.setCheckable(0)
#        self.connect(self.openLatticeDialogButton, QtCore.SIGNAL('clicked()'), lambda who="ref": self.openFileDialog(who))
#        row.addWidget(self.openLatticeDialogButton)
#        
#        latticeTabLayout.addWidget(self.latticeBox)
#        
#        # add read input box
#        self.inputBox = GenericForm(self.inputTab, self.toolbarWidth, "Load input lattice")
#        self.inputBox.show()
#        
#        # file name line
#        row = self.inputBox.newRow()
#        label = QtGui.QLabel("File name")
#        row.addWidget(label)
#        
#        self.inputLatticeLabel = QtGui.QLineEdit("lattice.dat")
#        self.inputLatticeLabel.setFixedWidth(150)
#        row.addWidget(self.inputLatticeLabel)
#        
#        self.loadLatticeInputButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
#        self.loadLatticeInputButton.setStatusTip("Load input")
#        self.connect(self.loadLatticeInputButton, QtCore.SIGNAL('clicked()'), lambda who="input": self.openFile(who))
#        row.addWidget(self.loadLatticeInputButton)
#        
#        # open dialog
#        row = self.inputBox.newRow()
#        self.openLatticeInputDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open input")
#        self.openLatticeInputDialogButton.setStatusTip("Open input")
#        self.openLatticeInputDialogButton.setCheckable(0)
#        self.connect(self.openLatticeInputDialogButton, QtCore.SIGNAL('clicked()'), lambda who="input": self.openFileDialog(who))
#        row.addWidget(self.openLatticeInputDialogButton)
#        
#        latticeTabLayout.addWidget(self.inputBox)
#    
#    def updateFileLabel(self, state, filename):
#        """
#        Update label with loaded file name
#        
#        """
#        if state == "ref":
#            self.latticeLabel.setText(str(filename))
#        
#        else:
#            self.inputLatticeLabel.setText(str(filename))
#    
#    def openFile(self, who):
#        """
#        Open the specified file
#        
#        """
#        self.mainWindow.setFileType("DAT")
#        
#        if who == "ref":
#            filename = self.latticeLabel.text()
#        else:
#            filename = self.inputLatticeLabel.text()
#        
#        result = self.mainWindow.openFile(str(filename), who)
#        
#        if result is not None:
#            self.updateFileLabel(who, result)
#    
#    def openFileDialog(self, who):
#        """
#        Open the file dialog
#        
#        """
#        # first set the file type
#        self.mainWindow.setFileType("DAT")
#        
#        # then open the dialog
#        result = self.mainWindow.openFileDialog(who)
#        
#        if result is not None:
#            self.updateFileLabel(who, result)
#    
#    
#
#
#################################################################################
#class LBOMDPage(QtGui.QWidget):
#    def __init__(self, parent, mainToolbar, mainWindow, width):
#        super(LBOMDPage, self).__init__(parent)
#        
#        self.inputTab = parent
#        self.mainToolbar = mainToolbar
#        self.mainWindow = mainWindow
#        self.toolbarWidth = width
#        
#        # layout
#        LBOMDTabLayout = QtGui.QVBoxLayout(self)
#        LBOMDTabLayout.setSpacing(0)
#        LBOMDTabLayout.setContentsMargins(0, 0, 0, 0)
#        LBOMDTabLayout.setAlignment(QtCore.Qt.AlignTop)
#        
#        # add read reference box
#        self.refBox = GenericForm(self.inputTab, self.toolbarWidth, "Load animation reference")
#        self.refBox.show()
#        
#        # file name line
#        row = self.refBox.newRow()
#        label = QtGui.QLabel("File name")
#        row.addWidget(label)
#        
#        self.LBOMDRefLabel = QtGui.QLineEdit("animation-reference.xyz")
#        self.LBOMDRefLabel.setFixedWidth(150)
#        row.addWidget(self.LBOMDRefLabel)
#        
#        self.loadRefButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
#        self.loadRefButton.setStatusTip("Load reference")
#        self.connect(self.loadRefButton, QtCore.SIGNAL('clicked()'), lambda who="ref": self.openFile(who))
#        row.addWidget(self.loadRefButton)
#        
#        # open dialog
#        row = self.refBox.newRow()
#        self.openRefDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open reference")
#        self.openRefDialogButton.setStatusTip("Open reference")
#        self.openRefDialogButton.setCheckable(0)
#        self.connect(self.openRefDialogButton, QtCore.SIGNAL('clicked()'), lambda who="ref": self.openFileDialog(who))
#        row.addWidget(self.openRefDialogButton)
#        
#        LBOMDTabLayout.addWidget(self.refBox)
#        
#        # add read input box
#        self.inputBox = GenericForm(self.inputTab, self.toolbarWidth, "Load input xyz")
#        self.inputBox.show()
#        
#        # file name line
#        row = self.inputBox.newRow()
#        label = QtGui.QLabel("File name")
#        row.addWidget(label)
#        
#        self.LBOMDInputLabel = QtGui.QLineEdit("track0000.xyz")
#        self.LBOMDInputLabel.setFixedWidth(150)
#        row.addWidget(self.LBOMDInputLabel)
#        
#        self.loadInputButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
#        self.loadInputButton.setStatusTip("Load input")
#        self.connect(self.loadInputButton, QtCore.SIGNAL('clicked()'), lambda who="input": self.openFile(who))
#        row.addWidget(self.loadInputButton)
#        
#        # open dialog
#        row = self.inputBox.newRow()
#        self.openInputDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open input")
#        self.openInputDialogButton.setStatusTip("Open input")
#        self.openInputDialogButton.setCheckable(0)
#        self.connect(self.openInputDialogButton, QtCore.SIGNAL('clicked()'), lambda who="input": self.openFileDialog(who))
#        row.addWidget(self.openInputDialogButton)
#        
#        LBOMDTabLayout.addWidget(self.inputBox)
#    
#    def updateFileLabel(self, state, filename):
#        """
#        Update label with loaded file name
#        
#        """
#        if state == "ref":
#            self.LBOMDRefLabel.setText(str(filename))
#        
#        else:
#            self.LBOMDInputLabel.setText(str(filename))
#    
#    def openFile(self, who):
#        """
#        Open the specified file
#        
#        """
#        self.mainWindow.setFileType("LBOMD")
#        
#        if who == "ref":
#            filename = self.LBOMDRefLabel.text()
#        else:
#            filename = self.LBOMDInputLabel.text()
#        
#        result = self.mainWindow.openFile(str(filename), who)
#        
#        if result is not None:
#            self.updateFileLabel(who, result)
#        
#    def openFileDialog(self, who):
#        """
#        Open the file dialog
#        
#        """
#        # first set the file type
#        self.mainWindow.setFileType("LBOMD")
#        
#        # then open the dialog
#        result = self.mainWindow.openFileDialog(who)
#        
#        if result is not None:
#            self.updateFileLabel(who, result)


################################################################################
#class InputTab(QtGui.QWidget):
#    def __init__(self, parent, mainWindow, width):
#        super(InputTab, self).__init__(parent)
#        
#        self.mainToolbar = parent
#        self.mainWindow = mainWindow
#        self.toolbarWidth = width
#        
#        self.fileTypeCurrentIndex = 0
#        
#        # layout
#        inputTabLayout = QtGui.QVBoxLayout(self)
#        
#        selector = GenericForm(self, self.toolbarWidth, "File type")
#        selector.show()
#        row = selector.newRow()
#        self.fileTypeComboBox = QtGui.QComboBox()
#        self.fileTypeComboBox.addItem("LBOMD XYZ")
#        self.fileTypeComboBox.addItem("LBOMD LATTICE")
#        self.connect(self.fileTypeComboBox, QtCore.SIGNAL("currentIndexChanged(QString)"), self.setWidgetStack)
#        row.addWidget(self.fileTypeComboBox)
#        
#        # clear reference button
#        self.clearRefButton = QtGui.QPushButton(QtGui.QIcon(iconPath("edit-delete.svg")), "Clear reference")
#        self.clearRefButton.setStatusTip("Clear current reference file")
#        self.clearRefButton.setCheckable(0)
#        self.clearRefButton.setChecked(1)
#        self.connect(self.clearRefButton, QtCore.SIGNAL('clicked()'), self.mainWindow.clearReference)
#        
#        row = selector.newRow()
#        row.addWidget(self.clearRefButton)
#        
#        inputTabLayout.addWidget(selector)
#        
##        self.connect(trashButton, QtCore.SIGNAL('clicked()'), self.filterTab.removeFilterList)
#        
#        # stacked widget
#        self.stackedWidget = QtGui.QStackedWidget(self)
#        
#        self.LBOMDPage = LBOMDPage(self, self.mainToolbar, self.mainWindow, self.toolbarWidth)
#        self.stackedWidget.addWidget(self.LBOMDPage)
#        
#        self.latticePage = LatticePage(self, self.mainToolbar, self.mainWindow, self.toolbarWidth)
#        self.stackedWidget.addWidget(self.latticePage)
#        
#        inputTabLayout.addWidget(self.stackedWidget)
#        
#        # periodic boundaries
#        group = QtGui.QGroupBox("Periodic boundaries")
#        group.setAlignment(QtCore.Qt.AlignHCenter)
#        
#        groupLayout = QtGui.QVBoxLayout(group)
#        
#        self.PBCXCheckBox = QtGui.QCheckBox("x")
#        self.PBCXCheckBox.setChecked(1)
#        self.PBCYCheckBox = QtGui.QCheckBox("y")
#        self.PBCYCheckBox.setChecked(1)
#        self.PBCZCheckBox = QtGui.QCheckBox("z")
#        self.PBCZCheckBox.setChecked(1)
#        
#        self.connect(self.PBCXCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.PBCXChanged)
#        self.connect(self.PBCYCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.PBCYChanged)
#        self.connect(self.PBCZCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.PBCZChanged)
#        
#        row = QtGui.QWidget(self)
#        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
#        rowLayout.addWidget(self.PBCXCheckBox)
#        rowLayout.addWidget(self.PBCYCheckBox)
#        rowLayout.addWidget(self.PBCZCheckBox)
#        
#        groupLayout.addWidget(row)
#        
#        inputTabLayout.addWidget(group)
#        inputTabLayout.addStretch(1)
#        
#    def PBCXChanged(self, val):
#        """
#        PBC changed.
#        
#        """
#        if self.PBCXCheckBox.isChecked():
#            self.mainWindow.PBC[0] = 1
#        
#        else:
#            self.mainWindow.PBC[0] = 0
#    
#    def PBCYChanged(self, val):
#        """
#        PBC changed.
#        
#        """
#        if self.PBCYCheckBox.isChecked():
#            self.mainWindow.PBC[1] = 1
#        
#        else:
#            self.mainWindow.PBC[1] = 0
#    
#    def PBCZChanged(self, val):
#        """
#        PBC changed.
#        
#        """
#        if self.PBCZCheckBox.isChecked():
#            self.mainWindow.PBC[2] = 1
#        
#        else:
#            self.mainWindow.PBC[2] = 0
#    
#    def setWidgetStack(self, text):
#        """
#        Set which stacked widget is displayed
#        
#        """
#        ok = self.okToChangeFileType()
#        
#        if ok:
#            if text == "LBOMD XYZ":
#                self.stackedWidget.setCurrentIndex(self.fileTypeComboBox.currentIndex())
#                self.fileTypeCurrentIndex = 0
#            
#            elif text == "LBOMD LATTICE":
#                self.stackedWidget.setCurrentIndex(self.fileTypeComboBox.currentIndex())
#                self.fileTypeCurrentIndex = 1
#                
#        #TODO: when the stacked widget is changed I should change file type here,
#        #      and warn that current data will be reset
#        
#    def okToChangeFileType(self):
#        """
#        Is it ok to change the filetype
#        
#        """
#        if self.mainWindow.refLoaded:
#            ok = 0
#            
#            if self.fileTypeComboBox.currentIndex() == self.fileTypeCurrentIndex:
#                pass
#            
#            else:
#                self.fileTypeComboBox.setCurrentIndex(self.fileTypeCurrentIndex)
#                self.warnClearReference()
#            
#        else:
#            ok = 1
#            
#        return ok
#        
#    def warnClearReference(self):
#        """
#        Warn user that they must clear the 
#        reference file before changing file
#        type
#        
#        """
#        QtGui.QMessageBox.warning(self, "Warning", "Reference must be cleared before changing file type!")
