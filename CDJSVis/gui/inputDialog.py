
"""
The input tab for the main toolbar

@author: Chris Scott

"""
import sys

from PySide import QtGui, QtCore

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
        self.setModal(True)
        
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
        clearRefButton.setToolTip("Clear current reference file")
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

