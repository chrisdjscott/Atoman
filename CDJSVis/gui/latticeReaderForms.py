
"""
Lattice reader forms for the inputTab.

@author: Chris Scott

"""
import os
import sys

from PyQt4 import QtGui, QtCore

from ..visutils.utilities import iconPath
from .genericForm import GenericForm
from .. import latticeReaders

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)




################################################################################

class GenericReaderForm(GenericForm):
    """
    Generic reader widget.
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, width, title, stateType):
        super(GenericReaderForm, self).__init__(parent, width, title)
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        self.widgetTitle = title
        
        self.stateType = stateType
        
        self.fileFormatString = None
        
        self.fileExtension = None
        
        # always show widget
        self.show()
        
    def openFile(self, label=None, filename=None, rouletteIndex=None):
        """
        This should be sub-classed to load the selected file.
        
        Should return the name of the loaded file.
        
        """
        self.mainWindow.displayError("GenericReaderWidget: openFile has not been overriden on:\n%s" % str(self))
        return 1
    
    def getFileName(self):
        pass
    
    def updateFileLabel(self, filename):
        pass
    
    def postOpenFile(self, stateType, state, filename):
        """
        Should always be called at the end of openFile.
        
        """
        self.updateFileLabel(filename)
        
        self.parent.fileLoaded(stateType, state, filename, self.fileExtension)
    
    def openFileDialog(self):
        """
        Open a file dialog to select a file.
        
        """
        if self.fileFormatString is None:
            self.mainWindow.displayError("GenericReaderWidget: fileFormatString not set on:\n%s" % str(self))
            return None
        
        fdiag = QtGui.QFileDialog()
        
        filesString = str(self.fileFormatString)
        
        filename = fdiag.getOpenFileName(self, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString)
        filename = str(filename)
        
        if not len(filename):
            return None
        
        (nwd, filename) = os.path.split(filename)        
        
        # change to new working directory
        if nwd != os.getcwd():
            self.mainWindow.console.write("Changing to dir "+nwd)
            os.chdir(nwd)
            self.mainWindow.updateCWD()
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        # open file
        result = self.openFile(filename=filename)
        
        return result
        
        

################################################################################

class LbomdDatReaderForm(GenericReaderForm):
    """
    LBOMD DAT input widget.
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, width, state):
        super(LbomdDatReaderForm, self).__init__(parent, mainToolbar, mainWindow, width, "LBOMD DAT READER", state)
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        self.fileExtension = "dat"
        
        # acceptable formats string
        self.fileFormatString = "Lattice files (*.dat *.dat.bz2 *.dat.gz)"
        
        # reader
        self.latticeReader = latticeReaders.LbomdDatReader(self.mainWindow.tmpDirectory, self.mainWindow.console.write, self.mainWindow.displayWarning)
        
        self.show()
        
        # file name line
        row = self.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        if state == "ref" and os.path.exists("ref.dat"):
            ininame = "ref.dat"
        elif state == "input" and os.path.exists("launch.dat"):
            ininame = "launch.dat"
        else:
            ininame = "lattice.dat"
        
        self.latticeLabel = QtGui.QLineEdit(ininame)
        self.latticeLabel.setFixedWidth(150)
        row.addWidget(self.latticeLabel)
        
        self.loadLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadLatticeButton.setStatusTip("Load file")
        self.connect(self.loadLatticeButton, QtCore.SIGNAL('clicked()'), self.openFile)
        row.addWidget(self.loadLatticeButton)
        
        # open dialog
        row = self.newRow()
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open file")
        self.openLatticeDialogButton.setStatusTip("Open file")
        self.openLatticeDialogButton.setCheckable(0)
        self.connect(self.openLatticeDialogButton, QtCore.SIGNAL('clicked()'), self.openFileDialog)
        row.addWidget(self.openLatticeDialogButton)
    
    def updateFileLabel(self, filename):
        """
        Update file label.
        
        """
        self.latticeLabel.setText(filename)
    
    def getFileName(self):
        """
        Returns file name from label.
        
        """
        return str(self.latticeLabel.text())
    
    def openFile(self, filename=None, rouletteIndex=None):
        """
        Open file.
        
        """
        if filename is None:
            filename = self.getFileName()
        
        filename = str(filename)
        
#        if not len(filename):
#            return None
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        status, state = self.latticeReader.readFile(filename, rouletteIndex=rouletteIndex)
        
        if not status:
            GenericReaderForm.postOpenFile(self, self.stateType, state, filename)
        
        return status
        
################################################################################

class LbomdRefReaderForm(GenericReaderForm):
    """
    LBOMD REF input widget.
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, width, state):
        super(LbomdRefReaderForm, self).__init__(parent, mainToolbar, mainWindow, width, "LBOMD REF READER", state)
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        self.fileExtension = "xyz"
        
        self.fileFormatString = "REF files (*.xyz *.xyz.bz2 *.xyz.gz)"
        
        self.latticeReader = latticeReaders.LbomdRefReader(self.mainWindow.tmpDirectory, self.mainWindow.console.write, self.mainWindow.displayWarning)
        
        self.show()
        
        # file name line
        row = self.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        self.latticeLabel = QtGui.QLineEdit("animation-reference.xyz")
        self.latticeLabel.setFixedWidth(150)
        row.addWidget(self.latticeLabel)
        
        self.loadLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadLatticeButton.setStatusTip("Load file")
        self.connect(self.loadLatticeButton, QtCore.SIGNAL('clicked()'), self.openFile)
        row.addWidget(self.loadLatticeButton)
        
        # open dialog
        row = self.newRow()
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open file")
        self.openLatticeDialogButton.setStatusTip("Open file")
        self.openLatticeDialogButton.setCheckable(0)
        self.connect(self.openLatticeDialogButton, QtCore.SIGNAL('clicked()'), self.openFileDialog)
        row.addWidget(self.openLatticeDialogButton)
    
    def updateFileLabel(self, filename):
        """
        Update file label.
        
        """
        self.latticeLabel.setText(filename)
    
    def getFileName(self):
        """
        Returns file name from label.
        
        """
        return str(self.latticeLabel.text())
    
    def openFile(self, filename=None, rouletteIndex=None):
        """
        Open file.
        
        """
        if filename is None:
            filename = self.getFileName()
        
        filename = str(filename)
        
#        if not len(filename):
#            return None
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        status, state = self.latticeReader.readFile(filename, rouletteIndex=rouletteIndex)
        
        if not status:
            GenericReaderForm.postOpenFile(self, self.stateType, state, filename)
            
            # set xyz input form to use this as ref
            self.parent.lbomdXyzWidget_input.setRefState(state, filename)
        
        return status

################################################################################

class LbomdXYZReaderForm(GenericReaderForm):
    """
    LBOMD XYZ reader.
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, width, state):
        super(LbomdXYZReaderForm, self).__init__(parent, mainToolbar, mainWindow, width, "LBOMD XYZ READER", state)
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        self.refLoaded = False
        
        self.fileExtension = "xyz"
        
        self.fileFormatString = "XYZ files (*.xyz *.xyz.bz2 *.xyz.gz)"
        self.fileFormatStringRef = "XYZ files (*.xyz *.xyz.bz2 *.xyz.gz)"
        
        self.latticeReader = latticeReaders.LbomdXYZReader(self.mainWindow.tmpDirectory, self.mainWindow.console.write, self.mainWindow.displayWarning)
        self.refReader = latticeReaders.LbomdRefReader(self.mainWindow.tmpDirectory, self.mainWindow.console.write, self.mainWindow.displayWarning)
        
        self.show()
        
        # ref file
        row = self.newRow()
        label = QtGui.QLabel("Ref name")
        row.addWidget(label)
        
        self.refLabel = QtGui.QLineEdit("animation-reference.xyz")
        self.refLabel.setFixedWidth(150)
        row.addWidget(self.refLabel)
        
        self.loadRefButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadRefButton.setStatusTip("Load ref")
        self.connect(self.loadRefButton, QtCore.SIGNAL('clicked()'), lambda isRef=True: self.openFile(isRef=isRef))
        row.addWidget(self.loadRefButton)
        
        # open dialog
        row = self.newRow()
        self.openRefDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open reference")
        self.openRefDialogButton.setStatusTip("Open ref")
        self.openRefDialogButton.setCheckable(0)
        self.connect(self.openRefDialogButton, QtCore.SIGNAL('clicked()'), lambda isRef=True: self.openFileDialog(isRef))
        row.addWidget(self.openRefDialogButton)
        
        # input file name line
        row = self.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        self.latticeLabel = QtGui.QLineEdit("PuGaH0000.xyz")
        self.latticeLabel.setFixedWidth(150)
        row.addWidget(self.latticeLabel)
        
        self.loadLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadLatticeButton.setStatusTip("Load file")
        self.connect(self.loadLatticeButton, QtCore.SIGNAL('clicked()'), lambda isRef=False: self.openFile(isRef=isRef))
        row.addWidget(self.loadLatticeButton)
        
        # open dialog
        row = self.newRow()
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open file")
        self.openLatticeDialogButton.setStatusTip("Open file")
        self.openLatticeDialogButton.setCheckable(0)
        self.connect(self.openLatticeDialogButton, QtCore.SIGNAL('clicked()'), lambda isRef=False: self.openFileDialog(isRef))
        row.addWidget(self.openLatticeDialogButton)
    
    def getFileName(self, isRef):
        """
        Get filename
        
        """
        if isRef:
            fn = str(self.refLabel.text())
        else:
            fn = str(self.latticeLabel.text())
        
        return fn
    
    def updateFileLabelCustom(self, filename, isRef):
        """
        Update file label
        
        """
        if isRef:
            self.refLabel.setText(filename)
        else:
            self.latticeLabel.setText(filename)
    
    def openFileDialog(self, isRef):
        """
        Open a file dialog to select a file.
        
        """
        if self.fileFormatString is None:
            self.mainWindow.displayError("GenericReaderWidget: fileFormatString not set on:\n%s" % str(self))
            return None
        
        if not isRef and not self.refLoaded:
            self.mainWindow.displayWarning("Must load corresponding reference first!")
            return None
        
        if isRef and self.refLoaded:
            reply = QtGui.QMessageBox.question(self, "Message", 
                                               "Ref is already loaded: do you want to overwrite it?",
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
            if reply != QtGui.QMessageBox.Yes:
                return None
        
        fdiag = QtGui.QFileDialog()
        
        if isRef:
            filesString = str(self.fileFormatStringRef)
        else:
            filesString = str(self.fileFormatString)
        
        filename = fdiag.getOpenFileName(self, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString)
        filename = str(filename)
        
        if not len(filename):
            return None
        
        (nwd, filename) = os.path.split(filename)        
        
        # change to new working directory
        if nwd != os.getcwd():
            self.mainWindow.console.write("Changing to dir "+nwd)
            os.chdir(nwd)
            self.mainWindow.updateCWD()
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        # open file
        result = self.openFile(filename=filename, isRef=isRef)
        
        return result
    
    def setRefState(self, state, filename):
        """
        Set the ref state.
        
        """
        self.currentRefState = state
        self.refLabel.setText(filename)
        self.refLoaded = True
    
    def openFile(self, filename=None, isRef=False, rouletteIndex=None):
        """
        Open file.
        
        """
        if not isRef and not self.refLoaded:
            self.mainWindow.displayWarning("Must load corresponding reference first!")
            return 2
        
        if isRef and self.refLoaded:
            reply = QtGui.QMessageBox.question(self, "Message", 
                                               "Ref is already loaded: do you want to overwrite it?",
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
            if reply != QtGui.QMessageBox.Yes:
                return 3
        
        if filename is None:
            filename = self.getFileName(isRef)
        
        filename = str(filename)
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        if isRef:
            status, state = self.refReader.readFile(filename)
        else:
            status, state = self.latticeReader.readFile(filename, self.currentRefState, rouletteIndex=rouletteIndex)
        
        if not status:
            if isRef:
                self.setRefState(state, filename)
                self.parent.lbomdXyzWidget_input.setRefState(state, filename)
            else:
                GenericReaderForm.postOpenFile(self, self.stateType, state, filename)
            
            self.updateFileLabelCustom(filename, isRef)
        
        return status
