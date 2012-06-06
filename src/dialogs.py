
"""
Additional dialogs.

@author: Chris Scott

"""
import os
import sys

from PyQt4 import QtGui, QtCore
import numpy as np

from atoms import elements
from utilities import resourcePath, iconPath

try:
    import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################
class ConsoleWindow(QtGui.QDialog):
    """
    Console window for displaying output to the user.
    
    """
    def __init__(self, parent=None):
        super(ConsoleWindow, self).__init__(parent)
        
        self.parent = parent
        self.setModal(0)
#        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Console")
        self.setWindowIcon(QtGui.QIcon(iconPath("console-icon.png")))
        self.resize(500,300)
        
        consoleLayout = QtGui.QVBoxLayout(self)
        consoleLayout.setAlignment(QtCore.Qt.AlignTop)
        consoleLayout.setContentsMargins(0, 0, 0, 0)
        consoleLayout.setSpacing(0)
        
        self.textWidget = QtGui.QTextEdit()
        self.textWidget.setReadOnly(1)
        
        consoleLayout.addWidget(self.textWidget)
        
        #TODO: add save text.
        
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


################################################################################
class ElementEditor(QtGui.QDialog):
    """
    Dialog to edit element properties.
    
    """
    def __init__(self, parent=None):
        super(ElementEditor, self).__init__(parent)
        
        self.parent = parent
        self.setModal(1)
        
        self.setWindowTitle("Element editor")
        self.setWindowIcon(QtGui.QIcon(iconPath("periodic-table-icon.png")))
        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.dirty = False
        
        layout = QtGui.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignHCenter)
#        layout.setContentsMargins(0, 0, 0, 0)
#        layout.setSpacing(0)
        
        # lattice objects
        self.inputLattice = self.parent.inputState
        self.refLattice = self.parent.refState
        
        # list of unique species
        uniqueSpecies = set()
        for sym in self.inputLattice.specieList:
            uniqueSpecies.add(sym)
        
        for sym in self.refLattice.specieList:
            uniqueSpecies.add(sym)
        
        # add elements to combo box
        self.fullSpecieList = []
        for sym in uniqueSpecies:
            self.fullSpecieList.append(sym)
        
        self.colourButtonDict = {}
        self.radiusSpinBoxDict = {}
        self.colourDict = {}
        self.radiusDict = {}
        for sym in self.fullSpecieList:
            group = QtGui.QGroupBox("%s - %s" % (sym, elements.atomName(sym)))
            group.setAlignment(QtCore.Qt.AlignCenter)
            groupLayout = QtGui.QVBoxLayout(group)
            groupLayout.setContentsMargins(0, 0, 0, 0)
            
            row = QtGui.QWidget(self)
            rowLayout = QtGui.QHBoxLayout(row)
            rowLayout.setContentsMargins(0, 0, 0, 0)
            rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
            
            # colour label
            label = QtGui.QLabel("Colour: ")
            rowLayout.addWidget(label)
            
            # colour
            RGB = elements.RGB(sym)
            col = QtGui.QColor(RGB[0]*255.0, RGB[1]*255.0, RGB[2]*255.0)
            self.colourDict[sym] = RGB
            
            # colour button
            button = QtGui.QPushButton("")
            button.setFixedWidth(50)
            button.setFixedHeight(30)
            button.setStyleSheet("QPushButton { background-color: %s }" % col.name())
            self.connect(button, QtCore.SIGNAL("clicked()"), lambda symbol=sym: self.showColourDialog(symbol))
            self.colourButtonDict[sym] = button
            rowLayout.addWidget(button)
            
            groupLayout.addWidget(row)
            
            row = QtGui.QWidget(self)
            rowLayout = QtGui.QHBoxLayout(row)
            rowLayout.setContentsMargins(0, 0, 0, 0)
            rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
            
            # radius label
            label = QtGui.QLabel("Radius: ")
            rowLayout.addWidget(label)
            
            # radius
            self.radiusDict[sym] = elements.covalentRadius(sym)
            
            # radius spin box
            spinBox = QtGui.QDoubleSpinBox(self)
            spinBox.setSingleStep(0.01)
            spinBox.setMinimum(0.0)
            spinBox.setMaximum(100.0)
            spinBox.setValue(elements.covalentRadius(sym))
            self.connect(spinBox, QtCore.SIGNAL('valueChanged(double)'), lambda x, symbol=sym: self.radiusChanged(x, symbol))
            self.radiusSpinBoxDict[sym] = spinBox
            rowLayout.addWidget(spinBox)
            
            groupLayout.addWidget(row)
            
            layout.addWidget(group)
        
        # buttons
        buttonContainer = QtGui.QWidget(self)
        buttonLayout = QtGui.QHBoxLayout(buttonContainer)
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.setSpacing(0)
        
        # apply button
        applyButton = QtGui.QPushButton(QtGui.QIcon(iconPath("redo_64.png")), "Apply")
        applyButton.setStatusTip("Apply changes to current session")
        applyButton.clicked.connect(self.applyChanges)
        
        saveButton = QtGui.QPushButton(QtGui.QIcon(iconPath("save_64.png")), "Save")
        applyButton.setStatusTip("Save changes for use in future sessions")
        saveButton.clicked.connect(self.saveChanges)
        
        resetButton = QtGui.QPushButton(QtGui.QIcon(iconPath("undo_64.png")), "Reset")
        applyButton.setStatusTip("Reset changes to last applied")
        resetButton.clicked.connect(self.resetChanges)
        
        buttonLayout.addWidget(applyButton)
        buttonLayout.addWidget(saveButton)
        buttonLayout.addWidget(resetButton)
        
        layout.addWidget(buttonContainer)
    
    def resetChanges(self):
        """
        Reset changes.
        
        """
        for sym in self.fullSpecieList:
            self.radiusSpinBoxDict[sym].setValue(elements.covalentRadius(sym))
            self.radiusDict[sym] = elements.covalentRadius(sym)
            
            RGB = elements.RGB(sym)
            col = QtGui.QColor(RGB[0]*255.0, RGB[1]*255.0, RGB[2]*255.0)
            self.colourDict[sym] = RGB
            self.colourButtonDict[sym].setStyleSheet("QPushButton { background-color: %s }" % col.name())
        
        self.parent.setStatus("Element properties reset")
    
    def saveChanges(self):
        """
        Save changes.
        
        """
        reply = QtGui.QMessageBox.question(self, "Message", 
                                           "This will overwrite the current element properties file. You should create a backup first!\n\nDo you wish to continue?",
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
        if reply == QtGui.QMessageBox.Yes:
            self.applyChanges()
            
            # save to file
            elements.write(resourcePath("data/atoms.IN"))
            
            self.parent.setStatus("Saved element properties")
    
    def applyChanges(self):
        """
        Apply changes.
        
        """
        for sym in self.fullSpecieList:
            
            radius = self.radiusDict[sym]
            
            RGB = self.colourDict[sym] 
            R = RGB[0]
            G = RGB[1]
            B = RGB[2]
            
            # first modify the Lattice objects
            if sym in self.inputLattice.specieList:
                index = np.where(self.inputLattice.specieList == sym)[0][0]
                
                # radius
                self.inputLattice.specieCovalentRadius[index] = radius
                
                # RGB
                self.inputLattice.specieRGB[index][0] = R
                self.inputLattice.specieRGB[index][1] = G
                self.inputLattice.specieRGB[index][2] = B
                
            
            if sym in self.refLattice.specieList:
                index = np.where(self.refLattice.specieList == sym)[0][0]
                
                # radius
                self.refLattice.specieCovalentRadius[index] = radius
                
                # RGB
                self.refLattice.specieRGB[index][0] = R
                self.refLattice.specieRGB[index][1] = G
                self.refLattice.specieRGB[index][2] = B
            
            # now modify elements structure
            elements.updateCovalentRadius(sym, radius)
            elements.updateRGB(sym, R, G, B)
            
        self.parent.setStatus("Element properties applied")
    
    def radiusChanged(self, val, sym):
        """
        Radius has been changed.
        
        """
        self.radiusDict[sym] = val
    
    def showColourDialog(self, sym):
        """
        Show the color dialog.
        
        """
        RGB = self.colourDict[sym]
        cur = QtGui.QColor(RGB[0] * 255.0, RGB[1] * 255.0, RGB[2] * 255.0)
        
        col = QtGui.QColorDialog.getColor(cur, self, "%s" % sym)
        
        if col.isValid():
            self.colourButtonDict[sym].setStyleSheet("QPushButton { background-color: %s }" % col.name())
            
            self.colourDict[sym][0] = float(col.red() / 255.0)
            self.colourDict[sym][1] = float(col.green() / 255.0)
            self.colourDict[sym][2] = float(col.blue() / 255.0)


################################################################################
class ImageViewer(QtGui.QDialog):
    """
    Image viewer.
    
    @author: Marc Robinson
    Rewritten by Chris Scott
    
    """
    def __init__(self, mainWindow, parent=None):
        super(ImageViewer, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.setWindowTitle("Image Viewer:")
        self.setWindowIcon(QtGui.QIcon(iconPath("applications-graphics.svg")))
        
        # main layout
        dialogLayout = QtGui.QHBoxLayout()
        
        # initial dir
        startDir = os.getcwd()
        
        # dir model
        self.model = QtGui.QFileSystemModel()
        self.model.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs | QtCore.QDir.Files)
        self.model.setNameFilters(["*.jpg", "*.tif","*.png","*.bmp"])
        self.model.setNameFilterDisables(0)
        self.model.setRootPath(startDir)
        
        # dir view
        self.view = QtGui.QTreeView(parent=self)
        self.view.setModel(self.model)
        self.view.clicked[QtCore.QModelIndex].connect(self.clicked)
        self.view.hideColumn(1)
        self.view.setRootIndex(self.model.index(startDir))
        self.view.setMinimumWidth(300)
        self.view.setColumnWidth(0, 150)
        self.view.setColumnWidth(2, 50)
        
        # add to main layout
        dialogLayout.addWidget(self.view)
        
        # image label
        self.imageLabel = QtGui.QLabel()
        
        column = QtGui.QWidget()
        columnLayout = QtGui.QVBoxLayout(column)
        columnLayout.setSpacing(0)
        columnLayout.setContentsMargins(0, 0, 0, 0)
        
        columnLayout.addWidget(self.imageLabel)
        
        # delete button
        deleteImageButton = QtGui.QPushButton(QtGui.QIcon(iconPath("edit-delete.svg")), "Delete image")
        deleteImageButton.clicked.connect(self.deleteImage)
        deleteImageButton.setStatusTip("Delete image")
        deleteImageButton.setAutoDefault(False)
        columnLayout.addWidget(deleteImageButton)
        
        # add to layout
        dialogLayout.addWidget(column)
        
        # set layout
        self.setLayout(dialogLayout)
    
    def clicked(self, index):
        """
        File clicked.
        
        """
        self.showImage(self.model.filePath(index))
    
    def showImage(self, filename):
        """
        Show image.
        
        """
        try:
            image = QtGui.QImage(filename)
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
            self.setWindowTitle("Image Viewer: %s" % filename)
        
        except:
            print "ERROR: could not display image in Image Viewer"
    
    def deleteImage(self):
        """
        Delete image.
        
        """
        success = self.model.remove(self.view.currentIndex())
        
        if success:
            self.clearImage()
    
    def clearImage(self):
        """
        Clear the image label.
        
        """
        self.imageLabel.clear()
        self.setWindowTitle("Image Viewer:")
    
    def changeDir(self, dirname):
        """
        Change directory
        
        """
        self.view.setRootIndex(self.model.index(dirname))
        self.clearImage()
    
    def keyReleaseEvent(self, event):
        """
        Handle up/down key press
        
        """
        if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_Down:
            self.model.filePath(self.view.currentIndex())
            self.showImage(self.model.filePath(self.view.currentIndex()))
