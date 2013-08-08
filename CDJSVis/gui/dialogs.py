
"""
Additional dialogs.

@author: Chris Scott

"""
import os
import sys

from PySide import QtGui, QtCore
import numpy as np

from . import genericForm
from ..atoms import elements
from ..visutils.utilities import resourcePath, iconPath
from ..visutils import vectors
from ..visutils import utilities
from ..rendering import highlight

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class NewRendererWindowDialog(QtGui.QDialog):
    """
    Dialog for opening new renderer window.
    
    """
    def __init__(self, parent=None):
        super(NewRendererWindowDialog, self).__init__(parent)
        
        self.parent = parent
        self.setModal(True)
        
        self.setWindowTitle("Open new render window")
        self.setWindowIcon(QtGui.QIcon(iconPath("window-new.svg")))
        
        layout = QtGui.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignHCenter)
        
        label = QtGui.QLabel("Number of viewports: ")
        
        NViewPortsSpin = QtGui.QSpinBox()
        NViewPortsSpin.setMinimum(1)
        NViewPortsSpin.setMaximum(2)
        NViewPortsSpin.setValue(1)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(NViewPortsSpin)
        
        layout.addLayout(row)
        
        # buttons
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        
        layout.addWidget(buttonBox)
        

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
        self.setModal(0)
        
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
        applyButton.setToolTip("Apply changes to current session")
        applyButton.clicked.connect(self.applyChanges)
        
        saveButton = QtGui.QPushButton(QtGui.QIcon(iconPath("save_64.png")), "Save")
        saveButton.setStatusTip("Save changes for use in future sessions")
        saveButton.setToolTip("Save changes for use in future sessions")
        saveButton.clicked.connect(self.saveChanges)
        
        resetButton = QtGui.QPushButton(QtGui.QIcon(iconPath("undo_64.png")), "Reset")
        resetButton.setStatusTip("Reset changes to last applied")
        resetButton.setToolTip("Reset changes to last applied")
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
            elements.write(resourcePath("atoms.IN"))
            
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
        reply = QtGui.QMessageBox.question(self, "Message", 
                                           "Delete file: %s?" % self.model.filePath(self.view.currentIndex()),
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
        if reply == QtGui.QMessageBox.Yes:
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

################################################################################

class OnScreenTextListWidget(QtGui.QListWidget):
    """
    Override QListWidget to allow drag/drops.
    
    """
    def __init__(self, parent=None):
        super(OnScreenTextListWidget, self).__init__(parent)
        
#        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
    
#    def dragEnterEvent(self, event):
#        """
#        
#        
#        """
##        print "DRAG ENTER EVENT", event
##        event.accept()
#    
#    def dragMoveEvent(self, event):
#        """
#        
#        
#        """
#        print "DRAG MOVE EVENT", event
#        event.accept()
    
#    def dropEvent(self, event):
#        """
#        
#        
#        """
#        print "DROP EVENT"

################################################################################

class TextSettingsDialog(QtGui.QDialog):
    """
    Dialog for setting text options.
    
    """
    def __init__(self, title, parent=None):
        super(TextSettingsDialog, self).__init__(parent)
        
        self.parent = parent
        
        self.setModal(1)
        
        titleText = "%s settings" % title
        self.setWindowTitle(titleText)
        
        dialogLayout = QtGui.QVBoxLayout(self)
        
        groupBox = genericForm.GenericForm(self, 0, titleText)
        groupBox.show()
        
        # defaults
        self.textPosition = "Top left"
        
        # location of text
        label = QtGui.QLabel("Text location: ")
        self.positionComboBox = QtGui.QComboBox()
        self.positionComboBox.addItem("Top left")
        self.positionComboBox.addItem("Top right")
        self.positionComboBox.currentIndexChanged.connect(self.positionChanged)
        
        row = groupBox.newRow()
        row.addWidget(label)
        row.addWidget(self.positionComboBox)
        
        dialogLayout.addWidget(groupBox)
        
        # add close button
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.acceptOverride)
        
        dialogLayout.addWidget(buttonBox)
    
    def acceptOverride(self):
        """
        Accepted.
        
        """
        self.hide()
    
    def positionChanged(self, item):
        """
        Position changed.
        
        """
        self.textPosition = str(self.positionComboBox.currentText())


################################################################################

class OnScreenInfoDialog(QtGui.QDialog):
    """
    On screen info selector.
    
    """
    def __init__(self, mainWindow, index, parent=None):
        super(OnScreenInfoDialog, self).__init__(parent)
        
        self.parent = parent
        self.rendererWindow = parent
        self.mainWindow = mainWindow
        
        self.setWindowTitle("On screen info - Render window %d" % index)
        self.setWindowIcon(QtGui.QIcon(iconPath("preferences-desktop-font.svg")))
        
        dialogLayout = QtGui.QVBoxLayout()
        self.setLayout(dialogLayout)
        
        # row
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        
        # list containing selected text
        col = QtGui.QWidget()
        colLayout = QtGui.QVBoxLayout(col)
        
        label = QtGui.QLabel("Selected text")
        colLayout.addWidget(label)
        
        self.selectedText = OnScreenTextListWidget(self)
        self.selectedText.setFixedHeight(200)
        colLayout.addWidget(self.selectedText)
        
        rowLayout.addWidget(col)
        
        # buttons
        col = QtGui.QWidget()
        colLayout = QtGui.QVBoxLayout(col)
        
        colLayout.addStretch()
        
        selectButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-previous.svg")), "")
        selectButton.setStatusTip("Select text")
        selectButton.setAutoDefault(False)
        selectButton.clicked.connect(self.selectButtonClicked)
        colLayout.addWidget(selectButton)
        
        removeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-next.svg")), "")
        removeButton.setStatusTip("Remove text")
        removeButton.setAutoDefault(False)
        removeButton.clicked.connect(self.removeButtonClicked)
        colLayout.addWidget(removeButton)
        
        colLayout.addStretch()
        
        rowLayout.addWidget(col)
        
        # list containing available (unselected) text
        col = QtGui.QWidget()
        colLayout = QtGui.QVBoxLayout(col)
        
        label = QtGui.QLabel("Available text")
        colLayout.addWidget(label)
        
        self.availableText = OnScreenTextListWidget(self)
        self.availableText.setFixedHeight(200)
        colLayout.addWidget(self.availableText)
        
        rowLayout.addWidget(col)
        
        dialogLayout.addWidget(row)
        
        # refresh button
        refreshButton = QtGui.QPushButton("Refresh")
        refreshButton.setAutoDefault(0)
        refreshButton.setStatusTip("Refresh on-screen information")
        refreshButton.clicked.connect(self.refresh)
        
        buttonWidget = QtGui.QWidget()
        buttonLayout = QtGui.QHBoxLayout(buttonWidget)
        buttonLayout.addStretch()
        buttonLayout.addWidget(refreshButton)
        buttonLayout.addStretch()
        
        dialogLayout.addWidget(buttonWidget)
        
        # add options
        self.selectedText.addItem("Atom count")
        self.selectedText.addItem("Visible count")
        self.availableText.addItem("Visible specie count")
        self.availableText.addItem("Simulation time")
        self.availableText.addItem("Energy barrier")
        self.availableText.addItem("Defect count")
        self.selectedText.addItem("Defect specie count")
        
        # add settings
        self.textSettings = {}
        
        for i in xrange(self.selectedText.count()):
            item = self.selectedText.item(i)
            text = str(item.text())
            
            self.textSettings[text] = TextSettingsDialog(text, self)
        
        for i in xrange(self.availableText.count()):
            item = self.availableText.item(i)
            text = str(item.text())
            
            self.textSettings[text] = TextSettingsDialog(text, self)
        
        # connect
        self.selectedText.itemDoubleClicked.connect(self.showTextSettingsDialog)
    
    def showTextSettingsDialog(self, item):
        """
        Show text settings dialog.
        
        """
        text = str(item.text())
        
        self.textSettings[text].hide()
        self.textSettings[text].show()
    
    def refresh(self):
        """
        Refresh on screen info.
        
        """
        self.parent.refreshOnScreenInfo()
    
    def selectButtonClicked(self):
        """
        Select text.
        
        """
        row = self.availableText.currentRow()
        
        if row < 0:
            return
        
        self.selectedText.addItem(self.availableText.takeItem(row))
    
    def removeButtonClicked(self):
        """
        Select text.
        
        """
        row = self.selectedText.currentRow()
        
        if row < 0:
            return
        
        self.availableText.addItem(self.selectedText.takeItem(row))
    
    def refreshLists(self):
        """
        Refresh lists.
        
        Remove options that are no longer available.
        Add options that are now available.
        
        """
        pass

################################################################################

class DefectInfoWindow(QtGui.QDialog):
    """
    Atom info window.
    
    """
    def __init__(self, rendererWindow, defectIndex, defectType, defList, parent=None):
        super(DefectInfoWindow, self).__init__(parent)
        
        self.parent = parent
        self.rendererWindow = rendererWindow
        self.defectIndex = defectIndex
        self.defectType = defectType
        self.defList = defList
        
        inputState = self.rendererWindow.getCurrentInputState()
        refState = self.rendererWindow.getCurrentRefState()
        
        self.setWindowTitle("Defect info")
        
        layout = QtGui.QVBoxLayout()
        
        if defectType == 1:
            vacancies = defList[0]
            
            # vacancy
            index = vacancies[defectIndex]
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Type: vacancy"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Specie: %s" % refState.specieList[refState.specie[index]]))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Position: (%f, %f, %f)" % (refState.pos[3*index], refState.pos[3*index+1], refState.pos[3*index+2])))
            layout.addLayout(row)
            
        elif defectType == 2:
            interstitials = defList[0]
            
            # vacancy
            index = interstitials[defectIndex]
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Type: interstitial"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Specie: %s" % inputState.specieList[inputState.specie[index]]))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Position: (%f, %f, %f)" % (inputState.pos[3*index], inputState.pos[3*index+1], inputState.pos[3*index+2])))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("PE: %f eV" % (inputState.PE[index],)))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("KE: %f eV" % (inputState.KE[index],)))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Charge: %f" % (inputState.charge[index],)))
            layout.addLayout(row)
        
        elif defectType == 3:
            antisites = defList[0]
            onAntisites = defList[1]
            
            # antisite
            index = antisites[defectIndex]
            index2 = onAntisites[defectIndex]
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Type: antisite"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Position: (%f, %f, %f)" % (refState.pos[3*index], refState.pos[3*index+1], refState.pos[3*index+2])))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Specie: %s" % refState.specieList[refState.specie[index]]))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Occupying atom:"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Specie: %s" % inputState.specieList[inputState.specie[index2]]))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    PE: %f eV" % (inputState.PE[index2],)))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    KE: %f eV" % (inputState.KE[index2],)))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Charge: %f" % (inputState.charge[index2],)))
            layout.addLayout(row)
        
        elif defectType == 4:
            splitInts = defList[0]
            
            # split interstitial
            vacIndex = splitInts[3*defectIndex]
            int1Index = splitInts[3*defectIndex+1]
            int2Index = splitInts[3*defectIndex+2]
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Type: split interstitial"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Vacancy position: (%f, %f, %f)" % (refState.pos[3*vacIndex], refState.pos[3*vacIndex+1], refState.pos[3*vacIndex+2])))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Vacancy specie: %s" % refState.specieList[refState.specie[vacIndex]]))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Interstitial 1:"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Position: (%f, %f, %f)" % (inputState.pos[3*int1Index], inputState.pos[3*int1Index+1], inputState.pos[3*int1Index+2])))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Specie: %s" % inputState.specieList[inputState.specie[int1Index]]))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    PE: %f eV" % (inputState.PE[int1Index],)))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    KE: %f eV" % (inputState.KE[int1Index],)))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Charge: %f" % (inputState.charge[int1Index],)))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Interstitial 2:"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Position: (%f, %f, %f)" % (inputState.pos[3*int2Index], inputState.pos[3*int2Index+1], inputState.pos[3*int2Index+2])))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Specie: %s" % inputState.specieList[inputState.specie[int2Index]]))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    PE: %f eV" % (inputState.PE[int2Index],)))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    KE: %f eV" % (inputState.KE[int2Index],)))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Charge: %f" % (inputState.charge[int2Index],)))
            layout.addLayout(row)
            
            # orientation
            pos1 = inputState.pos[3*int1Index:3*int1Index+3]
            pos2 = inputState.pos[3*int2Index:3*int2Index+3]
            
            pp = rendererWindow.getCurrentPipelinePage()
            
            sepVec = vectors.separationVector(pos1, pos2, inputState.cellDims, pp.PBC)
            norm = vectors.normalise(sepVec)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Orientation: (%f %f %f)" % (norm[0], norm[1], norm[2])))
            layout.addLayout(row)
        
        self.setLayout(layout)

################################################################################

class AtomInfoWindow(QtGui.QDialog):
    """
    Atom info window.
    
    """
    def __init__(self, rendererWindow, atomIndex, scalar, scalarType, parent=None):
        super(AtomInfoWindow, self).__init__(parent)
        
        self.parent = parent
        self.rendererWindow = rendererWindow
        self.atomIndex = atomIndex
        
        lattice = self.rendererWindow.getCurrentInputState()
        
        self.highlighter = highlight.AtomHighlighter(self, self.rendererWindow.vtkRen, self.rendererWindow.vtkRenWinInteract)
        
        self.setWindowTitle("Atom info")
        
        layout = QtGui.QVBoxLayout()
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Atom: %d" % atomIndex))
        layout.addLayout(row)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Specie: %s" % lattice.specieList[lattice.specie[atomIndex]]))
        layout.addLayout(row)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Position: (%f, %f, %f)" % (lattice.pos[3*atomIndex], lattice.pos[3*atomIndex+1], lattice.pos[3*atomIndex+2])))
        layout.addLayout(row)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("PE: %f eV" % (lattice.PE[atomIndex],)))
        layout.addLayout(row)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("KE: %f eV" % (lattice.KE[atomIndex],)))
        layout.addLayout(row)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Charge: %f" % (lattice.charge[atomIndex],)))
        layout.addLayout(row)
        
        if scalar is not None and scalarType is not None:
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("%s: %f" % (scalarType, scalar)))
            layout.addLayout(row)
        
        self.highlighter.add(lattice.atomPos(atomIndex), lattice.specieCovalentRadius[lattice.specie[atomIndex]])
        
        self.setLayout(layout)
    
    def closeEvent(self, event):
        """
        Override close event
        
        """
        self.highlighter.remove()
        
        event.accept()

################################################################################

class ConfirmCloseDialog(QtGui.QDialog):
    """
    Confirm close dialog.
    
    """
    def __init__(self, parent=None):
        super(ConfirmCloseDialog, self).__init__(parent)
        
        self.setModal(1)
        self.setWindowTitle("Exit application?")
        
        layout = QtGui.QVBoxLayout(self)
        
        # label
        label = QtGui.QLabel("<b>Are you sure you want to exit?</b>")
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        layout.addLayout(row)
        
        # clear settings
        self.clearSettingsCheck = QtGui.QCheckBox("Clear settings")
        row = QtGui.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(self.clearSettingsCheck)
        layout.addLayout(row)
        
        # buttons
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Yes | QtGui.QDialogButtonBox.No)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        
        layout.addWidget(buttonBox)
        
