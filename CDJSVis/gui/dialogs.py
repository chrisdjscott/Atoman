
"""
Additional dialogs.

@author: Chris Scott

"""
import os
import sys
import copy
import logging

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

class CameraSettingsDialog(QtGui.QDialog):
    """
    Camera settings dialog
    
    """
    def __init__(self, parent, renderer):
        super(CameraSettingsDialog, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.renderer = renderer
        
        self.setModal(True)
        
        self.setWindowTitle("Camera settings")
        self.setWindowIcon(QtGui.QIcon(iconPath("cam.png")))
        
        self.contentLayout = QtGui.QVBoxLayout(self)
#         self.contentLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        # ini vals
        self.campos = list(renderer.camera.GetPosition())
        self.camfoc = list(renderer.camera.GetFocalPoint())
        self.camvup = list(renderer.camera.GetViewUp())
        
        self.camposbkup = copy.deepcopy(self.campos)
        self.camfocbkup = copy.deepcopy(self.camfoc)
        self.camvupbkup = copy.deepcopy(self.camvup)
        
        # row
        row = self.newRow()
        
        label = QtGui.QLabel("Position: ")
        row.addWidget(label)
        
        # cam pos
        self.camPosXSpin = QtGui.QDoubleSpinBox()
        self.camPosXSpin.setMinimum(-99999.0)
        self.camPosXSpin.setMaximum(99999.0)
        self.camPosXSpin.setValue(self.campos[0])
        self.camPosXSpin.valueChanged[float].connect(self.camxposChanged)
        row.addWidget(self.camPosXSpin)
        
        self.camPosYSpin = QtGui.QDoubleSpinBox()
        self.camPosYSpin.setMinimum(-99999.0)
        self.camPosYSpin.setMaximum(99999.0)
        self.camPosYSpin.setValue(self.campos[1])
        self.camPosYSpin.valueChanged[float].connect(self.camyposChanged)
        row.addWidget(self.camPosYSpin)
        
        self.camPosZSpin = QtGui.QDoubleSpinBox()
        self.camPosZSpin.setMinimum(-99999.0)
        self.camPosZSpin.setMaximum(99999.0)
        self.camPosZSpin.setValue(self.campos[2])
        self.camPosZSpin.valueChanged[float].connect(self.camzposChanged)
        row.addWidget(self.camPosZSpin)
        
        # row
        row = self.newRow()
        
        label = QtGui.QLabel("Focal point: ")
        row.addWidget(label)
        
        # cam focal point
        self.camFocXSpin = QtGui.QDoubleSpinBox()
        self.camFocXSpin.setMinimum(-99999.0)
        self.camFocXSpin.setMaximum(99999.0)
        self.camFocXSpin.setValue(self.camfoc[0])
        self.camFocXSpin.valueChanged[float].connect(self.camxfocChanged)
        row.addWidget(self.camFocXSpin)
        
        self.camFocYSpin = QtGui.QDoubleSpinBox()
        self.camFocYSpin.setMinimum(-99999.0)
        self.camFocYSpin.setMaximum(99999.0)
        self.camFocYSpin.setValue(self.camfoc[1])
        self.camFocYSpin.valueChanged[float].connect(self.camyfocChanged)
        row.addWidget(self.camFocYSpin)
        
        self.camFocZSpin = QtGui.QDoubleSpinBox()
        self.camFocZSpin.setMinimum(-99999.0)
        self.camFocZSpin.setMaximum(99999.0)
        self.camFocZSpin.setValue(self.camfoc[2])
        self.camFocZSpin.valueChanged[float].connect(self.camzfocChanged)
        row.addWidget(self.camFocZSpin)
        
        # row
        row = self.newRow()
        
        label = QtGui.QLabel("View up: ")
        row.addWidget(label)
        
        # cam focal point
        self.camVupXSpin = QtGui.QDoubleSpinBox()
        self.camVupXSpin.setMinimum(-99999.0)
        self.camVupXSpin.setMaximum(99999.0)
        self.camVupXSpin.setValue(self.camvup[0])
        self.camVupXSpin.valueChanged[float].connect(self.camxvupChanged)
        row.addWidget(self.camVupXSpin)
        
        self.camVupYSpin = QtGui.QDoubleSpinBox()
        self.camVupYSpin.setMinimum(-99999.0)
        self.camVupYSpin.setMaximum(99999.0)
        self.camVupYSpin.setValue(self.camvup[1])
        self.camVupYSpin.valueChanged[float].connect(self.camyvupChanged)
        row.addWidget(self.camVupYSpin)
        
        self.camVupZSpin = QtGui.QDoubleSpinBox()
        self.camVupZSpin.setMinimum(-99999.0)
        self.camVupZSpin.setMaximum(99999.0)
        self.camVupZSpin.setValue(self.camvup[2])
        self.camVupZSpin.valueChanged[float].connect(self.camzvupChanged)
        row.addWidget(self.camVupZSpin)
        
        # reset button
        resetButton = QtGui.QPushButton(QtGui.QIcon(iconPath("undo_64.png")), "Reset")
        resetButton.setStatusTip("Reset changes")
        resetButton.setToolTip("Reset changes")
        resetButton.clicked.connect(self.resetChanges)
        
        row = self.newRow()
        row.addWidget(resetButton)
    
    def resetChanges(self):
        """
        Reset changes
        
        """
        self.campos = self.camposbkup
        self.camfoc = self.camfocbkup
        self.camvup = self.camvupbkup
        
        self.renderer.camera.SetPosition(self.campos)
        self.renderer.camera.SetFocalPoint(self.camfoc)
        self.renderer.camera.SetViewUp(self.camvup)
        
        self.renderer.reinit()
    
    def camxposChanged(self, val):
        """
        Cam x pos changed
        
        """
        self.campos[0] = val
        self.renderer.camera.SetPosition(self.campos)
        self.renderer.reinit()
    
    def camyposChanged(self, val):
        """
        Cam y pos changed
        
        """
        self.campos[1] = val
        self.renderer.camera.SetPosition(self.campos)
        self.renderer.reinit()
    
    def camzposChanged(self, val):
        """
        Cam z pos changed
        
        """
        self.campos[2] = val
        self.renderer.camera.SetPosition(self.campos)
        self.renderer.reinit()
    
    def camxfocChanged(self, val):
        """
        Cam x foc changed
        
        """
        self.camfoc[0] = val
        self.renderer.camera.SetFocalPoint(self.camfoc)
        self.renderer.reinit()
    
    def camyfocChanged(self, val):
        """
        Cam y foc changed
        
        """
        self.camfoc[1] = val
        self.renderer.camera.SetFocalPoint(self.camfoc)
        self.renderer.reinit()
    
    def camzfocChanged(self, val):
        """
        Cam z foc changed
        
        """
        self.camfoc[2] = val
        self.renderer.camera.SetFocalPoint(self.camfoc)
        self.renderer.reinit()
    
    def camxvupChanged(self, val):
        """
        Cam x foc changed
        
        """
        self.camvup[0] = val
        self.renderer.camera.SetViewUp(self.camvup)
        self.renderer.reinit()
    
    def camyvupChanged(self, val):
        """
        Cam y foc changed
        
        """
        self.camvup[1] = val
        self.renderer.camera.SetViewUp(self.camvup)
        self.renderer.reinit()
    
    def camzvupChanged(self, val):
        """
        Cam z foc changed
        
        """
        self.camvup[2] = val
        self.renderer.camera.SetViewUp(self.camvup)
        self.renderer.reinit()
    
    def newRow(self, align="Right"):
        """
        New row
        
        """
        row = genericForm.FormRow(align=align)
        self.contentLayout.addWidget(row)
        
        return row

################################################################################

class ConsoleWindow(QtGui.QDialog):
    """
    Console window for displaying output to the user.
    
    """
    def __init__(self, parent=None):
        super(ConsoleWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
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
        
        # logging handler
        handler = utilities.TextEditHandler(self.textWidget)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        handler.setFormatter(formatter)
        
        # add to root logger
        logging.getLogger().addHandler(handler)
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        root = logging.getLogger()
        print "HANDLERS", root.handlers
        
        
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
        
#         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
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
        self.systemsDialog = self.parent.systemsDialog
        
        # list of unique species
        uniqueSpecies = set()
        for latt in self.systemsDialog.lattice_list:
            for sym in latt.specieList:
                uniqueSpecies.add(sym)
            
            for sym in latt.specieList:
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
            
            for latt in self.systemsDialog.lattice_list:
                # first modify the Lattice objects
                if sym in latt.specieList:
                    index = np.where(latt.specieList == sym)[0][0]
                    
                    # radius
                    latt.specieCovalentRadius[index] = radius
                    
                    # RGB
                    latt.specieRGB[index][0] = R
                    latt.specieRGB[index][1] = G
                    latt.specieRGB[index][2] = B
            
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
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
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
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
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

class ConfirmCloseDialog(QtGui.QDialog):
    """
    Confirm close dialog.
    
    """
    def __init__(self, parent=None):
        super(ConfirmCloseDialog, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
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

################################################################################

class RotateViewPointDialog(QtGui.QDialog):
    """
    Rotate view point dialog
    
    """
    def __init__(self, rw, parent=None):
        super(RotateViewPointDialog, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.setModal(0)
        
        self.rw = rw
        self.parent = parent
        
        layout = QtGui.QVBoxLayout(self)
        
        # direction
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        
        label = QtGui.QLabel("Direction:")
        
        self.directionCombo = QtGui.QComboBox()
        self.directionCombo.addItems(["Right", "Left", "Up", "Down"])
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.directionCombo)
        
        layout.addWidget(row)
        
        # angle
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        
        label = QtGui.QLabel("Angle:")
        
        self.angleSpin = QtGui.QDoubleSpinBox()
        self.angleSpin.setSingleStep(0.1)
        self.angleSpin.setMinimum(0.0)
        self.angleSpin.setMaximum(360.0)
        self.angleSpin.setValue(90)
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.angleSpin)
        
        layout.addWidget(row)
        
        # apply button
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        
        applyButton = QtGui.QPushButton("Apply")
        applyButton.setStatusTip("Apply rotation")
        applyButton.setToolTip("Apply rotation")
        applyButton.clicked.connect(self.applyRotation)
        
        rowLayout.addWidget(applyButton)
        
        layout.addWidget(row)
    
    def applyRotation(self):
        """
        Apply the rotation
        
        """
        angle = self.angleSpin.value()
        direction = str(self.directionCombo.currentText())
        
        renderer = self.rw.renderer
        
        if direction == "Right" or direction == "Left":
            if direction == "Right":
                angle = - angle
            
            # apply rotation
            renderer.camera.Azimuth(angle)
        
        else:
            if direction == "Up":
                angle = - angle
            
            # apply rotation
            renderer.camera.Elevation(angle)
        
        renderer.reinit()
