# -*- coding: utf-8 -*-

"""
Additional dialogs.

@author: Chris Scott

"""
import os
import copy
import logging

from PySide import QtGui, QtCore
import numpy as np

from . import genericForm
from ..state.atoms import elements
from ..visutils.utilities import resourcePath, iconPath
from ..visutils import utilities


################################################################################

class CameraSettingsDialog(QtGui.QDialog):
    """
    Camera settings dialog
    
    """
    def __init__(self, parent, renderer):
        super(CameraSettingsDialog, self).__init__(parent)
        
#         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
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
        
        self.iniWinFlags = self.windowFlags()
        self.setWindowFlags(self.iniWinFlags | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.setModal(0)
#        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Console")
        self.setWindowIcon(QtGui.QIcon(iconPath("console-icon.png")))
        self.resize(800,400)
        
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
        self.clearButton.clicked.connect(self.clearText)
        
        self.saveButton = QtGui.QPushButton("Save")
        self.saveButton.setAutoDefault(0)
        self.saveButton.clicked.connect(self.saveText)
        
        self.closeButton = QtGui.QPushButton("Hide")
        self.closeButton.setAutoDefault(1)
        self.closeButton.clicked.connect(self.close)
        
        # logging handler
        handler = utilities.TextEditHandler(self.textWidget)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        handler.setFormatter(formatter)
        
        # add to root logger
        logging.getLogger().addHandler(handler)
        
        # set level, try settings first or fallback to INFO
        settings = QtCore.QSettings()
        level = int(settings.value("logging/console", logging.INFO))
        
        logger = logging.getLogger(__name__)
        handler.setLevel(int(level))
        logger.debug("Initial console window logging level: %s", logging.getLevelName(level))
        
        self.logger = logger
        
        self.loggingLevels = {"CRITICAL": logging.CRITICAL,
                              "ERROR": logging.ERROR,
                              "WARNING": logging.WARNING,
                              "INFO": logging.INFO,
                              "DEBUG": logging.DEBUG}
        
        self.loggingLevelsSorted = ["CRITICAL",
                                    "ERROR",
                                    "WARNING",
                                    "INFO",
                                    "DEBUG"]
        
        # should get these from settings
        consoleLevel = level
        consoleLevelIndex = self.getLevelIndex(consoleLevel)
        
        self.consoleLevelCombo = QtGui.QComboBox()
        self.consoleLevelCombo.addItems(self.loggingLevelsSorted)
        self.consoleLevelCombo.currentIndexChanged[str].connect(self.consoleLevelChanged)
        self.consoleLevelCombo.setCurrentIndex(consoleLevelIndex)
        label = QtGui.QLabel("Level:")
        
        buttonWidget = QtGui.QWidget()
        buttonLayout = QtGui.QHBoxLayout(buttonWidget)
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.saveButton)
        buttonLayout.addStretch()
        buttonLayout.addWidget(label)
        buttonLayout.addWidget(self.consoleLevelCombo)
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.closeButton)
        
        consoleLayout.addWidget(buttonWidget)
    
    def getLevelIndex(self, level):
        """
        Return index of level
        
        """
        levelKey = None
        for key, val in self.loggingLevels.iteritems():
            if val == level:
                levelKey = key
                break
        
        if levelKey is None:
            logger = logging.getLogger(__name__)
            logger.critical("No match for log level: %s", str(level))
            return 2
        
        return self.loggingLevelsSorted.index(levelKey)
    
    def consoleLevelChanged(self, levelKey):
        """
        Console window logging level has changed
        
        """
        levelKey = str(levelKey)
        level = self.loggingLevels[levelKey]
        
        # get handler (console window is second)
        handler = logging.getLogger().handlers[1]
        
        # set level
        handler.setLevel(level)
        
        # update settings
        settings = QtCore.QSettings()
        settings.setValue("logging/console", level)
    
    def saveText(self):
        """
        Save text to file
        
        """
        self.setWindowFlags(self.iniWinFlags)
        
        # get file name
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save Console Output', '.', "HTML files (*.html)")[0]
        
        self.setWindowFlags(self.iniWinFlags | QtCore.Qt.WindowStaysOnTopHint)
        self.show()
        
        if len(filename):
            if not filename.endswith(".html"):
                filename += ".html"
            
            self.logger.debug("Saving console output to file: '%s'", filename)
            
            # write to file
            f = open(filename, "w")
            f.write(self.textWidget.toHtml())
            f.close()
    
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
            for _ in xrange(indent):
                ind += "  "
            self.textWidget.append("%s %s%s" % (">", ind, string))
        
    def closeEvent(self, event):
        self.hide()
        self.parent.consoleOpen = 0

################################################################################

class BondEditorSettingsForm(genericForm.GenericForm):
    """
    Settings for bond
    
    """
    def __init__(self, parent, syma, symb, bondMin, bondMax):
        super(BondEditorSettingsForm, self).__init__(parent, None, "%s - %s" % (syma, symb))
        
        self.syma = syma
        self.symb = symb
        self.bondMin = bondMin
        self.bondMax = bondMax
        
        row = self.newRow()
        row.addWidget(QtGui.QLabel("Bond min:"))
        
        bondMinSpin = QtGui.QDoubleSpinBox()
        bondMinSpin.setSingleStep(0.1)
        bondMinSpin.setMinimum(0.0)
        bondMinSpin.setMaximum(99.99)
        bondMinSpin.setValue(self.bondMin)
        bondMinSpin.valueChanged.connect(self.bondMinChanged)
        row.addWidget(bondMinSpin)
        
        row = self.newRow()
        row.addWidget(QtGui.QLabel("Bond max:"))
        
        bondMaxSpin = QtGui.QDoubleSpinBox()
        bondMaxSpin.setSingleStep(0.1)
        bondMaxSpin.setMinimum(0.0)
        bondMaxSpin.setMaximum(99.99)
        bondMaxSpin.setValue(self.bondMax)
        bondMaxSpin.valueChanged.connect(self.bondMaxChanged)
        row.addWidget(bondMaxSpin)
    
    def updateBondData(self):
        """
        Update bond data in elements object
        
        """
        logger = logging.getLogger(__name__)
        logger.debug("Updating bond data: %s - %s : (%.2f, %.2f)" % (self.syma, self.symb, self.bondMin, self.bondMax))
        
        elements.bondDict[self.syma][self.symb] = (self.bondMin, self.bondMax)
        elements.bondDict[self.symb][self.syma] = (self.bondMin, self.bondMax)
        
        self.parent.settingModified("%s - %s" % (self.syma, self.symb))
    
    def bondMinChanged(self, val):
        """
        Bond min changed
        
        """
        self.bondMin = val
        self.updateBondData()
    
    def bondMaxChanged(self, val):
        """
        Bond max changed
        
        """
        self.bondMax = val
        self.updateBondData()

################################################################################

class AddBondDialog(QtGui.QDialog):
    """
    Add bond dialog
    
    """
    def __init__(self, parent=None):
        super(AddBondDialog, self).__init__(parent)
        
        self.parent = parent
        self.setModal(1)
        
        self.setWindowTitle("Add bond")
        self.setWindowIcon(QtGui.QIcon(iconPath("other/molecule1.png")))
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        layout = QtGui.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignHCenter)
        
        # list of elements
        elementsList = elements.listElements()
        
        # row
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)
        
        # first combo
        self.specieComboA = QtGui.QComboBox()
        self.specieComboA.addItems(elementsList)
        
        # second combo
        self.specieComboB = QtGui.QComboBox()
        self.specieComboB.addItems(elementsList)
        
        # add to row
        rowLayout.addStretch(1)
        rowLayout.addWidget(self.specieComboA)
        rowLayout.addWidget(QtGui.QLabel(" - "))
        rowLayout.addWidget(self.specieComboB)
        rowLayout.addStretch(1)
        
        # add to layout
        layout.addWidget(row)
        
        # button box
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)
        
        self.setLayout(layout)

################################################################################

class BondEditorDialog(QtGui.QDialog):
    """
    Bond editor dialog
    
    """
    def __init__(self, parent=None):
        super(BondEditorDialog, self).__init__(parent)
        
        self.iniWinFlags = self.windowFlags()
        self.setWindowFlags(self.iniWinFlags | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.mainWindow = parent
        self.setModal(0)
        
        self.setWindowTitle("Bonds editor")
        self.setWindowIcon(QtGui.QIcon(iconPath("other/molecule1.png")))
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.dirty = False
        self.modifiedList = set()
        self.bondsSet = set()
        
        layout = QtGui.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignHCenter)
#        layout.setContentsMargins(0, 0, 0, 0)
#        layout.setSpacing(0)
        
        # combo box with pairs
        self.bondsCombo = QtGui.QComboBox()
        self.bondsCombo.currentIndexChanged.connect(self.setWidgetStack)
        
        # add button
        addButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/list-add.png")), "")
        addButton.setFixedWidth(35)
        addButton.setToolTip("Add new bond pair")
        addButton.setStatusTip("Add new bond pair")
        addButton.clicked.connect(self.addBondClicked)
        
        row = QtGui.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.bondsCombo)
        row.addWidget(addButton)
        row.addStretch(1)
        layout.addLayout(row)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget()
        layout.addWidget(self.stackedWidget)
        
        # populate combo and stacked widget
        bondDict = elements.bondDict
        keyas = sorted(bondDict.keys())
        for keya in keyas:
            keybs = sorted(bondDict[keya].keys())
            for keyb in keybs:
                if keyb > keya:
                    continue
                
                # min, max
                bondMin, bondMax = bondDict[keya][keyb]
                
                # add
                self.addBond(keya, keyb, bondMin, bondMax)
        
        # save button
        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Save)
        self.buttonBox.button(QtGui.QDialogButtonBox.Save).setEnabled(False)
        self.buttonBox.accepted.connect(self.saveChanges)
        layout.addWidget(self.buttonBox)
    
    def addBond(self, syma, symb, bondMin=0.0, bondMax=0.0):
        """
        Add the bond to the combo and settings form to stacked widget
        
        """
        # settings form
        form = BondEditorSettingsForm(self, syma, symb, bondMin, bondMax)
        
        # add
        text = "%s - %s" % (syma, symb)
        self.bondsCombo.addItem(text)
        self.bondsSet.add(text)
        self.stackedWidget.addWidget(form)
    
    def addBondClicked(self):
        """
        Add a new bond
        
        """
        logger = logging.getLogger(__name__)
        
        # dialog
        dlg = AddBondDialog(self)
        ret = dlg.exec_()
        if ret:
            # check if already exists
            syma = str(dlg.specieComboA.currentText())
            symb = str(dlg.specieComboB.currentText())
            
            texta = "%s - %s" % (syma, symb)
            textb = "%s - %s" % (symb, syma)
            
            if texta in self.bondsSet or textb in self.bondsSet:
                pass
            
            else:
                logger.info("Adding new bond: '%s - %s'", syma, symb)
                
                # add
                self.addBond(syma, symb)
                
                # select
                self.bondsCombo.setCurrentIndex(self.bondsCombo.count() - 1)
                
                # add to bond dict (elements)
                elements.addBond(syma, symb, 0.0, 0.0)
    
    def settingModified(self, pairString):
        """
        Setting has been modified
        
        """
        self.dirty = True
        self.buttonBox.button(QtGui.QDialogButtonBox.Save).setEnabled(True)
        self.modifiedList.add(pairString)
    
    def saveChanges(self):
        """
        Save changes
        
        """
        logger = logging.getLogger(__name__)
        
        if not self.dirty:
            logger.info("No changes to save")
        
        else:
            # show dialog to make sure user understands what is happening
            text = "This will overwrite the bonds file, do you wish to continue?"
            text += "\n"
            text += "The following bonds have been modified:"
#             text += "\n"
            for bond in self.modifiedList:
                text += "\n    %s" % bond
            
            ret = QtGui.QMessageBox.question(self, "Save bonds settings", text, QtGui.QMessageBox.Ok | QtGui.QMessageBox.Cancel, 
                                             QtGui.QMessageBox.Cancel)
            
            if ret == QtGui.QMessageBox.Ok:
                fn = resourcePath("bonds.IN")
                logger.debug("Overwriting bonds file: '%s'", fn)
                elements.writeBonds(fn)
                self.dirty = False
                self.modifiedList.clear()
                self.buttonBox.button(QtGui.QDialogButtonBox.Save).setEnabled(False)
    
    def setWidgetStack(self, index):
        """
        Change stacked widget
        
        """
        self.stackedWidget.setCurrentIndex(index)

################################################################################

class ElementSettingsForm(genericForm.GenericForm):
    """
    Form for editing element settings
    
    """
    def __init__(self, sym, parent=None):
        self.sym = sym
        self.name = elements.atomName(sym)
        self.titleText = "%s - %s" % (sym, self.name)
        super(ElementSettingsForm, self).__init__(parent, None, self.titleText)
        
        # row
        row = self.newRow()
        
        # colour label
        label = QtGui.QLabel("Colour: ")
        row.addWidget(label)
        
        # colour
        rgb = copy.deepcopy(elements.RGB(sym))
        col = QtGui.QColor(rgb[0]*255.0, rgb[1]*255.0, rgb[2]*255.0)
        self.colour = rgb
        
        # colour button
        self.colourButton = QtGui.QPushButton("")
        self.colourButton.setFixedWidth(50)
        self.colourButton.setFixedHeight(30)
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
        self.colourButton.clicked.connect(self.showColourDialog)
        row.addWidget(self.colourButton)
        
        # row
        row = self.newRow()
        
        # radius label
        label = QtGui.QLabel("Radius: ")
        row.addWidget(label)
        
        # radius
        self.radius = elements.covalentRadius(sym)
        
        # radius spin box
        self.spinBox = QtGui.QDoubleSpinBox(self)
        self.spinBox.setSingleStep(0.01)
        self.spinBox.setMinimum(0.0)
        self.spinBox.setMaximum(100.0)
        self.spinBox.setValue(elements.covalentRadius(sym))
        self.spinBox.valueChanged[float].connect(self.radiusChanged)
        row.addWidget(self.spinBox)
    
    def showColourDialog(self):
        """
        Show colour dialog
        
        """
        sym = self.sym
        RGB = self.colour
        cur = QtGui.QColor(RGB[0] * 255.0, RGB[1] * 255.0, RGB[2] * 255.0)
        
        col = QtGui.QColorDialog.getColor(cur, self, "%s" % sym)
        
        if col.isValid():
            self.colourChanged(qtcolour=col)
            self.settingsModified()
    
    def colourChanged(self, qtcolour=None, colour=None):
        """
        Colour changed
        
        """
        if qtcolour is None and colour is None:
            return
        
        if qtcolour is not None:
            colour = [float(qtcolour.red() / 255.0), float(qtcolour.green() / 255.0), float(qtcolour.blue() / 255.0)]
        
        else:
            qtcolour = QtGui.QColor(colour[0] * 255.0, colour[1] * 255.0, colour[2] * 255.0)
        
        self.colour[0] = colour[0]
        self.colour[1] = colour[1]
        self.colour[2] = colour[2]
        
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % qtcolour.name())
    
    def radiusChanged(self, val):
        """
        Radius has changed
        
        """
        self.radius = val
        self.settingsModified()
    
    def settingsModified(self):
        """
        Settings have been modified
        
        """
        self.parent.settingModified(self.sym)

################################################################################

class ElementEditor(QtGui.QDialog):
    """
    Element editor dialog
    
    """
    def __init__(self, parent=None):
        super(ElementEditor, self).__init__(parent)
        
        self.iniWinFlags = self.windowFlags()
        self.setWindowFlags(self.iniWinFlags | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.mainWindow = parent
        self.setModal(0)
        
        self.setWindowTitle("Element editor")
        self.setWindowIcon(QtGui.QIcon(iconPath("other/periodic-table-icon.png")))
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        # lattice objects
        self.systemsDialog = self.mainWindow.systemsDialog
        
        # initial settings
        self.dirty = False
        self.modifiedListApply = set()
        self.modifiedListSave = set()
        self.formsDict = {}
        
        layout = QtGui.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignHCenter)
#        layout.setContentsMargins(0, 0, 0, 0)
#        layout.setSpacing(0)
        
        # combo box with elements
        self.elementsCombo = QtGui.QComboBox()
        self.elementsCombo.currentIndexChanged.connect(self.setWidgetStack)
        
        row = QtGui.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.elementsCombo)
        row.addStretch(1)
        layout.addLayout(row)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget()
        layout.addWidget(self.stackedWidget)
        
        # populate combo and stacked widget
        elementsList = elements.listElements()
        for sym in elementsList:
            # form for stacked widget
            form = ElementSettingsForm(sym, parent=self)
            self.formsDict[sym] = form
            
            # add to stacked widget
            self.stackedWidget.addWidget(form)
            
            # add to combo box
            self.elementsCombo.addItem("%s - %s" % (sym, elements.atomName(sym)))
        
        # buttons
        buttonContainer = QtGui.QWidget(self)
        buttonLayout = QtGui.QHBoxLayout(buttonContainer)
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.setSpacing(0)
        
        # apply button
        self.applyButton = QtGui.QPushButton(QtGui.QIcon(iconPath("redo_64.png")), "Apply")
        self.applyButton.setStatusTip("Apply changes to current session")
        self.applyButton.setToolTip("Apply changes to current session")
        self.applyButton.clicked.connect(self.applyChanges)
        self.applyButton.setEnabled(False)
        
        self.saveButton = QtGui.QPushButton(QtGui.QIcon(iconPath("save_64.png")), "Save")
        self.saveButton.setStatusTip("Save changes for use in future sessions")
        self.saveButton.setToolTip("Save changes for use in future sessions")
        self.saveButton.clicked.connect(self.saveChanges)
        self.saveButton.setEnabled(False)
        
        self.resetButton = QtGui.QPushButton(QtGui.QIcon(iconPath("undo_64.png")), "Reset")
        self.resetButton.setStatusTip("Reset changes to last applied")
        self.resetButton.setToolTip("Reset changes to last applied")
        self.resetButton.clicked.connect(self.resetChanges)
        self.resetButton.setEnabled(False)
        
        buttonLayout.addWidget(self.applyButton)
        buttonLayout.addWidget(self.saveButton)
        buttonLayout.addWidget(self.resetButton)
        
        layout.addWidget(buttonContainer)
    
    def applyChanges(self):
        """
        Apply changes.
        
        """
        logger = logging.getLogger(__name__+".ElementEditor")
        logger.debug("Applying element editor changes (%d)", len(self.modifiedListApply))
        
        for sym in self.modifiedListApply:
            settings = self.formsDict[sym]
            
            # radius
            radius = settings.radius
            
            # colour
            RGB = settings.colour
            R = RGB[0]
            G = RGB[1]
            B = RGB[2]
            
            logger.debug("Applying changes for '%s': rad %.3f; rgb <%.3f, %.3f, %.3f>", sym, radius, R, G, B)
            
            latticeList = self.systemsDialog.getLatticeList()
            for latt in latticeList:
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
        
        # disable buttons
        self.applyButton.setEnabled(False)
        self.resetButton.setEnabled(False)
        self.modifiedListApply.clear()
        
        self.parent.setStatus("Element properties applied")
    
    def saveChanges(self):
        """
        Save changes
        
        """
        logger = logging.getLogger(__name__+".ElementEditor")
        logger.debug("Saving element editor changes (%d)", len(self.modifiedListSave))
        
        msgtext = "This will overwrite the current element properties file. You should create a backup first!"
        msgtext += "\nModified elements:"
        for text in self.modifiedListSave:
            msgtext += "\n%s" % text
        msgtext += "\n\nDo you wish to continue?"
        
        reply = QtGui.QMessageBox.question(self, "Message", msgtext, QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
        if reply == QtGui.QMessageBox.Yes:
            # apply changes first
            self.applyChanges()
            
            # save to file
            elements.write(resourcePath("atoms.IN"))
            
            # disable save button
            self.saveButton.setEnabled(False)
            self.modifiedListSave.clear()
            
            self.parent.setStatus("Saved element properties")
    
    def resetChanges(self):
        """
        Reset changes.
        
        """
        logger = logging.getLogger(__name__+".ElementEditor")
        logger.debug("Resetting element editor changes (%d)", len(self.modifiedListApply))
        
        for sym in self.modifiedListApply:
            settings = self.formsDict[sym]
            
            # radius
            settings.spinBox.setValue(elements.covalentRadius(sym))
            assert settings.radius == elements.covalentRadius(sym)
            
            # colour
            rgb = elements.RGB(sym)
            settings.colourChanged(colour=rgb)
        
        # disable buttons
        self.applyButton.setEnabled(False)
        self.resetButton.setEnabled(False)
        
        self.parent.setStatus("Element properties reset")
    
    def settingModified(self, elementText):
        """
        Setting has been modified
        
        """
        self.dirty = True
        self.resetButton.setEnabled(True)
        self.applyButton.setEnabled(True)
        self.saveButton.setEnabled(True)
        self.modifiedListApply.add(elementText)
        self.modifiedListSave.add(elementText)
    
    def setWidgetStack(self, index):
        """
        Change stacked widget
        
        """
        self.stackedWidget.setCurrentIndex(index)

################################################################################

class ImageViewer(QtGui.QDialog):
    """
    Image viewer.
    
    @author: Marc Robinson
    Rewritten by Chris Scott
    
    """
    def __init__(self, mainWindow, parent=None):
        super(ImageViewer, self).__init__(parent)
        
#         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.setWindowTitle("Image Viewer:")
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/applications-graphics.png")))
        
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
        deleteImageButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/edit-delete.png")), "Delete image")
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

class NotifyFeatureWindow(QtGui.QDialog):
    """
    Notify user of new feature at startup
    
    """
    def __init__(self, parent=None):
        super(NotifyFeatureWindow, self).__init__(parent)
        
        self.notificationID = "onscreeninfo_updated"
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.setFixedWidth(300)
        
        self.setWindowTitle("New features")
        
        layout = QtGui.QVBoxLayout(self)
        
        # image
        pic = QtGui.QLabel()
        pic.resize(200, 200)
        pic.setPixmap(QtGui.QPixmap(iconPath("oxygen/preferences-desktop-font.png")))
        row = QtGui.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.addWidget(pic)
        layout.addLayout(row)
        
        # text
        dialogText = """<p>Check out the new 'on screen text' options.<ul><li>Double click the 
                        items to change settings</li><li>Items that are ticked are only displayed if 
                        they are available.</li><li>All format specifiers must be used</li><li>Entering
                        and incorrect format will just use the default</li></ul></p>"""
        
        # label
        label = QtGui.QLabel(dialogText)
        label.setWordWrap(True)
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        layout.addLayout(row)
        
        layout.addStretch()
        
        # show message next time
        self.dontShowAgainCheck = QtGui.QCheckBox("Do not show this again")
        row = QtGui.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(self.dontShowAgainCheck)
        layout.addLayout(row)
        
        # buttons
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        
        layout.addWidget(buttonBox)

################################################################################

class AboutMeDialog(QtGui.QMessageBox):
    """
    About me dialog.
    
    """
    def __init__(self, parent=None):
        super(AboutMeDialog, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        from ..visutils.version import getVersion
        import datetime
        import paramiko
        import matplotlib
        import platform
        import PySide
        import vtk
        import scipy
        version = getVersion()
        
        self.setWindowTitle("CDJSVis %s" % version)
        
        # message box layout (grid layout)
        l = self.layout()
        
        self.setText("""<p><b>CDJSVis</b> %s</p>
                          <p>Copyright &copy; %d Chris Scott</p>
                          <p>This application can be used to visualise atomistic simulations.</p>
                          <p>GUI based on <a href="http://sourceforge.net/projects/avas/">AVAS</a> 
                             by Marc Robinson.</p>""" % (
                          version, datetime.date.today().year))
        
        packageList = QtGui.QListWidget()
        
        packageList.addItem("Python %s" % platform.python_version())
        packageList.addItem("Qt %s" % QtCore.__version__)
        packageList.addItem("PySide %s" % PySide.__version__)
        packageList.addItem("VTK %s" % vtk.vtkVersion.GetVTKVersion())
        packageList.addItem("NumPy %s" % np.__version__)
        packageList.addItem("SciPy %s" % scipy.__version__)
        packageList.addItem("Matplotlib %s" % matplotlib.__version__)
        packageList.addItem("Paramiko %s" % paramiko.__version__)
        
        
        # Hide the default button
        button = l.itemAtPosition( l.rowCount() - 1, 1 ).widget()
        l.removeWidget(button)
        
        # add list widget to layout
        l.addWidget(packageList, l.rowCount(), 1, 1, l.columnCount(), QtCore.Qt.AlignHCenter)
        
        # add widget back in
        l.addWidget(button, l.rowCount(), 1, 1, 1, QtCore.Qt.AlignRight)
        
        self.setStandardButtons(QtGui.QMessageBox.Ok)
        self.setIcon(QtGui.QMessageBox.Information)

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
        
#         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.setWindowTitle("Rotate view point")
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/transform-rotate.png")))
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

################################################################################

class ReplicateCellDialog(QtGui.QDialog):
    """
    Ask user which directions they want to replicate the cell in
    
    """
    def __init__(self, pbc, parent=None):
        super(ReplicateCellDialog, self).__init__(parent)
        
        self.setWindowTitle("Replicate cell options")
        
        # layout
        layout = QtGui.QFormLayout()
        self.setLayout(layout)
        
        # x
        self.replicateInXSpin = QtGui.QSpinBox()
        self.replicateInXSpin.setMinimum(0)
        self.replicateInXSpin.setMaximum(10)
        self.replicateInXSpin.setValue(0)
        self.replicateInXSpin.setToolTip("Number of times to replicate the cell in the x direction")
        if not pbc[0]:
            self.replicateInXSpin.setEnabled(False)
        layout.addRow("Replicate in x", self.replicateInXSpin)
        
        # y
        self.replicateInYSpin = QtGui.QSpinBox()
        self.replicateInYSpin.setMinimum(0)
        self.replicateInYSpin.setMaximum(10)
        self.replicateInYSpin.setValue(0)
        self.replicateInYSpin.setToolTip("Number of times to replicate the cell in the y direction")
        if not pbc[1]:
            self.replicateInYSpin.setEnabled(False)
        layout.addRow("Replicate in y", self.replicateInYSpin)
        
        # z
        self.replicateInZSpin = QtGui.QSpinBox()
        self.replicateInZSpin.setMinimum(0)
        self.replicateInZSpin.setMaximum(10)
        self.replicateInZSpin.setValue(0)
        self.replicateInZSpin.setToolTip("Number of times to replicate the cell in the z direction")
        if not pbc[2]:
            self.replicateInYSpin.setEnabled(False)
        layout.addRow("Replicate in z", self.replicateInZSpin)
        
        # button box
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)
