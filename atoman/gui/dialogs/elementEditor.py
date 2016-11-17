# -*- coding: utf-8 -*-

"""
Dialogs and classes for editting element properties.

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import copy
import logging

from PyQt5 import QtGui, QtCore, QtWidgets

import numpy as np

from ...system.atoms import elements
from ...visutils.utilities import dataPath, iconPath


################################################################################

class ElementSettingsForm(QtWidgets.QGroupBox):
    """
    Form for editing element settings
    
    """
    settingsModified = QtCore.pyqtSignal(str)
    
    def __init__(self, sym, parent=None):
        self.sym = sym
        self.name = elements.atomName(sym)
        self.titleText = "%s - %s" % (sym, self.name)
        super(ElementSettingsForm, self).__init__(self.titleText, parent=parent)
        
        # form layout
        layout = QtWidgets.QFormLayout(self)
        self.setAlignment(QtCore.Qt.AlignHCenter)
        
        # colour
        rgb = copy.deepcopy(elements.RGB(sym))
        col = QtGui.QColor(rgb[0]*255.0, rgb[1]*255.0, rgb[2]*255.0)
        self.colour = rgb
        
        # colour button
        self.colourButton = QtWidgets.QPushButton("")
        self.colourButton.setFixedWidth(50)
        self.colourButton.setFixedHeight(30)
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
        self.colourButton.clicked.connect(self.showColourDialog)
        layout.addRow("Colour", self.colourButton)
        
        # radius
        self.radius = elements.covalentRadius(sym)
        
        # radius spin box
        self.spinBox = QtWidgets.QDoubleSpinBox(self)
        self.spinBox.setSingleStep(0.01)
        self.spinBox.setMinimum(0.0)
        self.spinBox.setMaximum(100.0)
        self.spinBox.setValue(elements.covalentRadius(sym))
        self.spinBox.valueChanged[float].connect(self.radiusChanged)
        layout.addRow("Radius", self.spinBox)
    
    def showColourDialog(self):
        """
        Show colour dialog
        
        """
        sym = self.sym
        RGB = self.colour
        cur = QtGui.QColor(RGB[0] * 255.0, RGB[1] * 255.0, RGB[2] * 255.0)
        
        col = QtWidgets.QColorDialog.getColor(cur, self, "%s" % sym)
        
        if col.isValid():
            self.colourChanged(qtcolour=col)
            self.settingsModified.emit(self.sym)
    
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
        self.settingsModified.emit(self.sym)

################################################################################

class ElementEditor(QtWidgets.QDialog):
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
        
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        
        # lattice objects
        self.systemsDialog = self.mainWindow.systemsDialog
        
        # initial settings
        self.dirty = False
        self.modifiedListApply = set()
        self.modifiedListSave = set()
        self.formsDict = {}
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignHCenter)
#        layout.setContentsMargins(0, 0, 0, 0)
#        layout.setSpacing(0)
        
        # combo box with elements
        self.elementsCombo = QtWidgets.QComboBox()
        self.elementsCombo.currentIndexChanged.connect(self.setWidgetStack)
        
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.elementsCombo)
        row.addStretch(1)
        layout.addLayout(row)
        
        # stacked widget
        self.stackedWidget = QtWidgets.QStackedWidget()
        layout.addWidget(self.stackedWidget)
        
        # populate combo and stacked widget
        elementsList = elements.listElements()
        for sym in elementsList:
            # form for stacked widget
            form = ElementSettingsForm(sym, parent=self)
            form.settingsModified.connect(self.settingModified)
            self.formsDict[sym] = form
            
            # add to stacked widget
            self.stackedWidget.addWidget(form)
            
            # add to combo box
            self.elementsCombo.addItem("%s - %s" % (sym, elements.atomName(sym)))
        
        # buttons
        buttonContainer = QtWidgets.QWidget(self)
        buttonLayout = QtWidgets.QHBoxLayout(buttonContainer)
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.setSpacing(0)
        
        # apply button
        self.applyButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath("redo_64.png")), "Apply")
        self.applyButton.setStatusTip("Apply changes to current session")
        self.applyButton.setToolTip("Apply changes to current session")
        self.applyButton.clicked.connect(self.applyChanges)
        self.applyButton.setEnabled(False)
        
        self.saveButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath("save_64.png")), "Save")
        self.saveButton.setStatusTip("Save changes for use in future sessions")
        self.saveButton.setToolTip("Save changes for use in future sessions")
        self.saveButton.clicked.connect(self.saveChanges)
        self.saveButton.setEnabled(False)
        
        self.resetButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath("undo_64.png")), "Reset")
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
                    index = latt.getSpecieIndex(sym)
                    
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
        
        reply = QtWidgets.QMessageBox.question(self, "Message", msgtext, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        
        if reply == QtWidgets.QMessageBox.Yes:
            # apply changes first
            self.applyChanges()
            
            # save to file
            elements.write(dataPath("atoms.IN"))
            
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
