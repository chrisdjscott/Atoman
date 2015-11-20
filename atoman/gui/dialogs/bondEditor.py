# -*- coding: utf-8 -*-

"""
Dialogs and classes for editting bonds settings.

@author: Chris Scott

"""
import logging

from PySide import QtGui, QtCore

from ...system.atoms import elements
from ...visutils.utilities import iconPath, dataPath


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
        form.settingModified.connect(self.settingModified)
        
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
                fn = dataPath("bonds.IN")
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

class BondEditorSettingsForm(QtGui.QGroupBox):
    """
    Settings for bond
    
    """
    settingModified = QtCore.Signal(str)
    
    def __init__(self, parent, syma, symb, bondMin, bondMax):
        super(BondEditorSettingsForm, self).__init__("%s - %s" % (syma, symb), parent=parent)
        
        self.syma = syma
        self.symb = symb
        self.bondMin = bondMin
        self.bondMax = bondMax
        
        # form layout
        layout = QtGui.QFormLayout(self)
        self.setAlignment(QtCore.Qt.AlignHCenter)
        
        # minimum value for the bond
        bondMinSpin = QtGui.QDoubleSpinBox()
        bondMinSpin.setSingleStep(0.1)
        bondMinSpin.setMinimum(0.0)
        bondMinSpin.setMaximum(99.99)
        bondMinSpin.setValue(self.bondMin)
        bondMinSpin.valueChanged.connect(self.bondMinChanged)
        layout.addRow("Bond minimum", bondMinSpin)
        
        # maximum value for the bond
        bondMaxSpin = QtGui.QDoubleSpinBox()
        bondMaxSpin.setSingleStep(0.1)
        bondMaxSpin.setMinimum(0.0)
        bondMaxSpin.setMaximum(99.99)
        bondMaxSpin.setValue(self.bondMax)
        bondMaxSpin.valueChanged.connect(self.bondMaxChanged)
        layout.addRow("Bond maximum", bondMaxSpin)
    
    def updateBondData(self):
        """
        Update bond data in elements object
        
        """
        logger = logging.getLogger(__name__)
        logger.debug("Updating bond data: %s - %s : (%.2f, %.2f)" % (self.syma, self.symb, self.bondMin, self.bondMax))
        
        elements.bondDict[self.syma][self.symb] = (self.bondMin, self.bondMax)
        elements.bondDict[self.symb][self.syma] = (self.bondMin, self.bondMax)
        
        self.settingModified.emit("%s - %s" % (self.syma, self.symb))
    
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
