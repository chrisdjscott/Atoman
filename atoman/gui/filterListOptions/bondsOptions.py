
"""
Bonds options
-------------

Selecting "Draw bonds" will result in bonds being drawn between visible 
atoms from this filter list. You must also select the bonds you want to 
draw from the list (eg. "Pu-Pu" or "Pu-Ga"). 

The "Bond thickness" settings determine the size of the bonds when they are
rendered.  "VTK" is the onscreen rendering while "POV" is used during 
POV-Ray rendering.

The "Number of sides" settings determines how many sides make up the tube
used to render the bond.  A higher setting will look better but will be 
much slower to render and interact with.

"""
import logging

from PySide import QtGui, QtCore

from ...visutils.utilities import iconPath


################################################################################

class BondListItem(QtGui.QListWidgetItem):
    """
    Item in the bonds list widget.
    
    """
    def __init__(self, syma, symb):
        super(BondListItem, self).__init__()
        
        # add check box
        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable)
        
        # don't allow it to be selected
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsSelectable)
        
        # set unchecked initially
        self.setCheckState(QtCore.Qt.Unchecked)
        
        # store bond pair
        self.syma = syma
        self.symb = symb
        
        # set text
        self.setText("%s - %s" % (syma, symb))

################################################################################

class BondsOptionsWindow(QtGui.QDialog):
    """
    Options dialog for drawing bonds.
    
    """
    modified = QtCore.Signal(str)
    
    def __init__(self, mainWindow, parent=None):
        super(BondsOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Bonds options")
        self.setWindowIcon(QtGui.QIcon(iconPath("other/molecule1.png")))
        
        self.mainWindow = mainWindow
        
        # logger
        self.logger = logging.getLogger(__name__)
        
        # options
        self.drawBonds = False
        self.bondThicknessPOV = 0.2
        self.bondThicknessVTK = 0.2
        self.bondNumSides = 5
        
        # layout
        layout = QtGui.QVBoxLayout(self)
#        layout.setSpacing(0)
#        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        
        # draw bonds group box
        self.drawBondsGroup = QtGui.QGroupBox("Draw bonds")
        self.drawBondsGroup.setCheckable(True)
        self.drawBondsGroup.setChecked(False)
#         self.drawBondsGroup.setAlignment(QtCore.Qt.AlignCenter)
        self.drawBondsGroup.toggled.connect(self.drawBondsToggled)
        layout.addWidget(self.drawBondsGroup)
        
        self.groupLayout = QtGui.QVBoxLayout()
#        self.groupLayout.setSpacing(0)
#        self.groupLayout.setContentsMargins(0, 0, 0, 0)
        self.groupLayout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        
        self.bondsList = QtGui.QListWidget(self)
        self.bondsList.setFixedHeight(100)
        self.bondsList.setFixedWidth(120)
        self.groupLayout.addWidget(self.bondsList)
        
        self.drawBondsGroup.setLayout(self.groupLayout)
        
        # thickness
        bondThicknessGroup = QtGui.QGroupBox("Bond thickness")
        bondThicknessGroup.setAlignment(QtCore.Qt.AlignCenter)
        bondThicknessLayout = QtGui.QVBoxLayout()
        bondThicknessGroup.setLayout(bondThicknessLayout)
        layout.addWidget(bondThicknessGroup)
        
        # vtk
        vtkThickSpin = QtGui.QDoubleSpinBox()
        vtkThickSpin.setMinimum(0.01)
        vtkThickSpin.setMaximum(10)
        vtkThickSpin.setSingleStep(0.01)
        vtkThickSpin.setValue(self.bondThicknessVTK)
        vtkThickSpin.valueChanged.connect(self.vtkThickChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("VTK:"))
        row.addWidget(vtkThickSpin)
        bondThicknessLayout.addLayout(row)
        
        # pov
        povThickSpin = QtGui.QDoubleSpinBox()
        povThickSpin.setMinimum(0.01)
        povThickSpin.setMaximum(10)
        povThickSpin.setSingleStep(0.01)
        povThickSpin.setValue(self.bondThicknessPOV)
        povThickSpin.valueChanged.connect(self.povThickChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("POV:"))
        row.addWidget(povThickSpin)
        bondThicknessLayout.addLayout(row)
        
        # thickness
        numSidesGroup = QtGui.QGroupBox("Number of sides")
        numSidesGroup.setAlignment(QtCore.Qt.AlignCenter)
        numSidesLayout = QtGui.QVBoxLayout()
        numSidesGroup.setLayout(numSidesLayout)
        layout.addWidget(numSidesGroup)
        
        # pov
        numSidesSpin = QtGui.QSpinBox()
        numSidesSpin.setMinimum(3)
        numSidesSpin.setMaximum(999)
        numSidesSpin.setSingleStep(1)
        numSidesSpin.setValue(self.bondNumSides)
        numSidesSpin.valueChanged.connect(self.numSidesChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(numSidesSpin)
        numSidesLayout.addLayout(row)
        
        # button box
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)
        
        # always refresh
        self.refresh()
    
    def numSidesChanged(self, val):
        """
        Number of sides changed.
        
        """
        self.bondNumSides = val
    
    def vtkThickChanged(self, val):
        """
        VTK thickness changed.
        
        """
        self.bondThicknessVTK = val
    
    def povThickChanged(self, val):
        """
        POV thickness changed.
        
        """
        self.bondThicknessPOV = val
    
    def drawBondsToggled(self, state):
        """
        Draw bonds changed.
        
        """
        self.drawBonds = state
        
        if self.drawBonds:
            text = "Bonds options: On"
        else:
            text = "Bonds options: Off"
        
        self.modified.emit(text)
    
    def refresh(self):
        """
        Refresh available bonds.
        
        Should be called whenever a new input is loaded!?
        If the species are the same don't change anything!?
        
        """
        inputState = self.parent.filterTab.inputState
        if inputState is None:
            return
        
        self.logger.debug("Refreshing bonds options (%d - %d)", self.parent.pipelinePage.pipelineIndex, self.parent.tab)
        
        specieList = inputState.specieList
        
        # set of added pairs
        currentPairs = set()
        
        # remove pairs that don't exist
        num = self.bondsList.count()
        for i in xrange(num - 1, -1, -1):
            item = self.bondsList.item(i)
            
            # make this 'and' so that if a lattice is missing one specie we still
            # keep the pair in case it comes back later... 
            if item.syma not in specieList and item.symb not in specieList:
                self.logger.debug("  Removing bond option: %s - %s", item.syma, item.symb)
                self.bondsList.takeItem(i) # does this delete it?
            
            else:
                currentPairs.add("%s - %s" % (item.syma, item.symb))
                currentPairs.add("%s - %s" % (item.symb, item.syma))
        
        # add pairs that aren't already added
        for i in xrange(len(inputState.specieList)):
            for j in xrange(i, len(inputState.specieList)):
                p1 = "%s - %s" % (specieList[i], specieList[j])
                p2 = "%s - %s" % (specieList[j], specieList[i])
                if p1 in currentPairs:
                    self.logger.debug("  Keeping bond option: %s", p1)
                
                elif p2 in currentPairs:
                    self.logger.debug("  Keeping bond option: %s", p2)
                
                else:
                    self.logger.debug("  Adding bond option: %s", p1)
                    item = BondListItem(specieList[i], specieList[j])
                    self.bondsList.addItem(item)
