
"""
Lattice generator forms.

@author: Chris Scott

"""
import os
import sys

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from .genericForm import GenericForm
from ..lattice_gen import lattice_gen_pu3ga
from ..lattice_gen import lattice_gen_fcc
from ..lattice_gen import lattice_gen_bcc
from ..lattice_gen import lattice_gen_fluorite
from ..lattice_gen import lattice_gen_rockSalt

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)

################################################################################

class GenericLatticeGeneratorForm(GenericForm):
    """
    Generic reader widget.
    
    """
    def __init__(self, parent, mainWindow, title):
        super(GenericLatticeGeneratorForm, self).__init__(parent, None, title)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.generatorArgs = None
        self.filename = "generated.dat"
        
        self.show()
    
    def log(self, message, indent=0):
        """
        Log message.
        
        """
        self.mainWindow.console.write(str(message), indent=0)
    
    def generateLattice(self):
        """
        Generate the lattice
        
        """
        status, lattice = self.generateLatticeMain()
        
        self.postGenerateLattice(status, lattice)
    
    def generateLatticeMain(self):
        """
        Main generate function to be overridden
        
        """
        return 1, None
    
    def postGenerateLattice(self, status, lattice):
        """
        Post generate lattice.
        
        """
        if not status and lattice is not None:
            self.parent.file_generated(lattice, self.filename)
    
    def add_specie_options(self, NSpecies):
        """
        Add specie options
        
        """
        pass
    
    def add_a0_option(self):
        """
        Add lattice constant option
        
        """
        # row
        row = self.newRow()
        
        # label
        label = QtGui.QLabel("Lattice constant:")
        row.addWidget(label)
        
        # spin
        latticeConstantSpin = QtGui.QDoubleSpinBox()
        latticeConstantSpin.setDecimals(5)
        latticeConstantSpin.setSingleStep(0.1)
        latticeConstantSpin.setMinimum(0.00001)
        latticeConstantSpin.setMaximum(99.99999)
        latticeConstantSpin.setValue(self.generatorArgs.a0)
        latticeConstantSpin.valueChanged.connect(self.latticeConstantChanged)
        row.addWidget(latticeConstantSpin)
        
    def latticeConstantChanged(self, val):
        """
        Lattice constant changed
        
        """
        self.generatorArgs.a0 = val
    
    def add_unit_cell_options(self):
        """
        Add unit cell options to layout
        
        """
        # new row
        row = self.newRow()
        
        # label
        label = QtGui.QLabel("Number of unit cells (x,y,z):")
        row.addWidget(label)
        
        # new row
        row = self.newRow()
        
        # num unit cells
        numUnitCellsXSpin = QtGui.QSpinBox()
        numUnitCellsXSpin.setMinimum(1)
        numUnitCellsXSpin.setMaximum(1000)
        numUnitCellsXSpin.setValue(self.generatorArgs.NCells[0])
        numUnitCellsXSpin.valueChanged.connect(self.numUnitCellsXChanged)
        row.addWidget(numUnitCellsXSpin)
        
        row.addWidget(QtGui.QLabel("x"))
        
        numUnitCellsYSpin = QtGui.QSpinBox()
        numUnitCellsYSpin.setMinimum(1)
        numUnitCellsYSpin.setMaximum(1000)
        numUnitCellsYSpin.setValue(self.generatorArgs.NCells[1])
        numUnitCellsYSpin.valueChanged.connect(self.numUnitCellsYChanged)
        row.addWidget(numUnitCellsYSpin)
        
        row.addWidget(QtGui.QLabel("x"))
        
        numUnitCellsZSpin = QtGui.QSpinBox()
        numUnitCellsZSpin.setMinimum(1)
        numUnitCellsZSpin.setMaximum(1000)
        numUnitCellsZSpin.setValue(self.generatorArgs.NCells[2])
        numUnitCellsZSpin.valueChanged.connect(self.numUnitCellsZChanged)
        row.addWidget(numUnitCellsZSpin)
    
    def numUnitCellsXChanged(self, val):
        """
        Number of unit cells changed.
        
        """
        self.generatorArgs.NCells[0] = val
    
    def numUnitCellsYChanged(self, val):
        """
        Number of unit cells changed.
        
        """
        self.generatorArgs.NCells[1] = val
    
    def numUnitCellsZChanged(self, val):
        """
        Number of unit cells changed.
        
        """
        self.generatorArgs.NCells[2] = val
    
    def filenameChanged(self, text):
        """
        Filename has changed
        
        """
        self.filename = str(text)
    
    def add_filename_option(self):
        """
        Add filename option
        
        """
        row = self.newRow()
        
        label = QtGui.QLabel("Display name:")
        row.addWidget(label)
        
        filenameLineEdit = QtGui.QLineEdit(self.filename)
        filenameLineEdit.setFixedWidth(130)
        filenameLineEdit.textChanged.connect(self.filenameChanged)
        row.addWidget(filenameLineEdit)
    
    def add_pbc_options(self):
        """
        Add pbc options
        
        """
        # periodic boundaries
        label = QtGui.QLabel("Periodic boundaries:")
        row = self.newRow()
        row.addWidget(label)
        
        PBCXCheckBox = QtGui.QCheckBox("x   ")
        PBCXCheckBox.setChecked(QtCore.Qt.Checked)
        PBCYCheckBox = QtGui.QCheckBox("y   ")
        PBCYCheckBox.setChecked(QtCore.Qt.Checked)
        PBCZCheckBox = QtGui.QCheckBox("z   ")
        PBCZCheckBox.setChecked(QtCore.Qt.Checked)
        
        PBCXCheckBox.stateChanged.connect(self.PBCXChanged)
        PBCYCheckBox.stateChanged.connect(self.PBCYChanged)
        PBCZCheckBox.stateChanged.connect(self.PBCZChanged)
        
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.addWidget(PBCXCheckBox)
        rowLayout.addWidget(PBCYCheckBox)
        rowLayout.addWidget(PBCZCheckBox)
        
        row = self.newRow()
        row.addWidget(PBCXCheckBox)
        row.addWidget(PBCYCheckBox)
        row.addWidget(PBCZCheckBox)
    
    def PBCXChanged(self, checked):
        """
        PBC changed.
        
        """
        if checked == QtCore.Qt.Checked:
            self.generatorArgs.pbcx = True
        
        else:
            self.generatorArgs.pbcx = False
    
    def PBCYChanged(self, checked):
        """
        PBC changed.
        
        """
        if checked == QtCore.Qt.Checked:
            self.generatorArgs.pbcy = True
        
        else:
            self.generatorArgs.pbcy = False
    
    def PBCZChanged(self, checked):
        """
        PBC changed.
        
        """
        if checked == QtCore.Qt.Checked:
            self.generatorArgs.pbcz = True
        
        else:
            self.generatorArgs.pbcz = False
    
    def add_generate_button(self):
        """
        Add generate button
        
        """
        row = self.newRow()
        generateButton = QtGui.QPushButton("Generate lattice")
        generateButton.clicked.connect(self.generateLattice)
        row.addWidget(generateButton)

################################################################################

class Pu3GaLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    Pu3Ga lattice generator
    
    """
    def __init__(self, parent, mainWindow):
        super(Pu3GaLatticeGeneratorForm, self).__init__(parent, mainWindow, "Pu3Ga lattice generator")
        
        self.generatorArgs = lattice_gen_pu3ga.Args()
        
        self.add_filename_option()
        
        self.add_unit_cell_options()
        
        self.add_a0_option()
        
        self.add_pbc_options()
        
        # percent Ga
        row = self.newRow()
        
        label = QtGui.QLabel("Percent Ga:")
        row.addWidget(label)
        
        percGaSpin = QtGui.QDoubleSpinBox()
        percGaSpin.setSingleStep(0.1)
        percGaSpin.setMinimum(0.0)
        percGaSpin.setMaximum(25.0)
        percGaSpin.setValue(self.generatorArgs.percGa)
        percGaSpin.valueChanged.connect(self.percGaChanged)
        row.addWidget(percGaSpin)
        
        # generate button
        self.add_generate_button()

    def percGaChanged(self, val):
        """
        Percent Ga changed
        
        """
        self.generatorArgs.percGa = val
    
    def generateLatticeMain(self):
        """
        Generate lattice
        
        """
        generator = lattice_gen_pu3ga.Pu3GaLatticeGenerator(log=self.mainWindow.console.write)
        
        status, lattice = generator.generateLattice(self.generatorArgs)
        
        return status, lattice

################################################################################

class FCCLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    FCC lattice generator
    
    """
    def __init__(self, parent, mainWindow):
        super(FCCLatticeGeneratorForm, self).__init__(parent, mainWindow, "FCC lattice generator")
        
        self.generatorArgs = lattice_gen_fcc.Args()
        
        self.add_filename_option()
        
        # specie
        row = self.newRow()
        
        label = QtGui.QLabel("Specie:")
        row.addWidget(label)
        
        self.specie_text = QtGui.QLineEdit(self.generatorArgs.sym)
        self.specie_text.setFixedWidth(30)
        self.specie_text.setMaxLength(2)
        self.specie_text.textEdited.connect(self.specie_text_edited)
        row.addWidget(self.specie_text)
        
        self.add_unit_cell_options()
        
        self.add_a0_option()
        
        self.add_pbc_options()
                
        # generate button
        self.add_generate_button()
    
    def specie_text_edited(self, text):
        """
        Specie text edited
        
        """
        self.generatorArgs.sym = str(text)
    
    def generateLatticeMain(self):
        """
        Generate lattice
        
        """
        generator = lattice_gen_fcc.FCCLatticeGenerator(log=self.mainWindow.console.write)
        
        status, lattice = generator.generateLattice(self.generatorArgs)
        
        return status, lattice

################################################################################

class BCCLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    BCC lattice generator
    
    """
    def __init__(self, parent, mainWindow):
        super(BCCLatticeGeneratorForm, self).__init__(parent, mainWindow, "BCC lattice generator")
        
        self.generatorArgs = lattice_gen_bcc.Args()
        
        self.add_filename_option()
        
        # specie
        row = self.newRow()
        
        label = QtGui.QLabel("Specie:")
        row.addWidget(label)
        
        self.specie_text = QtGui.QLineEdit(self.generatorArgs.sym)
        self.specie_text.setFixedWidth(30)
        self.specie_text.setMaxLength(2)
        self.specie_text.textEdited.connect(self.specie_text_edited)
        row.addWidget(self.specie_text)
        
        self.add_unit_cell_options()
        
        self.add_a0_option()
        
        self.add_pbc_options()
                
        # generate button
        self.add_generate_button()
    
    def specie_text_edited(self, text):
        """
        Specie text edited
        
        """
        self.generatorArgs.sym = str(text)
    
    def generateLatticeMain(self):
        """
        Generate lattice
        
        """
        generator = lattice_gen_bcc.BCCLatticeGenerator(log=self.mainWindow.console.write)
        
        status, lattice = generator.generateLattice(self.generatorArgs)
        
        return status, lattice

################################################################################

class FluoriteLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    Fluorite lattice generator
    
    """
    def __init__(self, parent, mainWindow):
        super(FluoriteLatticeGeneratorForm, self).__init__(parent, mainWindow, "Fluorite lattice generator")
        
        self.generatorArgs = lattice_gen_fluorite.Args()
        
        self.add_filename_option()
        
        # specie 1
        row = self.newRow()
        
        label = QtGui.QLabel("Specie 1:")
        row.addWidget(label)
        
        self.specie1_text = QtGui.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        row.addWidget(self.specie1_text)
        
        # charge 1
        label = QtGui.QLabel("Charge 1:")
        row.addWidget(label)
        
        charge1Spin = QtGui.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        row.addWidget(charge1Spin)
        
        # specie 2
        row = self.newRow()
        
        label = QtGui.QLabel("Specie 2:")
        row.addWidget(label)
        
        self.specie2_text = QtGui.QLineEdit(self.generatorArgs.sym2)
        self.specie2_text.setFixedWidth(30)
        self.specie2_text.setMaxLength(2)
        self.specie2_text.textEdited.connect(self.specie2_text_edited)
        row.addWidget(self.specie2_text)
        
        # charge 1
        label = QtGui.QLabel("Charge 2:")
        row.addWidget(label)
        
        charge2Spin = QtGui.QDoubleSpinBox()
        charge2Spin.setMinimum(-99.99)
        charge2Spin.setValue(self.generatorArgs.charge2)
        charge2Spin.valueChanged.connect(self.charge2_changed)
        row.addWidget(charge2Spin)
        
        self.add_unit_cell_options()
        
        self.add_a0_option()
        
        self.add_pbc_options()
                
        # generate button
        self.add_generate_button()
    
    def charge1_changed(self, val):
        """
        Charge 1 changed
        
        """
        self.generatorArgs.charge1 = val
    
    def charge2_changed(self, val):
        """
        Charge 2 changed
        
        """
        self.generatorArgs.charge2 = val
    
    def specie1_text_edited(self, text):
        """
        Specie 1 text edited
        
        """
        self.generatorArgs.sym1 = str(text)
    
    def specie2_text_edited(self, text):
        """
        Specie 2 text edited
        
        """
        self.generatorArgs.sym2 = str(text)
    
    def generateLatticeMain(self):
        """
        Generate lattice
        
        """
        generator = lattice_gen_fluorite.FluoriteLatticeGenerator(log=self.mainWindow.console.write)
        
        status, lattice = generator.generateLattice(self.generatorArgs)
        
        return status, lattice

################################################################################

class RockSaltLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    Rock Salt lattice generator
    
    """
    def __init__(self, parent, mainWindow):
        super(RockSaltLatticeGeneratorForm, self).__init__(parent, mainWindow, "Rock Salt lattice generator")
        
        self.generatorArgs = lattice_gen_rockSalt.Args()
        
        self.add_filename_option()
        
        # specie 1
        row = self.newRow()
        
        label = QtGui.QLabel("Specie 1:")
        row.addWidget(label)
        
        self.specie1_text = QtGui.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        row.addWidget(self.specie1_text)
        
        # charge 1
        label = QtGui.QLabel("Charge 1:")
        row.addWidget(label)
        
        charge1Spin = QtGui.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        row.addWidget(charge1Spin)
        
        # specie 2
        row = self.newRow()
        
        label = QtGui.QLabel("Specie 2:")
        row.addWidget(label)
        
        self.specie2_text = QtGui.QLineEdit(self.generatorArgs.sym2)
        self.specie2_text.setFixedWidth(30)
        self.specie2_text.setMaxLength(2)
        self.specie2_text.textEdited.connect(self.specie2_text_edited)
        row.addWidget(self.specie2_text)
        
        # charge 1
        label = QtGui.QLabel("Charge 2:")
        row.addWidget(label)
        
        charge2Spin = QtGui.QDoubleSpinBox()
        charge2Spin.setMinimum(-99.99)
        charge2Spin.setValue(self.generatorArgs.charge2)
        charge2Spin.valueChanged.connect(self.charge2_changed)
        row.addWidget(charge2Spin)
        
        self.add_unit_cell_options()
        
        self.add_a0_option()
        
        self.add_pbc_options()
                
        # generate button
        self.add_generate_button()
    
    def charge1_changed(self, val):
        """
        Charge 1 changed
        
        """
        self.generatorArgs.charge1 = val
    
    def charge2_changed(self, val):
        """
        Charge 2 changed
        
        """
        self.generatorArgs.charge2 = val
    
    def specie1_text_edited(self, text):
        """
        Specie 1 text edited
        
        """
        self.generatorArgs.sym1 = str(text)
    
    def specie2_text_edited(self, text):
        """
        Specie 2 text edited
        
        """
        self.generatorArgs.sym2 = str(text)
    
    def generateLatticeMain(self):
        """
        Generate lattice
        
        """
        generator = lattice_gen_rockSalt.RockSaltLatticeGenerator(log=self.mainWindow.console.write)
        
        status, lattice = generator.generateLattice(self.generatorArgs)
        
        return status, lattice
