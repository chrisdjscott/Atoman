
"""
Lattice generation
==================

In the *Generate new system* section of the *Input* tab of the main toolbar lattices
of different types and sizes can be generated.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from PySide2 import QtGui, QtCore, QtWidgets


from ..visutils.utilities import iconPath
from ..lattice_gen import lattice_gen_pu3ga
from ..lattice_gen import lattice_gen_fcc
from ..lattice_gen import lattice_gen_bcc
from ..lattice_gen import lattice_gen_fluorite
from ..lattice_gen import lattice_gen_rockSalt
from ..lattice_gen import lattice_gen_sic


################################################################################

class GenericLatticeGeneratorForm(QtWidgets.QWidget):
    """
    Generic reader widget.
    
    """
    def __init__(self, parent, mainWindow, title):
        super(GenericLatticeGeneratorForm, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.generatorArgs = None
        self.filename = "generated.dat"
        
        self.formLayout = QtWidgets.QFormLayout(self)
        
        self.show()
    
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
        latticeConstantSpin = QtWidgets.QDoubleSpinBox()
        latticeConstantSpin.setDecimals(5)
        latticeConstantSpin.setSingleStep(0.1)
        latticeConstantSpin.setMinimum(0.00001)
        latticeConstantSpin.setMaximum(99.99999)
        latticeConstantSpin.setValue(self.generatorArgs.a0)
        latticeConstantSpin.valueChanged.connect(self.latticeConstantChanged)
        latticeConstantSpin.setSuffix(" \u212B")
        latticeConstantSpin.setToolTip("Set the lattice constant")
        self.formLayout.addRow("Lattice constant", latticeConstantSpin)
        
    def latticeConstantChanged(self, val):
        """
        Lattice constant changed
        
        """
        self.generatorArgs.a0 = val
    
    def add_unit_cell_options(self):
        """
        Add unit cell options to layout
        
        """
        row = QtWidgets.QHBoxLayout()
        
        # num unit cells
        numUnitCellsXSpin = QtWidgets.QSpinBox()
        numUnitCellsXSpin.setMinimum(1)
        numUnitCellsXSpin.setMaximum(1000)
        numUnitCellsXSpin.setValue(self.generatorArgs.NCells[0])
        numUnitCellsXSpin.valueChanged.connect(self.numUnitCellsXChanged)
        numUnitCellsXSpin.setToolTip("Set the number of unit cells in x")
        row.addWidget(numUnitCellsXSpin)
        
        row.addWidget(QtWidgets.QLabel("x"))
        
        numUnitCellsYSpin = QtWidgets.QSpinBox()
        numUnitCellsYSpin.setMinimum(1)
        numUnitCellsYSpin.setMaximum(1000)
        numUnitCellsYSpin.setValue(self.generatorArgs.NCells[1])
        numUnitCellsYSpin.valueChanged.connect(self.numUnitCellsYChanged)
        numUnitCellsYSpin.setToolTip("Set the number of unit cells in y")
        row.addWidget(numUnitCellsYSpin)
        
        row.addWidget(QtWidgets.QLabel("x"))
        
        numUnitCellsZSpin = QtWidgets.QSpinBox()
        numUnitCellsZSpin.setMinimum(1)
        numUnitCellsZSpin.setMaximum(1000)
        numUnitCellsZSpin.setValue(self.generatorArgs.NCells[2])
        numUnitCellsZSpin.valueChanged.connect(self.numUnitCellsZChanged)
        numUnitCellsZSpin.setToolTip("Set the number of unit cells in z")
        row.addWidget(numUnitCellsZSpin)
        
        self.formLayout.addRow("Number of unit cells", row)
    
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
        filenameLineEdit = QtWidgets.QLineEdit(self.filename)
        filenameLineEdit.setFixedWidth(130)
        filenameLineEdit.textChanged.connect(self.filenameChanged)
        filenameLineEdit.setToolTip("Enter the display name")
        self.formLayout.addRow("Display name", filenameLineEdit)
    
    def add_pbc_options(self):
        """
        Add pbc options
        
        """
        # periodic boundaries
        PBCXCheckBox = QtWidgets.QCheckBox("x   ")
        PBCXCheckBox.setChecked(QtCore.Qt.Checked)
        PBCXCheckBox.setToolTip("Set periodic boundaries in x")
        PBCYCheckBox = QtWidgets.QCheckBox("y   ")
        PBCYCheckBox.setChecked(QtCore.Qt.Checked)
        PBCYCheckBox.setToolTip("Set periodic boundaries in y")
        PBCZCheckBox = QtWidgets.QCheckBox("z   ")
        PBCZCheckBox.setChecked(QtCore.Qt.Checked)
        PBCZCheckBox.setToolTip("Set periodic boundaries in z")
        
        PBCXCheckBox.stateChanged.connect(self.PBCXChanged)
        PBCYCheckBox.stateChanged.connect(self.PBCYChanged)
        PBCZCheckBox.stateChanged.connect(self.PBCZChanged)
        
        row = QtWidgets.QWidget(self)
        rowLayout = QtWidgets.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.addWidget(PBCXCheckBox)
        rowLayout.addWidget(PBCYCheckBox)
        rowLayout.addWidget(PBCZCheckBox)
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(PBCXCheckBox)
        hbox.addWidget(PBCYCheckBox)
        hbox.addWidget(PBCZCheckBox)
        self.formLayout.addRow("Periodic boundaries", hbox)
    
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
        generateButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath("oxygen/run-build.png")), "Generate lattice")
        generateButton.clicked.connect(self.generateLattice)
        generateButton.setToolTip("Generate the lattice")
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(generateButton)
        hbox.addStretch(1)
        self.formLayout.addRow(hbox)

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
        percGaSpin = QtWidgets.QDoubleSpinBox()
        percGaSpin.setSingleStep(0.1)
        percGaSpin.setMinimum(0.0)
        percGaSpin.setMaximum(25.0)
        percGaSpin.setValue(self.generatorArgs.percGa)
        percGaSpin.valueChanged.connect(self.percGaChanged)
        percGaSpin.setSuffix(" at.%")
        percGaSpin.setToolTip("Set the Ga concentration")
        self.formLayout.addRow("Ga concentration", percGaSpin)
        
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
        generator = lattice_gen_pu3ga.Pu3GaLatticeGenerator()
        
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
        self.specie_text = QtWidgets.QLineEdit(self.generatorArgs.sym)
        self.specie_text.setFixedWidth(30)
        self.specie_text.setMaxLength(2)
        self.specie_text.textEdited.connect(self.specie_text_edited)
        self.specie_text.setToolTip("Set the species")
        self.formLayout.addRow("Species", self.specie_text)
        
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
        generator = lattice_gen_fcc.FCCLatticeGenerator()
        
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
        self.specie_text = QtWidgets.QLineEdit(self.generatorArgs.sym)
        self.specie_text.setFixedWidth(30)
        self.specie_text.setMaxLength(2)
        self.specie_text.textEdited.connect(self.specie_text_edited)
        self.specie_text.setToolTip("Set the species")
        self.formLayout.addRow("Species", self.specie_text)
        
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
        generator = lattice_gen_bcc.BCCLatticeGenerator()
        
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
        self.specie1_text = QtWidgets.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        self.specie1_text.setToolTip("Set the symbol of the first species")
        
        # charge 1
        charge1Spin = QtWidgets.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        charge1Spin.setToolTip("Set the charge of the first species")
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.specie1_text)
        hbox.addWidget(charge1Spin)
        self.formLayout.addRow("Species 1", hbox)
        
        # specie 2
        self.specie2_text = QtWidgets.QLineEdit(self.generatorArgs.sym2)
        self.specie2_text.setFixedWidth(30)
        self.specie2_text.setMaxLength(2)
        self.specie2_text.textEdited.connect(self.specie2_text_edited)
        self.specie2_text.setToolTip("Set the symbol of the second species")
        
        # charge 2
        charge2Spin = QtWidgets.QDoubleSpinBox()
        charge2Spin.setMinimum(-99.99)
        charge2Spin.setValue(self.generatorArgs.charge2)
        charge2Spin.valueChanged.connect(self.charge2_changed)
        charge2Spin.setToolTip("Set the charge of the second species")
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.specie2_text)
        hbox.addWidget(charge2Spin)
        self.formLayout.addRow("Species 2", hbox)
        
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
        generator = lattice_gen_fluorite.FluoriteLatticeGenerator()
        
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
        self.specie1_text = QtWidgets.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        self.specie1_text.setToolTip("Set the symbol of the first species")
        
        # charge 1
        charge1Spin = QtWidgets.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        charge1Spin.setToolTip("Set the charge of the first species")
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.specie1_text)
        hbox.addWidget(charge1Spin)
        self.formLayout.addRow("Species 1", hbox)
        
        # specie 2
        self.specie2_text = QtWidgets.QLineEdit(self.generatorArgs.sym2)
        self.specie2_text.setFixedWidth(30)
        self.specie2_text.setMaxLength(2)
        self.specie2_text.textEdited.connect(self.specie2_text_edited)
        self.specie2_text.setToolTip("Set the symbol of the second species")
        
        # charge 2
        charge2Spin = QtWidgets.QDoubleSpinBox()
        charge2Spin.setMinimum(-99.99)
        charge2Spin.setValue(self.generatorArgs.charge2)
        charge2Spin.valueChanged.connect(self.charge2_changed)
        charge2Spin.setToolTip("Set the charge of the second species")
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.specie2_text)
        hbox.addWidget(charge2Spin)
        self.formLayout.addRow("Species 2", hbox)
        
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
        generator = lattice_gen_rockSalt.RockSaltLatticeGenerator()
        
        status, lattice = generator.generateLattice(self.generatorArgs)
        
        return status, lattice

################################################################################

class SiC4HLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    SiC 4H lattice generator
    
    """
    def __init__(self, parent, mainWindow):
        super(SiC4HLatticeGeneratorForm, self).__init__(parent, mainWindow, "CSi 4H lattice generator")
        
        self.generatorArgs = lattice_gen_sic.Args()
        
        self.add_filename_option()
        
        # specie 1
        self.specie1_text = QtWidgets.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        self.specie1_text.setToolTip("Set the symbol of the first species")
        
        # charge 1
        charge1Spin = QtWidgets.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        charge1Spin.setToolTip("Set the charge of the first species")
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.specie1_text)
        hbox.addWidget(charge1Spin)
        self.formLayout.addRow("Species 1", hbox)
        
        # specie 2
        self.specie2_text = QtWidgets.QLineEdit(self.generatorArgs.sym2)
        self.specie2_text.setFixedWidth(30)
        self.specie2_text.setMaxLength(2)
        self.specie2_text.textEdited.connect(self.specie2_text_edited)
        self.specie2_text.setToolTip("Set the symbol of the second species")
        
        # charge 2
        charge2Spin = QtWidgets.QDoubleSpinBox()
        charge2Spin.setMinimum(-99.99)
        charge2Spin.setValue(self.generatorArgs.charge2)
        charge2Spin.valueChanged.connect(self.charge2_changed)
        charge2Spin.setToolTip("Set the charge of the second species")
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.specie2_text)
        hbox.addWidget(charge2Spin)
        self.formLayout.addRow("Species 2", hbox)
        
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
        generator = lattice_gen_sic.SiC4HLatticeGenerator()
        
        status, lattice = generator.generateLattice(self.generatorArgs)
        
        return status, lattice
