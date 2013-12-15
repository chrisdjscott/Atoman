
"""
Info dialogs

@author: Chris Scott

"""
import sys

from PySide import QtGui, QtCore

from ..visutils import vectors
from ..rendering import highlight

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)

################################################################################

class AtomInfoWindow(QtGui.QDialog):
    """
    Atom info window.
    
    """
    def __init__(self, rendererWindow, atomIndex, scalar, scalarType, filterList, parent=None):
        super(AtomInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
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
        
        # radius
        radius = lattice.specieCovalentRadius[lattice.specie[atomIndex]] * filterList.displayOptions.atomScaleFactor
        
        self.highlighter.add(lattice.atomPos(atomIndex), radius * 1.1)
        
        self.setLayout(layout)
    
    def closeEvent(self, event):
        """
        Override close event
        
        """
        self.highlighter.remove()
        
        event.accept()

################################################################################

class DefectInfoWindow(QtGui.QDialog):
    """
    Atom info window.
    
    """
    def __init__(self, rendererWindow, defectIndex, defectType, defList, parent=None):
        super(DefectInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
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
