
"""
Info dialogs

@author: Chris Scott

"""
import sys
import uuid

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
    def __init__(self, pipelinePage, atomIndex, scalar, scalarType, filterList, parent=None):
        super(AtomInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.windowID = uuid.uuid4()
        
        self.parent = parent
        self.pipelinePage = pipelinePage
        self.atomIndex = atomIndex
        self.filterList = filterList
        
        lattice = self.pipelinePage.inputState
        
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
        
        row = QtGui.QHBoxLayout()
        row.addStretch(1)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.close)
        row.addWidget(closeButton)
        row.addStretch(1)
        layout.addLayout(row)
        
        self.setLayout(layout)
    
    def getHighlighters(self):
        """
        Return highlighter for this atom
        
        """
        # lattice
        lattice = self.pipelinePage.inputState
        
        # radius
        radius = lattice.specieCovalentRadius[lattice.specie[self.atomIndex]] * self.filterList.displayOptions.atomScaleFactor
        
        # highlighter
        highlighter = highlight.AtomHighlighter(lattice.atomPos(self.atomIndex), radius * 1.1)
        
        return self.windowID, [highlighter,]
    
    def closeEvent(self, event):
        """
        Override close event
        
        """
        # remove highlighters
        self.pipelinePage.broadcastToRenderers("removeHighlighters", (self.windowID,))
        
        event.accept()

################################################################################

class DefectInfoWindow(QtGui.QDialog):
    """
    Atom info window.
    
    """
    def __init__(self, pipelinePage, defectIndex, defectType, defList, filterList, parent=None):
        super(DefectInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.windowID = uuid.uuid4()
        
        self.parent = parent
        self.pipelinePage = pipelinePage
        self.defectIndex = defectIndex
        self.defectType = defectType
        self.defList = defList
        self.filterList = filterList
        
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        
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
            row.addWidget(QtGui.QLabel("Site index: %d" % index))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Specie: %s" % refState.specieList[refState.specie[index]]))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Position: (%f, %f, %f)" % (refState.pos[3*index], refState.pos[3*index+1], refState.pos[3*index+2])))
            layout.addLayout(row)
        
        elif defectType == 2:
            interstitials = defList[0]
            
            # interstitial
            index = interstitials[defectIndex]
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Type: interstitial"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Atom: %d" % index))
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
            row.addWidget(QtGui.QLabel("Site index: %d" % index))
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
            row.addWidget(QtGui.QLabel("    Atom: %d" % index2))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Position: (%f, %f, %f)" % (inputState.pos[3*index2], inputState.pos[3*index2+1], inputState.pos[3*index2+2])))
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
            row.addWidget(QtGui.QLabel("Site index: %d" % vacIndex))
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
            row.addWidget(QtGui.QLabel("    Atom: %d" % int1Index))
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
            row.addWidget(QtGui.QLabel("    Atom: %d" % int2Index))
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
            
            sepVec = vectors.separationVector(pos1, pos2, inputState.cellDims, self.pipelinePage.PBC)
            norm = vectors.normalise(sepVec)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Orientation: (%f %f %f)" % (norm[0], norm[1], norm[2])))
            layout.addLayout(row)
        
        row = QtGui.QHBoxLayout()
        row.addStretch(1)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.close)
        row.addWidget(closeButton)
        row.addStretch(1)
        layout.addLayout(row)
        
        self.setLayout(layout)
    
    def getHighlighters(self):
        """
        Return highlighter for this defect
        
        """
        highlighters = []
        
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        
        if self.defectType == 1:
            vacancies = self.defList[0]
            
            # vacancy
            index = vacancies[self.defectIndex]
            
            # radius
            radius = refState.specieCovalentRadius[refState.specie[index]] * self.filterList.displayOptions.atomScaleFactor
            
            # can do this because defect filter is always by itself
            vacScaleSize = self.filterList.currentSettings[0].vacScaleSize
            radius *= vacScaleSize * 2.0
            
            # highlighter
            highlighter = highlight.VacancyHighlighter(refState.atomPos(index), radius * 1.1)
            
            highlighters.append(highlighter)
        
        elif self.defectType == 2:
            interstitials = self.defList[0]
            
            # interstitial
            index = interstitials[self.defectIndex]
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlighter
            highlighter = highlight.AtomHighlighter(inputState.atomPos(index), radius * 1.1)
            
            highlighters.append(highlighter)
        
        elif self.defectType == 3:
            antisites = self.defList[0]
            onAntisites = self.defList[1]
            
            # antisite
            index = antisites[self.defectIndex]
            index2 = onAntisites[self.defectIndex]
            
            #### highlight antisite ####
            
            # radius
            radius = 2.0 * refState.specieCovalentRadius[refState.specie[index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighter = highlight.AntisiteHighlighter(refState.atomPos(index), radius)
            highlighters.append(highlighter)
            
            #### highlight occupying atom ####
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[index2]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighter = highlight.AtomHighlighter(inputState.atomPos(index2), radius * 1.1)
            highlighters.append(highlighter)
        
        elif self.defectType == 4:
            splitInts = self.defList[0]
            
            # split interstitial
            vacIndex = splitInts[3*self.defectIndex]
            int1Index = splitInts[3*self.defectIndex+1]
            int2Index = splitInts[3*self.defectIndex+2]
            
            #### highlight vacancy ####
            
            # radius
            radius = refState.specieCovalentRadius[refState.specie[vacIndex]] * self.filterList.displayOptions.atomScaleFactor
            
            # can do this because defect filter is always by itself
            vacScaleSize = self.filterList.currentSettings[0].vacScaleSize
            radius *= vacScaleSize * 2.0
            
            # highlight
            highlighter = highlight.VacancyHighlighter(refState.atomPos(vacIndex), radius * 1.1)
            highlighters.append(highlighter)
            
            #### highlight int 1 ####
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[int1Index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighter = highlight.AtomHighlighter(inputState.atomPos(int1Index), radius * 1.1)
            highlighters.append(highlighter)
            
            #### highlight int 2 ####
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[int2Index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighter = highlight.AtomHighlighter(inputState.atomPos(int2Index), radius * 1.1)
            highlighters.append(highlighter)
        
        return self.windowID, highlighters
    
    def closeEvent(self, event):
        """
        Close event
        
        """
        # remove highlighters
        self.pipelinePage.broadcastToRenderers("removeHighlighters", (self.windowID,))
        
        event.accept()
