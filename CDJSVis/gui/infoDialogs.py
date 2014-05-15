
"""
Info dialogs

@author: Chris Scott

"""
import sys
import uuid
import functools

from PySide import QtGui, QtCore

from ..visutils import vectors
from ..rendering import highlight

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)

################################################################################

class ClusterInfoWindow(QtGui.QDialog):
    """
    Cluster info window
    
    """
    def __init__(self, pipelinePage, filterList, clusterIndex, parent=None):
        super(ClusterInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.setWindowTitle("Cluster %d info" % clusterIndex)
        self.windowID = uuid.uuid4()
        
        self.pipelinePage = pipelinePage
        self.filterList = filterList
        self.clusterIndex = clusterIndex
        
        lattice = self.pipelinePage.inputState
        
        layout = QtGui.QVBoxLayout(self)
        
        # cluster
        self.cluster = filterList.filterer.clusterList[clusterIndex]
        
        # volume
        if self.cluster.volume is not None:
            layout.addWidget(QtGui.QLabel("Volume: %f units^3" % self.cluster.volume))
        
        # facet area
        if self.cluster.facetArea is not None:
            layout.addWidget(QtGui.QLabel("Facet area: %f units^2" % self.cluster.facetArea))
        
        # label
        layout.addWidget(QtGui.QLabel("Cluster atoms (%d):" % len(self.cluster)))
        
        # list widget
        self.listWidget = QtGui.QListWidget(self)
        self.listWidget.setFixedHeight(175)
        self.listWidget.setFixedWidth(300)
        layout.addWidget(self.listWidget)
        
        # populate list widget
        
        for index in self.cluster:
            item = QtGui.QListWidgetItem()
            item.setText("Atom %d: %s (%.3f, %.3f, %.3f)" % (lattice.atomID[index], lattice.atomSym(index), lattice.pos[3*index], 
                                                        lattice.pos[3*index+1], lattice.pos[3*index+2]))
            self.listWidget.addItem(item)
        
        
        
        # close button
        row = QtGui.QHBoxLayout()
        row.addStretch(1)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.close)
        closeButton.setAutoDefault(True)
        row.addWidget(closeButton)
        row.addStretch(1)
        layout.addLayout(row)

################################################################################

class AtomInfoWindow(QtGui.QDialog):
    """
    Atom info window.
    
    """
    def __init__(self, pipelinePage, atomIndex, scalarsDict, filterList, parent=None):
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
        row.addWidget(QtGui.QLabel("Atom: %d" % lattice.atomID[atomIndex]))
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
        
        # add scalars
        for scalarType, scalar in scalarsDict.iteritems():
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("%s: %f" % (scalarType, scalar)))
            layout.addLayout(row)
        
        # check if belongs to a cluster?
        clusterIndex = None
        for i, cluster in enumerate(filterList.filterer.clusterList):
            if atomIndex in cluster:
                clusterIndex = i
                break
        
        # add button to show cluster info
        if clusterIndex is not None:
            clusterButton = QtGui.QPushButton("Cluster %d info" % clusterIndex)
            clusterButton.clicked.connect(functools.partial(filterList.showClusterInfoWindow, clusterIndex))
            clusterButton.setAutoDefault(False)
            row = QtGui.QHBoxLayout()
            row.addStretch()
            row.addWidget(clusterButton)
            row.addStretch()
            layout.addLayout(row)
        
        # close button
        row = QtGui.QHBoxLayout()
        row.addStretch(1)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.close)
        closeButton.setAutoDefault(True)
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
        voronoiOptions = filterList.voronoiOptions
        
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        
        vor = None
        voroKey = voronoiOptions.getVoronoiDictKey()
        if voroKey in inputState.voronoiDict:
            vor = inputState.voronoiDict[voroKey]
        
        self.setWindowTitle("Defect info")
        
        self.mainLayout = QtGui.QVBoxLayout()
        layout = self.mainLayout
        
        if defectType == 1:
            vacancies = defList[0]
            
            # vacancy
            index = vacancies[defectIndex]
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Type: vacancy"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Site index: %d" % refState.atomID[index]))
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
            row.addWidget(QtGui.QLabel("Atom: %d" % inputState.atomID[index]))
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
            
            if vor is not None:
                self.voroVolLine(vor.atomVolume(index))
        
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
            row.addWidget(QtGui.QLabel("Site index: %d" % refState.atomID[index]))
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
            row.addWidget(QtGui.QLabel("    Atom: %d" % inputState.atomID[index2]))
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
            
            if vor is not None:
                self.voroVolLine(vor.atomVolume(index2))
        
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
            row.addWidget(QtGui.QLabel("Site index: %d" % refState.atomID[vacIndex]))
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
            row.addWidget(QtGui.QLabel("    Atom: %d" % inputState.atomID[int1Index]))
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
            
            if vor is not None:
                self.voroVolLine(vor.atomVolume(int1Index))
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("Interstitial 2:"))
            layout.addLayout(row)
            
            row = QtGui.QHBoxLayout()
            row.addWidget(QtGui.QLabel("    Atom: %d" % inputState.atomID[int2Index]))
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
            
            if vor is not None:
                self.voroVolLine(vor.atomVolume(int2Index))
            
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
    
    def voroVolLine(self, vol):
        """
        Return line containing Voronoi volume
        
        """
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Voronoi volume: %f" % vol))
        self.mainLayout.addLayout(row)
    
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
            vacScaleSize = self.filterList.getCurrentFilterSettings()[0].vacScaleSize
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
            vacScaleSize = self.filterList.getCurrentFilterSettings()[0].vacScaleSize
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
