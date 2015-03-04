
"""
Info dialogs

@author: Chris Scott

"""
import uuid
import functools
import logging

from PySide import QtGui, QtCore

from ..algebra import vectors
from ..rendering import highlight
from . import utils


################################################################################

class ClusterListWidgetItem(QtGui.QListWidgetItem):
    """
    Item for cluster info window list widget
    
    """
    def __init__(self, atomIndex, defectType=None):
        super(ClusterListWidgetItem, self).__init__()
        self.atomIndex = atomIndex
        self.defectType = defectType

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
        
        # highlighter colour
        self.highlightColour = QtGui.QColor(255, 20, 147)
        self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                   float(self.highlightColour.green()) / 255.0,
                                   float(self.highlightColour.blue()) / 255.0]
        
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
        self.listWidget.setMinimumWidth(300)
        self.listWidget.setMinimumHeight(175)
        layout.addWidget(self.listWidget)
        
        #TODO: open atom info window on right click select from context menu
        
        
        # populate list widget
        for index in self.cluster:
            item = ClusterListWidgetItem(index)
            item.setText("Atom %d: %s (%.3f, %.3f, %.3f)" % (lattice.atomID[index], lattice.atomSym(index), lattice.pos[3*index], 
                                                             lattice.pos[3*index+1], lattice.pos[3*index+2]))
            self.listWidget.addItem(item)
        
        # colour button
        self.colourButton = QtGui.QPushButton("")
        self.colourButton.setFixedWidth(60)
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
        self.colourButton.clicked.connect(self.changeHighlighterColour)
        self.colourButton.setAutoDefault(False)
        self.highlightCheck = QtGui.QCheckBox("Highlight")
        self.highlightCheck.setChecked(True)
        self.highlightCheck.stateChanged.connect(self.highlightChanged)
        
        # close button
        row = QtGui.QHBoxLayout()
        row.addWidget(self.colourButton)
        row.addWidget(self.highlightCheck)
        row.addStretch(1)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.close)
        closeButton.setAutoDefault(True)
        row.addWidget(closeButton)
        layout.addLayout(row)
    
    def highlightChanged(self, state):
        """
        Highlight check changed
        
        """
        if state == QtCore.Qt.Unchecked:
            self.removeHighlighters()
            self.colourButton.setEnabled(False)
        
        else:
            self.addHighlighters()
            self.colourButton.setEnabled(True)
    
    def changeHighlighterColour(self):
        """
        Change highlighter colour
        
        """
        col = QtGui.QColorDialog.getColor(initial=self.highlightColour, title="Set highlighter colour")
        
        if col.isValid():
            self.highlightColour = col
            self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
            
            self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                       float(self.highlightColour.green()) / 255.0,
                                       float(self.highlightColour.blue()) / 255.0]
            
            # make change
            self.removeHighlighters()
            self.addHighlighters()
    
    def addHighlighters(self):
        """
        Return highlighters for this cluster
        
        """
        # lattice
        lattice = self.pipelinePage.inputState
        
        highlighters = []
        for atomIndex in self.cluster:
            # radius
            radius = lattice.specieCovalentRadius[lattice.specie[atomIndex]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlighter
            highlighters.append(highlight.AtomHighlighter(lattice.atomPos(atomIndex), radius * 1.1, rgb=self.highlightColourRGB))
        
        self.pipelinePage.broadcastToRenderers("addHighlighters", (self.windowID, highlighters))
    
    def show(self):
        """
        We override show() to first add highlighters to renderer windows
        
        """
        self.addHighlighters()
        
        return super(ClusterInfoWindow, self).show()
    
    def removeHighlighters(self):
        """
        Remove highlighters
        
        """
        self.pipelinePage.broadcastToRenderers("removeHighlighters", (self.windowID,))
    
    def closeEvent(self, event):
        """
        Override close event
        
        """
        # remove highlighters
        self.removeHighlighters()
        
        event.accept()

################################################################################

class NeighbourListWidgetItem(QtGui.QListWidgetItem):
    """
    Item for neighbours info window list widget
    
    """
    def __init__(self, atomIndex, separation):
        super(NeighbourListWidgetItem, self).__init__()
        self.atomIndex = atomIndex
        self.separation = separation

################################################################################

class AtomNeighboursInfoWindow(QtGui.QDialog):
    """
    Neighbours info window
    
    """
    def __init__(self, pipelinePage, atomIndex, nebList, nebFilterList, title, parent=None):
        super(AtomNeighboursInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.windowID = uuid.uuid4()
        
        self.pipelinePage = pipelinePage
        self.atomIndex = atomIndex
        self.nebList = nebList
        self.nebFilterList = nebFilterList
        
        lattice = self.pipelinePage.inputState
        
        self.setWindowTitle(title)
        
        # highlighter colour
        self.highlightColour = QtGui.QColor(255, 20, 147)
        self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                   float(self.highlightColour.green()) / 255.0,
                                   float(self.highlightColour.blue()) / 255.0]
        
        layout = QtGui.QVBoxLayout(self)
        
        # label
        layout.addWidget(QtGui.QLabel("Neighbours (%d):" % len(self.nebList)))
        
        # list widget
        self.listWidget = QtGui.QListWidget(self)
        self.listWidget.setMinimumWidth(300)
        self.listWidget.setMinimumHeight(175)
        layout.addWidget(self.listWidget)
        
        #TODO: open atom info window on right click select from context menu
        
        
        # populate list widget
        for index in self.nebList:
            sep = lattice.atomSeparation(atomIndex, index, self.pipelinePage.PBC)
            item = NeighbourListWidgetItem(index, sep)
            item.setText("Atom %d: %s; Separation = %f" % (lattice.atomID[index], lattice.atomSym(index), sep))
            self.listWidget.addItem(item)
        
        # colour button
        self.colourButton = QtGui.QPushButton("")
        self.colourButton.setFixedWidth(60)
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
        self.colourButton.clicked.connect(self.changeHighlighterColour)
        self.colourButton.setAutoDefault(False)
        self.highlightCheck = QtGui.QCheckBox("Highlight")
        self.highlightCheck.setChecked(True)
        self.highlightCheck.stateChanged.connect(self.highlightChanged)
        
        # close button
        row = QtGui.QHBoxLayout()
        row.addWidget(self.colourButton)
        row.addWidget(self.highlightCheck)
        row.addStretch(1)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.close)
        closeButton.setAutoDefault(True)
        row.addWidget(closeButton)
        layout.addLayout(row)
    
    def highlightChanged(self, state):
        """
        Highlight check changed
        
        """
        if state == QtCore.Qt.Unchecked:
            self.removeHighlighters()
            self.colourButton.setEnabled(False)
        
        else:
            self.addHighlighters()
            self.colourButton.setEnabled(True)
    
    def changeHighlighterColour(self):
        """
        Change highlighter colour
        
        """
        col = QtGui.QColorDialog.getColor(initial=self.highlightColour, title="Set highlighter colour")
        
        if col.isValid():
            self.highlightColour = col
            self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
            
            self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                       float(self.highlightColour.green()) / 255.0,
                                       float(self.highlightColour.blue()) / 255.0]
            
            # make change
            self.removeHighlighters()
            self.addHighlighters()
    
    def addHighlighters(self):
        """
        Return highlighters for this atoms neighbour list
        
        """
        # lattice
        lattice = self.pipelinePage.inputState
        
        highlighters = []
        for i, atomIndex in enumerate(self.nebList):
            # radius
            radius = lattice.specieCovalentRadius[lattice.specie[atomIndex]] * self.nebFilterList[i].displayOptions.atomScaleFactor
            
            # highlighter
            highlighters.append(highlight.AtomHighlighter(lattice.atomPos(atomIndex), radius * 1.1, rgb=self.highlightColourRGB))
        
        self.pipelinePage.broadcastToRenderers("addHighlighters", (self.windowID, highlighters))
    
    def show(self):
        """
        We override show() to first add highlighters to renderer windows
        
        """
        self.addHighlighters()
        
        return super(AtomNeighboursInfoWindow, self).show()
    
    def removeHighlighters(self):
        """
        Remove highlighters
        
        """
        self.pipelinePage.broadcastToRenderers("removeHighlighters", (self.windowID,))
    
    def closeEvent(self, event):
        """
        Override close event
        
        """
        # remove highlighters
        self.removeHighlighters()
        
        event.accept()

################################################################################

class DefectClusterInfoWindow(QtGui.QDialog):
    """
    Defect cluster info window
    
    """
    def __init__(self, pipelinePage, filterList, clusterIndex, parent=None):
        super(DefectClusterInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.setWindowTitle("Cluster %d info" % clusterIndex)
        self.windowID = uuid.uuid4()
        
        self.pipelinePage = pipelinePage
        self.filterList = filterList
        self.clusterIndex = clusterIndex
        
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        
        # highlighter colour
        self.highlightColour = QtGui.QColor(255, 20, 147)
        self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                   float(self.highlightColour.green()) / 255.0,
                                   float(self.highlightColour.blue()) / 255.0]
        
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
        layout.addWidget(QtGui.QLabel("Cluster defects (%d):" % self.cluster.getNDefects()))
        
        # list widget
        self.listWidget = QtGui.QListWidget(self)
        self.listWidget.setMinimumWidth(380)
        self.listWidget.setMinimumHeight(200)
        layout.addWidget(self.listWidget)
        
        #TODO: open atom info window on right click select from context menu
        
        
        # populate list widget
        for index in self.cluster.vacancies:
            item = ClusterListWidgetItem(index, defectType=1)
            item.setText("Vacancy %d: %s (%.3f, %.3f, %.3f)" % (refState.atomID[index], refState.atomSym(index), refState.pos[3*index], 
                                                                refState.pos[3*index+1], refState.pos[3*index+2]))
            self.listWidget.addItem(item)
        
        for index in self.cluster.interstitials:
            item = ClusterListWidgetItem(index, defectType=2)
            item.setText("Interstitial %d: %s (%.3f, %.3f, %.3f)" % (inputState.atomID[index], inputState.atomSym(index), inputState.pos[3*index], 
                                                                     inputState.pos[3*index+1], inputState.pos[3*index+2]))
            self.listWidget.addItem(item)
        
        for i in xrange(self.cluster.getNAntisites()):
            index = self.cluster.antisites[i]
            index1 = self.cluster.onAntisites[i]
            
            item = ClusterListWidgetItem(index, defectType=3)
            item.setText("Antisite %d: %s on %s (%.3f, %.3f, %.3f)" % (refState.atomID[index], refState.atomSym(index1), refState.atomSym(index), 
                                                                       refState.pos[3*index], refState.pos[3*index+1], refState.pos[3*index+2]))
            self.listWidget.addItem(item)
        
        for i in xrange(self.cluster.getNSplitInterstitials()):
            index = self.cluster.splitInterstitials[3*i]
            index1 = self.cluster.splitInterstitials[3*i+1]
            index2 = self.cluster.splitInterstitials[3*i+2]
            item = ClusterListWidgetItem(index, defectType=4)
            item.setText("Split interstitial %d: %s-%s (%.3f, %.3f, %.3f)" % (refState.atomID[index], refState.atomSym(index1), refState.atomSym(index2),
                                                                              refState.pos[3*index], refState.pos[3*index+1], refState.pos[3*index+2]))
            self.listWidget.addItem(item)
        
        # colour button
        self.colourButton = QtGui.QPushButton("")
        self.colourButton.setFixedWidth(60)
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
        self.colourButton.clicked.connect(self.changeHighlighterColour)
        self.colourButton.setAutoDefault(False)
        self.highlightCheck = QtGui.QCheckBox("Highlight")
        self.highlightCheck.setChecked(True)
        self.highlightCheck.stateChanged.connect(self.highlightChanged)
        
        # close button
        row = QtGui.QHBoxLayout()
        row.addWidget(self.colourButton)
        row.addWidget(self.highlightCheck)
        row.addStretch(1)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.close)
        closeButton.setAutoDefault(True)
        row.addWidget(closeButton)
        layout.addLayout(row)
    
    def changeHighlighterColour(self):
        """
        Change highlighter colour
        
        """
        col = QtGui.QColorDialog.getColor(initial=self.highlightColour, title="Set highlighter colour")
        
        if col.isValid():
            self.highlightColour = col
            self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
            
            self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                       float(self.highlightColour.green()) / 255.0,
                                       float(self.highlightColour.blue()) / 255.0]
            
            # make change
            self.removeHighlighters()
            self.addHighlighters()
    
    def highlightChanged(self, state):
        """
        Highlight check changed
        
        """
        if state == QtCore.Qt.Unchecked:
            self.removeHighlighters()
            self.colourButton.setEnabled(False)
        
        else:
            self.addHighlighters()
            self.colourButton.setEnabled(True)
    
    def addHighlighters(self):
        """
        Return highlighters for this defect cluster
        
        """
        highlighters = []
        
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        
        for index in self.cluster.vacancies:
            # radius
            radius = refState.specieCovalentRadius[refState.specie[index]] * self.filterList.displayOptions.atomScaleFactor
            
            # can do this because defect filter is always by itself
            vacScaleSize = self.filterList.getCurrentFilterSettings()[0].vacScaleSize
            radius *= vacScaleSize * 2.0
            
            # highlighter
            highlighters.append(highlight.VacancyHighlighter(refState.atomPos(index), radius * 1.1, rgb=self.highlightColourRGB))
        
        for index in self.cluster.interstitials:
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlighter
            highlighters.append(highlight.AtomHighlighter(inputState.atomPos(index), radius * 1.1, rgb=self.highlightColourRGB))
        
        for i in xrange(self.cluster.getNAntisites()):
            index = self.cluster.antisites[i]
            index2 = self.cluster.onAntisites[i]
            
            #### highlight antisite ####
            
            # radius
            radius = 2.0 * refState.specieCovalentRadius[refState.specie[index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighters.append(highlight.AntisiteHighlighter(refState.atomPos(index), radius, rgb=self.highlightColourRGB))
            
            #### highlight occupying atom ####
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[index2]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighters.append(highlight.AtomHighlighter(inputState.atomPos(index2), radius * 1.1, rgb=self.highlightColourRGB))
        
        for i in xrange(self.cluster.getNSplitInterstitials()):
            vacIndex = self.cluster.splitInterstitials[3*i]
            int1Index = self.cluster.splitInterstitials[3*i+1]
            int2Index = self.cluster.splitInterstitials[3*i+2]
            
            #### highlight vacancy ####
            
            # radius
            radius = refState.specieCovalentRadius[refState.specie[vacIndex]] * self.filterList.displayOptions.atomScaleFactor
            
            # can do this because defect filter is always by itself
            vacScaleSize = self.filterList.getCurrentFilterSettings()[0].vacScaleSize
            radius *= vacScaleSize * 2.0
            
            # highlight
            highlighters.append(highlight.VacancyHighlighter(refState.atomPos(vacIndex), radius * 1.1, rgb=self.highlightColourRGB))
            
            #### highlight int 1 ####
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[int1Index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighters.append(highlight.AtomHighlighter(inputState.atomPos(int1Index), radius * 1.1, rgb=self.highlightColourRGB))
            
            #### highlight int 2 ####
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[int2Index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighters.append(highlight.AtomHighlighter(inputState.atomPos(int2Index), radius * 1.1, rgb=self.highlightColourRGB))
        
        self.pipelinePage.broadcastToRenderers("addHighlighters", (self.windowID, highlighters))
    
    def show(self):
        """
        We override show() to first add highlighters to renderer windows
        
        """
        self.addHighlighters()
        
        return super(DefectClusterInfoWindow, self).show()
    
    def removeHighlighters(self):
        """
        Remove highlighters
        
        """
        self.pipelinePage.broadcastToRenderers("removeHighlighters", (self.windowID,))
    
    def closeEvent(self, event):
        """
        Override close event
        
        """
        # remove highlighters
        self.removeHighlighters()
        
        event.accept()

################################################################################

class SystemInfoWindow(QtGui.QDialog):
    """
    System info window.
    
    """
    def __init__(self, item, parent=None):
        super(SystemInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("System info: %s" % item.displayName)
        
        # attributes
        lattice = item.lattice
        
        layout = QtGui.QVBoxLayout()
        
        listWidget = QtGui.QListWidget(self)
        listWidget.setMinimumWidth(400)
        listWidget.setMinimumHeight(200)
        layout.addWidget(listWidget)
        
        listWidget.addItem("Display name: '%s'" % item.displayName)
        listWidget.addItem("Abspath: '%s'" % item.abspath)
        listWidget.addItem("Number of atoms: %d" % lattice.NAtoms)
        listWidget.addItem("Cell dimensions: [%f, %f, %f]" % (lattice.cellDims[0], lattice.cellDims[1], lattice.cellDims[2]))
        listWidget.addItem("Species list: %r" % list(lattice.specieList))
        listWidget.addItem("Species count: %r" % list(lattice.specieCount))
        
        # add lattice attributes
        if "Temperature" not in lattice.attributes:
            temperature = lattice.calcTemperature()
            if temperature is not None:
                listWidget.addItem("Temperature: %f" % temperature)
        
        for key, value in lattice.attributes.iteritems():
            listWidget.addItem("%s: %s" % (key, value))
        
        # button box
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

################################################################################

class AtomInfoWindow(QtGui.QDialog):
    """
    Atom info window.
    
    """
    def __init__(self, pipelinePage, atomIndex, scalarsDict, vectorsDict, filterList, parent=None):
        super(AtomInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.windowID = uuid.uuid4()
        
        self.parent = parent
        self.pipelinePage = pipelinePage
        self.atomIndex = atomIndex
        self.filterList = filterList
        self.neighbourInfoWindow = None
        
        lattice = self.pipelinePage.inputState
        
        self.setWindowTitle("Atom info")
        
        # logger
        self.logger = logging.getLogger(__name__+".AtomInfoWindow")
        self.logger.debug("Constructing info window for atom %d (%d)", lattice.atomID[atomIndex], atomIndex)
        
        # highlighter colour
        self.highlightColour = QtGui.QColor(158, 0, 196)
        self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                   float(self.highlightColour.green()) / 255.0,
                                   float(self.highlightColour.blue()) / 255.0]
        
        layout = QtGui.QVBoxLayout()
        
        listWidget = QtGui.QListWidget(self)
        listWidget.setMinimumWidth(300)
        listWidget.setMinimumHeight(175)
        layout.addWidget(listWidget)
        
        listWidget.addItem("Atom: %d" % lattice.atomID[atomIndex])
        listWidget.addItem("Specie: %s" % lattice.specieList[lattice.specie[atomIndex]])
        listWidget.addItem("Position: (%f, %f, %f)" % (lattice.pos[3*atomIndex], lattice.pos[3*atomIndex+1], lattice.pos[3*atomIndex+2]))
        listWidget.addItem("Charge: %f" % lattice.charge[atomIndex])
        
        # add scalars
        for scalarType, scalar in scalarsDict.iteritems():
            listWidget.addItem("%s: %f" % (scalarType, scalar))
        
        # add vectors
        for vectorType, vector in vectorsDict.iteritems():
            listWidget.addItem("%s: (%f, %f, %f)" % (vectorType, vector[0], vector[1], vector[2]))
        
        # add Voronoi neighbour info (if available)
        voro = filterList.filterer.voronoi
        self.voroNebList = []
        self.voroNebFilterList = []
        if voro is not None:
            voroNebList = voro.atomNebList(atomIndex)
            self.logger.debug("Voro neighbours exists: len = %d", len(voroNebList))
            
            # only show visible neighbours
            for nebIndex in voroNebList:
                self.logger.debug("Checking if neighbour %d (%d) is visible", lattice.atomID[nebIndex], nebIndex)
                
                visible, visFiltList = pipelinePage.checkIfAtomVisible(nebIndex)
                if visible:
                    self.logger.debug("Atom %d (%d) is visible; adding to neb list", lattice.atomID[nebIndex], nebIndex)
                    self.voroNebList.append(nebIndex)
                    self.voroNebFilterList.append(visFiltList)
            
            if len(self.voroNebList):
                # add button
                nebButton = QtGui.QPushButton("Neighbour list")
                nebButton.clicked.connect(self.showVoroNeighbourInfoWindow)
                nebButton.setAutoDefault(False)
                row = QtGui.QHBoxLayout()
                row.addStretch()
                row.addWidget(nebButton)
                row.addStretch()
                layout.addLayout(row)
        
        # check if belongs to clusters
        clusterIndexes = []
        for i, cluster in enumerate(filterList.filterer.clusterList):
            if atomIndex in cluster:
                clusterIndexes.append(i)
        
        # add button to show cluster info
        if len(clusterIndexes):
            for clusterIndex in clusterIndexes:
                clusterButton = QtGui.QPushButton("Cluster %d info" % clusterIndex)
                clusterButton.clicked.connect(functools.partial(filterList.showClusterInfoWindow, clusterIndex))
                clusterButton.setAutoDefault(False)
                row = QtGui.QHBoxLayout()
                row.addStretch()
                row.addWidget(clusterButton)
                row.addStretch()
                layout.addLayout(row)
        
        # colour button
        self.colourButton = QtGui.QPushButton("")
        self.colourButton.setFixedWidth(60)
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
        self.colourButton.clicked.connect(self.changeHighlighterColour)
        self.colourButton.setAutoDefault(False)
        self.highlightCheck = QtGui.QCheckBox("Highlight")
        self.highlightCheck.setChecked(True)
        self.highlightCheck.stateChanged.connect(self.highlightChanged)
        
        # close button
        row = QtGui.QHBoxLayout()
        row.addWidget(self.colourButton)
        row.addWidget(self.highlightCheck)
        row.addStretch(1)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.close)
        closeButton.setAutoDefault(True)
        row.addWidget(closeButton)
        layout.addLayout(row)
        
        self.setLayout(layout)
    
    def showVoroNeighbourInfoWindow(self):
        """
        Show the Voronoi neighbour info window
        
        """
        # make the window if it doesn't exist
        if self.neighbourInfoWindow is None:
            title = "Voronoi neighbours of atom %d" % self.pipelinePage.inputState.atomID[self.atomIndex]
            self.neighbourInfoWindow = AtomNeighboursInfoWindow(self.pipelinePage, self.atomIndex, self.voroNebList, 
                                                                self.voroNebFilterList, title, parent=self)
        
        # position window
        utils.positionWindow(self.neighbourInfoWindow, self.neighbourInfoWindow.size(), self.pipelinePage.mainWindow.desktop, self)
        
        # show the window
        self.neighbourInfoWindow.show()
    
    def changeHighlighterColour(self):
        """
        Change highlighter colour
        
        """
        col = QtGui.QColorDialog.getColor(initial=self.highlightColour, title="Set highlighter colour")
        
        if col.isValid():
            self.highlightColour = col
            self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
            
            self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                       float(self.highlightColour.green()) / 255.0,
                                       float(self.highlightColour.blue()) / 255.0]
            
            # make change
            self.removeHighlighters()
            self.addHighlighters()
    
    def addHighlighters(self):
        """
        Add highlighter for this atom
        
        """
        # lattice
        lattice = self.pipelinePage.inputState
        
        # radius
        radius = lattice.specieCovalentRadius[lattice.specie[self.atomIndex]] * self.filterList.displayOptions.atomScaleFactor
        
        # highlighter
        highlighter = highlight.AtomHighlighter(lattice.atomPos(self.atomIndex), radius * 1.1, rgb=self.highlightColourRGB)
        
        self.pipelinePage.broadcastToRenderers("addHighlighters", (self.windowID, [highlighter,]))
    
    def highlightChanged(self, state):
        """
        Highlight check changed
        
        """
        if state == QtCore.Qt.Unchecked:
            self.removeHighlighters()
            self.colourButton.setEnabled(False)
        
        else:
            self.addHighlighters()
            self.colourButton.setEnabled(True)
    
    def show(self):
        """
        We override show() to first add highlighters to renderer windows
        
        """
        self.addHighlighters()
        
        return super(AtomInfoWindow, self).show()
    
    def removeHighlighters(self):
        """
        Remove highlighters
        
        """
        self.pipelinePage.broadcastToRenderers("removeHighlighters", (self.windowID,))
    
    def closeEvent(self, event):
        """
        Override close event
        
        """
        if self.neighbourInfoWindow is not None:
            self.neighbourInfoWindow.close()
        
        # remove highlighters
        self.removeHighlighters()
        
        event.accept()

################################################################################

class DefectInfoWindow(QtGui.QDialog):
    """
    Atom info window.
    
    """
    def __init__(self, pipelinePage, defectIndex, defectType, defList, scalarsDict, vectorsDict, filterList, parent=None):
        super(DefectInfoWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.windowID = uuid.uuid4()
        
        self.parent = parent
        self.pipelinePage = pipelinePage
        self.defectIndex = defectIndex
        self.defectType = defectType
        self.defList = defList
        self.filterList = filterList
        self.scalarsDict = scalarsDict
        self.vectorsDict = vectorsDict
        
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        
        # highlighter colour
        self.highlightColour = QtGui.QColor(158, 0, 196)
        self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                   float(self.highlightColour.green()) / 255.0,
                                   float(self.highlightColour.blue()) / 255.0]
        
        vor = filterList.filterer.voronoi
        
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
            
            listWidget = QtGui.QListWidget(self)
            listWidget.setMinimumWidth(300)
            listWidget.setMinimumHeight(150)
            layout.addWidget(listWidget)
            
            listWidget.addItem("Atom: %d" % inputState.atomID[index])
            listWidget.addItem("Specie: %s" % inputState.specieList[inputState.specie[index]])
            listWidget.addItem("Position: (%f, %f, %f)" % (inputState.pos[3*index], inputState.pos[3*index+1], inputState.pos[3*index+2]))
            listWidget.addItem("Charge: %f" % inputState.charge[index])
            
            # voro vol (why separate?)
            if vor is not None:
                listWidget.addItem("Voronoi volume: %f" % vor.atomVolume(index))
            
            # add scalars
            for scalarType, scalar in scalarsDict.iteritems():
                listWidget.addItem("%s: %f" % (scalarType, scalar))
            
            # add vectors
            for vectorType, vector in vectorsDict.iteritems():
                listWidget.addItem("%s: (%f, %f, %f)" % (vectorType, vector[0], vector[1], vector[2]))
        
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
        
        # check if belongs to a cluster?
        clusterIndex = None
        for i, cluster in enumerate(filterList.filterer.clusterList):
            if defectType == 1:
                index = filterList.filterer.vacancies[defectIndex]
            elif defectType == 2:
                index = filterList.filterer.interstitials[defectIndex]
            elif defectType == 3:
                index = filterList.filterer.antisites[defectIndex]
            elif defectType == 4:
                index = filterList.filterer.splitInterstitials[3*defectIndex]
            
            if cluster.belongsInCluster(defectType, index):
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
        
        # colour button
        self.colourButton = QtGui.QPushButton("")
        self.colourButton.setFixedWidth(60)
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
        self.colourButton.clicked.connect(self.changeHighlighterColour)
        self.colourButton.setAutoDefault(False)
        self.highlightCheck = QtGui.QCheckBox("Highlight")
        self.highlightCheck.setChecked(True)
        self.highlightCheck.stateChanged.connect(self.highlightChanged)
        
        # close button
        row = QtGui.QHBoxLayout()
        row.addWidget(self.colourButton)
        row.addWidget(self.highlightCheck)
        row.addStretch(1)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.close)
        row.addWidget(closeButton)
        layout.addLayout(row)
        
        self.setLayout(layout)
    
    def voroVolLine(self, vol):
        """
        Return line containing Voronoi volume
        
        """
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Voronoi volume: %f" % vol))
        self.mainLayout.addLayout(row)
    
    def highlightChanged(self, state):
        """
        Highlight check changed
        
        """
        if state == QtCore.Qt.Unchecked:
            self.removeHighlighters()
            self.colourButton.setEnabled(False)
        
        else:
            self.addHighlighters()
            self.colourButton.setEnabled(True)
    
    def addHighlighters(self):
        """
        Add highlighters for this defect
        
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
            highlighter = highlight.VacancyHighlighter(refState.atomPos(index), radius * 1.1, rgb=self.highlightColourRGB)
            
            highlighters.append(highlighter)
        
        elif self.defectType == 2:
            interstitials = self.defList[0]
            
            # interstitial
            index = interstitials[self.defectIndex]
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlighter
            highlighter = highlight.AtomHighlighter(inputState.atomPos(index), radius * 1.1, rgb=self.highlightColourRGB)
            
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
            highlighter = highlight.AntisiteHighlighter(refState.atomPos(index), radius, rgb=self.highlightColourRGB)
            highlighters.append(highlighter)
            
            #### highlight occupying atom ####
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[index2]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighter = highlight.AtomHighlighter(inputState.atomPos(index2), radius * 1.1, rgb=self.highlightColourRGB)
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
            highlighter = highlight.VacancyHighlighter(refState.atomPos(vacIndex), radius * 1.1, rgb=self.highlightColourRGB)
            highlighters.append(highlighter)
            
            #### highlight int 1 ####
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[int1Index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighter = highlight.AtomHighlighter(inputState.atomPos(int1Index), radius * 1.1, rgb=self.highlightColourRGB)
            highlighters.append(highlighter)
            
            #### highlight int 2 ####
            
            # radius
            radius = inputState.specieCovalentRadius[inputState.specie[int2Index]] * self.filterList.displayOptions.atomScaleFactor
            
            # highlight
            highlighter = highlight.AtomHighlighter(inputState.atomPos(int2Index), radius * 1.1, rgb=self.highlightColourRGB)
            highlighters.append(highlighter)
        
        self.pipelinePage.broadcastToRenderers("addHighlighters", (self.windowID, highlighters))
    
    def changeHighlighterColour(self):
        """
        Change highlighter colour
        
        """
        col = QtGui.QColorDialog.getColor(initial=self.highlightColour, title="Set highlighter colour")
        
        if col.isValid():
            self.highlightColour = col
            self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.highlightColour.name())
            
            self.highlightColourRGB = [float(self.highlightColour.red()) / 255.0, 
                                       float(self.highlightColour.green()) / 255.0,
                                       float(self.highlightColour.blue()) / 255.0]
            
            # make change
            self.removeHighlighters()
            self.addHighlighters()
    
    def show(self):
        """
        We override show() to first add highlighters to renderer windows
        
        """
        self.addHighlighters()
        
        return super(DefectInfoWindow, self).show()
    
    def removeHighlighters(self):
        """
        Remove highlighters
        
        """
        self.pipelinePage.broadcastToRenderers("removeHighlighters", (self.windowID,))
    
    def closeEvent(self, event):
        """
        Close event
        
        """
        # remove highlighters
        self.removeHighlighters()
        
        event.accept()
