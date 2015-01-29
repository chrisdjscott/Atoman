
"""
The main toolbar

@author: Chris Scott

"""
import logging

from PySide import QtGui, QtCore

from .genericForm import GenericForm
from .pipelineForm import PipelineForm
from ..visutils.utilities import iconPath


################################################################################
class MainToolbar(QtGui.QDockWidget):
    def __init__(self, parent, width, height):
        super(MainToolbar, self).__init__(parent)
        
        self.mainWindow = parent
        
        self.setWindowTitle("Toolbar")
        
        self.setFeatures(self.DockWidgetMovable | self.DockWidgetFloatable)
        self.logger = logging.getLogger(__name__)
        
        # set size
        self.toolbarWidth = width
        self.toolbarHeight = height
        self.setFixedWidth(self.toolbarWidth)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        
        self.NPipelines = 0
        
        # create container for widgets
        self.container = QtGui.QWidget(self)
        containerLayout = QtGui.QVBoxLayout(self.container)
        containerLayout.setSpacing(0)
        containerLayout.setContentsMargins(0,0,0,0)
        containerLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # tab widget
        self.tabWidget = QtGui.QTabWidget(self)
        self.tabWidget.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        containerLayout.addWidget(self.tabWidget)
        
        # input form
        self.tabWidget.addTab(self.mainWindow.systemsDialog, "Input")
        
        # analysis pipelines form
        self.currentPipelineString = "Pipeline 0"
        self.currentPipelineIndex = 0
        self.analysisPipelinesForm = GenericForm(self, 0, "")
        
        row = self.analysisPipelinesForm.newRow()
        
        self.pipelineCombo = QtGui.QComboBox()
        self.pipelineCombo.currentIndexChanged.connect(self.currentPipelineChanged)
        row.addWidget(self.pipelineCombo)
        
        addPipelineButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/list-add.png")), "")
        addPipelineButton.setStatusTip("Add analysis pipeline")
        addPipelineButton.setToolTip("Add analysis pipeline")
        addPipelineButton.clicked.connect(self.addPipeline)
        row.addWidget(addPipelineButton)
        
        removePipelineButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/list-remove.png")), "")
        removePipelineButton.setStatusTip("Remove analysis pipeline")
        removePipelineButton.setToolTip("Remove analysis pipeline")
        removePipelineButton.clicked.connect(self.removePipeline)
        row.addWidget(removePipelineButton)
        
        applyAllButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/view-refresh.png")), "")
        applyAllButton.setStatusTip("Run all pipelines")
        applyAllButton.setToolTip("Run all pipelines")
        applyAllButton.clicked.connect(self.runAllPipelines)
        row.addWidget(applyAllButton)
        
        # divider
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        row = self.analysisPipelinesForm.newRow()
        row.addWidget(line)
        
        # stacked widget (for pipelines)
        self.stackedWidget = QtGui.QStackedWidget()
        row = self.analysisPipelinesForm.newRow()
        row.addWidget(self.stackedWidget)
        
        # list for holding pipeline
        self.pipelineList = []
        
        # add first pipeline
        self.addPipeline()
        
        # add to layout
        self.tabWidget.addTab(self.analysisPipelinesForm, "Analysis pipelines")
        self.tabWidget.setTabEnabled(1, False)
        
        # set the main widget
        self.setWidget(self.container)
    
    def changeStateDisplayName(self, index, displayName):
        """
        Change display name for a state.
        
        """
        for p in self.pipelineList:
            p.changeStateDisplayName(index, displayName)
    
    def addStateOptionToPipelines(self, filename):
        """
        Add state option to pipeline combos
        
        """
        for p in self.pipelineList:
            p.addStateOption(filename)
    
    def removeStateFromPipelines(self, index):
        """
        Remove selected state from pipelines
        
        """
        for p in self.pipelineList:
            p.removeStateOption(index)
    
    def getSelectedStatesFromPipelines(self):
        """
        Return set of currently selected states
        
        """
        currentStates = set()
        for p in self.pipelineList:
            refIndex, inputIndex = p.getCurrentStateIndexes()
            
            currentStates.add(refIndex)
            currentStates.add(inputIndex)
        
        return currentStates
    
    def runAllPipelines(self):
        """
        Run all pipelines.
        
        """
        iniIndex = self.currentPipelineIndex
        
        try:
            for count, p in enumerate(self.pipelineList):
                self.pipelineCombo.setCurrentIndex(count)
                status = p.runAllFilterLists()
                if status:
                    break
        
        finally:
            self.pipelineCombo.setCurrentIndex(iniIndex)
    
    def addPipeline(self):
        """
        Add a new analysis pipeline
        
        """
        self.logger.debug("Adding new pipeline form: %d", self.NPipelines)
        
        # add to pipeline combos
        name = "Pipeline %d" % self.NPipelines
        self.pipelineCombo.addItem(name)
        for rw in self.mainWindow.rendererWindows:
            rw.newPipeline(name)
        
        # form
        form = PipelineForm(self, self.mainWindow, self.toolbarWidth, self.NPipelines, name)
        
        self.pipelineList.append(form)
        self.stackedWidget.addWidget(form)
        
        self.pipelineCombo.setCurrentIndex(len(self.pipelineList) - 1)
        
        self.NPipelines += 1
    
    def removePipeline(self):
        """
        Remove pipeline.
        
        """
        remIndex = self.currentPipelineIndex
        
        # not allowed to remove last one
        if len(self.pipelineList) == 1:
            self.mainWindow.displayWarning("Cannot remove last pipeline")
            return
        
        # clear all actors from any windows and select diff pipeline for them (or close them?)
        pipeline = self.pipelineList[remIndex]
        
        # remove actors
        for filterList in pipeline.filterLists:
            filterList.filterer.removeActors()
        
        # remove pipeline
        form = self.pipelineList.pop(remIndex)
        
        # remove from stacked widget
        self.stackedWidget.removeWidget(form)
        
        # remove from combo boxes
        self.pipelineCombo.removeItem(remIndex)
        
        for rw in self.mainWindow.rendererWindows:
            rw.removePipeline(remIndex)
        
        # update current pipeline indexes and strings
        self.currentPipelineString = str(self.pipelineCombo.currentText())
        self.currentPipelineIndex = self.pipelineCombo.currentIndex()
        
    
    def currentPipelineChanged(self, index):
        """
        Current pipeline changed.
        
        """
        # Update stacked widget
        self.stackedWidget.setCurrentIndex(index)
        
        # update variable
        self.currentPipelineString = str(self.pipelineCombo.currentText())
        self.currentPipelineIndex = index
        
        



