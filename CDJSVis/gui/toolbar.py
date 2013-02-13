
"""
The main toolbar

@author: Chris Scott

"""
import sys

from PyQt4 import QtGui, QtCore

from .genericForm import GenericForm
from .filterTab import FilterTab
from ..visutils.utilities import iconPath
try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)



################################################################################
class MainToolbar(QtGui.QDockWidget):
    def __init__(self, parent, width, height):
        super(MainToolbar, self).__init__(parent)
        
        self.mainWindow = parent
        
        self.setWindowTitle("Main Toolbar")
        
        self.setFeatures(self.DockWidgetMovable | self.DockWidgetFloatable)
        
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
        
        self.currentPipelineString = "Pipeline 0"
        self.currentPipelineIndex = 0
        
        # display current file info
        self.currentFileBox = GenericForm(self, self.toolbarWidth, "Current file")
        self.currentFileBox.show()
        
        row = self.currentFileBox.newRow()
        self.currentRefLabel = QtGui.QLabel("Reference: " + str(self.mainWindow.refFile))
        row.addWidget(self.currentRefLabel)
        
        row = self.currentFileBox.newRow()
        self.currentInputLabel = QtGui.QLabel("Input: " + str(self.mainWindow.inputFile))
        row.addWidget(self.currentInputLabel)
        
        containerLayout.addWidget(self.currentFileBox)
        
        # load input form
        self.loadInputForm = GenericForm(self, 0, "Load input")
        self.loadInputForm.show()
        
        loadInputButton = QtGui.QPushButton(QtGui.QIcon(iconPath("document-open.svg")), "Open input")
        loadInputButton.clicked.connect(self.mainWindow.showLoadInputDialog)
        
        row = self.loadInputForm.newRow()
        row.addWidget(loadInputButton)
        
        containerLayout.addWidget(self.loadInputForm)
        
        # analysis pipelines form
        self.analysisPipelinesForm = GenericForm(self, 0, "Analysis pipelines")
        
        row = self.analysisPipelinesForm.newRow()
        
        self.pipelineCombo = QtGui.QComboBox()
        self.pipelineCombo.currentIndexChanged.connect(self.currentPipelineChanged)
        row.addWidget(self.pipelineCombo)
        
        addPipelineButton = QtGui.QPushButton(QtGui.QIcon(iconPath("list-add.svg")), "")
        addPipelineButton.setStatusTip("Add analysis pipeline")
        addPipelineButton.clicked.connect(self.addPipeline)
        row.addWidget(addPipelineButton)
        
        removePipelineButton = QtGui.QPushButton(QtGui.QIcon(iconPath("list-remove.svg")), "")
        removePipelineButton.setStatusTip("Remove analysis pipeline")
        removePipelineButton.clicked.connect(self.removePipeline)
        row.addWidget(removePipelineButton)
        
        # stacked widget (for pipelines)
        self.stackedWidget = QtGui.QStackedWidget()
        row = self.analysisPipelinesForm.newRow()
        row.addWidget(self.stackedWidget)
        
        # list for holding pipeline
        self.pipelineList = []
        
        # add first pipeline
        self.addPipeline()
        
        # add to layout
        containerLayout.addWidget(self.analysisPipelinesForm)
        
        # set the main widget
        self.setWidget(self.container)
    
    def addPipeline(self):
        """
        Add a new analysis pipeline
        
        """
        # add to pipeline combos
        name = "Pipeline %d" % self.NPipelines
        self.pipelineCombo.addItem(name)
        for rw in self.mainWindow.rendererWindows:
            rw.newPipeline(name)
        
        # form
        form = FilterTab(self, self.mainWindow, self.toolbarWidth, self.NPipelines, name)
        
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
        
        



