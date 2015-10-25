
"""
Contains GUI forms for the bubbles filter.

"""
import logging

from PySide import QtGui, QtCore

from . import base
from .speciesSettingsDialog import SpeciesListItem
from ...filtering.filters import bubblesFilter
from ...filtering import filterer


################################################################################

class BubblesSettingsDialog(base.GenericSettingsDialog):
    """
    Settings dialog for the bubbles filter.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(BubblesSettingsDialog, self).__init__(title, parent, "Bubbles")
        
        self.logger = logging.getLogger(__name__)
        
        # settings
        self._settings = bubblesFilter.BubblesFilterSettings()
        
        # bubble species
        self.speciesList = QtGui.QListWidget(self)
        self.speciesList.setFixedHeight(80)
        self.speciesList.setFixedWidth(100)
        self.speciesList.itemChanged.connect(self.speciesListChanged)
        self.speciesList.setToolTip("<p>The bubble atom species.<p>")
        self.contentLayout.addRow("Bubble species", self.speciesList)
        
        self.addHorizontalDivider()
        
        # vacancy radius
        self.addDoubleSpinBox("vacancyRadius", minVal=0.01, maxVal=10, step=0.1, label="Vacancy radius",
                              toolTip="<p>The vacancy radius is used when identifying vacancy clusters.</p>")
        
        self.addHorizontalDivider()
        
        # vacancy neighbour radius
        tip = "<p>If two vacancies are within this distance of one another they form part of the same vacancy cluster.</p>"
        self.addDoubleSpinBox("vacNebRad", minVal=0.01, maxVal=20, step=0.1, label="Vacancy neighbour radius",
                              toolTip=tip)
        
        # vacancy bubble radius
        tip = "<p>A bubble atom will be associated with a vacancy if they are separated by less than this value.<p>"
        self.addDoubleSpinBox("vacancyBubbleRadius", minVal=0.01, maxVal=20, step=0.1, label="Vacancy bubble radius",
                              toolTip=tip)
        
        # vac int association radius
        tip = "<p>A vacancy is ignored if it is not occupied by a bubble atom and there is an interstitial wihtin this distance of it.</p>"
        self.addDoubleSpinBox("vacIntRad", minVal=0.01, maxVal=20, step=0.1, label="Vac-int association radius",
                              toolTip=tip)
        
        self.refresh(firstRun=True)
    
    def speciesListChanged(self, *args):
        """Species selection has changed."""
        bubbleSpeciesList = []
        for i in xrange(self.speciesList.count()):
            item = self.speciesList.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                bubbleSpeciesList.append(item.symbol)
        
        self.logger.debug("Changed bubble species: %r", bubbleSpeciesList)
        
        self._settings.updateSetting("bubbleSpecies", bubbleSpeciesList)
    
    def refresh(self, firstRun=False):
        """Refresh the species list."""
        self.logger.debug("Refreshing bubble species options")
        
        refState = self.pipelinePage.refState
        inputState = self.pipelinePage.inputState
        refSpecieList = refState.specieList
        inputSpecieList = inputState.specieList
        
        # set of added species
        currentSpecies = set()
        
        # remove species that don't exist
        num = self.speciesList.count()
        for i in xrange(num - 1, -1, -1):
            item = self.speciesList.item(i)
            
            # remove if doesn't exist both ref and input
            if item.symbol not in inputSpecieList and item.symbol not in refSpecieList:
                self.logger.debug("  Removing species option: %s", item.symbol)
                self.speciesList.takeItem(i) # does this delete it?
            
            else:
                currentSpecies.add(item.symbol)
        
        # unique species from ref/input
        combinedSpecieList = list(inputSpecieList) + list(refSpecieList)
        uniqueCurrentSpecies = set(combinedSpecieList)
        
        # add species that aren't already added
        for sym in uniqueCurrentSpecies:
            if sym in currentSpecies:
                self.logger.debug("  Keeping species option: %s", sym)
            
            else:
                self.logger.debug("  Adding species option: %s", sym)
                item = SpeciesListItem(sym)
                if firstRun:
                	if sym == "He":
                		state = QtCore.Qt.Checked
                	elif sym == "H_":
                		state = QtCore.Qt.Checked
                	elif sym == "Ar":
                		state = QtCore.Qt.Checked
                	else:
                		state = QtCore.Qt.Unchecked
                else:
                	state = QtCore.Qt.Unchecked
                item.setCheckState(state)
                self.speciesList.addItem(item)
        
        # update bubble species list on first run
        if firstRun:
        	self.speciesListChanged()
