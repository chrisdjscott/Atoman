
"""
Contains GUI forms for the species filter.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from PyQt5 import QtCore, QtWidgets


from . import base
from ...system.atoms import elements
from ...filtering.filters import speciesFilter
from six.moves import range


################################################################################

class SpeciesListItem(QtWidgets.QListWidgetItem):
    """
    Item in a species list widget.
    
    """
    def __init__(self, symbol, name=None):
        super(SpeciesListItem, self).__init__()
        
        # add check box
        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable)
        
        # don't allow it to be selected
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsSelectable)
        
        # set unchecked initially
        self.setCheckState(QtCore.Qt.Checked)
        
        # store bond pair
        self.symbol = symbol
        
        # set text
        if name is not None:
            self.setText("%s - %s" % (symbol, name))
        
        else:
            self.setText("%s" % symbol)

################################################################################

class SpeciesSettingsDialog(base.GenericSettingsDialog):
    """
    Species filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(SpeciesSettingsDialog, self).__init__(title, parent, "Species")
        
        # settings
        self._settings = speciesFilter.SpeciesFilterSettings()
        
        # species list
        self.specieList = QtWidgets.QListWidget(self)
#         self.specieList.setFixedHeight(100)
        self.specieList.setFixedWidth(200)
        self.specieList.itemChanged.connect(self.speciesListChanged)
        self.contentLayout.addRow(self.specieList)
        
        self.refresh()
    
    def speciesListChanged(self, *args):
        """Species selection has changed."""
        visibleSpeciesList = []
        for i in range(self.specieList.count()):
            item = self.specieList.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                visibleSpeciesList.append(item.symbol)
        
        self.logger.debug("Species selection has changed: %r", visibleSpeciesList)
        self._settings.updateSetting("visibleSpeciesList", visibleSpeciesList)
    
    def refresh(self):
        """
        Refresh the specie list
        
        """
        self.logger.debug("Refreshing species filter options")
        
        inputSpecieList = self.pipelinePage.inputState.specieList
        refSpecieList = self.pipelinePage.refState.specieList
        
        # set of added species
        currentSpecies = set()
        
        # remove species that don't exist
        num = self.specieList.count()
        for i in range(num - 1, -1, -1):
            item = self.specieList.item(i)
            
            # remove if doesn't exist in both ref and input
            if item.symbol not in inputSpecieList and item.symbol not in refSpecieList:
                self.logger.debug("  Removing species option: %s", item.symbol)
                self.specieList.takeItem(i) # does this delete it?
            
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
                name = elements.atomName(sym)
                item = SpeciesListItem(sym, name=name)
                self.specieList.addItem(item)
        
        # update visible species list
        self.speciesListChanged()
