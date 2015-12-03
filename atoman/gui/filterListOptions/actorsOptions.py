
"""
Actors options
--------------

It is possible to set display options on the different groups of actors
that are created, for example the atoms actor or bonds actor.

Work in progress...

"""
import functools
import logging

from PySide import QtGui, QtCore


################################################################################

class ActorsOptionsWindow(QtGui.QDialog):
    """
    Actors options dialog.
    
    """
    def __init__(self, mainWindow, parent=None):
        super(ActorsOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Actors options")
#         self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        self.mainWindow = mainWindow
        
        # logger
        self.logger = logging.getLogger(__name__+".ActorsOptionsWindow")
        
        # defaults
        self.refreshing = False
        self.ambientSpins = []
        self.specularSpins = []
        self.specularPowerSpins = []
        
        # layout
        layout = QtGui.QFormLayout(self)
        self.setLayout(layout)
        
        # draw vectors list widget
        self.tree = QtGui.QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.itemChanged.connect(self.itemChanged)
        self.tree.setHeaderLabels(("Visibility", "Ambient", "Specular", "Specular power"))
        layout.addRow(self.tree)
        
        # button box
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)
    
    def itemChanged(self, item, column):
        """
        Item has changed.
        
        """
        if self.refreshing:
            return
        
        self.logger.debug("Item changed: %r, %r", item, column)
        if column != 0:
            self.logger.debug("Ignoring changed item")
            return
        
        if item.checkState(0) == QtCore.Qt.Unchecked:
            # hide actor
            self.parent.renderer.hideActor(item.text(0))
        
        else:
            # show actor
            self.parent.renderer.addActor(item.text(0))
    
    def addCheckedActors(self):
        """
        Add all actors that are checked (but not already added)
        
        """
        globalChanges = False
        it = QtGui.QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            
            if item.childCount() == 0:
                if item.checkState(0) == QtCore.Qt.Checked:
                    changes = self.parent.filterer.addActor(item.text(0), reinit=False)
                    if changes:
                        globalChanges = True
            
            it += 1
        
        if globalChanges:
            self.parent.filterer.reinitialiseRendererWindows()
    
    def refresh(self, actorsDict):
        """
        Refresh actor visibility options
        
        Should be called whenever the filters are run
        
        """
        self.refreshing = True
        
        try:
            inputState = self.parent.filterTab.inputState
            if inputState is None:
                return
            
            self.logger.debug("Refreshing actor visibility options")
            
            # clear the tree
            self.tree.clear()
            del self.ambientSpins[:]
            del self.specularSpins[:]
            del self.specularPowerSpins[:]
            
            # populate
            for key in sorted(actorsDict.keys()):
                self.addItem(self.tree, key)
        
        finally:
            self.refreshing = False
    
    def addItem(self, parent, name):
        """
        Add item with parent and name
        
        """
        flt = self.parent.renderer
        
        item = QtGui.QTreeWidgetItem(parent)
        item.setText(0, name)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsSelectable)
        item.setCheckState(0, QtCore.Qt.Checked)
        
        spin = QtGui.QDoubleSpinBox()
        minval = 0
        maxval = 1
        spin.setMinimum(minval)
        spin.setMaximum(maxval)
        current = flt.getActorAmbient(name)
        assert current <= maxval and current >= minval
        spin.setValue(current)
        spin.setSingleStep(0.1)
        spin.valueChanged.connect(functools.partial(self.ambientSpinChanged, name))
        self.ambientSpins.append(spin)
        self.tree.setItemWidget(item, 1, spin)
        
        spin = QtGui.QDoubleSpinBox()
        minval = 0
        maxval = 1
        spin.setMinimum(minval)
        spin.setMaximum(maxval)
        current = flt.getActorSpecular(name)
        assert current <= maxval and current >= minval
        spin.setValue(current)
        spin.setSingleStep(0.1)
        spin.valueChanged.connect(functools.partial(self.specularSpinChanged, name))
        self.specularSpins.append(spin)
        self.tree.setItemWidget(item, 2, spin)
        
        spin = QtGui.QDoubleSpinBox()
        minval = 0
        maxval = 1000
        spin.setMinimum(minval)
        spin.setMaximum(maxval)
        current = flt.getActorSpecularPower(name)
        assert current <= maxval and current >= minval
        spin.setValue(current)
        spin.setSingleStep(1)
        spin.valueChanged.connect(functools.partial(self.specularPowerSpinChanged, name))
        self.specularPowerSpins.append(spin)
        self.tree.setItemWidget(item, 3, spin)
    
    def ambientSpinChanged(self, name, val):
        """
        Ambient spin changed
        
        """
        self.parent.renderer.setActorAmbient(name, val)
    
    def specularSpinChanged(self, name, val):
        """
        Specular spin changed
        
        """
        self.parent.renderer.setActorSpecular(name, val)
    
    def specularPowerSpinChanged(self, name, val):
        """
        Specular power spin changed
        
        """
        self.parent.renderer.setActorSpecularPower(name, val)
