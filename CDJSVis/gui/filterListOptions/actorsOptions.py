
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
        
        #TODO: if child is unchecked, parent should be too
        #TODO: when all children are checked, parent should be checked too
        
        print "ITEM CHANGED", item, column
        if column != 0:
            print "  IGNORING"
            return
        
        if item.checkState(0) == QtCore.Qt.Unchecked:
            if item.childCount():
                # uncheck all children that are checked
                for i in xrange(item.childCount()):
                    child = item.child(i)
                    if child.checkState(0) == QtCore.Qt.Checked:
                        child.setCheckState(0, QtCore.Qt.Unchecked)
            
            else:
                # hide actor
                parentName = None
                parent = item.parent()
                if parent is not None:
                    parentName = parent.text(0)
                self.parent.filterer.hideActor(item.text(0), parentName=parentName)
                
                # also uncheck parent
                if parent is not None and parent.checkState(0) == QtCore.Qt.Checked:
                    self.refreshing = True
                    parent.setCheckState(0, QtCore.Qt.Unchecked)
                    self.refreshing = False
        
        else:
            if item.childCount():
                # check all children that aren't checked
                for i in xrange(item.childCount()):
                    child = item.child(i)
                    if child.checkState(0) == QtCore.Qt.Unchecked:
                        child.setCheckState(0, QtCore.Qt.Checked)
            
            else:
                # show actor
                parentName = None
                parent = item.parent()
                if parent is not None:
                    parentName = parent.text(0)
                self.parent.filterer.addActor(item.text(0), parentName=parentName)
                
                # if all parents children are checked, make sure parent is too
                if parent is not None and parent.checkState(0) == QtCore.Qt.Unchecked:
                    # count children
                    allChecked = True
                    for i in xrange(parent.childCount()):
                        child = parent.child(i)
                        if child.checkState(0) == QtCore.Qt.Unchecked:
                            allChecked = False
                            break
                    
                    if allChecked:
                        self.refreshing = True
                        parent.setCheckState(0, QtCore.Qt.Checked)
                        self.refreshing = False
    
    def addCheckedActors(self):
        """
        Add all actors that are checked (but not already added)
        
        """
        it = QtGui.QTreeWidgetItemIterator(self.tree)
        
        globalChanges = False
        while it.value():
            item = it.value()
            
            if item.childCount() == 0:
                if item.checkState(0) == QtCore.Qt.Checked:
                    
                    parent = item.parent()
                    parentName = None
                    if parent is not None:
                        parentName = parent.text(0)
                    
                    changes = self.parent.filterer.addActor(item.text(0), parentName=parentName, reinit=False)
                    
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
                val = actorsDict[key]
                
                if isinstance(val, dict):
                    parent = QtGui.QTreeWidgetItem(self.tree)
                    parent.setText(0, key)
                    parent.setFlags(parent.flags() | QtCore.Qt.ItemIsUserCheckable)
                    parent.setFlags(parent.flags() & ~QtCore.Qt.ItemIsSelectable)
                    parent.setCheckState(0, QtCore.Qt.Checked)
                    
                    for actorName in sorted(val.keys()):
                        self.addItem(parent, key, actorName)
                
                else:
                    self.addItem(self.tree, None, key)
        
        finally:
            self.refreshing = False
    
    def addItem(self, parent, parentName, name):
        """
        Add item with parent and name
        
        """
        flt = self.parent.filterer
        
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
        current = flt.getActorAmbient(name, parentName)
        assert current <= maxval and current >= minval
        spin.setValue(current)
        spin.setSingleStep(0.1)
        spin.valueChanged.connect(functools.partial(self.ambientSpinChanged, name, parentName))
        self.ambientSpins.append(spin)
        self.tree.setItemWidget(item, 1, spin)
        
        spin = QtGui.QDoubleSpinBox()
        minval = 0
        maxval = 1
        spin.setMinimum(minval)
        spin.setMaximum(maxval)
        current = flt.getActorSpecular(name, parentName)
        assert current <= maxval and current >= minval
        spin.setValue(current)
        spin.setSingleStep(0.1)
        spin.valueChanged.connect(functools.partial(self.specularSpinChanged, name, parentName))
        self.specularSpins.append(spin)
        self.tree.setItemWidget(item, 2, spin)
        
        spin = QtGui.QDoubleSpinBox()
        minval = 0
        maxval = 1000
        spin.setMinimum(minval)
        spin.setMaximum(maxval)
        current = flt.getActorSpecularPower(name, parentName)
        assert current <= maxval and current >= minval
        spin.setValue(current)
        spin.setSingleStep(1)
        spin.valueChanged.connect(functools.partial(self.specularPowerSpinChanged, name, parentName))
        self.specularPowerSpins.append(spin)
        self.tree.setItemWidget(item, 3, spin)
        
    
    def ambientSpinChanged(self, name, parentName, val):
        """
        Ambient spin changed
        
        """
        self.parent.filterer.setActorAmbient(name, parentName, val)
    
    def specularSpinChanged(self, name, parentName, val):
        """
        Specular spin changed
        
        """
        self.parent.filterer.setActorSpecular(name, parentName, val)
    
    def specularPowerSpinChanged(self, name, parentName, val):
        """
        Specular power spin changed
        
        """
        self.parent.filterer.setActorSpecularPower(name, parentName, val)
