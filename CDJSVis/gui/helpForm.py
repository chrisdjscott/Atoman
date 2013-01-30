
"""
Help form

@author: Chris Scott

"""
from PyQt4 import QtGui, QtCore


from ..visutils.utilities import iconPath


################################################################################
class HelpForm(QtGui.QDialog):
    """
    The applications help form.
    
    """
    def __init__(self, page, parent=None):
        super(HelpForm, self).__init__(parent)
        
        self.parent = parent
        
        self.setModal(0)
        
#        self.setWindowTitle("Help")
        self.setWindowIcon(QtGui.QIcon(iconPath("Help-icon.png")))
        
        self.open = 0
        
        backAction = QtGui.QAction(QtGui.QIcon(iconPath("go-previous.svg")), "&Back", self)
        backAction.setShortcut(QtGui.QKeySequence.Back)
        homeAction = QtGui.QAction(QtGui.QIcon(iconPath("go-home.svg")), "&Home", self)
        homeAction.setShortcut("Home")
        self.pageLabel = QtGui.QLabel()
        
        toolbar = QtGui.QToolBar()
        toolbar.addAction(backAction)
        toolbar.addAction(homeAction)
        toolbar.addWidget(self.pageLabel)
        
        self.textBrowser = QtGui.QTextBrowser()
        
        layout = QtGui.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.textBrowser, 1)
        self.setLayout(layout)
        
        self.connect(backAction, QtCore.SIGNAL("triggered()"), self.textBrowser, QtCore.SLOT("backward()"))
        self.connect(homeAction, QtCore.SIGNAL("triggered()"), self.textBrowser, QtCore.SLOT("home()"))
        self.connect(self.textBrowser, QtCore.SIGNAL("sourceChanged(QUrl)"), self.updatePageTitle)
        
        self.textBrowser.setSearchPaths([":/help"])
        self.textBrowser.setSource(QtCore.QUrl(page))
        self.resize(500, 600)
        self.setWindowTitle("CDJSVis Help")
        
    def updatePageTitle(self):
        """
        Update the page title.
        
        """
        self.pageLabel.setText(self.textBrowser.documentTitle())
    
    def closeEvent(self, event):
        """
        Close event
        
        """
        self.parent.helpOpen = 0
        self.hide()
