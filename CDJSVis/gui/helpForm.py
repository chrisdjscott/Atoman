
"""
Help form

@author: Chris Scott

"""
import logging

from PySide import QtGui, QtCore, QtWebKit

from ..visutils.utilities import iconPath
from .. import resources


################################################################################
class HelpForm(QtGui.QDialog):
    """
    The applications help form.
    
    """
    def __init__(self, page, parent=None):
        super(HelpForm, self).__init__(parent)
        
        self.parent = parent
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
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

################################################################################
class HelpFormSphinx(QtGui.QDialog):
    """
    The applications help form.
    
    """
    def __init__(self, parent=None):
        super(HelpFormSphinx, self).__init__(parent)
        
        self.parent = parent
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.setModal(0)
        
        self.setWindowTitle("CDJSVis Help")
        self.setWindowIcon(QtGui.QIcon(iconPath("Help-icon.png")))
        
        self.helpFormOpen = False
        
        self.webView = QtWebKit.QWebView(self)
        
        logger = logging.getLogger(__name__)
        self.logger = logger
        logger.debug("Setting up help form")
        
        self.webView.load("qrc:///doc/index.html")
        self.webView.show()
        
        layout = QtGui.QVBoxLayout(self)
        row = QtGui.QHBoxLayout()
        row.addWidget(self.webView)
        layout.addLayout(row)
        
        self.resize(900, 700)
    
    def loadUrl(self, url):
        """
        Load given url
        
        """
        self.logger.debug("Loading URL: '%s'", url)
        self.webView.load(url)
    
    def loadPage(self, page):
        """
        Load given page
        
        """
        url = "qrc:///doc/%s" % page
        self.loadUrl(url)
    
    def show(self):
        """
        Show window
        
        """
        if self.helpFormOpen:
            self.logger.debug("Raising helpWindow")
            self.raise_()
        
        else:
            self.logger.debug("Showing helpWindow")
            super(HelpFormSphinx, self).show()
            self.helpFormOpen = True
    
    def closeEvent(self, event):
        """
        Close event
        
        """
        self.logger.debug("HelpWindow close event")
        self.helpFormOpen = False
        self.hide()
