
"""
Help form

@author: Chris Scott

"""
import logging
import functools

from PySide import QtGui, QtCore, QtWebKit

from ..visutils.utilities import iconPath
from .. import resources


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
        
        # browser
        self.webView = QtWebKit.QWebView(self)
        
        # toolbar actions
        backAction = QtGui.QAction(QtGui.QIcon(iconPath("go-previous.svg")), "&Back", self)
        backAction.triggered.connect(self.webView.back)
        homeAction = QtGui.QAction(QtGui.QIcon(iconPath("go-home.svg")), "&Home", self)
        homeAction.triggered.connect(functools.partial(self.loadUrl, "qrc:///doc/index.html"))
        forwardAction = QtGui.QAction(QtGui.QIcon(iconPath("go-next.svg")), "&Foward", self)
        forwardAction.triggered.connect(self.webView.forward)
        
        # tool bar
        toolbar = QtGui.QToolBar()
        toolbar.addAction(backAction)
        toolbar.addAction(homeAction)
        toolbar.addAction(forwardAction)
        
        logger = logging.getLogger(__name__)
        self.logger = logger
        logger.debug("Setting up help form")
        
        self.webView.load("qrc:///doc/index.html")
        self.webView.show()
        
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.webView)
        
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
