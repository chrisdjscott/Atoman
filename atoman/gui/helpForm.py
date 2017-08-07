
"""
Help form

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import functools

from PySide2 import QtGui, QtCore, QtWidgets #, QtWebEngineWidgets


from ..visutils.utilities import iconPath, helpPath


################################################################################

class HelpFormSphinx(QtWidgets.QDialog):
    """
    The applications help form.
    
    """
    def __init__(self, parent=None):
        super(HelpFormSphinx, self).__init__(parent)
        
        self.parent = parent
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.setModal(0)
        
        self.setWindowTitle("Atoman Help")
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/help-browser.png")))
        
        self.helpFormOpen = False
        
        # browser
#        self.webView = QtWebEngineWidgets.QWebEngineView(self)
        self.webView = None
        
        # toolbar actions
        backAction = QtWidgets.QAction(QtGui.QIcon(iconPath("oxygen/go-previous.png")), "&Back", self)
#        backAction.triggered.connect(self.webView.back)
        homeAction = QtWidgets.QAction(QtGui.QIcon(iconPath("oxygen/go-home.png")), "&Home", self)
        homeAction.triggered.connect(functools.partial(self.loadPage, "index.html"))
        forwardAction = QtWidgets.QAction(QtGui.QIcon(iconPath("oxygen/go-next.png")), "&Foward", self)
#        forwardAction.triggered.connect(self.webView.forward)
        
        # tool bar
        toolbar = QtWidgets.QToolBar()
        toolbar.setFixedHeight(50)
        toolbar.addAction(backAction)
        toolbar.addAction(homeAction)
        toolbar.addAction(forwardAction)
        
        logger = logging.getLogger(__name__)
        self.logger = logger
        logger.debug("Setting up help form")
        
        self.loadPage("index.html")
#        self.webView.show()
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
#        layout.addWidget(self.webView)
        
        self.resize(950, 700)
    
    def loadUrl(self, url):
        """
        Load given url
        
        """
        self.logger.debug("Loading URL: '%s'", url)
#        self.webView.load(QtCore.QUrl(url))
    
    def loadPage(self, page):
        """
        Load given page
        
        """
        url = helpPath(page)
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
