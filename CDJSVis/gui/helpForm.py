
"""
Help form

@author: Chris Scott

"""
from PySide import QtGui, QtCore, QtWebKit


from ..visutils.utilities import iconPath


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
        
        self.open = False
        
        self.webView = QtWebKit.QWebView(self)
        
        self.webView.load("/Volumes/Users_HD/Users/macdjs/git/CDJSVis/doc/_build/html/index.html")
        self.webView.show()
        
        layout = QtGui.QVBoxLayout(self)
        row = QtGui.QHBoxLayout()
        row.addWidget(self.webView)
        layout.addLayout(row)
        
        # create splitter
#         self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
#         
#         # create helper and set it up
#         self.helpEngine = QtHelp.QHelpEngine("/Volumes/Users_HD/Users/macdjs/git/CDJSVis/doc/_build/qthelp/CDJSVis.qhc", self)
#         self.helpEngine.setupData()
#         
#         # get the content widget and browser
#         self.contentWidget = self.helpEngine.contentWidget()
#         self.helpBrowser = HelpBrowser(self.splitter, self.helpEngine)
#         
#         # build widget
#         self.splitter.insertWidget(0, self.contentWidget) 
#         self.splitter.insertWidget(1, self.helpBrowser) 
#         self.splitter.setStretchFactor(1, 1) 
#         
#         layout = QtGui.QVBoxLayout(self)
#         row = QtGui.QHBoxLayout()
#         row.addWidget(self.splitter)
#         layout.addLayout(row)
#         
#         # connect the TOC to the help browser
#         self.helpEngine.contentWidget().linkActivated.connect(self.helpBrowser.setSource)
        
        self.resize(900, 700)
    
    def closeEvent(self, event):
        """
        Close event
        
        """
        self.parent.helpOpen = 0
        self.hide()

################################################################################

class HelpBrowser(QtGui.QTextBrowser):
    """
    Help browser
    
    """
    def __init__(self, parent, helpEngine):
        super(HelpBrowser, self).__init__(parent)
        
        self.helpEngine = helpEngine
    
    def loadResource(self, loadType, url):
        """
        Load resource
        
        """
        if url.scheme() != "qthelp":
            return super(HelpBrowser, self).loadResource(loadType, url)
        
        return self.helpEngine.fileData(url)

