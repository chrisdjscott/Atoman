
"""
Generic form class.

@author: Marc Robinson
Modified by Chris Scott

"""

import sys

from PyQt4 import QtGui, QtCore


################################################################################
class GenericForm(QtGui.QWidget):
    def __init__(self, parent,width,title):
        super(GenericForm, self).__init__(parent)
        
        self.width = width
        self.parent = parent
        self.title = title
        #self.setFixedWidth(self.width-10)
        
        self.FormLayout = QtGui.QVBoxLayout()
        self.FormLayout.setSpacing(0)
        self.FormLayout.setContentsMargins(0,0,0,0)
        self.FormLayout.setAlignment(QtCore.Qt.AlignTop)
        
        self.Group = QtGui.QGroupBox(self.title)
        self.Group.setAlignment(QtCore.Qt.AlignCenter)
        
        self.ContentLayout = QtGui.QVBoxLayout()
        self.ContentLayout.setSpacing(0)
        self.ContentLayout.setContentsMargins(0,0,0,0)
        self.ContentLayout.setAlignment(QtCore.Qt.AlignTop)
        
        self.Group.setLayout(self.ContentLayout)
        
        self.FormLayout.addWidget(self.Group)
        
        self.setLayout(self.FormLayout)
        
        self.show()
        self.hide()
        
    def newRow(self,align=None):
        row = FormRow(align=align)
        self.ContentLayout.addWidget(row)
        return row
    
    def removeRow(self,row):
        self.ContentLayout.removeWidget(row)    
    
    def removeAllRows(self):
        for i in xrange(0,self.ContentLayout.count()):
            temp = self.ContentLayout.removeItem(self.ContentLayout.itemAt(0))
            del temp    


################################################################################
class FormRow(QtGui.QWidget):
    def __init__(self,align=None):        
        super(FormRow, self).__init__()
        
        self.RowLayout = QtGui.QHBoxLayout(self)
        self.RowLayout.setAlignment(QtCore.Qt.AlignCenter)
        
        if(align=="Right"):
            self.RowLayout.setAlignment(QtCore.Qt.AlignRight)
        elif(align=="Left"):
            self.RowLayout.setAlignment(QtCore.Qt.AlignLeft)
        elif(align=="Center"):
            self.RowLayout.setAlignment(QtCore.Qt.AlignCenter)    
    
    def align(self,align):
        
        if(align=="Right"):
            self.RowLayout.setAlignment(QtCore.Qt.AlignRight)
        elif(align=="Left"):
            self.RowLayout.setAlignment(QtCore.Qt.AlignLeft)
        elif(align=="Center"):
            self.RowLayout.setAlignment(QtCore.Qt.AlignCenter) 
           
    def addWidget(self,widget):
        
        self.RowLayout.addWidget(widget)
        self.RowLayout.setSpacing(0)
        self.RowLayout.setContentsMargins(0,0,0,0)
    
    def removeWidget(self,widget):    
        
        self.RowLayout.removeWidget(widget)
