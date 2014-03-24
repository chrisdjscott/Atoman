
"""
Threading module

@author: Chris Scott

"""
import functools

from PySide import QtCore


################################################################################

class GenericRunnable(QtCore.QRunnable):
    """
    Creates runnable for running object on thread pool.
    
    """
    def __init__(self, worker, method="run", args=(), kwargs={}):
        super(GenericRunnable, self).__init__()
        
        self.worker = worker
        self.method = method
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        """
        Run the thread.
        
        """
        workerMethod = getattr(self.worker, self.method, None)
        
        if workerMethod is not None:
            workerMethod(*self.args, **self.kwargs)

################################################################################

class GenericThreader(object):
    """
    GenericThreader threads the given method on the object.
    
    """
    def __init__(self, parent, log=None):
        self.parent = parent
        self.log = log
    
    def thread(self, worker, startMethod, startargs, startkwargs, finishedSlot):
        """
        Create and connect the thread (but doesn't start it).
        
        """
        # worker thread
        workerThread = QtCore.QThread(self.parent)
        
        # connect signals
        workerThread.started.connect(functools.partial(startMethod, *startargs, **startkwargs))
        worker.finished.connect(finishedSlot)
        if self.log is not None:
            worker.logMessage.connect(self.log)
        
        worker.finishedThread.connect(workerThread.quit)
        worker.finishedThread.connect(worker.deleteLater)
        worker.destroyed.connect(workerThread.deleteLater)
        
        # move reader to thread
        worker.moveToThread(workerThread)
        
        return workerThread
