
import os
print "CDJSVis loaded from: %s" % os.path.dirname(__file__)

from .visutils import version
__version__ = version.getVersion()
