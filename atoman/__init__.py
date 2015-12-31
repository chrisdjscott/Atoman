from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
print("Atoman loaded from: %s" % os.path.dirname(__file__))

from .visutils import version
__version__ = version.getVersion()
