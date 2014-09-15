
from setuptools import setup, find_packages
from setuptools.extension import Extension

import numpy as np

from CDJSVis.visutils import version



ext_acna = Extension("CDJSVis.visclibs.acna", ["CDJSVis/visclibs/acna.c", "CDJSVis/visclibs/neb_list.c", 
                                               "CDJSVis/visclibs/utilities.c", "CDJSVis/visclibs/boxeslib.c"])


pkgs = [
    'CDJSVis',
#     'CDJSVis.filtering',
#     'CDJSVis.gui',
#     'CDJSVis.lattice_gen',
# #     'CDJSVis.md',
#     'CDJSVis.rendering',
#     'CDJSVis.state',
# #     'CDJSVis.tests',
    'CDJSVis.visclibs',
#     'CDJSVis.visutils',
]

setup(name="CDJSVis",
      version=version.getVersion(),
      author="Chris Scott",
      author_email="chris@chrisdjscott.co.uk",
      url="http://chrisdjscott.com",
      description="CDJSVis Atomistic Visualisation and Analysis Library",
      include_dirs=[np.get_include()],
      packages=[pkgs],
      ext_modules=[ext_acna],
)
