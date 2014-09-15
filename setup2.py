
import numpy as np
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

from CDJSVis.visutils import version


# ext_acna = Extension("CDJSVis.visclibs.acna", ["CDJSVis/visclibs/acna.c", "CDJSVis/visclibs/neb_list.c", 
#                                                "CDJSVis/visclibs/utilities.c", "CDJSVis/visclibs/boxeslib.c"])

def configuration(parent_package='', top_path=None):
    # configuration object
    config = Configuration(package_name='CDJSVis', parent_name=parent_package, top_path=top_path, 
                           author="Chris Scott",
                           author_email="chris@chrisdjscott.co.uk",
                           version=version.getVersion())
    
    config.add_scripts(["CDJSVis.py"])
    
    config.add_subpackage("filtering", "CDJSVis/filtering")
    config.add_subpackage("gui", "CDJSVis/gui")
    config.add_subpackage("lattice_gen", "CDJSVis/lattice_gen")
    config.add_subpackage("rendering", "CDJSVis/rendering")
    config.add_subpackage("state", "CDJSVis/state")
    config.add_subpackage("visclibs", "CDJSVis/visclibs")
    config.add_subpackage("visutils", "CDJSVis/visutils")
    
#     config.add_subpackage("CDJSVis.filtering", "CDJSVis/filtering")
#     config.add_subpackage("CDJSVis.gui", "CDJSVis/gui")
#     config.add_subpackage("CDJSVis.lattice_gen", "CDJSVis/lattice_gen")
#     config.add_subpackage("CDJSVis.rendering", "CDJSVis/rendering")
#     config.add_subpackage("CDJSVis.state", "CDJSVis/state")
#     config.add_subpackage("CDJSVis.visclibs", "CDJSVis/visclibs")
#     config.add_subpackage("CDJSVis.visutils", "CDJSVis/visutils")
    
    config.add_extension("CDJSVis.visclibs.acna", 
                         ["CDJSVis/visclibs/acna.c", "CDJSVis/visclibs/utilities.c",
                          "CDJSVis/visclibs/boxeslib.c", "CDJSVis/visclibs/neb_list.c",
                        "CDJSVis/visclibs/array_utils.c"],
                         libraries=["-lgsl", "-lgslcblas"])
#                          include_dirs=[np.get_include()])
    
    
#     config.set_options(quiet=True)
    
    return config

if __name__ == "__main__":
    setup(**configuration().todict())


# from setuptools import setup, find_packages
# from setuptools.extension import Extension
# 
# import numpy as np
# 
# from CDJSVis.visutils import version
# 
# 
# 
# ext_acna = Extension("CDJSVis.visclibs.acna", ["CDJSVis/visclibs/acna.c", "CDJSVis/visclibs/neb_list.c", 
#                                                "CDJSVis/visclibs/utilities.c", "CDJSVis/visclibs/boxeslib.c"])
# 
# 
# pkgs = [
#     'CDJSVis',
# #     'CDJSVis.filtering',
# #     'CDJSVis.gui',
# #     'CDJSVis.lattice_gen',
# # #     'CDJSVis.md',
# #     'CDJSVis.rendering',
# #     'CDJSVis.state',
# # #     'CDJSVis.tests',
#     'CDJSVis.visclibs',
# #     'CDJSVis.visutils',
# ]
# 
# setup(name="CDJSVis",
#       version=version.getVersion(),
#       author="Chris Scott",
#       author_email="chris@chrisdjscott.co.uk",
#       url="http://chrisdjscott.com",
#       description="CDJSVis Atomistic Visualisation and Analysis Library",
#       include_dirs=[np.get_include()],
#       packages=[pkgs],
#       ext_modules=[ext_acna],
# )
