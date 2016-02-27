
from __future__ import print_function
from __future__ import absolute_import

import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, os.pardir))
    
    # config
    config = Configuration("rendering", parent_package, top_path)
    
    arraydeps = [os.path.join("..", "visclibs", "array_utils.c"),
                 os.path.join("..", "visclibs", "array_utils.h")]
    
    config.add_extension("_rendering",
                         ["rendering.c"],
                         include_dirs=[incdir],
                         depends=arraydeps,
                         libraries=["array_utils"])
    
    config.add_subpackage("renderers")
    
    return config

if __name__ == "__main__":
    print("This is the wrong setup.py to run")
