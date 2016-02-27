
from __future__ import print_function
from __future__ import absolute_import

import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, os.pardir))
    
    # config
    config = Configuration("plotting", parent_package, top_path)
    
    boxesdeps = [os.path.join("..", "visclibs", "boxeslib.c"),
                 os.path.join("..", "visclibs", "boxeslib.h")]
    utildeps = [os.path.join("..", "visclibs", "utilities.c"),
                os.path.join("..", "visclibs", "utilities.h")]
    arraydeps = [os.path.join("..", "visclibs", "array_utils.c"),
                 os.path.join("..", "visclibs", "array_utils.h")]
    
    config.add_extension("_rdf",
                         ["rdf.c"],
                         include_dirs=[incdir],
                         depends=boxesdeps + utildeps + arraydeps,
                         libraries=["boxeslib", "array_utils", "utilities"])
    
    return config

if __name__ == "__main__":
    print("This is the wrong setup.py to run")
