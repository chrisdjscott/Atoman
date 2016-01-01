
from __future__ import print_function
from __future__ import absolute_import

import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, os.pardir))
    
    boxesdeps = [os.path.join("..", "visclibs", "boxeslib.c"),
                 os.path.join("..", "visclibs", "boxeslib.h")]
    utildeps = [os.path.join("..", "visclibs", "utilities.c"),
                os.path.join("..", "visclibs", "utilities.h")]
    arraydeps = [os.path.join("..", "visclibs", "array_utils.c"),
                 os.path.join("..", "visclibs", "array_utils.h")]
    
    # config
    config = Configuration("filtering", parent_package, top_path)
    
    config.add_subpackage("filters")
    
    config.add_extension("bonds",
                         ["bonds.c"],
                         include_dirs=[incdir],
                         depends=boxesdeps + utildeps + arraydeps,
                         libraries=["boxeslib", "utilities", "array_utils"])
    
    config.add_extension("_clusters",
                         ["clusters.c"],
                         include_dirs=[incdir],
                         depends=boxesdeps + utildeps + arraydeps,
                         libraries=["boxeslib", "utilities", "array_utils"])
    
    config.add_extension("_voronoi",
                         ["voronoi.c", "voro_iface.cpp",
                          "voro++/src/voro++.cc"],
                         depends=["voro_iface.h"] + arraydeps,
                         libraries=["array_utils"],
                         include_dirs=[incdir])
    
    return config

if __name__ == "__main__":
    print("This is the wrong setup.py to run")
