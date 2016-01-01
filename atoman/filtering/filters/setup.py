
from __future__ import print_function
from __future__ import absolute_import

import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # dependencies
    boxesdeps = [os.path.join("..", "..", "visclibs", "boxeslib.c"),
                 os.path.join("..", "..", "visclibs", "boxeslib.h")]
    utildeps = [os.path.join("..", "..", "visclibs", "utilities.c"),
                os.path.join("..", "..", "visclibs", "utilities.h")]
    arraydeps = [os.path.join("..", "..", "visclibs", "array_utils.c"),
                 os.path.join("..", "..", "visclibs", "array_utils.h")]
    nebdeps = [os.path.join("..", "..", "visclibs", "neb_list.c"),
               os.path.join("..", "..", "visclibs", "neb_list.h")]
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, os.pardir, os.pardir))
    
    # config
    config = Configuration("filters", parent_package, top_path)
    
    config.add_extension("_acna",
                         ["acna.c"],
                         depends=[os.path.join("..", "atom_structure.h")] + boxesdeps + utildeps + nebdeps + arraydeps,
                         include_dirs=[incdir],
                         libraries=["boxeslib", "utilities", "neb_list", "array_utils"])
    
    config.add_extension("_bond_order",
                         ["bond_order.c"],
                         include_dirs=[incdir],
                         depends=boxesdeps + utildeps + nebdeps + arraydeps,
                         libraries=["boxeslib", "utilities", "neb_list", "array_utils"])
     
    config.add_extension("_defects",
                         ["defects.c"],
                         depends=[os.path.join("..", "atom_structure.h")] + boxesdeps + utildeps + nebdeps + arraydeps,
                         include_dirs=[incdir],
                         libraries=["boxeslib", "utilities", "neb_list", "array_utils"])
     
    config.add_extension("_filtering",
                         ["filtering.c"],
                         include_dirs=[incdir],
                         depends=boxesdeps + utildeps + arraydeps,
                         libraries=["boxeslib", "utilities", "array_utils"])
    
    config.add_extension("_bubbles",
                         ["bubbles.c"],
                         include_dirs=[incdir],
                         depends=boxesdeps + utildeps + nebdeps + arraydeps,
                         libraries=["boxeslib", "utilities", "neb_list", "array_utils"])
    
    return config

if __name__ == "__main__":
    print("This is the wrong setup.py to run")
