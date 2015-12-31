from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration("visclibs", parent_package, top_path)

    # root path for includes
    cwd = os.path.dirname(__file__)
    incdirs = [os.path.join(cwd, os.pardir)]

    # add libraries
    config.add_library("boxeslib", ["boxeslib.c"], depends=["boxeslib.h"], include_dirs=[incdirs])
    config.add_library("utilities", ["utilities.c"], depends=["utilities.h"], include_dirs=[incdirs])
    config.add_library("neb_list", ["neb_list.c", "boxeslib.c", "utilities.c"],
                       depends=["neb_list.h", "boxeslib.h", "utilities.h"],
                       include_dirs=[incdirs])
    config.add_library("array_utils", ["array_utils.c"], depends=["array_utils.h"],
                       include_dirs=[incdirs])

    # add extensions (for testing)
    config.add_extension("tests._test_boxeslib",
                         ["tests/test_boxeslib.c"],
                         libraries=["boxeslib", "array_utils"],
                         include_dirs=[incdirs],
                         depends=["boxeslib.h", "boxeslib.c", "array_utils.h", "array_utils.c"])
    
    return config

if __name__ == "__main__":
    print("This is the wrong setup.py to run")
