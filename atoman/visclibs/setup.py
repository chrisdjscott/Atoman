
import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration("visclibs", parent_package, top_path)

    # add libraries
    config.add_library("boxeslib", ["boxeslib.c"], depends=["boxeslib.h"])
    config.add_library("utilities", ["utilities.c"], depends=["utilities.h"])
    config.add_library("neb_list", ["neb_list.c", "boxeslib.c", "utilities.c"],
                       depends=["neb_list.h", "boxeslib.h", "utilities.h"])
    config.add_library("array_utils", ["array_utils.c"], depends=["array_utils.h"])

    # add extensions (for testing)
    incdirs = [os.getcwd()]
    config.add_extension("tests._test_boxeslib",
                         ["tests/test_boxeslib.c"],
                         libraries=["boxeslib", "array_utils"],
                         depends=["boxeslib.h", "boxeslib.c", "array_utils.h", "array_utils.c"])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
