# -*- coding: utf-8 -*-

"""
Setup script for atoman

@author: Chris Scott

"""
from __future__ import print_function
from __future__ import absolute_import
import os
import glob
import sys
import subprocess
import shutil
import platform
import tempfile

# setuptools is required for entry point
import setuptools
import distutils.sysconfig

import versioneer


# check for openmp following
# http://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
# see http://openmp.org/wp/openmp-compilers/
omp_test = br"""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
"""

TEST_OMP_FLAGS = [
    "-fopenmp",
    "-qopenmp",
]


def check_for_openmp():
    try:
        cc = os.environ['CC']
    except KeyError:
        cc = 'gcc'
    curdir = os.getcwd()
    for omp_flag in TEST_OMP_FLAGS:
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        try:
            filename = r'test.c'
            with open(filename, 'wb', 0) as file:
                file.write(omp_test)
            with open(os.devnull, 'wb') as fnull:
                result = subprocess.call([cc, omp_flag, filename],
                                         stdout=fnull, stderr=fnull)
            print('check_for_openmp() result for {}: '.format(omp_flag), result)
            if result == 0:
                break
        finally:
            # clean up
            shutil.rmtree(tmpdir)
            os.chdir(curdir)

    return result == 0, omp_flag


HAVE_OMP, OMP_FLAG = check_for_openmp()
print("Have OpenMP: ", HAVE_OMP)
if HAVE_OMP:
    print("OpenMP flag: ", OMP_FLAG)

# sphinx build
try:
    from sphinx.setup_command import BuildDoc
    HAVE_SPHINX = True
except ImportError:
    HAVE_SPHINX = False

if HAVE_SPHINX:
    class AtomanBuildDoc(BuildDoc):
        """Compile resources and run in-place build before Sphinx doc-build"""
        def run(self):
            # in place build
            ret = subprocess.call([sys.executable, sys.argv[0], 'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building atoman failed (%d)" % ret)

            # build doc
            BuildDoc.run(self)


# package configuration method
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path, version=versioneer.get_version())
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage("atoman")
    config.add_data_dir(("atoman/doc", os.path.join("doc", "build", "html")))

    return config


# clean
def do_clean():
    cwd = os.getcwd()
    os.chdir("atoman")
    try:
        for root, dirs, files in os.walk(os.getcwd()):
            so_files = glob.glob(os.path.join(root, "*.so"))
            for so_file in so_files:
                print("rm atoman/%s" % os.path.relpath(so_file))
                os.unlink(so_file)

            if "resources.py" in files:
                os.unlink(os.path.join(root, "resources.py"))

            pyc_files = glob.glob(os.path.join(root, "*.pyc"))
            for pyc_file in pyc_files:
                os.unlink(pyc_file)

    finally:
        os.chdir(cwd)

    for root, dirs, files in os.walk(os.getcwd()):
        cachepth = os.path.join(root, "__pycache__")
        if os.path.isdir(cachepth):
            shutil.rmtree(cachepth)

    if os.path.isdir("atoman/doc"):
        print("rm -rf atoman/doc")
        shutil.rmtree(os.path.join("atoman", "doc"))

#    if os.path.isdir(os.path.join("doc", "build")):
#        print("rm -rf doc/build/*")
#        os.system("rm -rf doc/build/*")

    if os.path.isdir("dist"):
        print("rm -rf dist/")
        shutil.rmtree("dist")

    if os.path.isdir("build"):
        print("rm -rf build/")
        shutil.rmtree("build")
    if os.path.isdir("atoman.egg-info"):
        print("rm -rf atoman.egg-info/")
        shutil.rmtree("atoman.egg-info")


# setup the package
def setup_package():
    # clean?
    if "clean" in sys.argv:
        do_clean()

    # documentation (see scipy...)
    cmdclass = versioneer.get_cmdclass()
    if HAVE_SPHINX:
        cmdclass['build_sphinx'] = AtomanBuildDoc

    # metadata
    metadata = dict(
        name="Atoman",
        maintainer="Chris Scott",
        maintainer_email="chris@chrisdjscott.co.uk",
        description="Analysis and visualisation of atomistic simulations",
        long_description="Analysis and visualisation of atomistic simulations",
        author="Chris Scott",
        author_email="chris@chrisdjscott.co.uk",
        license="MIT",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Environment :: X11 Applications",
            "Environment :: X11 Applications :: Qt",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Operating System :: MacOS",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: C",
            "Programming Language :: C++",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Visualization",
        ],
        platforms=["Linux", "Mac OS-X"],
        cmdclass=cmdclass,
        entry_points={
            'gui_scripts': [
                'Atoman = atoman.__main__:main',
            ]
        },
        zip_safe=False,
    )

    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or sys.argv[1] in ('--help-commands', 'egg_info',
                                                                           '--version', 'clean', 'nosetests',
                                                                           'test')):
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = versioneer.get_version()
        metadata['test_suite'] = "nose.collector"

    else:
        from numpy.distutils.core import setup
        from numpy.distutils.command.build_ext import build_ext
        from numpy.distutils.command.build_clib import build_clib

        # subclass build_ext to use additional compiler options (eg. for OpenMP)
        class build_ext_subclass(build_ext):
            def build_extensions(self, *args, **kwargs):
                for e in self.extensions:
                    if HAVE_OMP:
                        e.extra_compile_args.append(OMP_FLAG)
                        e.extra_link_args.append(OMP_FLAG)
                    e.include_dirs.append(distutils.sysconfig.get_python_inc())

                return build_ext.build_extensions(self, *args, **kwargs)

        # subclass build_clib to use additional compiler options (eg. for OpenMP)
        class build_clib_subclass(build_clib):
            def build_libraries(self, *args, **kwargs):
                for libtup in self.libraries:
                    opts = libtup[1]
                    if HAVE_OMP:
                        if "extra_compiler_args" not in opts:
                            opts["extra_compiler_args"] = []
                        opts["extra_compiler_args"].append(OMP_FLAG)
                    if "include_dirs" not in opts:
                        opts["include_dirs"] = []
                    opts["include_dirs"].append(distutils.sysconfig.get_python_inc())

                return build_clib.build_libraries(self, *args, **kwargs)

        cmdclass["build_ext"] = build_ext_subclass
        cmdclass["build_clib"] = build_clib_subclass
        metadata["configuration"] = configuration

    # run setup
    setup(**metadata)


if __name__ == "__main__":
    setup_package()
    if not HAVE_OMP:
        print("Warning: building without OpenMP - it will be slow")
