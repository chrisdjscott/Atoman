#!/bin/bash -e

# This script will install a conda environment with all the requirements for running Atoman
# NOTE: if you already have conda installed you can pass the path to the conda install
#       directory to the -d flag.
# Author: Chris Scott

# usage
display_usage() {
    echo
    echo "Usage: $0 [-d conda_dir] [-p python_version] [-V vtk_version] [-e conda_env] [-g] [-h]"
    echo
    echo "Arguments:"
    echo
    echo "    [-d conda_dir] The location to install conda. If conda already exists at that"
    echo "                   location we use it. (default is '$HOME/miniconda')"
    echo "    [-p python_version] The version of python to use (either 2 or 3, default is 3)"
    echo "    [-e conda_env] The name of the conda environment to create (default is 'atoman')"
    echo "    [-g] Install gcc from conda"
    echo "    [-h] Display help"
    echo
}

# default args
# NOTE: Python 3 will be much slower as it has to install PySide from source
CONDIR=${HOME}/miniconda
PYVER=3
CONDENV=atoman
WITH_GCC=0

# parse args
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -d|--dir)
        CONDIR="$2"
        shift
        ;;
        -p|--pyver)
        PYVER="$2"
        shift
        ;;
        -e|--cenv)
        CONDENV="$2"
        shift
        ;;
        -g|--gcc)
        WITH_GCC=1
        ;;
        -h|--help)
        display_usage
        exit 0
        ;;
        *)
        # unknown option
        echo "Error: unrecognised argument '${key}'"
        display_usage
        exit 1
        ;;
    esac
    shift # past argument or value
done

# check args
NEED_CONDA=1
if [ -d ${CONDIR} ]
then
    # if conda exists we try and use it, otherwise raise error
    if [ -x "${CONDIR}/bin/conda" ]; then
        echo "Found conda, checking if we can use it..."
        NEED_CONDA=0
        export PATH=${CONDIR}/bin:$PATH

        # check if conda env exists already
        set +e
        if source activate ${CONDENV} &> /dev/null; then
            echo
            echo "Error: conda env \"${CONDENV}\" already exists. Please specify a different one."
            display_usage
            exit 8
        fi
        set -e
        echo "...conda looks good"
    else
        echo "Error: install dir already exists and does not contain conda: \"${CONDIR}\". Please specify a different directory."
        display_usage
        exit 2
    fi
fi

# machine hardware name
MACHW=`uname -m`
echo Installing for machine: "${MACHW}"

# conda OS
ostmp=`uname -s`
case $ostmp in
    Darwin)
    CONDOS="MacOSX"
    ;;
    Linux)
    CONDOS="Linux"
    ;;
    *)
    echo "Error: unsupported OS $ostmp"
    exit 3
    ;;
esac
echo CONDA OS = "${CONDOS}"

case ${PYVER} in
    2)
    PYVER=2.7
    ;;
    3)
    PYVER=3.6
    ;;
    *)
    echo "Error: Python version must be '2' or '3' ('2' is default)"
    display_usage
    exit 5
    ;;
esac

# echo args
echo CONDA INSTALL DIR  = "${CONDIR}"
echo CONDA ENV          = "${CONDENV}"
echo PYTHON VERSION     = "${PYVER}"

# source travis_retry function
RETRY=""
if [[ ! -z "${TRAVIS_OS_NAME}" ]]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    source ${SCRIPT_DIR}/travis_retry.sh
    RETRY=travis_retry
fi

# install conda if required
if [ "$NEED_CONDA" = "1" ]; then
    # download and run miniconda (to /tmp for now)
    mincon=/tmp/miniconda.sh
    echo "Downloading miniconda..."
    ${RETRY} curl -L -o "${mincon}" https://repo.continuum.io/miniconda/Miniconda3-latest-${CONDOS}-${MACHW}.sh
    chmod +x "${mincon}"
    echo "Installing miniconda..."
    ${RETRY} "${mincon}" -b -p ${CONDIR}
    rm "${mincon}"
    # set PATH
    export PATH=${CONDIR}/bin:${PATH}
fi

# fix SSL issue on travis??
[[ -z "${TRAVIS_OS_NAME}" ]] || conda config --set ssl_verify false

# info about conda
echo "Conda installation info..."
type conda
conda info -a

# enable conda-forge channel
conda config --add channels conda-forge

# update conda
echo Updating conda...
${RETRY} conda update --yes --quiet conda

# create conda environment
echo Creating conda environment: \"${CONDENV}\"...
${RETRY} conda create -y -q -n ${CONDENV} python=${PYVER} pip

# install GCC if required
if [ "$WITH_GCC" = "1" ]; then
    ${RETRY} conda install -y -q -n ${CONDENV} gcc
fi

# activate the environment
source activate ${CONDENV}

# install dependencies from pypi
pip install --upgrade pip
pip install numpy
pip install scipy
pip install matplotlib
pip install pillow
pip install nose
pip install setuptools
pip install sphinx
pip install sphinx_rtd_theme
pip install paramiko
pip install vtk
pip install pyhull || true
pip install pyinstaller
pip install --index-url=http://download.qt.io/snapshots/ci/pyside/5.11/latest/ pyside2 --trusted-host download.qt.io

echo
echo ==============================================================================================================
echo Install complete.
echo Add \"source ${CONDIR}/etc/profile.d/conda.sh\" to your ~/.bashrc to use conda.
echo Enable the environment by running \"conda activate ${CONDENV}\".
echo ==============================================================================================================
echo
