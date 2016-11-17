#!/bin/bash -e

# This script will install a conda environment with all the requirements for running Atoman
# NOTE: if you already have conda installed you can pass the path to the conda install
#       directory to the -d flag.
# Author: Chris Scott

# usage
display_usage() {
    echo
    echo "Usage: $0 [-d conda_dir] [-p python_version] [-V vtk_version] [-e conda_env] [-h]"
    echo
}

# default args
# NOTE: Python 3 will be much slower as it has to install PySide from source
CONDIR=${HOME}/miniconda
PYVER=2
VTKVER=7
CONDENV=atoman

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
        -V|--vtkver)
        VTKVER="$2"
        shift
        ;;
        -e|--env)
        CONDENV="$2"
        shift
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

case ${PYVER} in
    2)
    PYVER=2.7
    ;;
    3)
    PYVER=3.5
    ;;
    *)
    echo "Error: Python version must be '2' or '3' ('2' is default)"
    display_usage
    exit 5
    ;;
esac

case ${VTKVER} in
    5)
    VTKVER=5.10.1
    ;;
    6)
    VTKVER=6.3.0
    ;;
    7)
    VTKVER=7.0.0
    ;;
    *)
    echo "Error: VTK version must be '5' or '6' or '7' ('7' is default)"
    display_usage
    exit 6
    ;;
esac

# echo args
echo CONDA INSTALL DIR  = "${CONDIR}"
echo CONDA ENV          = "${CONDENV}"
echo PYTHON VERSION     = "${PYVER}"
echo VTK VERSION        = "${VTKVER}"

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

if [ "$NEED_CONDA" = "1" ]; then
    # download and run miniconda (to /tmp for now)
    mincon=/tmp/miniconda.sh
    echo "Downloading miniconda..."
    wget http://repo.continuum.io/miniconda/Miniconda-latest-${CONDOS}-${MACHW}.sh -O "${mincon}"
    chmod +x "${mincon}"
    echo "Installing miniconda..."
    "${mincon}" -b -p ${CONDIR}
    rm "${mincon}"
    # set PATH
    export PATH=${CONDIR}/bin:${PATH}
fi

# update conda
echo Updating conda...
conda update --yes conda

# create conda environment
echo Creating conda environment: \"${CONDENV}\"...
conda create -y -n ${CONDENV} python=${PYVER}
echo Installing gcc...
conda install -y -n ${CONDENV} gcc
echo Installing numpy...
conda install -y -n ${CONDENV} numpy
echo Installing scipy...
conda install -y -n ${CONDENV} scipy
echo Installing matplotlib...
conda install -y -n ${CONDENV} matplotlib
echo Installing pillow...
conda install -y -n ${CONDENV} pillow
echo Installing pip...
conda install -y -n ${CONDENV} pip
echo Installing nose...
conda install -y -n ${CONDENV} nose
echo Installing setuptools...
conda install -y -n ${CONDENV} setuptools
echo Installing sphinx...
conda install -y -n ${CONDENV} sphinx sphinx_rtd_theme
echo Installing paramiko
conda install -y -n ${CONDENV} paramiko
echo Installing PyQt5
conda install -y -n ${CONDENV} pyqt

# install VTK
case $VTKVER in
    7.0.0)
    echo Installing VTK 7.0.0 ...
    conda install -y -n ${CONDENV} -c menpo vtk=${VTKVER}
    ;;
    *)
    echo Installing VTK ${VTKVER} ...
    conda install -y -n ${CONDENV} vtk=${VTKVER}
    ;;
esac

# activate the environment
source activate ${CONDENV}

# install additional packages using pip
echo Installing additional packages using pip...
pip install pyhull
pip install pyinstaller

echo
echo ==============================================================================================================
echo Install complete.
echo Add \"export PATH=${CONDIR}/bin:\$\{PATH\}\" to your ~/.bashrc to use conda.
echo Enable the environment by running \"source activate ${CONDENV}\", or by adding that line to your .bashrc too.
echo ==============================================================================================================
echo
