#!/usr/bin/env bash

export python_version="3.9"
export name="prosailvae"

export GITLAB_TOKEN=hssC4vRMTyMzbU6Gi86k 
export GITLAB_TOKEN_USER=readonly
export VERSION=tensorport

export python_version="3.9"
export name="prosailvae"
if ! [ -z "$1" ]
then
    export name=$1
fi

source ~/source_proxy.sh
if [ -z "$https_proxy" ]
then
    echo "Please set https_proxy environment variable before running this script"
    exit 1
fi

export target=/work/scratch/$USER/virtualenv/$name

if ! [ -z "$2" ]
then
    export target="$2/$name"
fi

echo "Installing $name in $target ..."

if [ -d "$target" ]; then
   echo "Cleaning previous conda env"
   rm -rf $target
fi

# Create blank virtualenv
module load conda
module load gcc
conda activate
conda create -p $target

# Enter virtualenv
conda activate $target
conda install python==${python_version} pip

conda install -y pandas scikit-learn tqdm numpy pip
# Installing Pytorch. Please change option for GPU use.
conda install pytorch-gpu torchvision cudatoolkit=11.0 -c pytorch -c conda-forge
conda install -y -c conda-forge zenodo_get jax numpyro

which python
python --version

conda deactivate
conda activate $target

# Install thirdparties
rm -rf thirdparties
mkdir thirdparties

git clone https://${GITLAB_TOKEN_USER}:${GITLAB_TOKEN}@src.koda.cnrs.fr/yoel.zerah.1/prosailpython.git@${VERSION} thirdparties/prosailpython
pip install -e ./thirdparties/prosailpython

# Install the current project in edit mode
pip install -e .[testing]

# End
conda deactivate

