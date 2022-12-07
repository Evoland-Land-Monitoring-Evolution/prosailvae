#!/usr/bin/env bash

export python_version="3.9"
export name="juanvae"

if ! [ -z "$1" ]
then
    export name=$1
fi

source ~/set_proxy_iota2.sh

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
conda create -p $target

# Enter virtualenv
conda activate $target
conda install -y python==${python_version} pip

#conda install -y pandas scikit-learn tqdm numpy pip
# Installing Pytorch. Please change option for GPU use.
#conda install -y pytorch-gpu torchvision cudatoolkit=11.0 -c pytorch -c conda-forge
#conda install -y -c conda-forge zenodo_get jax numpyro

which python
python --version

conda deactivate
conda activate $target

pip install -e /home/uz/$USER/scratch/src/thirdparties/prosailpython
pip install -e /home/uz/$USER/src/thirdparties/mmdc-singledate/
pip install -e /home/uz/$USER/src/thirdparties/sensorsio/
pip install -e /home/uz/$USER/src/thirdparties/torchutils/

# Install requirements
pip install -r requirements.txt

# Install the current project in edit mode
pip install -e .

# End
conda deactivate
