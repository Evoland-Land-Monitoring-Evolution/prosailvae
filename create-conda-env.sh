#!/usr/bin/env bash

export python_version="3.9"
export name="prosailvae"

conda create -n $name python=$python_version

# Enter virtualenv
source activate $name

which python
python --version

export GITLAB_TOKEN=hssC4vRMTyMzbU6Gi86k 
export GITLAB_TOKEN_USER=readonly
export VERSION=tensorport

conda install -y pandas scikit-learn tqdm numpy pip 
pip install psutil
# Installing Pytorch. Please change option for GPU use.
conda install -y pytorch torchvision cpuonly -c pytorch
conda install -y -c conda-forge zenodo_get jax numpyro
conda install -y  spyder
pip install git+https://${GITLAB_TOKEN_USER}:${GITLAB_TOKEN}@src.koda.cnrs.fr/yoel.zerah.1/prosailpython.git@${VERSION}
pip install -e .

# End
source deactivate
exec $SHELL

