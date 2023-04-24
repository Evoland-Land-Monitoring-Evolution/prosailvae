#!/usr/bin/env bash

export python_version="3.10"
export name="prosailvae"

conda create -n $name python=$python_version

# Enter virtualenv
source activate $name

which python
python --version

export GITLAB_TOKEN=hssC4vRMTyMzbU6Gi86k 
export GITLAB_TOKEN_USER=readonly
export VERSION=cnn

pip install -e /home/yoel/Documents/Dev/PROSAIL-VAE/prosailpython
pip install -e /home/yoel/Documents/Dev/PROSAIL-VAEmmdc-singledate/
pip install -e /home/yoel/Documents/Dev/PROSAIL-VAE/sensorsio/
pip install -e /home/yoel/Documents/Dev/PROSAIL-VAE/torchutils/

# Install requirements
pip install -r requirements.txt

# Install the current project in edit mode
pip install -e .

# End
source deactivate
exec $SHELL

