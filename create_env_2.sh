#!/usr/bin/env bash

export python_version="3.9"
export name="juanvae"

conda create -n $name python=$python_version

# Enter virtualenv
conda activate $name

which python
python --version

pip install spyder
pip install -e /home/yoel/Documents/Dev/PROSAIL-VAE/prosailpython
pip install -e /home/yoel/Documents/Dev/PROSAIL-VAE/thirdparties/mmdc-singledate/
pip install -e /home/yoel/Documents/Dev/PROSAIL-VAE/thirdparties/sensorsio
pip install -e /home/yoel/Documents/Dev/PROSAIL-VAE/thirdparties/torchutils
pip install -r requirements.txt
pip install -e .


# End
conda deactivate

