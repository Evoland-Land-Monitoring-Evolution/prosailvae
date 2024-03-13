conda create --yes --prefix $target python==${python_version} pip


# Enter virtualenv
conda activate $target

which python
python --version

conda deactivate
conda activate $target

# Install requirements
pip install -r requirements.txt

# Install sensorsio
rm -rf thirdparties/sensorsio
git clone https://src.koda.cnrs.fr/mmdc/sensorsio.git thirdparties/sensorsio
pip install thirdparties/sensorsio

# Install torchutils
rm -rf thirdparties/torchutils
git clone https://src.koda.cnrs.fr/mmdc/torchutils.git thirdparties/torchutils
pip install thirdparties/torchutils

# Install MMDC-Single-Date
rm -rf thirdparties/mmdc-singledate
git clone https://src.koda.cnrs.fr/mmdc/mmdc-singledate.git thirdparties/mmdc-singledate
pip install -e thirdparties/mmdc-singledate

# Install ProsailPython
rm -rf thirdparties/prosailpython
git clone -b downsampled_tensor https://src.koda.cnrs.fr/mmdc/prosailpython.git thirdparties/prosailpython
pip install -e thirdparties/prosailpython

# Install the current project in edit mode
pip install -e .[testing]

# Activate pre-commit hooks
pre-commit install

# End
conda deactivate
