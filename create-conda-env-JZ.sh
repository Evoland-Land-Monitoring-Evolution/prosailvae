#!/usr/bin/env bash

export python_version="3.11"
export name="prosailvae"

export target=$WORK/.conda/envs/$name


echo "Installing $name in $target ..."

if [ -d "$target" ]; then
   echo "Cleaning previous conda env"
   rm -rf $target
fi

# Create blank virtualenv
module purge
module load anaconda-py3/2023.09

source create-conda-env-common.sh
