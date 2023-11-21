#!/usr/bin/env bash

export python_version="3.11"
export name="prosailvae"

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

export target=/work/scratch/env/$USER/virtualenv/$name

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
module purge
module load conda

source create-conda-env-common.sh
