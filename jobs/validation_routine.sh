#!/usr/bin/env sh
#PBS -N validation
#PBS -l select=1:ncpus=4:mem=24G
#PBS -l walltime=1:00:00

module load conda
conda activate /work/scractch/vinascj/virtualenv/geovisualize
# export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
# conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/vinascj/src/
export VAEDIR=${SRCDIR}/prosailvae/prosailvae
# # export DATADIR=/work/scratch/zerahy/prosailvae/data/1e7_dataset/
# export RSR_DIR=/work/scratch/zerahy/prosailvae/data/
# export N_SAMPLES=10000
# export NOISE=0.01

python ${VAEDIR}/validation_routine.py \
    -input_config ~/src/prosailvae/config/validation.json \
    --export_path /work/scratch/$USER/ \
