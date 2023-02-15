#!/usr/bin/env sh
#PBS -N validation
#PBS -l select=1:ncpus=4:mem=24G
#PBS -l walltime=1:00:00

# clean previous conda
conda deactivate
module purge

# load conda
module load conda
conda activate /home/uz/vinascj/scratch/virtualenv/geovisualize
export SRCDIR=/home/uz/vinascj/src/
export VAEDIR=${SRCDIR}/prosailvae/prosailvae

# Export Frace 1
python ${VAEDIR}/validation_routine.py \
    --input_config ~/src/prosailvae/config/validation_france1.json \
    --export_path /home/uz/vinascj/src/prosailvae/field_data/processed/ \

# Export Italy 1
python ${VAEDIR}/validation_routine.py \
    --input_config ~/src/prosailvae/config/validation_italy1.json \
    --export_path /home/uz/vinascj/src/prosailvae/field_data/processed/ \

# Export Italy 2
python ${VAEDIR}/validation_routine.py \
    --input_config ~/src/prosailvae/config/validation_italy1.json \
    --export_path /home/uz/vinascj/src/prosailvae/field_data/processed/ \

# Export Spain 1
python ${VAEDIR}/validation_routine.py \
    --input_config ~/src/prosailvae/config/validation_spain1.json \
    --export_path /home/uz/vinascj/src/prosailvae/field_data/processed/ \






# python ${VAEDIR}/validation_routine.py \
#     --input_config ~/src/prosailvae/config/validation.json \
#     --export_path /home/uz/vinascj/src/prosailvae/field_data/processed/ \
