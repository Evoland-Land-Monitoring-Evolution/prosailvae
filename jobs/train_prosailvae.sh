#!/bin/bash
#SBATCH --job-name=trainpvae
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH -N 1                       # number of nodes ( or --nodes=1)
#SBATCH --ntasks-per-node=4                      # number of tasks ( or --tesks=8)
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH --partition=gpu_a100        # partition
#SBATCH --qos=gpu_all               # QoS
#SBATCH --time=20:00:00            # Walltime
#SBATCH --mem-per-cpu=24G          # memory per core
#SBATCH --account=cesbio           # MANDATORY : account (launch myaccounts to list your accounts)
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc

module purge
module load conda
# export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/env/$USER/virtualenv/prosailvae
export SRCDIR=/work/scratch/data/$USER/src/MMDC/prosailvae
export VAEDIR=${SRCDIR}/src/prosailvae
export RSR_DIR=${SRCDIR}/data/
export OUTDIR=/work/scratch/data/$USER/prosailvae/results/${PBS_JOBID:0:8}_config_rnn_pvae_b2_n3_best/
export NUM_WORKERS=12
export CONFIG=config.json
export CONFIG_DIR=${SRCDIR}/config/

mkdir -p ${OUTDIR}
python ${VAEDIR}/train.py  -r $OUTDIR -rsr $RSR_DIR -c $CONFIG -p True -a True -cd $CONFIG_DIR  >> output_$SLURM_JOBID.log
conda deactivate
