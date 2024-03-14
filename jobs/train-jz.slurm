#!/bin/bash
#SBATCH --job-name=psvae
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1                       # number of nodes ( or --nodes=1)
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH --cpus-per-task=4
#SBATCH --time=01:59:00            # Walltime
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --account=adz@v100 # MANDATORY : account (launch myaccounts to list your accounts)
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc


experiment=jz

# be sure no modules loaded
module purge
export SCRATCH=/gpfsdswork/projects/rech/adz/uzh16pa
export SRCDIR=${SCRATCH}/src/MMDC/prosailvae
export WORKING_DIR=${SCRATCH}/PROSAILVAE/jobs

cd ${WORKING_DIR}
module purge
module load anaconda-py3/2023.09
conda activate prosailvae
export PYTHONOPTIMIZE=TRUE
export HYDRA_FULL_ERROR=1
python ${SRCDIR}/train.py experiment=$experiment  >> output_$SLURM_JOBID.log
