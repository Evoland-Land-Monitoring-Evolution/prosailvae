#!/bin/bash
#SBATCH --job-name=mmpsvae
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1                       # number of nodes ( or --nodes=1)
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH --cpus-per-task=32
#SBATCH --time=5:00:00            # Walltime
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=adz@v100 # MANDATORY : account (launch myaccounts to list your accounts)
#SBATCH --signal=SIGUSR1@90


experiment=mmdc

# be sure no modules loaded
module purge
module load anaconda-py3/2023.09
conda activate prosailvae
export BASE_DIR=/gpfsdswork/projects/rech/adz/uzh16pa
export PV_SRCDIR=${BASE_DIR}/src/MMDC/prosailvae
export PV_WORKING_DIR=${SCRATCH}/JOBS/PROSAILVAE/jobs

cd ${PV_WORKING_DIR}
export PYTHONOPTIMIZE=TRUE
export HYDRA_FULL_ERROR=1
srun python -u ${PV_SRCDIR}/train.py experiment=$experiment \
       >> output_$SLURM_JOBID.log
