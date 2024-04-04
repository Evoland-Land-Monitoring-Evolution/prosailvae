#!/bin/bash
#SBATCH --job-name=pvmktest
#SBATCH --output=outputfile-%j.out
#SBATCH --error=errorfile-%j.err
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1                       # number of nodes ( or --nodes=1)
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00            # Walltime
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --account=adz@v100 # MANDATORY : account (launch myaccounts to list your accounts)
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc


experiment=jz

# be sure no modules loaded
module purge
export BASE_DIR=/gpfsdswork/projects/rech/adz/uzh16pa
export PV_SRCDIR=${BASE_DIR}/src/MMDC/prosailvae
export PV_WORKING_DIR=${BASE_DIR}/PROSAILVAE/jobs

cd ${PV_SRCDIR}
module purge
module load anaconda-py3/2023.09
conda activate prosailvae
make test
