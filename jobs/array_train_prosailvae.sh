#PBS -N arrtrprosail
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=12:00:00
#PBS -J 0-0:1

export CONFIG_LIST=( 
"config"
)
module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export RSR_DIR=/work/scratch/zerahy/prosailvae/data/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/pvae_all_others/${PBS_JOBID:0:8}_prospect_D/${PBS_ARRAY_INDEX}_${CONFIG_LIST[${PBS_ARRAY_INDEX}]}/
export NUM_WORKERS=12
export CONFIG=${CONFIG_LIST[${PBS_ARRAY_INDEX}]}.json
export CONFIG_DIR=/home/uz/zerahy/projects/prosailvae/prosailvae/config/

CUDA_LAUNCH_BLOCKING=1 python ${VAEDIR}/prosailvae/train.py -r $OUTDIR -rsr $RSR_DIR -c $CONFIG -a True -cd $CONFIG_DIR -p True
conda deactivate


