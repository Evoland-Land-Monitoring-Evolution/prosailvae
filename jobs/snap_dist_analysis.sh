#PBS -N trainsnap
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/scratch/zerahy/prosailvae/data/1e5_dataset_V2/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_snap_regressions/

CUDA_LAUNCH_BLOCKING=1 python ${VAEDIR}/snap_regression/snap_regression.py -d ${DATADIR} -r $OUTDIR 
conda deactivate
