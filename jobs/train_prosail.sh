#PBS -N trainprosail
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=92G:ngpus=1  
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/scratch/zerahy/prosailvae/data/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_jobarray/
export NUM_WORKERS=12

python ${VAEDIR}/prosailvae/train.py -d ${DATADIR} -r $OUTDIR -p "sim_"
conda deactivate



