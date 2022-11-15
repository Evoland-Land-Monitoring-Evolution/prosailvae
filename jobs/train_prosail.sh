#PBS -N trainprosail
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=92G:ngpus=1  
#PBS -l walltime=12:00:00


export PATH="/work/scratch/zerahy/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/lib

module load conda
module load gcc/10.2.0
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/scratch/zerahy/prosailvae/data/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_jobarray/
export NUM_WORKERS=12

python ${VAEDIR}/prosailvae/train.py -d ${DATADIR} -r $OUTDIR -p "sim_"
conda deactivate



