#PBS -N genprosail
#PBS -l select=1:ncpus=4:mem=92G
#PBS -l walltime=72:00:00


export PATH="/work/scratch/zerahy/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/lib

module load conda
module load gcc
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/scratch/zerahy/prosailvae/data/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_jobarray/
export N_SAMPLES=10000

python ${VAEDIR}/dataset/generate_dataset.py -d ${DATADIR} -n $N_SAMPLES -p "sim_"
conda deactivate


