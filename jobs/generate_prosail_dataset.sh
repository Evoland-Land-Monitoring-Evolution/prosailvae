#PBS -N genprosail
#PBS -l select=1:ncpus=4:mem=92G
#PBS -l walltime=72:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/scratch/zerahy/prosailvae/data/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_jobarray/
export N_SAMPLES=100000
export NOISE=0

python ${VAEDIR}/dataset/generate_dataset.py -d ${DATADIR} -n $N_SAMPLES -p "noiseless_" -s $NOISE
conda deactivate


