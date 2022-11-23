#PBS -N genprobatch
#PBS -l select=1:ncpus=4:mem=92G
#PBS -l walltime=72:00:00
#PBS -N jobArray
#PBS -J 0-1000:1

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/scratch/zerahy/prosailvae/data/1e7_dataset/
export N_SAMPLES=10000
export NOISE=0.01

python ${VAEDIR}/dataset/generate_dataset.py -d ${DATADIR} -n $N_SAMPLES -p "${PBS_ARRAY_INDEX}_" -s $NOISE
conda deactivate


