#PBS -N foldgather
#PBS -l select=1:ncpus=2:mem=5G
#PBS -l walltime=00:10:00
#PBS -J 172-294:1

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/BIG_37963322_jobarray//${PBS_ARRAY_INDEX}/


python ${VAEDIR}/metrics/fold_results.py -r $OUTDIR 
conda deactivate



