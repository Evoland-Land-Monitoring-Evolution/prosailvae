#PBS -N resgather
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=1:00:00
#PBS -J 1-2:1

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/BIG_37758640_jobarray/${PBS_ARRAY_INDEX}/


python ${VAEDIR}/metrics/fold_results.py -r $OUTDIR 
conda deactivate



