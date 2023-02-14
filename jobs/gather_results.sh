#PBS -N resgather
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G
#PBS -l walltime=1:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/37650650_jobarray/

python ${VAEDIR}/metrics/gather_results.py -r $OUTDIR 
conda deactivate



