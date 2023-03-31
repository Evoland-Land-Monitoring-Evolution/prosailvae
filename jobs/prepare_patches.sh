#PBS -N preparepatch
#PBS -l select=1:ncpus=4:mem=60G
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/CESBIO/projects/MAESTRIA/prosail_validation/validation_sites/torchfiles/
export OUTDIR=/work/scratch/zerahy/prosailvae/data/patches/

CUDA_LAUNCH_BLOCKING=1 python ${VAEDIR}/prosailvae/prepare_patch_dataset.py -d ${DATADIR} -o $OUTDIR 
conda deactivate
