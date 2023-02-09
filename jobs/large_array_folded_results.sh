#PBS -N prslhyperres
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=1:00:00
#PBS -J 1-2:1

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export RSR_DIR=/work/scratch/zerahy/prosailvae/data/
export DATADIR=/work/scratch/zerahy/prosailvae/data/1e5_dataset_V2/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/BIG_37797341_jobarray/${PBS_ARRAY_INDEX}/3_kfold_3_supervised_True_full_/
export TENSOR_DIR=/work/CESBIO/projects/MAESTRIA/prosail_validation/toyexample/torchfiles/
export NUM_WORKERS=12

cd /work/scratch/zerahy/temp/
qsub -v FRSR_DIR=$RSR_DIR,FDATADIR=$DATADIR,FOUTDIR=$OUTDIR,FTENSOR_DIR=$TENSOR_DIR ${VAEDIR}/jobs/train_prosail_folds.sh




