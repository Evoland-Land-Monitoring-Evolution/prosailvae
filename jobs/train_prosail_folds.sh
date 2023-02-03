#PBS -N prosailvaefolds
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=02:00:00
#PBS -J 1-5:1

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export FRSR_DIR=$1
export FDATADIR=$2
export FOUTDIR=$3
export FTENSOR_DIR=$4
export FCONFIG=$5
export K_FOLD=${PBS_ARRAY_INDEX}
export NUMBER_OF_FOLDS=5


python ${VAEDIR}/prosailvae/train.py -d ${FDATADIR} -r $FOUTDIR -rsr $FRSR_DIR -t $FTENSOR_DIR -c $FCONFIG -a True -n $NUMBER_OF_FOLDS -x $K_FOLD
conda deactivate




