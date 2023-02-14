#PBS -N prosailresbatch
#PBS -q qgpgpudev
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=00:30:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/


python ${VAEDIR}/metrics/results_batch.py -d ${FDATADIR} -r $FOUTDIR -rsr $FRSR_DIR -t $FTENSOR_DIR 
conda deactivate




