#PBS -N trainprosail
#PBS -l select=1:ncpus=4:mem=60G
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export RSR_DIR=/work/scratch/zerahy/prosailvae/data/
export DATADIR=/work/scratch/zerahy/prosailvae/data/patches_V3/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_compare_models/
export CONFIG=/work/scratch/zerahy/prosailvae/configs/model_dict.json

python ${VAEDIR}/metrics/comparative_results.py -d ${DATADIR} -r $OUTDIR -rsr $RSR_DIR -c $CONFIG 
conda deactivate
