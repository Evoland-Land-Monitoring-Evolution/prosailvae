#PBS -N trainprosail
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export RSR_DIR=/work/scratch/zerahy/prosailvae/data/
export DATADIR=/work/scratch/zerahy/prosailvae/data/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_jobarray/
export TENSOR_DIR=/work/CESBIO/projects/MAESTRIA/prosail_validation/toyexample/torchfiles/
export NUM_WORKERS=12
export CONFIG=config_supervised_kl.json
export SUPERVISED_CONFIG=path/to/sup_config.json
export SUPERVISED_WEIGHTS=path/to/sup_weights.tar

python ${VAEDIR}/prosailvae/train.py -d ${DATADIR} -r $OUTDIR -rsr $RSR_DIR -t $TENSOR_DIR -c $CONFIG -cs $SUPERVISED_CONFIG -ws $SUPERVISED_WEIGHTS
conda deactivate



