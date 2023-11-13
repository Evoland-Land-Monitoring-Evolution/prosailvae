#PBS -N trainpvae
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export RSR_DIR=/work/scratch/zerahy/prosailvae/data/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_config_rnn_pvae_b2_n3_best/
export NUM_WORKERS=12
export CONFIG=config.json
export CONFIG_DIR=/home/uz/zerahy/projects/prosailvae/prosailvae/config/

python ${VAEDIR}/prosailvae/train.py -d ${DATADIR} -r $OUTDIR -rsr $RSR_DIR -c $CONFIG -p True -a True -cd $CONFIG_DIR
conda deactivate
