#PBS -N suptrainprosail
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/juanvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export RSR_DIR=/work/scratch/zerahy/prosailvae/data/
export DATADIR=/work/scratch/zerahy/prosailvae/data/1e5_dataset_V2/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_jobarray/
export TENSOR_DIR=/work/CESBIO/projects/MAESTRIA/prosail_validation/toyexample/torchfiles/
export NUM_WORKERS=12
export CONFIG=config_unsupervised_b1_idx_1_diag_normed_sample_10.json
export SUPERVISED_CONFIG=/work/scratch/zerahy/prosailvae/results/37012944_jobarray/1_d2023_01_17_17_07_55_supervised_True_full_/config.json
export SUPERVISED_WEIGHTS=/work/scratch/zerahy/prosailvae/results/37012944_jobarray/1_d2023_01_17_17_07_55_supervised_True_full_/prosailvae_weigths.tar

python ${VAEDIR}/prosailvae/train.py -d ${DATADIR} -r $OUTDIR -rsr $RSR_DIR -t $TENSOR_DIR -c $CONFIG -cs $SUPERVISED_CONFIG -ws $SUPERVISED_WEIGHTS
conda deactivate



