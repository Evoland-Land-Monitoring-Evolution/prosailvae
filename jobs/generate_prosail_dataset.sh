#PBS -N genprosail
#PBS -l select=1:ncpus=4:mem=92G
#PBS -l walltime=01:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/scratch/zerahy/prosailvae/data/1e5_simulated_full_bands_new_dist_2_corr_type_2/
export RSR_DIR=/work/scratch/zerahy/prosailvae/data/
export NOISE=0.01

python ${VAEDIR}/dataset/generate_dataset.py -d ${DATADIR} -n 10000 -p test_ -s $NOISE -rsr $RSR_DIR -dt new_v2 -m v2
python ${VAEDIR}/dataset/generate_dataset.py -d ${DATADIR} -n 100000 -p train_ -s $NOISE -rsr $RSR_DIR -dt new_v2 -m v2


conda deactivate
