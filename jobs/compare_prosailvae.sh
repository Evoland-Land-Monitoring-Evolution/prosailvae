#PBS -N comparepvae
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export RSR_DIR=/work/scratch/zerahy/prosailvae/data/
export DATADIR=/work/scratch/zerahy/prosailvae/data/curated_europe_theia/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/pvae_all_others/52507020_pvae_b0/compare_perfs/
export CONFIG=/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/model_compare_dict.json

python ${VAEDIR}/metrics/comparative_results.py -d ${DATADIR} -r $OUTDIR -rsr $RSR_DIR -m $CONFIG 
conda deactivate

