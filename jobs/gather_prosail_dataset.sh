#PBS -N nllsimvaebatch
#PBS -l select=1:ncpus=4:mem=92G
#PBS -l walltime=02:00:00

export PATH="/work/scratch/zerahy/miniconda3/bin:$PATH"
module load conda
conda activate /work/scratch/zerahy/dotconda/envs/vae_test
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/scratch/zerahy/prosailvae/data/1e7_dataset/

python ${VAEDIR}/dataset/gather_sub_datasets.py -d ${DATADIR} 
conda deactivate


