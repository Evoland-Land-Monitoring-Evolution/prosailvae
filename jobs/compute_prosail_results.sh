#PBS -N resprosail
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
export OUTDIR=/work/scratch/zerahy/prosailvae/results/BIG_37797341_jobarray/240/3_kfold_3_supervised_True_full_/3_kfold_3_n_0_d2023_02_08_20_44_12_supervised_True_full_/
export TENSOR_DIR=/work/CESBIO/projects/MAESTRIA/prosail_validation/toyexample/torchfiles/


python ${VAEDIR}/metrics/results.py -d ${DATADIR} -r $OUTDIR -rsr $RSR_DIR -t $TENSOR_DIR -p True
conda deactivate



