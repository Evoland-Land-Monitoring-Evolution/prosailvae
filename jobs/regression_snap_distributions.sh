#PBS -N snapdist
#PBS -q qgpgpudev
#PBS -l select=1:ncpus=4:mem=12G:ngpus=1  
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/
export DATADIR=/work/scratch/zerahy/prosailvae/data/snap_distribution_dataset/
export OUTDIR=/work/scratch/zerahy/prosailvae/results/bvnet_dist_regression/${PBS_JOBID:0:8}_snap_regressions_new_sim_data_dist_with_validation/
export RSR_DIR=/work/scratch/zerahy/prosailvae/data/

CUDA_LAUNCH_BLOCKING=1 python ${VAEDIR}/bvnet_regression/bvnet_distribution_regression_simulated.py -d ${DATADIR} -r $OUTDIR -e 300 -lr 0.001 -n 10 -rsr $RSR_DIR -s True -v True -si 4
conda deactivate
