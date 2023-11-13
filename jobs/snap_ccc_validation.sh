#PBS -N snapccc
#PBS -q qgpgpudev
#PBS -l select=1:ncpus=4:mem=60G:ngpus=1  
#PBS -l walltime=12:00:00

module load conda
export LD_LIBRARY_PATH=/work/scratch/zerahy/dotconda/prosailvae/lib:$LD_LIBRAY_PATH
conda activate /work/scratch/zerahy/dotconda/prosailvae
export SRCDIR=/home/uz/zerahy/projects/    
export VAEDIR=${SRCDIR}/prosailvae/prosailvae/

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_snap_ccc_validation_jordi
#python ${VAEDIR}/snap_regression/validate_snap_on_weiss.py -r $OUTDIR 

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_snap_lai_validation_jordi
#python ${VAEDIR}/snap_regression/validate_snap_on_weiss.py -r $OUTDIR -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_snap_ccc_validation_jordi_projected
#python ${VAEDIR}/snap_regression/validate_snap_on_weiss.py -r $OUTDIR -p True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_snap_lai_validation_jordi_projected
#python ${VAEDIR}/snap_regression/validate_snap_on_weiss.py -r $OUTDIR -p True -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_snap_lai_validation_new_sim_v2_corr_type_1
#export DATADIR=/work/scratch/zerahy/prosailvae/data/1e5_simulated_full_bands_new_dist_corr_type_1/
#python ${VAEDIR}/snap_regression/validate_snap_on_weiss.py -r $OUTDIR -sd True -d $DATADIR -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_snap_ccc_validation_new_sim_v2_corr_type_1
#export DATADIR=/work/scratch/zerahy/prosailvae/data/1e5_simulated_full_bands_new_dist_corr_type_1/
#python ${VAEDIR}/snap_regression/validate_snap_on_weiss.py -r $OUTDIR -sd True -d $DATADIR

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_snap_lai_validation_new_sim_v2_corr_type_2
#export DATADIR=/work/scratch/zerahy/prosailvae/data/1e5_simulated_full_bands_new_dist_corr_type_2/
#python ${VAEDIR}/snap_regression/validate_snap_on_weiss.py -r $OUTDIR -sd True -d $DATADIR -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/${PBS_JOBID:0:8}_snap_ccc_validation_new_sim_v2_corr_type_2
#export DATADIR=/work/scratch/zerahy/prosailvae/data/1e5_simulated_full_bands_new_dist_corr_type_2/
#python ${VAEDIR}/snap_regression/validate_snap_on_weiss.py -r $OUTDIR -sd True -d $DATADIR 


############
#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_ccc_validation_new_sim_v2_corr_type_2_prospect_5
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECT5_corr_v2_

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_lai_validation_new_sim_v2_corr_type_2_prospect_5
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECT5_corr_v2_ -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_ccc_validation_new_sim_v2_corr_type_1_prospect_5
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECT5_corr_v1_

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_lai_validation_new_sim_v2_corr_type_1_prospect_5
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECT5_corr_v1_ -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_ccc_validation_new_sim_v2_corr_type_2_prospect_D
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECTD_corr_v2_

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_lai_validation_new_sim_v2_corr_type_2_prospect_D
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECTD_corr_v2_ -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_ccc_validation_new_sim_v2_corr_type_1_prospect_D
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECTD_corr_v1_

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_lai_validation_new_sim_v2_corr_type_1_prospect_D
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECTD_corr_v1_ -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_ccc_validation_new_sim_v2_corr_type_2_prospect_PRO
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECTPRO_corr_v2_

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_lai_validation_new_sim_v2_corr_type_2_prospect_PRO
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECTPRO_corr_v2_ -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_ccc_validation_new_sim_v2_corr_type_1_prospect_PRO
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECTPRO_corr_v1_

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/model_corr_bvnet/${PBS_JOBID:0:8}_snap_lai_validation_new_sim_v2_corr_type_1_prospect_PRO
#export DATADIR=/work/scratch/zerahy/prosailvae/data/simulated_data_set_various_prospect_models_and_corr/
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -sd True -d $DATADIR -fp bvnet_dataset_PROSPECTPRO_corr_v1_ -l True

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/bvnet_regression/${PBS_JOBID:0:8}_snap_ccc_validation_jordi
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR 

#export OUTDIR=/work/scratch/zerahy/prosailvae/results/bvnet_regression/${PBS_JOBID:0:8}_snap_lai_validation_jordi
#python ${VAEDIR}/bvnet_regression/validate_bvnet_on_sim_data.py -r $OUTDIR -l True


conda deactivate
