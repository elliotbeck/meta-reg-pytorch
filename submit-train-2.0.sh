#!/bin/bash -l

bsub -J $JOB_NAME -n 1 -W $ttime -R "rusage[mem=$mem_per_gpu, ngpus_excl_p=$num_gpu_per_job]" -R "select[gpu_model0==GeForceRTX2080Ti]" \
python -W ignore main.py \
--batch_size ${BATCH_SIZE} \
--epochs_full ${EPOCHS_FULL} \
--epochs_metatrain ${EPOCHS_METATRAIN} \
--meta_train_steps ${META_TRAIN_STEPS} \
--hidden_dim ${HIDDEN_DIM} \
