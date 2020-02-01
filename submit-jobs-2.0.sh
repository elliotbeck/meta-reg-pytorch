#!/bin/sh

# Hard coded settings for resources
# time limit
export ttime=4:00
# number of gpus per job
export num_gpu_per_job=1
# memory per job
export mem_per_gpu=30000

export JOB_NAME='MetaReg_grid'

# load python
module load eth_proxy python_gpu/3.6.4
module load cuda/10.0.130
module load cudnn/7.6.4

for VAR_BATCH_SIZE in 16 24 
do
    export BATCH_SIZE=$VAR_BATCH_SIZE
    for  VAR_EPOCHS_FULL in 10 30 50 
    do
        export EPOCHS_FULL=$VAR_EPOCHS_FULL
        for  VAR_EPOCHS_METATRAIN in 10 30 50
        do
            export EPOCHS_METATRAIN=$VAR_EPOCHS_METATRAIN
            for  VAR_META_TRAIN_STEPS in 10 30 50 
            do
                export META_TRAIN_STEPS=$VAR_META_TRAIN_STEPS
                for  VAR_HIDDEN_DIM in 256 512 1024 
                do
                    export HIDDEN_DIM=$VAR_HIDDEN_DIM
                    sh submit-train-2.0.sh
                done
            done
        done
    done    
done

