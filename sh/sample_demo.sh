#!/bin/bash
#SBATCH --job-name=bash          # avoid lightning auto-debug configuration
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=6      # shall >#GPU to avoid overtime thread distribution 
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem=128GB               
#SBATCH --time 02:59:59           # time limit (D-HH:MM:ss)
#SBATCH --gres=gpu:4
#SBATCH --constraint='80gb'       # 80GB GPU
#SBATCH --partition=short-unkillable

#########################
####### Configs #########
#########################
CONDA_ENV_NAME=py311
CONDA_HOME=$(expr match $CONDA_PREFIX '\(.*miniconda\)')

source $CONDA_HOME/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}
module load cuda/12.1.1 # for deepspeed compilation

dt=$(date '+%d/%m/%Y-%H:%M:%S')
echo "[$0] | Start: ${dt}"

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export CUTLASS_PATH="./cutlass"

# checkpoint_path="~/scratch/data/pdb/protenix/model_v0.2.0.pt"
# run_name="pretrain"
checkpoint_path="./release.pt"
run_name="eba_release"


# 36min | 4 GPU
# torchrun --nproc_per_node=4 ./runner/train.py \
python ./runner/train.py \
--run_name ${run_name} \
--seed 42 \
--base_dir ./outputs/inference \
--dtype bf16 \
--project protenix \
--use_wandb false \
--diffusion_batch_size 8 \
--eval_interval 400 \
--log_interval 50 \
--ema_decay 0.999 \
--max_steps 100000 \
--sample_diffusion.noise_scale_lambda 1.75 \
--sample_diffusion.step_scale_eta 1.25 \
--sample_diffusion.N_step 20 \
--sample_diffusion.N_sample 250 \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--model.N_cycle 4 \
--data.train_sets example \
--data.test_sets example_test \
--predict_only true 
