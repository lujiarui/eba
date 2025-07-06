#!/bin/bash
#SBATCH --job-name=bash          # avoid lightning auto-debug configuration
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=6      # shall >#GPU to avoid overtime thread distribution 
#SBATCH --cpus-per-task=4        # number of OpenMP threads per MPI process
#SBATCH --mem=192GB               
#SBATCH --time 47:59:59           # time limit (D-HH:MM:ss)
#SBATCH --gres=gpu:4
#SBATCH --constraint='80gb'       # 80GB GPU

#########################
####### Configs #########
#########################
CONDA_ENV_NAME=py311
CONDA_HOME=$(expr match $CONDA_PREFIX '\(.*miniconda\)')
WORKDIR=$(pwd)

#########################
####### Env loader ######
#########################
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}
module load cuda/12.1.1 # for deepspeed compilation

dt=$(date '+%d/%m/%Y-%H:%M:%S')
echo "[$0] | Start: ${dt}"

export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export CUTLASS_PATH="./cutlass"

checkpoint_path="./release.pt"
T=5000
k=2

python ./runner/train.py \
--run_name example_EBA \
--seed 42 \
--base_dir ./outputs/EBA \
--dtype bf16 \
--project af3_atlas_eba \
--use_wandb false \
--diffusion_batch_size 8 \
--diffusion_chunk_size None \
--blocks_per_ckpt 1 \
--eval_interval 2000 \
--log_interval 1 \
--checkpoint_interval 500 \
--ema_decay 0.999 \
--max_steps 100000 \
--warmup_steps 100 \
--lr 0.0000001 \
--iters_to_accumulate 4 \
--sample_diffusion.N_step 20 \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.train_sets example_eba \
--data.test_sets example_test \
--skip_confidence_and_permutation true \
--dpo_training true \
--data.atlas_train_eba.base_info.retrieval_k ${k} \
--loss.diffusion.beta_dpo ${T} \
--loss.diffusion.dpo_enabled true \
--loss.diffusion.linear_ew false \
--loss.diffusion.eba_enabled true \
--loss.diffusion.eba_with_ref false \
--loss.diffusion.norm_by_len true