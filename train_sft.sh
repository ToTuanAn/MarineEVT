export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='/project/marieninst/an/marineevt'

export WG_BACKEND="ray"
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_gsc_rpc_server_reconnect_timeout_s=100
export BASE_DIR=$(pwd)

WAND_PROJECT="EVT-R1"
RAY_DASHBOARD_ADDRESS="http://127.0.0.1:8266" # your head node address
N_NODES=1

n_gpus_per_node=2
train_batch_size=$[$n_gpus_per_node * 2]
val_batch_size=$[$n_gpus_per_node * 1]

export BASE_MODEL='Qwen/Qwen3-VL-8B-Instruct'
export EXPERIMENT_NAME=evt-r1-qwen3vl-2b

ulimit -n 65535

# FIX: Wrap the command with torchrun
ray job submit --address=$RAY_DASHBOARD_ADDRESS \
    --runtime-env=verl/trainer/runtime_env.yaml \
    -- \
    python -m torch.distributed.run  --standalone --nnodes=1 --nproc_per_node=$n_gpus_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    model.partial_pretrain=$BASE_MODEL \
    model.trust_remote_code=true \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.max_length=40000 \
    +data.val_batch_size=$val_batch_size \
    trainer.logger="['wandb', 'console']" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    +trainer.save_freq=60 \
    +trainer.test_freq=60 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.total_training_steps=180 \
    trainer.default_local_dir=${BASE_DIR}/verl_checkpoints/$EXPERIMENT_NAME \
    2>&1 | tee verl_log/$EXPERIMENT_NAME.log