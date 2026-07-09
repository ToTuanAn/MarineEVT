export CUDA_VISIBLE_DEVICES=0,1,2
export DATA_DIR='/project/marieninst/an/marineevt'

export WG_BACKEND="ray"
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_gsc_rpc_server_reconnect_timeout_s=100
export BASE_DIR=$(pwd)

WAND_PROJECT="EVT-R1"
RAY_DASHBOARD_ADDRESS="http://127.0.0.1:8266" # your head node address
N_NODES=1

n_gpus_per_node=3
train_batch_size=$[$n_gpus_per_node * 2]
val_batch_size=$[$n_gpus_per_node * 1]
actor_ppo_mini_batch_size=$[$n_gpus_per_node * 8]
actor_ppo_micro_batch_size=$[$n_gpus_per_node * 4]
log_prob_micro_batch_size=$[$n_gpus_per_node * 4]

export BASE_MODEL='Qwen/Qwen3-VL-2B-Instruct'
export EXPERIMENT_NAME=evt-r1-qwen3vl-2b

ulimit -n 65535

ray job submit --address=$RAY_DASHBOARD_ADDRESS \
    --runtime-env=verl/trainer/runtime_env.yaml \
    -- \
    python3 -m verl.trainer.main_ppo_format \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=250000 \
    data.max_response_length=10000 \
    data.max_start_length=100000 \
    data.max_obs_length=100000 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.policy_loss=grpo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.5 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$actor_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$actor_ppo_micro_batch_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    +actor_rollout_ref.rollout.min_pixel_tokens=4 \
    +actor_rollout_ref.rollout.max_pixel_tokens=128 \
    +actor_rollout_ref.rollout.disable_log_stats=true \
    +actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.rollout.n_agent=4 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger="['wandb', 'console']" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=60 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.total_training_steps=180 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${BASE_DIR}/verl_checkpoints/$EXPERIMENT_NAME \
    reward_model.structure_format_score=0.5 \
    reward_model.final_answer_score=0.5 \
    reward_model.tool_name_score=0.5 \
    reward_model.tool_output_score=0.5 \
    do_search=true \
    max_turns=3 \
    tool.url="http://127.0.0.1:8111/sam" \
    tool.emb="http://127.0.0.1:8111/emb" \
    tool.topk=3 \
    2>&1 | tee verl_log/$EXPERIMENT_NAME.log