LOCAL_DIR=verl_checkpoints/evt-r1-qwen3vl-8b/actor/global_step_60
TARGET_DIR=./merged_hf_model

python scripts/model_merger.py merge --backend fsdp --local_dir $LOCAL_DIR --target_dir $TARGET_DIR