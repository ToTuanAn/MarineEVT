# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_em_format, qa_f1_format
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import json
import requests

def calculate_tool_correct(response, groundtruth):
    for tool in response:
        if tool not in groundtruth:
            return 0
    return 1

def calculate_tool_step_correct(response, groundtruth):
    idx = 0
    while idx < len(response) and idx < len(groundtruth):
        if response[idx] != groundtruth[idx]:
            return 0
        idx += 1
    return 1

def calculate_correct_temporal_IoU(response, groundtruth):
    s1, e1 = int(response[0]), int(response[1])
    s2, e2 = int(groundtruth[0]), int(groundtruth[1])
    if s1 >= e1 or s2 >= e2:
        return 0

    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter

    if inter / union >= 0.5:
        return 1
    return 0

def calculate_correct_spatial_IoU(response, groundtruth):
    def parse_box(raw_box):
        """Safely extract & normalize [x1, y1, x2, y2] from nested list/strings."""
        if not raw_box or len(raw_box) == 0:
            return None
        try:
            coords = [float(c) for c in raw_box[0]]
            x1, y1, x2, y2 = coords
            # Normalize to (min_x, min_y, max_x, max_y)
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        except (ValueError, IndexError):
            return None
    
    def compute_iou(box1, box2):
        """Calculate 2D Intersection over Union for (x1, y1, x2, y2) tuples."""
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    matches = []
    boxes1 = [parse_box(b) for b in response if b != []]
    boxes2 = [parse_box(b) for b in groundtruth if b != []]
    
    for i, b1 in enumerate(boxes1):
        if b1 is None: continue
        for j, b2 in enumerate(boxes2):
            if b2 is None: continue
            iou = compute_iou(b1, b2)
            if iou > 0.5:
                matches.append({"idx_1": i, "idx_2": j, "iou": round(iou, 4)})
    
    if len(matches) >= len(boxes1):
        return 1

    return 0


def parse_tool_blocks(raw_string: str, ordered: bool = True) -> dict:
    # decoded = html.unescape(raw_string)

    # Single regex to capture both tag types in order of appearance
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.finditer(pattern, str(raw_string), re.DOTALL)

    call_str = matches.group(1)
    call_json = json.loads(call_str)

    return call_json["function"]["name"], call_json["function"]["arguments"]

def compute_score_fn(think_str,
                     solution_str, 
                     ground_truth, 
                     structure_format_score,
                     final_answer_score, 
                     tool_name_score,
                     tool_output_score,
                     question_format,
                     image_paths,
                     sam3_url):
    
    def is_valid_sequence(think_str, solution_str):
        has_reason = "<reason>" in think_str and "</reason>" in think_str

        has_tool_call = "<tool_call>" in solution_str and "</tool_call>" in solution_str
        has_answer = "<answer>" in solution_str and "</answer>" in solution_str
    
        if has_reason and has_tool_call:
            return True, "tool"
        
        if has_reason and has_answer:
            return True, "answer"
            
        return False, None

    def extract_solution(solution_str):
        """Extract the equation from the solution string."""

        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.finditer(answer_pattern, solution_str, re.DOTALL)
        matches = list(match)
        
        # If there are 0 or exactly 1 matches, return None
        if len(matches) == 0:
            return None
        
        # If there are 2 or more matches, return the last one
        return matches[-1].group(1).strip()
    
    def is_correct_answer(response, ground_truth_response, question_format):
        return 1

    is_valid_format, response_typing = is_valid_sequence(think_str, solution_str)
    ground_truth_typing, ground_truth_tool_name, ground_truth_response = ground_truth["type"], ground_truth["tool_name"], ground_truth["output_target"]
    
    response_tool_name, response = None, None

    if response_typing == "tool":
        print("Solution str: ", solution_str)
        response_tool_name, response_params = parse_tool_blocks(solution_str)
        if response_tool_name != ground_truth_tool_name:
            return 0
        else:
            if response_tool_name == "temporal_grounding":
                if calculate_correct_temporal_IoU(response=[(lambda m, s: int(m) * 60 + int(s))(*response_params["start_time"].split(':')),
                                                            (lambda m, s: int(m) * 60 + int(s))(*response_params["end_time"].split(':'))],
                                                  groundtruth=[ground_truth_response["start_time"], ground_truth_response["end_time"]]):
                    return tool_output_score + tool_name_score
                else:
                    return tool_name_score
            elif response_tool_name == "spatial_grounding":
                result = None

                try:
                    # 3. Make the POST request
                    response = requests.post(sam3_url, json={
                        "prompt": response_params["prompt"],
                        "ground_type": response_params.get("ground_type", None),
                        "image_paths": image_paths
                    })
                    
                    # 4. Check for HTTP errors
                    response.raise_for_status()
                    
                    # 5. Parse and print the response
                    result = response.json()
                    print("Success!")
                    print(json.dumps(result, indent=4))

                except requests.exceptions.HTTPError as http_err:
                    print(http_err)
                
                if result is None:
                    return tool_name_score
            
                if calculate_correct_spatial_IoU(response=result["result"]["boxes"],
                                                 groundtruth=response_params["boxes"]):
                    return tool_output_score + tool_name_score
                else:
                    return tool_name_score
            else:
                return 0
    else:
        response = extract_solution(solution_str=solution_str)
        is_correct = is_correct_answer(response, ground_truth_response, question_format)

    if response_typing != ground_truth_typing:
        return 0
    else:
        if ground_truth_typing == "tool":
            return 1
        elif ground_truth_typing == "answer":
            if is_valid_format:
                if is_correct:
                    return structure_format_score + final_answer_score
                else:
                    return structure_format_score
            else:
                return 0
        else:
            return 0


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, sam3_url, structure_format_score=0., final_answer_score=0., tool_name_score=0., tool_output_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.sam3_url = sam3_url
        self.final_answer_score = final_answer_score
        self.structure_format_score = structure_format_score
        self.tool_name_score = tool_name_score
        self.tool_output_score = tool_output_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        print("Calculate reward")
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            data_item = data[i]
            input_ids = data_item.batch['input_ids']

            prompt_length = input_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            response_str = data_item.non_tensor_batch["responses_str"]
            think_str = data_item.non_tensor_batch["thinks_str"]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            question_format = data_item.non_tensor_batch['question_format']
            image_paths = data_item.non_tensor_batch['images']

            score = compute_score_fn(think_str=think_str,
                                     solution_str=response_str, 
                                     ground_truth=ground_truth, 
                                     structure_format_score=self.structure_format_score, 
                                     final_answer_score=self.final_answer_score, 
                                     tool_name_score=self.tool_name_score,
                                     tool_output_score=self.tool_output_score,
                                     question_format=question_format,
                                     image_paths=image_paths,
                                     sam3_url=self.sam3_url)

            reward_tensor[i, valid_response_length - 1] = score
        
        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        print("Ray is not initialized! run ray.init()...")
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    print("Ray already initialized. Get remote ...")
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              sam3_url=config.tool.url,
                              structure_format_score=config.reward_model.structure_format_score, 
                              final_answer_score=config.reward_model.final_answer_score,
                              tool_name_score=config.reward_model.tool_name_score,
                              tool_output_score=config.reward_model.tool_output_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, sam3_url=config.tool.url)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    print(f"=================init_workers================")
    trainer.init_workers()
    print(f"=================fit================")
    trainer.fit()


if __name__ == '__main__':
    main()
