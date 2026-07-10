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
Preprocess the QA dataset to parquet format
"""

import re
import os
import datasets
import json
import random
import glob

from verl.utils.hdfs_io import copy, makedirs
import argparse

from PIL import Image, ImageDraw


def make_prefix(dp, template_type):
    question = dp

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'vqa':
        # """This works for any base model"""
        prefix = f"""
/* Task Description */ 
You are given a question and a video containing $$N$$ frames. Your goal is to answer the question by selecting appropriate tools to assist in modifying the video to ground important visual cues.

/* Decision Protocol */ 
After receiving the video and tool calling history information, you sequentially perform two steps: 
- Step 1: Start by documenting your thought process wrapped inside <reason></reason> tags. Analyze the frames and information to decide whether to call a tool or deliver a direct answer. Do not skip this step; it requires careful and deliberate evaluation.
- Step 2:
    Option 1: Invoke a tool 
        - Use this when you need more context to answer, wrapped inside <tool_call></tool_call> tags.
        - Output one <tool_call> action to temporally or spatially ground the visuals:
            Temporal grounding of the video:  <tool_call>""" +'{"function": {"name": "temporal_grounding", "arguments": {"start_time": "(specify starting time in "MM:SS" format)", "end_time": "(specify ending time in "MM:SS" format)"}}}'+"""</tool_call>
            Spatial grounding of the video: <tool_call>"""+'{"function": {"name": "spatial_grounding", "arguments": {"prompt": "(specify a text description for the spatial grounding area)", "ground_type": "(choose one: highest/multiple)"}}}'+"""</tool_call>
    Option 2: Answer (direct answer)
        - Use this ONLY when you have sufficient information or at turn 3.
        - Output the final answer wrapped inside <answer></answer> tags.
        - Once <answer> is output, the task ends immediately.

/* Example Output */
    Example 1:
        <reason>The question asks for the specific time when the diver first makes contact with the sea turtle. I need to locate the exact timeframe of this interaction in the video to provide an accurate answer. Therefore, I will invoke the temporal grounding tool to extract the relevant clip.</reason><tool_call>{"function": {"name": "temporal_grounding", "arguments": {"start_time": "00:10", "end_time": "00:12"}}}</tool_call>
    Example 2:
        <reason>The question asks to identify the specific area in the frame where the interaction between the marine life and the equipment is occurring. I need to visually locate this region to understand the context. Therefore, I will invoke the spatial grounding tool to highlight the area containing the diver and the turtle, requesting the highest confidence bounding box.</reason><tool_call>{"function": {"name": "spatial_grounding", "arguments": {"prompt": "A region of a diver interacting with a hawksbill sea turtle underwater", "ground_type": "highest"}}}</tool_call>
    Example 3:
        <reason>The question asks about the current state of the ship based on the provided frames. I can clearly see from the visual cues that the mooring lines are attached to the pier and the ship is stationary. I have sufficient information to answer the question directly without needing to extract further temporal or spatial details. Therefore, I will deliver a direct answer.</reason><answer>A. The ship is secured to the dock</answer>
"""+f"""\n/* Information */
    Turn 1, Question: {question}
"""   
    else:
        raise NotImplementedError
    return prefix

def format_time(total_seconds):
    minutes, seconds = divmod(int(total_seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"

def build_history(turn_data, episode_turns):
    """Build dialogue history up to current turn (text-only after turn 1)"""
    history = []
    for t in episode_turns:
        if t["turn_id"] > turn_data["turn_id"]:
            break
        # User message
        if t["turn_id"] == turn_data["turn_id"]:

            if t["turn_id"] == 1:
                content = [{"type": "text", "text": make_prefix(t["user_query"], "vqa")}]
                history.append({"role": "user", "content": content})
            else:
                content = [{"type": "text", "text": f'Turn {t["turn_id"]}:, question: ' + t["user_query"]}]
                history.append({"role": "user", "content": content})

        else:

            if t["turn_id"] == 1:
                history.append({"role": "user", "content": [{"type": "text", "text": make_prefix(t["user_query"], "vqa")}]})
            else:
                content = [{"type": "text", "text": f'Turn {t["turn_id"]}:, question: ' + t["user_query"]}]
                history.append({"role": "user", "content": content})

            if "tool_output" in t:
                assistance_response = json.loads(t["assistant_response"])
                tool_name = assistance_response[0]["function"]["name"]

                if tool_name == "video_sampling":
                    assistance_response[0]["function"]["name"] = "temporal_grounding"
                
                if tool_name == "object_grounding":
                    assistance_response[0]["function"]["name"] = "spatial_grounding"

                history.append({"role": "assistant", "content": [{"type": "text", "text": ""}], "tool_calls": assistance_response})
                # print(json.loads(t["assistant_response"]))
                history.append({
                    "role": "tool",
                    "content": [{"type": "text", "name": assistance_response[0]["function"]["name"], "text": ""}]
                })
            else:
                history.append({"role": "assistant", "content": [{"type": "text", "text": t["assistant_response"]}]})


    return history

def process_groudntruth(turn_data, frames):
    typing, tool_name, response = None, None, None

    if "<answer>" not in turn_data["assistant_response"]:
        typing = "tool"
        assistance_response = json.loads(turn_data["assistant_response"])
        tool_name = assistance_response[0]["function"]["name"]

        if tool_name == "video_sampling":
            tool_name = "temporal_grounding"
            response =  json.dumps({"start_time": f'{assistance_response[0]["function"]["arguments"]["start_time"]}', "end_time": f'{assistance_response[0]["function"]["arguments"]["end_time"]}'})
        
        if tool_name == "object_grounding":
            tool_name = "spatial_grounding"
            response = turn_data["tool_output"]

            # tool_content_json = json.loads(turn_data["tool_output"])
            # tool_boundingbox = tool_content_json["boxes"]
            # for idx in range(len(frames)):
            #     if len(tool_boundingbox[idx]) == 0:
            #         continue 
                
            #     try:
            #         for jdx in range(len(tool_boundingbox[idx])):
            #             bbox = [float(value) for value in tool_boundingbox[idx][jdx]]
            #             draw = ImageDraw.Draw(frames[idx])
            #             draw.rectangle(bbox, outline="yellow", width=20)
            #     except:
            #         for jdx in range(len(tool_boundingbox[idx][0])):
            #             try:
            #                 bbox = [float(value) for value in tool_boundingbox[idx][0][jdx]]
            #                 draw = ImageDraw.Draw(frames[idx])
            #                 draw.rectangle(bbox, outline="yellow", width=20)
            #             except:
            #                 import ast 
            #                 bbox = [float(value) for value in ast.literal_eval(tool_boundingbox[idx][0][jdx])]
            #                 draw = ImageDraw.Draw(frames[idx])
            #                 draw.rectangle(bbox, outline="yellow", width=20)
        
    else:
        typing = "answer"
        assistance_response = turn_data["assistant_response"]
        tool_name = None
        response = turn_data["assistant_response"]
    

    return {
        "type": typing,
        "tool_name": tool_name ,
        "output_target": response,
    }


def process_response(response):
    if "<answer>" in response:
        return response
    else:
        response = response[1:-1]
        response = response.replace("object_grounding", "spatial_grounding")
        response = response.replace("video_sampling", "temporal_grounding")
        return "<tool_call>" + response + "</tool_call>"

import base64
from io import BytesIO

def pil_to_base64(img):
    buffered = BytesIO()
    
    # Auto-detect format; default to PNG if format is unknown
    img_format = img.format if img.format else "PNG"
    
    # Save the image to the byte buffer
    img.save(buffered, format=img_format)
    
    # Encode the bytes to base64 and decode to a string
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Format as a standard Data URI
    return f"data:image/{img_format.lower()};base64,{img_str}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--data_sources', default='nq')

    args = parser.parse_args()

    data_sources = args.data_sources.split(',')
    local_dir = args.local_dir
    template_type = args.template_type

    all_dataset = []

    for data_source in data_sources:
        for dimension_path in glob.glob(f"{local_dir}/{data_source}/*"):
            dimension_name = dimension_path.split("/")[-1]
            for subdimension_path in glob.glob(f"{dimension_path}/*"):
                subdimension_name = subdimension_path.split("/")[-1]

                if not os.path.exists(f"{subdimension_path}/multi_turn_data_ver2.json"):
                    continue
                
                with open(f"{subdimension_path}/multi_turn_data_ver2.json", "r") as f:
                    json_data = json.load(f)
                
                cnt = 0
                
                for item in json_data["data"]:

                    video_id = item["video_id"]
                    train_dataset = []
                    
                    for idx, turn in enumerate(item["turns"]):

                        history = build_history(turn, item["turns"])
                        frames = []
                        frames_path = []

                        frame_folder = f"{local_dir}/{data_source}/{dimension_name}/{subdimension_name}/videos/{video_id}"

                        for frame_path in turn["visual_input"]:
                            # print(os.path.join(frame_folder, frame_path))
                            if os.path.exists(os.path.join(frame_folder, frame_path)):
                                frames_path.append(os.path.join(frame_folder, frame_path))
                                image = Image.open(os.path.join(frame_folder, frame_path)).convert("RGB")
                                frames.append(image)
                        
                        groundtruth = process_groudntruth(turn, frames)
                        procesed_response = process_response(turn["assistant_response"])

                        if len(frames_path) == 0:
                            continue

                        data = {
                            "data_source": data_source,
                            "images": frames_path,
                            "prompt": history,
                            "response": procesed_response,
                            "raw_question": item["question"],
                            "reward_model": {
                                "style": "rule",
                                "ground_truth": groundtruth
                            },
                            "question_format": item["question_format"],
                            "extra_info": {
                                'split': "train",
                                "dimension": item["dimension"],
                                "subdimension": item["subdimension"],
                                "question_task": item["question_task"],
                                "question_format": item["question_format"],
                                'index': f"{data_source}#{item['dimension']}#{item['subdimension']}#{cnt:06d}",
                            }
                        }

                        cnt += 1
                        # print(data)
                        train_dataset.append(data)
                    #     break
                    # break
                    all_dataset.extend(train_dataset)

    random.shuffle(all_dataset)

    print("Total dataset len: ", len(all_dataset))
    hdfs_dir = args.hdfs_dir

    all_train_dataset = datasets.Dataset.from_list(all_dataset)
    all_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
