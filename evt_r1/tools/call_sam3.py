from evt_r1.tools.sam3.sam3 import build_sam3_image_model
from evt_r1.tools.sam3.sam3.model.box_ops import box_xywh_to_cxcywh
from evt_r1.tools.sam3.sam3.model.sam3_image_processor import Sam3Processor
from typing import Optional, Union, Tuple, List, Any, Dict
import re 
import glob
import os
import json
from PIL import Image, ImageDraw
import math
import torch
import requests

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

runtime_device="cuda"
bpe_path = f"/home/tato/MarineEVT/evt_r1/tools/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
sam_model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path="/home/tato/MarineEVT/model/SAM-3/sam3.pt",  device=runtime_device)

def call_sam(json_content, image_paths):
    prompts = []

    if "function" in json_content:
        prompts.append(json_content["function"]["arguments"]["prompt"])
        if "ground_type" in json_content["function"]["arguments"]:
            ground_type = json_content["function"]["arguments"]["ground_type"]
        else:
            ground_type = "highest"
    else:
        prompts.append(json_content["prompt"])
        if "ground_type" in json_content:
            ground_type = json_content["ground_type"]
        else:
            ground_type = "highest"

    if ground_type == "highest":
        sam_processor = Sam3Processor(sam_model, confidence_threshold=0.02, device=runtime_device)
    else:
        sam_processor = Sam3Processor(sam_model, confidence_threshold=0.1, device=runtime_device)

    bounding_box = {"boxes" : []}

    for image_path in image_paths:
        image = Image.open(image_path)
        bounding_boxes = []

        for prompt in prompts:
            current_boxes = []
            with torch.amp.autocast(device_type=runtime_device, enabled=True):
                inference_state = sam_processor.set_image(image)
                sam_processor.reset_all_prompts(inference_state)
                results = sam_processor.set_text_prompt(state=inference_state, prompt=prompt)
            
            current_score = 0
            scoring = [t.item() for t in results["scores"]] 

            nb_objects = len(results["scores"])
            print(f"Number of {prompt} detected: {nb_objects}")

            if ground_type == "highest":
                for i in range(nb_objects):
                    if scoring[i] > current_score:
                        float_boxes = [str(float(bb)) for bb in results["boxes"][i]]
                        current_boxes = [float_boxes]
                        current_score = scoring[i]
            else:
                for i in range(nb_objects):
                    float_boxes = [str(float(bb)) for bb in results["boxes"][i]]
                    current_boxes.append(float_boxes)
            
            bounding_boxes.extend(current_boxes)
            bounding_box["boxes"].append(bounding_boxes)
    return bounding_box

if __name__ == "__main__":
    pass