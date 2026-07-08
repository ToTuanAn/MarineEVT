import os

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from evt_r1.tools.sam3.sam3 import build_sam3_image_model
from evt_r1.tools.sam3.sam3.model.box_ops import box_xywh_to_cxcywh
from evt_r1.tools.sam3.sam3.model.sam3_image_processor import Sam3Processor
from evt_r1.tools.sam3.sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
import torch
from tqdm import tqdm
import glob 
import json
import time 
import litellm

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

ROOT_PATH = "/mnt/hdd/an/marinesci/"
SEG_PATH = "/mnt/hdd/an/marinesci/data/seg_objects"

def return_commonname(sp):
    prompt = """
    You are a marine biology expert. When given a species' scientific name, respond with only the common name of that species. 
    
    Output format
    Do not include any additional text, explanations, or formatting—just the common name.

    Input: Orcinus_orca 
    Output: Killer whale

    Input: Vincentia_novaehollandiae
    Output: New Holland seahorse
    """

    os.environ['DEEPSEEK_API_KEY'] = "sk-6173c85d2478459599dc4aadd9353d66"
    query = f"\nInput: {sp}\nOutput:"

    for _ in range(50):
        try:
            response = litellm.completion(
                                            model="deepseek/deepseek-chat",
                                            messages=[
                                                {"role": "system", "content": prompt},
                                                {"role": "user", "content": query}
                                            ]
                                        )
            response = response['choices'][0]['message']['content'].strip().lower()
            break
        except Exception as e:
            time.sleep(5)
            continue
    
    return response

with open("/mnt/hdd/an/marinesci/data/connected_annotations_v2.json", "r") as f:
    annotation_json = json.load(f)

os.makedirs(SEG_PATH, exist_ok=True)

def generate_sam_gif():
    bpe_path = f"./assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path="/mnt/hdd/an/marinesci/SAM-3/sam3.pt")
    processor = Sam3Processor(model, confidence_threshold=0.5)

    for json_data in tqdm(annotation_json["data"]):
        
        image_path = ROOT_PATH + json_data["image_path"][2:]
        image = Image.open(image_path)
        seg_folder = f"{SEG_PATH}/{image_path.split('/')[-1].replace('.jpg', '')}"

        if os.path.exists(seg_folder):
            continue
        
        os.makedirs(seg_folder, exist_ok=True)
        common_name_list = set()

        for anno in json_data["annotations"]:
            common_name = return_commonname(anno["species"])
            if common_name not in common_name_list:
                common_name_list.add(common_name)

            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=anno["species"])
            plot_results(image, inference_state, output_path=f"{seg_folder}/scientific_name_{anno['species']}.png", prompt=anno["species"])
        
        for common_name in list(common_name_list):
            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=common_name)
            plot_results(image, inference_state, output_path=f"{seg_folder}/common_name_{'_'.join(common_name.split(' '))}.png", prompt=common_name)

if __name__ == "__main__":
    generate_sam_gif()