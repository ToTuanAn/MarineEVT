from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
import re 
import os
import glob
import os
import json
from PIL import Image

from evt_r1.tools.call_sam3 import call_sam
from evt_r1.tools.call_time import call_video_sampling
from evt_r1.utils.image_processing import smart_resize, process_image, MAX_LENGTH

from scripts.data_process.marineevt_test_merge import make_prefix

def evt_r1_loop(question, image_frames, model, tokenizer):
    previous_name, previous_tool = None, None
    messages = []
    raw_response, response = None, None

    for i in range(5):
        image_frames, images = process_image(image_frames, previous_name, previous_tool)

        for i, image in enumerate(images):
            width, height = image.size
            resize_h, resize_w = smart_resize(height=height, width=width, factor=2, min_pixels=256**2, max_pixels=512**2)
            images[i] = image.resize((resize_w, resize_h), Image.Resampling.LANCZOS)

        messages.append({"role": "user", "content": [{"type": "text", "text": question}]})
        assert messages[-1]["role"] == "user"

        messages[0] = {"role": "user", "content": [{"type": "text", "text": question}]}
        content = messages[0]["content"]
        for j in range(len(images)):
            content.insert(0, {"type": "image", "image": images[j], "max_pixels": 512 * 512})

        prompts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,        # ← critical
        )
                
        inputs = tokenizer(
            text=prompts,
            images=images,
            add_special_tokens = False,
            return_tensors = "pt",
            truncation=False,
            max_length=MAX_LENGTH,  
            padding="longest",
        ).to("cuda")

        outputs = model.generate(**inputs, max_new_tokens = 2048,
            use_cache = True, temperature = 1.0, min_p = 0.1)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = output_text[output_text.find("assistant"):]
        output_text = output_text.replace("\n", "").replace("assistant", "")

        if "<answer>" in output_text:
            raw_response = output_text
            print(f"Response: {raw_response}")
            match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
            
            if match:
                # group(1) is already stripped of surrounding whitespace by the \s* in regex
                response = match.group(1) 
                print(f"Response: {output_text}") # Optional: if you still want to see the raw text
                print(f"Processed Answer: {response}")
                
                break # Only break when we have successfully extracted the complete answer
            else:
                # If you are in a loop waiting for the model to finish generating:
                # print("Waiting for complete </answer> tag...")
                pass 

        elif "<tool_call>" in output_text:
            match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', output_text, re.DOTALL)
            if match:
                inner_content = match.group(1).strip()
                print(f"Processed Tool JSON {inner_content}")

                try:
                    json_content = json.loads(inner_content)
                except Exception as e:
                    print(f"Exception: {inner_content}")
                    continue 

                if json_content["function"]["name"] == "temporal_grounding":
                    previous_name = "temporal_grounding"
                    previous_tool = call_video_sampling(json_content)
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": ""}], "tool_calls": [json_content]})
                    messages.append({
                        "role": "tool",
                        "content": [{"type": "text", "name": "temporal_grounding", "text": ""}]
                    })
                elif json_content["function"]["name"] == "spatial_grounding":
                    previous_name = "spatial_grounding"
                    previous_tool = call_sam(json_content, image_frames)
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": ""}], "tool_calls": [json_content]})
                    messages.append({
                        "role": "tool",
                        "content": [{"type": "text", "name": "spatial_grounding", "text": ""}]
                    })
            else:
                print("No tool_call tag found")
        else:
            continue
    
    return response, raw_response

if __name__ == "__main__":
    TEST_DATA_PATH = "/project/marieninst/an/marineevt/test"
    RESULT_PATH = "/home/tato/MarineEVT/output"
    MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"

    model = AutoModelForVision2Seq.from_pretrained(
        "/home/tato/MarineEVT/merged_hf_model",
        torch_dtype=torch.float16,  # or torch.bfloat16 depending on your hardware
        device_map="auto",
        trust_remote_code=True      # Highly recommended for Qwen models
    )

    tokenizer = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    for dimension in glob.glob(f"{TEST_DATA_PATH}/*"):
        dimension_name = dimension.split("/")[-1]
        for subdimension in glob.glob(f"{dimension}/*"):
            subdimension_name = subdimension.split("/")[-1]

            if not os.path.exists(f"{subdimension}/multi_turn_data_ver2.json"):
                continue
            
            if os.path.exists(f"{RESULT_PATH}/{dimension_name}/{subdimension_name}/response.json"):
                continue

            with open(f"{subdimension}/multi_turn_data_ver2.json") as f:
                data = json.load(f)

            for item in data["data"]:
                previous_name, previous_tool = None, None
                video_id = item["video_id"]
      
                image_frames = sorted(glob.glob(f"{subdimension}/videos/{video_id}/frames/*"))

                prompt = make_prefix(item["question"], template_type="vqa")
                response, raw_response = evt_r1_loop(prompt, image_frames, model, tokenizer)

                item["response"] = response
                item["raw_response"] = raw_response
            
            os.makedirs(f"{RESULT_PATH}/{dimension_name}", exist_ok=True)
            os.makedirs(f"{RESULT_PATH}/{dimension_name}/{subdimension_name}", exist_ok=True)

            with open(f"{RESULT_PATH}/{dimension_name}/{subdimension_name}/response.json", "w") as f:
                json.dump(data, f, indent=4) 