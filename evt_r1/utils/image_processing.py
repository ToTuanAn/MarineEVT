from typing import Tuple
import re 
import os
import glob
import os
from PIL import Image, ImageDraw
import math

MAX_LENGTH = 50000
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int, min_pixels = None, max_pixels= None) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    IMAGE_MAX_TOKEN_NUM = 256
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)
    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def process_image(image_frames, previous_name, previous_tool):
    if not previous_tool:
        return image_frames, [Image.open(img) for img in image_frames if os.path.exists(img)]
    else:
        if previous_name == "temporal_grounding":
            start_time, end_time = previous_tool.split("; ")
            new_image_frames = []
            for frame in image_frames:
                if int(start_time) <= int(frame.split("/")[-1].split("_")[1].split(".")[0]) <= int(end_time):
                    new_image_frames.append(frame)
            return new_image_frames, [Image.open(img) for img in new_image_frames if os.path.exists(img)]
        elif previous_name == "spatial_grounding":
            tool_content_json = previous_tool
            tool_boundingbox = tool_content_json["boxes"]
            frames = [Image.open(img) for img in image_frames if os.path.exists(img)]
            for idx in range(len(frames)):
                if len(tool_boundingbox[idx]) == 0:
                    continue

                for jdx in range(len(tool_boundingbox[idx])):
                    bbox = [float(value) for value in tool_boundingbox[idx][jdx]]
                    draw = ImageDraw.Draw(frames[idx])
                    draw.rectangle(bbox, outline="yellow", width=20)
            return image_frames, frames