import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
from evt_r1.tools.call_sam3 import call_sam
from evt_r1.tools.call_emb import call_text_emb_model

import torch.nn.functional as F

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class QueryRequest(BaseModel):
    prompt: str
    image_paths: List[str]
    ground_type: Optional[str] = None

class EmbRequest(BaseModel):
    answer: str
    groundtruth: str

app = FastAPI()

@app.post("/sam")
def sam_endpoint(request: QueryRequest):
    resp = call_sam(json_content={
        "prompt": request.prompt,
        "ground_type": request.ground_type
        }, 
        image_paths=request.image_paths)
    return {"result": resp}

@app.post("/emb")
def emb_endpoint(request: EmbRequest):
    answer_embeddings = call_text_emb_model(completions=request.answer)
    completion_embeddings = call_text_emb_model(completions=request.groundtruth)

    similarity_score = F.cosine_similarity(answer_embeddings.unsqueeze(0), completion_embeddings.unsqueeze(0)).tolist()[0]
    # print(f"Similarity score: {similarity_score}")

    del answer_embeddings
    del completion_embeddings
    # similarity_score = 0.9
    
    return {"result": {
        "similarity_score": similarity_score
    }}

if __name__ == "__main__":
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8111
    uvicorn.run(app, host="0.0.0.0", port=8111)