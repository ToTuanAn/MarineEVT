import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
from evt_r1.tools.call_sam3 import call_sam

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class QueryRequest(BaseModel):
    prompt: str
    image_paths: List[str]
    ground_type: Optional[str] = None


app = FastAPI()

@app.post("/sam")
def sam_endpoint(request: QueryRequest):
    resp = call_sam(json_content={
        "prompt": request.prompt,
        "ground_type": request.ground_type
        }, 
        image_paths=request.image_paths)
    return {"result": resp}


if __name__ == "__main__":
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8111
    uvicorn.run(app, host="0.0.0.0", port=8111)