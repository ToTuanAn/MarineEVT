# MarineEVT: Advancing Event-Centric Marine Video Understanding via Visual Tool Reasoning

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ECCV%202026-blue)](https://marineevt.hkustvgd.com/)
[![Website](https://img.shields.io/badge/Website-Project%20Page-green)](https://marineevt.hkustvgd.com/)
[![Dataset](https://img.shields.io/badge/Dataset-MarineEVT-orange)](https://marineevt.hkustvgd.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

**Official implementation of "MarineEVT: Advancing Event-Centric Marine Video Understanding via Visual Tool Reasoning"**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Dataset](#dataset)
- [Method: EVT-R1](#method-evt-r1)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## 📰 News
- **[June 2026]** 🎉 Our paper has been accepted to **ECCV 2026**!
- **[Coming Soon]** 🚀 Project website and code repository released.
- **[Coming Soon]** 📊 Full dataset and annotations will be released.
- **[Coming Soon]** 🤗 Pre-trained model checkpoints will be available on Hugging Face.

## 🌊 Overview

**MarineEVT** is a comprehensive, event-centric marine video dataset and benchmark integrating visual tool reasoning. It features **20K multi-task video visual question-answering pairs** that span **20 dimensions** of marine understanding and analysis.

In marine videos, informative events are often **sparse**, **ephemeral**, and **unevenly distributed**, posing significant challenges for existing Vision-Language Models (VLMs). MarineEVT addresses these challenges by providing a structured benchmark for evaluating semantic, contextual, spatial-temporal, and causal reasoning in underwater scenarios.

---

## ✨ Key Contributions

1. **MarineEVT Dataset**: A large-scale event-centric dataset comprising 20,000 richly annotated underwater video QA pairs spanning 20 fine-grained dimensions, including:
   - Marine species identification
   - Human activities recognition
   - Environmental conditions analysis
   - Behavioral interactions
   - Rare ecological events

2. **Five Reasoning Dimensions**:
   - **Semantic Reasoning**: Interpreting what events happened
   - **Contextual Reasoning**: Identifying which entities are present
   - **Spatial Reasoning**: Understanding temporal structures and when-dynamics
   - **Temporal Reasoning**: Localizing and grounding where entities are
   - **Causal Reasoning**: Inferring why certain events occur

3. **EVT-R1 Method**: A novel approach that decomposes event-centric marine video understanding into a multi-turn visual tool-integrated reasoning process with:
   - Turn-level reinforcement learning
   - Dual-component reward model (tool-reasoning reward + multi-task answer reward)
   - Effective intermediate tool-use decision learning

---

## 📊 Dataset

The MarineEVT dataset contains **20,000 video QA pairs** with annotations across 20 dimensions of marine understanding.

### Dataset Statistics
- **Total QA pairs**: 20,000
- **Evaluation set**: 2,000 pairs (reserved for testing)
- **Dimensions**: 20 fine-grained marine understanding categories
- **Reasoning types**: Semantic, Contextual, Spatial, Temporal, Causal

### Dataset Structure
```
MarineEVT/
├── CasualReasoning/
│   ├── Human-SpeciesCasualDynamics/
|       ├──train
|           ├──videos
|           └──multi_turn_data_ver2.json
|       ├──test
│   ├── Inter-SpeciesCausalDynamics/
│   └── ReasonInference/
├── SpatialReasoning/
└── README.md
```

### Download Dataset
The dataset will be available at: [https://marineevt.hkustvgd.com/](https://marineevt.hkustvgd.com/)

---

## 🚀 Method: EVT-R1

**EVT-R1** decomposes event-centric marine video understanding into a multi-turn visual tool-integrated reasoning process, leveraging powerful visual tools to localize and interpret critical information from redundant video frames with sparse and unevenly distributed events.

### Key Features
- **Multi-turn reasoning**: Breaks down complex video understanding into manageable steps
- **Visual tool integration**: Leverages external tools for localization and interpretation
- **Dual-component reward model**:
  - **Tool-reasoning reward (R_tool)**: Assesses validity and accuracy of tool invocations
  - **Multi-task answer reward (R_ans)**: Evaluates correctness of final responses
- **Turn-level RL**: Guides the model to produce correct answers while learning effective tool usage

### Architecture
```
Input Video → Multi-turn Reasoning → Visual Tool Invocation → Final Answer
                    ↓
            Turn-level RL Training
                    ↓
        Dual Reward (R_tool + R_ans)
```

---

## 🛠️ Installation

### Requirements
```bash
python >= 3.8
torch >= 2.0.0
transformers >= 4.35.0
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/MarineEVT.git
cd MarineEVT
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Inference with EVT-R1

```python
from evt_r1 import EVTR1

# Initialize model
model = EVTR1.from_pretrained("path/to/evt-r1-checkpoint")

# Load video and question
video_path = "path/to/video.mp4"
question = "What species is interacting with the coral reef?"

# Generate answer with visual tool reasoning
answer = model.generate(video_path, question)
print(answer)
```

### Training EVT-R1

```bash
# Training script
python train.py \
    --model_name qwen3-vl-8b \
    --data_path ./data/MarineEVT \
    --output_dir ./checkpoints/evt-r1 \
    --num_epochs 10 \
    --batch_size 4 \
    --learning_rate 1e-5
```

### Evaluation

```bash
# Evaluate on MarineEVT benchmark
python evaluate.py \
    --checkpoint ./checkpoints/evt-r1 \
    --test_file ./data/MarineEVT/annotations/test.json \
    --output_dir ./results
```

---

## 📈 Experimental Results

We compare EVT-R1 against state-of-the-art VLMs across five reasoning dimensions:

| Model | Training | Tools | SemR. | ConR. | SpaR. | TemR. | CasR. | Avg. |
|-------|----------|-------|-------|-------|-------|-------|-------|------|
| **Ours (EVT-R1)** | ✓ | ✓ | **65.80** | **53.33** | **30.60** | **20.75** | **74.00** | **48.89** |
| Qwen3-VL-8B (SFT) | ✓ | ✓ | 61.40 | 53.33 | 22.60 | 15.00 | 74.00 | 45.27 |
| InternVL3-8B | ✗ | ✗ | 53.40 | 50.33 | 17.20 | 10.33 | 71.77 | 40.61 |
| GPT-5-Mini | ✗ | ✓ | 58.40 | 43.67 | 22.80 | 13.00 | 64.33 | 40.35 |

**Key Findings:**
- EVT-R1 achieves **48.89% average accuracy**, outperforming all baselines
- Significant improvements in **Spatial Reasoning** (+8% over SFT) and **Temporal Reasoning** (+5.75% over SFT)
- Visual tool integration is crucial for handling sparse and ephemeral marine events

---

## 📝 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@inproceedings{to2026marineevt,
  title={MarineEVT: Advancing Event-Centric Marine Video Understanding via Visual Tool Reasoning},
  author={To, Tuan-An and Wong, Yuk-Kwan and Vu, Tuan-Anh and Zheng, Ziqiang and Yeung, Sai-Kit},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
