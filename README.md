<div align="center">

# QUART-Online 🤖

### Latency-Free Large Multimodal Language Model for Quadruped Robot Learning

[![arXiv](https://img.shields.io/badge/arXiv-2412.15557-b31b1b.svg)](https://arxiv.org/abs/2412.15576) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://quart-online.github.io/) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)

[**Homepage**](https://quart-online.github.io/) | [**arXiv**](https://arxiv.org/abs/2412.15576) | [**Model Weights**](https://huggingface.co/Tong314/Quart-Online)

---

</div>

## 📖 Overview

**QUART-Online** is a cutting-edge large multimodal language model designed for **zero-latency** quadruped robot learning. By integrating visual and language inputs, QUART-Online enables real-time decision-making and complex task execution for legged robots in simulation environments.

### Key Features

- 🚀 **Zero-Latency Inference**: Real-time action generation for quadruped robots
- 🎯 **Vision-Language Integration**: Combines visual perception with natural language instructions
- 🏃 **Multi-Task Capability**: Navigation, obstacle avoidance, object manipulation, and more
- ⚡ **Efficient Action Encoding**: Residual Vector Quantization (RVQ) for compact action representation
- 🎮 **IsaacGym Integration**: Seamless testing and deployment in physics simulation

---

## 🎬 Quick Start

### Prerequisites

- **Python**: 3.8
- **GPU Memory**: >19GB (float16) or >37GB (float32)
- **Recommended GPU**: NVIDIA A100 / RTX 3090 (V100 does not support float32)
- **CUDA**: 11.8 (for IsaacGym)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yuan48/QUART-Online.git
cd QUART-Online
```

2. **Create conda environment**

```bash
conda create -n quart python=3.8
conda activate quart
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download model checkpoints**

Download the QUART-Online and VQ checkpoints from [HuggingFace](https://huggingface.co/Tong314/Quart-Online/tree/main):

```bash
# Example: Place checkpoints in ./ckpts/
mkdir -p ckpts/vq_state_dict
# Download quart_online checkpoint to ./ckpts/
# Download Sequence_vq_10_each_conv.pt to ./ckpts/vq_state_dict/
```

5. **Test the installation**

```bash
python test_quart.py
```

---

## 🏗️ System Architecture

QUART-Online consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                     QUART-Online Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Raw Data]  →  [Preprocessing]  →  [VQ Training]           │
│                      ↓                    ↓                 │
│                  [Commands.npy]     [VQ Codebook]           │
│                      ↓                    ↓                 │
│              [VLA Model Training]  ←──────┘                 │
│                      ↓                                      │
│              [Trained QUART Model]                          │
│                      ↓                                      │
│              [IsaacGym Deployment]                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Data Preprocessing** (`preprocess/`)

   - Downsample data from 50Hz → 5Hz
   - Generate proprioception and command data
   - Create LLM-compatible JSON datasets
2. **Vector Quantization** (`models/RVQ/`)

   - Residual VQ (RVQ) for action encoding
   - 10-step sequence prediction
   - Compact action representation
3. **Vision-Language Model** (`models/fuyu/`)

   - Fuyu-8B based architecture
   - Multimodal encoder-decoder
   - Action token prediction
4. **IsaacGym Evaluation** (`gym_eval_scripts/`)

   - Real-time robot control
   - Multi-environment testing
   - Performance metrics collection

---

## 🎯 Usage

### 1. Testing Pre-trained Model

Run inference on sample data:

```bash
python test_quart.py \
    --exp_id Fuyu_v0 \
    --ckpt_path ./ckpts/quart_online \
    --vq_ckpt_path ./ckpts/vq_state_dict/Sequence_vq_10_each_conv.pt \
    --vocab_path ./vocabs/vocab_fuyu.json \
    --dataset_path ./sample_data/sim_quadruped_data_unload \
    --dataset_type Full \
    --detype float16
```

### 2. IsaacGym Deployment

First, install IsaacGym:

```bash
# Download from https://developer.nvidia.com/isaac-gym
tar -zxvf IsaacGym_Preview_4_Package.tar.gz -C /path/to/isaacgym
cd /path/to/isaacgym/python
pip install -e .
```

Then run the evaluation script:

```bash
# Update paths in the script first
bash ./gym_eval_scripts/quart_isaacgym_test.sh
```

**Configuration** (`quart_isaacgym_test.sh`):

```bash
PROJECT_PATH='your/quart/path'
CKPT_PATH="${PROJECT_PATH}/ckpts"
VQ_CKPT_PATH="${PROJECT_PATH}/ckpts/vq_state_dict/Sequence_vq_10_each_conv.pt"
TEST_TYPE="seen"           # or "unseen"
HEADLESS=True              # Run without visualization
ENV_NUM=10                 # Number of parallel environments
DETYPE=float16             # Precision: float16 or float32
```

### 3. Data Preprocessing (Optional)

If you want to process your own data:

```bash
python preprocess/vqdata_process.py
```

This will:

- Convert raw episode data (50Hz) → downsampled data (5Hz)
- Generate `commands.npy` and `proprioceptions.npy`
- Create task-specific JSON files for training

Alternatively, download preprocessed data:

- [Preprocessed 5Hz Data](https://drive.google.com/file/d/1B3ldn49Wpd3G5t9OCmmBK740vgAao4-F/view?usp=sharing)

### 4. Training

#### Complete Training Pipeline

The QUART-Online training follows a **three-stage pipeline**:

1. **Data Preprocessing** (50Hz → 5Hz downsampling)
2. **VQ Model Training** (Action sequence compression)
3. **VLA Model Training** (Vision-language-action learning)

---

#### Stage 1: Data Preprocessing

Convert raw robot demonstration data from 50Hz to 5Hz and generate training-ready formats.

**Input**: Raw episode data with images, proprioception, and commands
**Output**: `commands.npy`, `proprioceptions.npy`, and task JSON files

```bash
python preprocess/vqdata_process.py \
    --sim_path /path/to/raw/simulation/data \
    --output_path ./datasets/Full/sim_quadruped_data_info \
    --sample_rate 10  # Downsample from 50Hz to 5Hz
```

**What this does**:

- Reads raw episode data (`.npy` files at 50Hz)
- Downsamples to 5Hz by taking every 10th frame
- Extracts 12-dimensional action commands
- Saves consolidated `commands.npy` for VQ training
- Generates proprioception data (joint positions, velocities, etc.)

**Generated Files**:

```
datasets/Full/sim_quadruped_data_info/
├── commands.npy          # [N, 12] action sequences for VQ training
├── proprioceptions.npy   # Robot state observations
└── ranges.npy            # Normalization statistics
```

---

#### Stage 2: VQ Model Training

Train the Residual Vector Quantization model to compress action sequences into discrete tokens.

**Purpose**: Convert continuous 12-dimensional actions → discrete VQ codes (4 tokens per 10-step sequence)

##### Step 2.1: Train VQ Model

```bash
python models/train_vq_Sequence.py
```

**Key Configuration** (edit in script):

```python
input_dim = 12              # Action dimensions (see Action Space table)
timestep = "n_Seq"          # Sequence-based VQ (10 steps)
step = 10                   # Predict 10 future steps (0.5s at 5Hz)
```

**Model Architecture**:

- **Encoder**: Conv1D layers [12 → 512 → 512 → 512 → 512]
- **Quantizer**: 2-level residual VQ with 512 codebooks
- **Decoder**: Transposed Conv1D [512 → 512 → 512 → 12]

**Training Parameters**:

```python
batch_size = 1024
learning_rate = 3e-4
epochs = 50
train_split = 0.85  # 85% train, 15% validation
```

**Output**: `Sequence_vq_10_each_conv.pt` (VQ model checkpoint)

**VQ Data Format**:
The VQ model expects sequences of shape `[batch, timesteps, 12]`:

```python
# Example: 10-step action sequence
[
  [0, 2.07, 0.0, 0.0, 0.046, 3.0, 0.5, 0.0, 0.0, 0.084, 0.0, 0.104],  # t=0
  [0, 2.05, 0.0, 0.0, 0.045, 3.0, 0.5, 0.0, 0.0, 0.083, 0.0, 0.103],  # t=1
  ...  # t=2 to t=8
  [1, 1.95, 0.0, 0.0, 0.040, 3.0, 0.5, 0.0, 0.0, 0.080, 0.0, 0.100]   # t=9
]
# ↓ VQ Encoding ↓
# Compressed to 4 tokens: [13, 320, 16, 276]
```

##### Step 2.2: Generate VQ Tokenized Training Data

Use the trained VQ model to convert actions into discrete tokens for VLA training:

```bash
python preprocess/vq_ahead_n_simjson.py
```

**Configuration** (edit in script):

```python
n_step = 10  # Must match VQ model
vq_path = './ckpts/vq_state_dict/Sequence_vq_10_each_conv.pt'
sim_path = '/path/to/raw/simulation/data'
sim_info_path = './datasets/Full/sim_quadruped_data_info'
sim_json_path = './datasets/Full/sim_json_path'
```

**What this does**:

1. Loads the trained VQ model
2. For each episode, creates 10-step action windows
3. Encodes action sequences → VQ tokens
4. Generates JSON files with image paths and VQ token labels

**Output Format**:

```json
{
  "id": "000000000001",
  "image": "/path/to/episode/image/000.png",
  "conversations": [
    {
      "from": "human",
      "value": "What action should the legged robot take to go to the red cube?",
      "type": "sim"
    },
    {
      "from": "gpt",
      "value": "<0x04> "
    }
  ],
  "vq": "<0x04> 13 320 16 276"  # 4 VQ tokens (2 quantizers × 2 temporal codes)
}
```

**Generated Files**:

```
datasets/Full/sim_json_path/sim_vq_ahead_10_seq/
├── go_to_red_cube.json
├── go_avoid_obstacle.json
├── crawl_gate.json
└── ... (one JSON per task)
```

##### Step 2.3: Merge Task JSONs (Optional)

Combine individual task JSON files into a single training file:

```bash
# Edit the script to set task list and paths
python preprocess/vq_ahead_n_simjson_concurrent.py  # Parallel version (faster)
```

This creates `sim_ahead_10_seq.json` for VLA training.

---

#### Stage 3: Train QUART Model

Train the vision-language-action model using VQ-tokenized data:

```bash
bash ./train_script/train_fuyu_v2_step_10_sequence.sh
```

**Key Training Parameters**:

```bash
PRETRAINED_CKPT_PATH=./models--adept--fuyu-8b  # Fuyu-8B base model
TRAINING_DATA_PATH=./dataset/Full/sim_json_path/sim_ahead_10_seq.json
LEARNING_RATE=2e-5
GPU_NUM=4                 # Number of GPUs
BATCHSIZE_PERDEVICE=32    # Batch size per GPU
GRADACC_PERDEVICE=1       # Gradient accumulation steps
EPOCHS=10
TUNE_MM_MLP_ADAPTER=True  # Fine-tune vision adapter
EXP_ID=Fuyu_v0            # Experiment identifier
```

**Training Features**:

- DeepSpeed ZeRO-3 optimization for distributed training
- Mixed precision (BF16) for faster training
- Gradient checkpointing to reduce memory
- Effective batch size: 32 × 4 × 1 = 128

**Model Output**: Saved to `./ckpts/Fuyu_v0/<timestamp>/`

---

### 🔄 VQ Training Summary

| Stage                        | Input                     | Output                         | Script                                |
| ---------------------------- | ------------------------- | ------------------------------ | ------------------------------------- |
| **1. Preprocessing**   | Raw 50Hz data             | `commands.npy` (5Hz)         | `vqdata_process.py`                 |
| **2. VQ Training**     | `commands.npy`          | VQ model checkpoint            | `train_vq_Sequence.py`              |
| **3. VQ Tokenization** | Raw data + VQ model       | Task JSON files with VQ tokens | `vq_ahead_n_simjson.py`             |
| **4. VLA Training**    | JSON + Images + VQ tokens | QUART model                    | `train_fuyu_v2_step_10_sequence.sh` |

**Complete Pipeline**:

```
Raw Data 
  → [vqdata_process.py] → 
Commands.npy 
  → [train_vq_Sequence.py] → 
VQ Model (Sequence_vq_10.pt) 
  → [vq_ahead_n_simjson.py] → 
VQ Training JSONs 
  → [train_fuyu_v2.sh] → 
QUART Model
```

---

## 📊 Datasets

QUART-Online supports training on:

1. **Simulation Data**: Generated in IsaacGym environments

   - Navigation tasks (go-to, avoid obstacles)
   - Manipulation tasks (unload balls)
   - Letter recognition
   - Crawling under barriers
2. **Custom Data**: Process your own robot demonstrations

   - Use `preprocess/vqdata_process.py` to convert raw data
   - Ensure data includes RGB images, proprioception, and action commands

**Data Format**:

```json
{
  "image": "path/to/image.png",
  "conversations": [
    {
      "from": "human",
      "value": "What action should the robot take to go to the red cube?",
      "type": "sim"
    },
    {
      "from": "gpt",
      "value": "70003 70004 70005 70006"
    }
  ],
  "vq": "70003 70004 70005 70006"
}
```

---

## 🗂️ Project Structure

```
QUART-Online/
├── models/                      # Model architectures
│   ├── RVQ/                     # Residual Vector Quantization
│   │   ├── residual_vq.py       # RVQ implementation
│   │   ├── vq_Sequence.py       # Sequence VQ models
│   │   └── dataset.py           # VQ dataset loader
│   ├── fuyu/                    # Fuyu vision-language model
│   │   ├── modeling_fuyu.py     # Model architecture
│   │   └── processing_fuyu.py   # Data processing
│   └── quart_fuyu.py            # QUART model definition
├── preprocess/                  # Data preprocessing scripts
│   ├── vqdata_process.py        # Main preprocessing pipeline
│   ├── vq_ahead_n_simjson.py    # VQ tokenization
│   └── init_path.py             # Task instruction definitions
├── gym_eval_scripts/            # IsaacGym evaluation
│   ├── gym_task_loop.py         # Multi-task evaluation loop
│   ├── task_configs.py          # Task configurations
│   └── quart_isaacgym_test.sh   # Evaluation script
├── train_script/                # Training scripts
│   └── train_fuyu_v2_step_10_sequence.sh
├── scripts/                     # DeepSpeed configurations
│   ├── zero2.json
│   └── zero3.json
├── train_ahead_n.py             # Main training code
├── test_quart.py                # Inference script
├── utils.py                     # Utility functions
└── requirements.txt             # Python dependencies
```

---

## 🔬 Technical Details

### Action Space

QUART-Online predicts 12-dimensional continuous actions:

| Dimension | Description                 | Range      |
| --------- | --------------------------- | ---------- |
| 0         | Terminate flag              | {0, 1}     |
| 1         | Forward velocity (dx)       | Variable   |
| 2         | Lateral velocity (dy)       | Variable   |
| 3         | Yaw velocity (dyaw)         | Variable   |
| 4         | Body height                 | Variable   |
| 5         | Step frequency              | [1.0, 4.0] |
| 6-8       | Gait parameters (trot/pace) | [0.0, 1.0] |
| 9         | Foot swing height           | Variable   |
| 10        | Pitch angle                 | Variable   |
| 11        | Stance width                | Variable   |

### Vector Quantization

- **Codebook Size**: 512
- **Number of Quantizers**: 2 (hierarchical)
- **Sequence Length**: 10 steps (0.5 seconds at 5Hz)
- **Token Range**: 70003-70514 (added to vocabulary)

### Model Architecture

- **Base Model**: Fuyu-8B
- **Vision Encoder**: Patch-based image tokenization
- **Language Model**: Persimmon decoder
- **Training Strategy**: Mixed precision (BF16), DeepSpeed ZeRO-3
- **Batch Size**: 32 per device × 4 GPUs × 1 grad accumulation = 128 effective

---

## 📈 Performance

QUART-Online achieves state-of-the-art performance on various quadruped robot tasks:

- **Navigation Success Rate**: >90% on seen environments
- **Obstacle Avoidance**: >85% success rate
- **Inference Speed**: ~20ms per action (float16 on A100)
- **Generalization**: Strong performance on unseen objects and scenes

For detailed results, please refer to our [paper](https://arxiv.org/abs/2412.15576).

---

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA out of memory**

```bash
# Use float16 precision
--detype float16

# Reduce batch size
--per_device_train_batch_size 16

# Use gradient checkpointing
--gradient_checkpointing True
```

**2. IsaacGym installation fails**

```bash
# Ensure CUDA 11.8 is installed
# Check compatibility: https://developer.nvidia.com/isaac-gym
```

**3. Model loading errors**

```bash
# Verify checkpoint paths
# Ensure all checkpoint files are downloaded completely
```

---

## 📝 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{quart2024,
  title={QUART-Online: Latency-Free Large Multimodal Language Model for Quadruped Robot Learning},
  author={Your Authors},
  journal={arXiv preprint arXiv:2412.15557},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Fuyu-8B](https://www.adept.ai/blog/fuyu-8b) by Adept AI for the base vision-language model
- [IsaacGym](https://developer.nvidia.com/isaac-gym) by NVIDIA for the simulation environment
- [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways) for the quadruped control baseline

---

## 📧 Contact

For questions and discussions, please:

- Open an [issue](https://github.com/yuan48/QUART-Online/issues)
- Visit our [project page](https://quart-online.github.io/)
- Read the [paper](https://arxiv.org/abs/2412.15576)

---

<div align="center">

**Made with ❤️ for the robotics community**

⭐ Star us on GitHub if you find this project useful!

</div>
