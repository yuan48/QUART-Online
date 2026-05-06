<div align="center">

# QUART-Online 🤖

### Latency-Free Large Multimodal Language Model for Quadruped Robot Learning

[![arXiv](https://img.shields.io/badge/arXiv-2412.15557-b31b1b.svg)](https://arxiv.org/abs/2412.15576) [![Project Page](https://img.shields.io/badge/Project-Page-green)](https://quart-online.github.io/) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)

[**Homepage**](https://quart-online.github.io/) | [**arXiv**](https://arxiv.org/abs/2412.15576) | [**Model Weights**](https://huggingface.co/Tong314/Quart-Online) | [**Dataset**](https://huggingface.co/datasets/Tong314/QUARD) | [**Preprocessed Data**](https://huggingface.co/datasets/Tong314/Quart-Online-preprocess/tree/main)

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

Download the QUART-Online and VQ checkpoints from [HuggingFace](https://huggingface.co/Tong314/Quart-Online/tree/main).

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
│                  [vq_data.npy]     [VQ Codebook]           │
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

### Data Preprocessing

If you want to process your own data, see [preprocess/PREPROCESSING.md](preprocess/PREPROCESSING.md) for the complete pipeline documentation.

Alternatively, download [preprocessed data](https://huggingface.co/datasets/Tong314/Quart-Online-preprocess/tree/main).

---

## 🏋️ Training

The QUART-Online training consists of two stages: **VQ Model Training** and **VLA Model Training**.

### Stage 1: VQ Model Training

Train the Residual Vector Quantization model to compress action sequences into discrete tokens.

#### 1.1 Data Preprocessing

```bash
# Step 1: Downsample 50Hz data to 5Hz
python preprocess/1_quart_data_process.py

# Step 2: Prepare sliding window sequences
python preprocess/2_vq_train_preprocess.py
```

#### 1.2 Train VQ Model

```bash
python preprocess/3_train_vq_Sequence.py
```

**Model Config**:

- Architecture: Sequence-based RVQ (Conv1D encoder)
- Codebook size: 512
- Quantizers: 2 (hierarchical)
- Input: 10-step sequences `[10, 12]`
- Output: 4 discrete tokens

**Training Params**:

- Batch size: 1024
- Learning rate: 3e-4
- Epochs: 50
- Split: 85% train / 15% validation

**Output**: `vq_state_dict/VQ/Sequence_vq_10_each_conv.pt`

---

### Stage 2: VLA Model Training

Train the vision-language-action model using VQ-tokenized data.

#### 2.1 Generate VQ Tokenized Data

```bash
# Generate per-task JSONs with VQ tokens
python preprocess/4_quart_online_vq_process_concurrent.py

# Merge into single training file
python preprocess/5_merge_json.py
```

**Output**: `datasets/Full/sim_json_path/sim_ahead_10_seq.json`

#### 2.2 Train QUART Model

⚠️ You may need to change the directory of images in the JSON dataset.

```bash
bash ./train_script/train_fuyu_v2_step_10_sequence.sh
```

**Key Parameters**:

- Base model: Fuyu-8B
- Training data: `sim_ahead_10_seq.json`
- Batch size: 32 × 4 GPUs = 128 effective
- Precision: BF16 with DeepSpeed ZeRO-3
- Learning rate: 2e-5
- Epochs: 10

**Output**: `./ckpts/Fuyu_v0/<timestamp>/`

---

## 🧪 Testing & Evaluation

### 1. Inference Test

Run inference on sample data to verify the installation:

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

### 2. IsaacGym Evaluation

Test the trained model in IsaacGym simulation environment.

#### 2.1 Install IsaacGym

```bash
# Download from https://developer.nvidia.com/isaac-gym
tar -zxvf IsaacGym_Preview_4_Package.tar.gz -C /path/to/isaacgym
cd /path/to/isaacgym/python
pip install -e .
```

#### 2.2 Run Evaluation

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

### 3. Performance Metrics

QUART-Online achieves:

- **Navigation Success Rate**: >90% on seen environments
- **Obstacle Avoidance**: >85% success rate
- **Inference Speed**: ~20ms per action (float16 on A100)
- **Generalization**: Strong performance on unseen objects and scenes

For detailed results, see our [paper](https://arxiv.org/abs/2412.15576).

---

## 📊 Datasets

QUART-Online supports training on:

1. **Simulation Data**: Generated in IsaacGym environments

   - Navigation tasks (go-to, avoid obstacles)
   - Manipulation tasks (unload balls)
   - Letter recognition
   - Crawling under barriers
2. **Custom Data**: Process your own robot demonstrations

   - Use `preprocess/1_quart_data_process.py` to convert raw data
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
│   ├── 1_quart_data_process.py  # 50Hz → 5Hz downsampling
│   ├── 2_vq_train_preprocess.py # VQ training data prep
│   ├── 3_train_vq_Sequence.py   # VQ model training
│   ├── 4_quart_online_vq_process.py          # VQ tokenization
│   ├── 4_quart_online_vq_process_concurrent.py # Parallel version
│   ├── 5_merge_json.py          # Merge task JSONs
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
@article{tong2024quartonline,
  title={QUART-Online: Latency-Free Large Multimodal Language Model for Quadruped Robot Learning},
  author= {Tong, Xinyang and Ding, Pengxiang and Wang, Donglin and Zhang, Wenjie and Cui, Can and Sun, Mingyang and Fan, Yiguo and Han, Zhao and Zhang, Hongyin and Dang, Yonghao and Huang, Siteng and Lyu, Shangke},
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
