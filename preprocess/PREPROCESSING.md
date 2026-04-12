# Preprocessing Pipeline

## Overview

The preprocessing pipeline converts raw robot demonstration data (50Hz) into VQ-tokenized training data for the QUART-Online model.

You can download the preprocessed data at [Preprocessed Data](https://huggingface.co/datasets/Tong314/Quart-Online-preprocess/tree/main) in order to skip the following procedures.

## Complete Data Flow

```
Raw Data (50Hz)
    │
    ▼ [1_quart_data_process.py]
    ├── commands.npy          ← 12-dim action commands (5Hz)
    ├── vq_Dataset_{task}.npy ← Per-task action sequences for VQ training
    └── sim/{task}.json       ← Raw JSON files (not used in final training)
    │
    ▼ [2_vq_train_preprocess.py]
    └── vq_data_step_10_each.npy ← Sliding window sequences [N, 10, 12]
    │
    ▼ [3_train_vq_Sequence.py]
    └── Sequence_vq_10_each_conv.pt ← Trained VQ model
    │
    ▼ [4_quart_online_vq_process.py]
    └── sim_vq_ahead_10_seq/{task}.json ← VQ-tokenized JSONs
    │
    ▼ [5_merge_json.py]
    └── sim_ahead_10_seq.json ← Final training data
```

---

## Stage 1: Data Preprocessing

### Script: `1_quart_data_process.py`

**Purpose**: Downsample raw 50Hz data to 5Hz and generate training-ready files.

**Input**:

```
{RAW_DATA_PATH}/
└── sim_quadruped_data/              # Merged data folder (v1 + unload)
    └── {task}/
        └── {episode}/
            ├── command/{step}.npy    # Raw commands at 50Hz
            ├── action/{step}.npy     # Proprioception data
            └── image/{step}.png      # Camera images
```

**Output**:

```
datasets/Full/sim_quadruped_data_info/
├── commands.npy           # Dict of 12-dim actions per episode
├── proprioceptions.npy    # Robot state observations
├── ranges.npy             # Min/max values for normalization
└── vq_data/
    └── vq_Dataset_{task}.npy  # Per-task action sequences

datasets/Full/sim_json_path/sim/
└── {task}.json            # Raw JSON files (not used in final training)
```

**Key Variables**:

- `sim_sample_rate = 10` : Downsample from 50Hz to 5Hz

**Usage**:

```bash
# Edit RAW_DATA_PATH in the script first
python preprocess/1_quart_data_process.py
```

---

## Stage 2: VQ Data Preparation

### Script: `2_vq_train_preprocess.py`

**Purpose**: Prepare sliding window sequences for VQ model training.

**Input**:

```
datasets/Full/sim_quadruped_data_info/vq_data/
└── vq_Dataset_{task}.npy  # From Stage 1
```

**Output**:

```
datasets/Full/sim_quadruped_data_info/
└── vq_data_step_10_each.npy  # Shape: [N, 10, 12]
```

**How it works**:

1. Loads all `vq_Dataset_{task}.npy` files
2. Creates sliding windows of 10 consecutive timesteps
3. Each window has shape `[10, 12]` (10 steps × 12 action dims)

**Usage**:

```bash
python preprocess/2_vq_train_preprocess.py
```

---

## Stage 3: VQ Model Training

### Script: `3_train_vq_Sequence.py`

**Purpose**: Train the Residual Vector Quantization model.

**Input**:

```
datasets/Full/sim_quadruped_data_info/
└── vq_data_step_10_each.npy
```

**Output**:

```
vq_state_dict/VQ/
└── Sequence_vq_10_each_conv.pt
```

**Model Architecture**:

- **Type**: Sequence-based RVQ (RVQ_Seq_10)
- **Encoder**: Conv1D [12 → 512 → 512 → 512 → 512]
- **Quantizers**: 2-level residual VQ
- **Codebook Size**: 512
- **Output**: 4 tokens per 10-step sequence (2 quantizers × 2 temporal codes)

**Training Parameters**:

```python
batch_size = 1024
learning_rate = 3e-4
epochs = 50
train_split = 0.85  # 85% train, 15% validation
```

**Usage**:

```bash
python preprocess/3_train_vq_Sequence.py
```

---

## Stage 4: VQ Tokenization

### Scripts: `4_quart_online_vq_process.py` / `4_quart_online_vq_process_concurrent.py`

**Purpose**: Encode action sequences into VQ tokens and generate training JSONs.

**Input**:

```
{RAW_DATA_PATH}/sim_quadruped_data/      # Raw data with images
datasets/Full/sim_quadruped_data_info/
└── commands.npy                           # From Stage 1
vq_state_dict/VQ/
└── Sequence_vq_10_each_conv.pt           # Trained VQ model from Stage 3
```

**Output**:

```
datasets/Full/sim_json_path/
└── sim_vq_ahead_10_seq/
    ├── go.json
    ├── crawl.json
    └── {task}.json  # One file per task
```

**How it works**:

1. Loads trained VQ model
2. For each timestep in each episode:
   - Takes 10-step action window
   - Encodes to 4 VQ tokens via `VQ_model.tokenize()`
   - Saves JSON with image path and VQ tokens

**Output JSON Format**:

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
  "vq": "<0x04> 13 320 16 276"
}
```

**Usage**:

```bash
# Sequential version
python preprocess/4_quart_online_vq_process.py

# Parallel version (faster)
python preprocess/4_quart_online_vq_process_concurrent.py
```

---

## Stage 5: Merge Task JSONs

### Script: `5_merge_json.py`

**Purpose**: Combine all task JSONs into a single training file.

**Input**:

```
datasets/Full/sim_json_path/
└── sim_vq_ahead_10_seq/
    ├── go.json
    ├── crawl.json
    └── ... (all task JSONs)
```

**Output**:

```
datasets/Full/sim_json_path/
└── sim_ahead_10_seq.json  # Final training data
```

**Usage**:

```bash
python preprocess/5_merge_json.py
```

---

## Directory Structure Summary

```
QUART-Online/
├── datasets/Full/
│   ├── sim_quadruped_data_info/     # Processed data
│   │   ├── commands.npy
│   │   ├── proprioceptions.npy
│   │   ├── ranges.npy
│   │   ├── vq_data/
│   │   │   └── vq_Dataset_{task}.npy
│   │   └── vq_data_step_10_each.npy
│   └── sim_json_path/               # Training JSONs
│       ├── sim_vq_ahead_10_seq/
│       │   └── {task}.json
│       └── sim_ahead_10_seq.json
├── vq_state_dict/VQ/
│   └── Sequence_vq_10_each_conv.pt
└── preprocess/
    ├── 1_quart_data_process.py
    ├── 2_vq_train_preprocess.py
    ├── 3_train_vq_Sequence.py
    ├── 4_quart_online_vq_process.py
    ├── 4_quart_online_vq_process_concurrent.py
    └── 5_merge_json.py
```

## Configuration

Edit `RAW_DATA_PATH` in each script to point to your raw data directory:

```python
RAW_DATA_PATH = '/path/to/your/Datasets'  # Contains sim_quadruped_data/
```

Task definitions are in `preprocess/init_path.py`:

- `SIM_INSTRUCTION_DICT` - Simulation task instructions
- `REAL_INSTRUCTION_DICT` - Real-world task instructions
