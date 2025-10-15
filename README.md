# QUART-Online
Official code repository of "QUART-Online: Latency-Free Large Multimodal Language Model for Quadruped Robot Learning"


### Create model environment

```bash
git clone https://github.com/yuan48/QUART-Online.git
cd QUART-Online
```

Create a conda environment
```python
conda create -n quart python=3.8
conda activate quart
```

Install the environment
```python
pip install -r requirements.txt
```

Download the Quart-online checkpoint
Test the environment by running model inference (Memory required float16: >19GB, float32: >37GB)
Note: if using V100 it didn't support float32. We use A100 as the training GPU, and inference could run on 3090.

```python
python test_quart.py
```

### Download isaacgym

It is recommanded to follow the official environment (cuda11.8): [Isaacgym](https://developer.nvidia.com/isaac-gym)
```bash
tar -zxvf IsaacGym_Preview_4_Package.tar.gz -C ${your_path_isaacgym}
cd ${your_path_isaacgym}
cd python
pip install -e .
```

### RUN Quart-online in isaacgym

you need to change a few path in the script, then run:
```bash
bash ./gym_eval_scripts/quart_isaacgym_test.sh
```

## Fine-tune

If you want to train a new model on the base of QUARD dataset, you could either reprocess the [origin data](https://pan.baidu.com/share/init?surl=Gu9Xlb_ETbqxtSaVky0d3Q&pwd=ok0h), or train the vq 10 step sequence version using the processed json data files.

```bash
bash ./train_script/train_fuyu_v2_step_10_sequence.sh
```
