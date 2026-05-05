# MAPViS

MAPViS is a molecular property prediction project. The workflow has two main stages:

1. Pretraining with `molclr.py`.
2. Fine-tuning downstream tasks with `main.py`.

### Installation

Python 3.9 is recommended.

```bash
conda create -n mapvis python=3.9 -y
conda activate mapvis
pip install -r requirements.txt
```

Run all commands from the `MAPViS/` project root.

### 1. Pretraining

The pretraining script is `molclr.py`, and the configuration file is `config.yaml`.

Run:

```bash
python molclr.py
```

Common settings:

- `dataset.data_path` in `config.yaml`: path to the pretraining data
- `gpu` in `config.yaml`: device name, such as `cuda:0` or `cpu`
- `batch_size`, `epochs`, `init_lr`: training hyperparameters

After pretraining, the checkpoint is saved to:

```text
ckpt/<time>/checkpoints/model.pth
```

### Pretraining Data

The pretraining data file is large, so it is not uploaded to GitHub. Data for pre-training can be obtained by contacting the author.

The data is also available through Baidu Netdisk:

```text
File: pretrain_data
Link: https://pan.baidu.com/s/12DalLf6E7Bj_ZS4pUP-Bhw?pwd=5ikj
Extraction code: 5ikj
```

### 2. Fine-tuning

The fine-tuning script is `main.py`, and the configuration file is `config_finetune_ulog.yaml`.

First edit `config_finetune_ulog.yaml`:

```yaml
task_name: S. aureus growth inhibition
gpu: cuda:0
```

Supported downstream tasks:

- `S. aureus growth inhibition`
- `HepG2 cytotoxicity`
- `HSkMC cytotoxicity`
- `IMR-90 viability`
- `neisseria_gonorrhoeae_binarized`

Run:

```bash
python main.py
```

To use your own pretrained checkpoint, update the checkpoint path in `main.py` to:

```text
ckpt/<time>/checkpoints/model.pth
```

If no pretrained checkpoint is found, the script will continue training from scratch.

### Outputs

Fine-tuning outputs are saved under:

```text
scaffold/finetune/
```

Experiment summaries are saved under:

```text
scaffold/experiments/
```

Common outputs include:

- Best checkpoint: `checkpoints/model.pth`
- Copied run configuration: `config_finetune_ulog.yaml`
- Prediction and metric CSV files
