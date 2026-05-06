<div align="center">

# [Accelerating Vision Transformers With Adaptive Patch Sizes (ICLR 2026)](https://arxiv.org/abs/2510.18091)

**Rohan Choudhury\*<sup>1</sup>, JungEun Kim\*<sup>2,3</sup>, Jinhyung Park<sup>1</sup>, Eunho Yang<sup>2</sup>, László A. Jeni<sup>1</sup>, Kris M. Kitani<sup>1</sup>**

<sup>1</sup>Carnegie Mellon University, <sup>2</sup>KAIST, <sup>3</sup>General Robotics

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2510.18091)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://rccchoudhury.github.io/apt/)

<table>
  <tr>
    <td width="50%" align="center">
      <img src="./assets/apt_vis_720p_5s_slow_16x16_grid.gif" width="100%">
      <br>
      <b>Standard</b>
    </td>
    <td width="50%" align="center">
      <img src="./assets/apt_vis_720p_5s_slow.gif" width="100%">
      <br>
      <b>APT (ours)</b>
    </td>
  </tr>
</table>

</div>

## TL;DR

We accelerate Vision Transformers by using **adaptive patch sizes** based on image content. Instead of using fixed-size patches for all regions, our method dynamically selects smaller patches for detailed areas and larger patches for uniform regions, reducing computational cost while maintaining accuracy. This reduces the total number of patches while maintaining performance. 

We are releasing an initial version of our code, and in the next few days will add detailed training code and pretrained checkpoints. Please bear with us as we continue to clean our code and add more capabilities. If you have anything in particular you would like to see, don't hesitate to file a GitHub issue!

## Setup

We use mamba and uv for installing packages, but conda and pip should work too.

```bash
# Create a new environment with mamba
mamba create -n apt python=3.10 -y
mamba activate apt

# Install PyTorch (adjust for your CUDA version)
mamba install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install remaining dependencies with uv pip
uv pip install -r requirements.txt
```

## Dataset Setup

Datasets require configuration through YAML files located in `configs/data/`. To set up ImageNet:

1. Edit `configs/data/imagenet.yaml` and update the `data_dir` field to point to your ImageNet dataset:

```yaml
data_dir: /path/to/ILSVRC2012
```

2. Ensure your ImageNet directory has the following structure:

```
ILSVRC2012/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

The `data_dir` should point to the `ILSVRC2012` directory containing the `train` and `val` folders.

## Model Support

We support a wide range of timm models through automatic checkpoint downloading. The models we've tested are covered by the configs in `configs/model_variants/`, and our code automatically downloads the required checkpoint as needed. While we've primarily tested on ViT variants, we cannot guarantee support for all timm models. If you encounter any issues, please file a GitHub issue.

## Running A Model

### Validate Vision Transformer on ImageNet

```bash
# Run validation with default settings (8 GPUs, batch size 64)
python src/eval.py experiment=validate_vit

# Override specific parameters
python src/eval.py experiment=validate_vit \
  trainer.devices=4 \
  data.batch_size=128

# Adjust thresholds and scales
python src/eval.py experiment=validate_vit \
  num_scales=3 \
  thresholds=[6.0, 4.0]
```

## Training

The below script fine-tunes a ViT with APT from a pretrained checkpoint for 50 epochs; you can modify the config to do shorter fine-tuning from an already fine-tuned version as well.

```bash
# Fine-tune ViT-B/16 on ImageNet with APT (8 GPUs, 5 epochs)
python src/train.py experiment=train_vit_finetune 
```

## Visualizing APT Patches

Generate a visualization showing how APT selects patches for an image with the following command:

```bash
python scripts/gen_visualization_single.py \
  --input /your/input/image.jpg \
  --output /your/output/visualization.jpg \
  --method entropy \
  --vis_type grid
```

Essential parameters: `--input` (required), `--output`, `--method` (entropy/laplacian/upsample_mse), `--vis_type` (entropy/grid/none), `--patch_size`, `--num_scales`, `--thresholds`.

## Citation

If you use our code or the paper, please cite our work:

```bibtex
@article{choudhury2025apt,
  title={Accelerating Vision Transformers with Adaptive Patch Sizes},
  author={Choudhury, Rohan and Kim, JungEun and Park, Jinhyung and Yang, Eunho and Jeni, L{\'a}szl{\'o} A. and Kitani, Kris M.},
  journal={arXiv preprint arXiv:2510.18091},
  year={2025},
  url={https://arxiv.org/abs/2510.18091},
  doi={10.48550/arXiv.2510.18091}
}