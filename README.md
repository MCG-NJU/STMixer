# STMixer
This repository gives the official PyTorch implementation of [STMixer: A One-Stage Sparse Action Detector](https://arxiv.org/abs/2303.15879) (CVPR 2023)

## Installation
- PyTorch == 1.8 or 1.12 (other versions are not tested)
- tqdm
- yacs
- opencv-python
- tensorboardX
- SciPy
- fvcore
- timm
- iopath

## Data Preparation
Please refer to [ACAR-Net repo DATA.md](https://github.com/Siyu-C/ACAR-Net/blob/master/DATA.md)  for AVA dataset preparation.

## Model Zoo
| Backbone          | Config | Pre-train Model | Frames | Sampling Rate | Model |
|-------------------|:------:|:---------------:|:------:|:-------------:|:-----:|
| SlowOnly-R50      |   [cfg](https://github.com/MCG-NJU/STMixer/blob/main/config_files/PySlowonly-R50-K400-4x16.yaml)     |       [K400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWONLY_4x16_R50.pkl)      |    4   |       16      |  [Link](https://drive.google.com/file/d/1qJdnCGwi5NeqpHFYxPpIixLY6Mpsuync/view?usp=share_link) |
| SlowFast-R50      |   [cfg](https://github.com/MCG-NJU/STMixer/blob/main/config_files/PySlowfast-R50-K400-8x8.yaml)      |       [K400](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl)      |    8   |       8       |  [Link](https://drive.google.com/file/d/1pwXBC-g-OS71wzd9lxHDITASRw1cxRWm/view?usp=share_link) |
| SlowFast-R101-NL  |   [cfg](https://github.com/MCG-NJU/STMixer/blob/main/config_files/PySlowfast-R101-NL-K600-8x8.yaml)  |       [K600](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_32x2_R101_50_50.pkl)      |    8   |       8       |  [Link](https://drive.google.com/file/d/1oouF7IZFxs-vXhUDXSkLpy7FE5oh6Vp2/view?usp=share_link) |
| ViT-B(VideoMAE)   |   TODO  |       K400      |   16   |       4       |  TODO |
| ViT-B(VideoMAEv2) |   TODO  |    K710+K400    |   16   |       4       |  TODO |


## Training
```shell
python -m torch.distributed.launch --nproc_per_node=8 train_net.py --config-file "config_files/config_file.yaml" --transfer --no-head --use-tfboard
```

## Validation
```shell
python -m torch.distributed.launch --nproc_per_node=8 test_net.py --config-file "config_files/config_file.yaml" MODEL.WEIGHT "/path/to/model"
```

## Acknowledgements
We would like to thank Ligeng Chen for his help in drawing the figures in the paper and thank Lei Chen for her surpport in experiments. This project is built upon [AlphaAction](https://github.com/MVIG-SJTU/AlphAction), [AdaMixer](https://github.com/MCG-NJU/AdaMixer) and [PySlowFast](https://github.com/facebookresearch/SlowFast). We also reference and use some code from [SparseR-CNN](https://github.com/PeizeSun/SparseR-CNN), [WOO](https://github.com/ShoufaChen/WOO) and [VideoMAE](https://github.com/MCG-NJU/VideoMAE). Very sincere thanks to the contributors to these excellent codebases.

## Citation

If this project helps you in your research or project, please cite
our paper:

```
@inproceedings{wu2023stmixer,
      title={STMixer: A One-Stage Sparse Action Detector}, 
      author={Tao Wu and Mengqi Cao and Ziteng Gao and Gangshan Wu and Limin Wang},
      booktitle={{CVPR}},
      year={2023}
}
```


