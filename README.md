# FP4DiT: Towards Effective Floating Point Quantization for Diffusion Transformers [[paper](https://arxiv.org/abs/2503.15465)]
<p align="center">
    <a href="https://www.python.org/" alt="Python">
        <img src="https://img.shields.io/badge/Python-3.9-yellow" /></a>
    <a href="https://pytorch.org/" alt="PyTorch">
        <img src="https://img.shields.io/badge/PyTorch-1.13-orange" /></a>
    <a href="https://pytorch.org/" alt="Diffusers">
        <img src="https://img.shields.io/badge/Diffusers-0.29-red" /></a>
<p/>

## 🚀 News
 Our paper "FP4DiT: Towards Effective Floating Point Quantization for Diffusion Transformers" has been accepted by **Transactions on Machine Learning Research (TMLR)**!

## Overview
The code of FP4DiT is developed on top of the following repos:
1. Q-diffusion (ICCV 2023; https://github.com/Xiuyu-Li/q-diffusion), which performs quantization calibration and inference on diffusion models.
2. FP8 Quantization (Qualcomm; https://github.com/Qualcomm-AI-research/FP8-quantization), which provide data type support for floating-point quantization (FPQ).

In addition, the code for this paper is split into 3 branches, performing floating-point quantization (FPQ) on PixArt-alpha, Sigma and Hunyuan DiT respectively. 

## Setting up the environment
We adopt the original conda environment of `q-diffusion` (see their repo at github for setup instructions) 
```
conda env create -f environment.yml
conda activate qdiff_PixArt
```

but make some notable dependency changes:
```
pytorch==1.13.0+cu117
diffusers==0.29.2
transformers==4.42.3
accelerate==0.27.2
pytorch-fid==0.3.0
pytorch_lightning==1.5.0
networkx==3.1
sentencepiece
pycocotools
```

## Quantization
To perform FPQ inference, we need to follow the below steps:
1. Generate the calibration data.
```
python scripts/pixart_alpha_calib.py
```
This command will prepare the calibration data as `pixart_calib_brecq.pt`.

2. Calibrate PixArt alpha and generate 10k images using COCO prompt.
```
python scripts/pixart_alpha_brecq.py --plms --cond --n_samples 1 --outdir <output_dir> --ptq --weight_bit 4 --quant_mode qdiff --cond --cali_data_path pixart_calib_brecq.pt --cali_batch_size 16 --cali_iters 2500 --cali_iters_a 1 --quant_act --act_bit <6 or 8> --act_mantissa_bits <3 for A6, 4 for A8> --weight_group_size 128 --weight_mantissa_bits 1 --ff_weight_mantissa 0 --res 512 --coco_10k 
```
This command will generate a calibrated checkpoint under the directory `<output_dir>`.

3. If calibrated checkpoint `<ckpt>` has already been prepared, we can resume the quantized model using the checkpoint and skip the calibration step.
```
python scripts/pixart_alpha_brecq.py --plms --cond --n_samples 1 --outdir <output_dir> --ptq --weight_bit 4 --quant_mode qdiff --cond --cali_data_path pixart_calib_brecq.pt --cali_batch_size 16 --cali_iters 2500 --cali_iters_a 1 --quant_act --act_bit <6 or 8> --act_mantissa_bits <3 for A6, 4 for A8> --cali_ckpt <ckpt> --resume_w --weight_group_size 128 --weight_mantissa_bits 1 --ff_weight_mantissa 0 --res 512 --coco_10k 
```

## Seed
All seeds are default set as 42. If you want to change it, use --seed <112312 any number>

## Bibtex 
If you find our work useful, we kindly ask that you cite our paper:
```
@article{chen2025fp4dit,
  title={FP4DiT: Towards Effective Floating Point Quantization for Diffusion Transformers},
  author={Chen, Ruichen and Mills, Keith G. and Niu, Di},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={[https://openreview.net/forum?id=CcnH4mSQbP](https://openreview.net/forum?id=CcnH4mSQbP)}
}
```
