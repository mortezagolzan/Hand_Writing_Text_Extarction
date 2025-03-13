# 📙 HTR-VT (Pattern Recognition)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/htr-vt-handwritten-text-recognition-with/handwritten-text-recognition-on-lam-line)](https://paperswithcode.com/sota/handwritten-text-recognition-on-lam-line?p=htr-vt-handwritten-text-recognition-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/htr-vt-handwritten-text-recognition-with/handwritten-text-recognition-on-read2016-line)](https://paperswithcode.com/sota/handwritten-text-recognition-on-read2016-line?p=htr-vt-handwritten-text-recognition-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/htr-vt-handwritten-text-recognition-with/handwritten-text-recognition-on-iam-line)](https://paperswithcode.com/sota/handwritten-text-recognition-on-iam-line?p=htr-vt-handwritten-text-recognition-with)

# Handwritten Text Recognition (HTR) Model

This repository contains a Handwritten Text Recognition model implementation that supports both CUDA and Apple Silicon (MPS) devices. The model is specifically designed to handle noisy handwritten text, making it suitable for real-world applications like bank check processing.

## Features

- Supports both CUDA and Apple Silicon (MPS) devices
- Handles noisy handwritten text with:
  - Background variations
  - Horizontal lines
  - Smudges
  - Salt and pepper noise
- Trained on combined IAM and RIMES datasets
- No pre-processing required for inference

## Sample Images

### Training Data with Noise
Here are some examples of our training data with artificially added noise to simulate real-world conditions:

![Sample 1](img/picture1.png)
*Sample 1: Handwritten text with horizontal lines and background noise*

![Sample 2](img/picture2.png)
*Sample 2: French text sample with dotted line background - demonstrates model's ability to handle different writing styles and line patterns*

![Sample 3](img/picture3.png)
*Sample 3: Another example of French text with dotted line patterns - showing consistency in handling structured backgrounds*

### Model Output
The model processes these noisy inputs and produces clean text output. Output images can be found in the `output/predictions/` directory after running inference.

## Requirements

- Python 3.x
- PyTorch
- Other dependencies listed in requirements.txt

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Testing the Model

To run inference on test images:

```bash
# For Apple Silicon Macs (M1/M2)
export PYTORCH_ENABLE_MPS_FALLBACK=1
python test.py --out-dir ./output --exp-name iam IAM --test-data-list ./Test_Data/ --show-images True
```

The model will process the images and save the predictions with visualizations in `output/predictions/`.

## Notes

- For Apple Silicon (M1/M2) Macs, the model uses MPS (Metal Performance Shaders) backend with CPU fallback for certain operations
- Prediction images are saved in the `output/predictions/` directory
- The model includes spell-checking for improved accuracy
- No pre-processing required for noisy images

## Dataset Information

The model is trained on a combination of:
1. IAM Dataset (English)
   - 13,350 line-level samples
   - 657 different writers
2. RIMES Dataset (French characters)
   - Additional line-level samples
   - Over 1,300 participants

### Data Augmentation
To make the model robust for real-world applications, we augment the training data with:
- Horizontal lines (simulating check fields)
- Various backgrounds
- Smudges
- Salt and pepper noise

The training uses a 50:50 ratio of clean and noisy images to ensure the model learns both clean character structure and noise handling.

### Introduction
This is the official implementation of our **Pattern Recognition(PR)** 2024 paper *"HTR-VT: Handwritten Text Recognition with Vision Transformer".* It's a new and effective baseline for handwritten text recognition solely using Vision Transformer and CTC Loss. 

[[Project Page]](https://yutingli0606.github.io/HTR-VT/)
[[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0031320324007180)
[[arXiv]](https://arxiv.org/pdf/2409.08573) 
[[Google Drive]](https://drive.google.com/drive/folders/1g8ber-YqOeytDA4SqNah1otTt3AKCYXb?usp=drive_link)


## Table of Content
* [1. Overview](#1-overview)
* [2. Visual Results](#2-visual-results)
* [3. Installation](#3-installation)
* [4. Quick Start](#4-quick-start)
* [5. Citation](#5-citation)
* [6. Acknowledgement](#6-acknowledgement)

## 1. Overview
<p align="center">
<img src="img/HTR-VT.png" width="500px" alt="teaser">
</p>

## 2. Visual Results
<p align="center">
<img src="img/visual.png" width="900px" alt="method">
</p>

## 3. Installation

### 3.1. Environment

Our model can be learnt in a **single GPU RTX-4090 24G**
```bash
conda env create -f environment.yml
conda activate htr
```

The code was tested on Python 3.9 and PyTorch 1.13.0.


### 3.2. Datasets

* Using **IAM, READ2016 and LAM** for handwritten text recognition.

</summary>
  <details>
   <summary>
   IAM
   </summary>
    
    Register at the FKI's webpage :https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
    Download the dataset from here :https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database
  </details>
  <details>
   <summary>
   READ2016
   </summary>
    
    wget https://zenodo.org/record/1164045/files/{Test-ICFHR-2016.tgz,Train-And-Val-ICFHR-2016.tgz}
  </details>
  <details>
   <summary>
   LAM
   </summary>
    
    Download the dataset from here: https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=46
  </details>
  
* Download datasets to ./data/.
Take IAM for an example:
The structure of the file should be:

```
./data/iam/
├── train.ln
├── val.ln
├── test.ln
└── lines
      ├──a01-000u-00.png
      ├──a01-000u-00.txt
      ├──a01-000u-01.png
      ├──a01-000u-01.txt
      ...
```


## 4. Quick Start
* We provide convenient and comprehensive commands in ./run/ to train and test on different datasets to help researchers reproducing the results of the paper.

## 5. Citation
If our project is helpful for your research, please consider citing :
```
@article{li2024htr,
  title={HTR-VT: Handwritten text recognition with vision transformer},
  author={Li, Yuting and Chen, Dexiong and Tang, Tinglong and Shen, Xi},
  journal={Pattern Recognition},
  pages={110967},
  year={2024},
  publisher={Elsevier}
}
```

## 6. Acknowledgement

We appreciate helps from public code: [VAN](https://github.com/FactoDeepLearning/VerticalAttentionOCR) and [OrigamiNet](https://github.com/IntuitionMachines/OrigamiNet).  
