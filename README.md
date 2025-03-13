# 📙 HTR-VT (Pattern Recognition)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/htr-vt-handwritten-text-recognition-with/handwritten-text-recognition-on-lam-line)](https://paperswithcode.com/sota/handwritten-text-recognition-on-lam-line?p=htr-vt-handwritten-text-recognition-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/htr-vt-handwritten-text-recognition-with/handwritten-text-recognition-on-read2016-line)](https://paperswithcode.com/sota/handwritten-text-recognition-on-read2016-line?p=htr-vt-handwritten-text-recognition-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/htr-vt-handwritten-text-recognition-with/handwritten-text-recognition-on-iam-line)](https://paperswithcode.com/sota/handwritten-text-recognition-on-iam-line?p=htr-vt-handwritten-text-recognition-with)

# Handwritten Text Recognition with Noise Handling and Post-Processing

This repository is based on the HTR-VT model from the paper ["HTR-VT: Handwritten Text Recognition with Vision Transformer"](https://github.com/YutingLi0606/HTR-VT). We have extended the original implementation with:
1. Enhanced noise handling capabilities
2. Post-processing using Symspell for improved accuracy
3. Support for Apple Silicon (MPS) devices

## Key Enhancements

### 1. Noise-Robust Training
We retrained the model from scratch with augmented data that includes:
- Horizontal lines (simulating check fields)
- Various backgrounds
- Smudges
- Salt and pepper noise

A 50:50 ratio of clean and noisy images was used during training to ensure the model learns both clean character structure and noise handling.

### 2. Post-Processing with Symspell
We've integrated Symspell for fast and accurate spell-checking, specifically optimized for bank check processing:
- Custom dictionary for numeric words commonly found in check amounts
- Fast lookup based on Damerau-Levenshtein edit distance
- Compound spell checking for sentence-level text
- Processing time in order of milliseconds

#### Custom Dictionary Contents
The spell-checker includes commonly used terms in bank checks:
- Basic numbers: one, two, ..., ten
- Teens: eleven, twelve, ..., twenty
- Tens: thirty, forty, ..., ninety
- Magnitudes: hundred, thousand, lakhs, million

## Sample Images

### Training Data with Noise
Here are examples of our noisy training data that simulate real-world conditions:

![Sample 1](img/Picture1.png)
*Sample 1: Handwritten text with horizontal lines and background noise*

![Sample 2](img/Picture2.png)
*Sample 2: French text sample with dotted line background - demonstrates model's ability to handle different writing styles and line patterns*

![Sample 3](img/Picture3.png)
*Sample 3: Another example of French text with dotted line patterns - showing consistency in handling structured backgrounds*

### Model Output
The model processes these noisy inputs and produces clean text output, which is then enhanced through Symspell post-processing. Output images can be found in the `output/predictions/` directory after running inference.

## Requirements

- Python 3.x
- PyTorch
- Symspell (for post-processing)
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

## Notes

- For Apple Silicon (M1/M2) Macs, the model uses MPS (Metal Performance Shaders) backend with CPU fallback for certain operations
- Prediction images are saved in the `output/predictions/` directory
- Post-processing with Symspell significantly improves Character Error Rate (CER) and Word Error Rate (WER)
- No pre-processing required for noisy images

## Acknowledgements

This work builds upon the [HTR-VT project](https://github.com/YutingLi0606/HTR-VT) by Yuting Li et al. We extend our gratitude to the original authors for their foundational work in handwritten text recognition using Vision Transformers.

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
