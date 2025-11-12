# Crop-Weed Detection

## Overview
This repository focuses on **Crop-Weed Detection (CWD)** using deep learning models capable of handling both **RGB** and **RGB+NIR** image data.  
It includes implementations of UNet-based architectures, advanced segmentation networks (DeepLabV3, PSPNet), and robustness mechanisms such as **NIR Dropping** and **Student-Teacher Learning**.

---

## Model Approaches

### **1. Base Models**
- **UNet** – Standard encoder-decoder for semantic segmentation.  
- **MANNET** – Modified attention-based network designed for multispectral data.  
Both models incorporate **NIR Dropping** for robustness against missing NIR channels.

---

### **2. NIR Dropping**
To ensure the model performs consistently on datasets with or without NIR input, a probabilistic **NIR Drop Mechanism** replaces the NIR channel with zeros during training.  
This enhances generalization to both RGB and RGB+NIR datasets.

---

### **3. Student-Teacher Learning**
A **teacher model** (UNet/MANNET) is used to predict the **NIR channel** from RGB images.  
The **student model** is then trained on RGB + generated NIR inputs, maintaining high performance even when NIR data is unavailable.

---

### **4. Crop-Weed Detection (CWD) Experiments**
The `CWD/` folder includes experiments on the RGB-only dataset using various UNet-based architectures:
- UNet  
- UNet BN (Batch Normalization)  
- UNet SA (Spatial Attention)  
- UNet DSC (Depthwise Separable Convolutions)  
- UNet BN+SA+DSC (Combined)

Additionally, **DeepLabV3 (ASPP)** and **PSPNet (Pyramid Scene Parsing)** were implemented.  
These models achieved **≈60% fewer parameters** while providing **superior segmentation quality** compared to earlier UNet variants.

---

## Configurations
Training and inference hyperparameters are defined in:
- `train_config.yaml`
- `infer_config.yaml`

Dependencies are listed in:
- `weed_requirements.txt`

--- 


