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

Dependencies are listed in:
- `weed_requirements.txt`

--- 