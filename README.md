# 🧠 NeuroScan: DL-Based Brain Tumor Detection

## 🔬 Project Purpose & Overview
Brain tumors are highly critical neurological conditions where early and accurate detection is a matter of life and death. While Magnetic Resonance Imaging (MRI) is the gold standard for scanning, manual interpretation of these complex images is time-consuming, highly subjective, and prone to human error—especially when identifying diffuse tumor boundaries like Gliomas.

NeuroScan is a complete, end-to-end Deep Learning pipeline and web application designed to automate this diagnosis. Moving beyond simple “black-box” AI classification, this project focuses heavily on clinical interpretability and accessibility.

It classifies MRI scans into four categories:
- Glioma  
- Meningioma  
- Pituitary Tumor  
- No Tumor  

---

## ✨ Key Features

- Advanced Deep Learning using EfficientNet-B0  
- Mathematical Edge Enhancement using IFFT  
- Explainable AI using GradCAM  
- CPU-friendly deployment  
- Dark themed medical UI  

---

## ⚙️ System Architecture

User → Flask Backend → Inference Pipeline → Output Visualization

---

## 📁 File Structure
```
brain-tumor-detection/
│
├── notebook/
│   └── train_model.ipynb
├── app/
│   ├── app.py
│   ├── pipeline.py
│   ├── model_utils.py
│   ├── models/
│   ├── uploads/
│   ├── static/
│   └── templates/
└── requirements.txt
```
---

## 🚀 Setup

1. Clone repo  
2. Install requirements  
3. Add model (.pth)  
4. Run Flask  

---

## 📊 Dataset

- Training: 5600 images  
- Validation: 840 (15%)  
- Training used: 4760 (85%)  
- Testing: 1600  

---

## 📈 Results

Accuracy: ~94%

---

## ⚠️ Disclaimer

For educational purposes only.
