<h1 align="center">🛡️ Multimodal Phishing Detection System</h1>

<p align="center">
  AI-powered phishing detection using Email, URL, and Image analysis with Federated Learning 🚀
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Multimodal-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Federated Learning-Enabled-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NLP-BERT-orange?style=for-the-badge" />
</p>

---

## 🎯 Project Overview

<p align="center">
A research-grade phishing detection system that analyzes <b>Email Text</b>, <b>URLs</b>, and <b>Images</b> using Machine Learning, Deep Learning, and Federated Learning (FL).  
</p>

<p align="center">
It is designed to simulate decentralized training environments while maintaining high accuracy and data privacy.
</p>

---

## 🖼️ System Overview

<p align="center">
  <img src="https://github.com/srisaktic/multimodal_phishing_detection/blob/main/image.png?raw=true" width="700"/>
</p>

---

## ⚙️ How It Works

<table align="center">
<tr>
<td>

🔹 **Email Detection** → NLP + BERT + ML models  
🔹 **URL Detection** → Feature Engineering + ML models  
🔹 **Image Detection** → CNN architectures (ResNet, DenseNet, EfficientNet)  
🔹 **Fusion Layer** → Combines predictions using voting  
🔹 **Federated Learning** → Simulates decentralized model training  

</td>
</tr>
</table>

---

## 🔍 Module Breakdown

### 📩 Email Phishing Detection
- TF-IDF + ML models (LogReg, SVM, Naive Bayes, XGBoost)
- BERT fine-tuning using HuggingFace
- F1-score weighted soft voting

---

### 🌐 URL Phishing Detection
- Feature engineering (length, entropy, domain tokens)
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Evaluated using F1-score & confusion matrix

---

### 🖼️ Image Phishing Detection
- CNN Models:
  - ResNet50  
  - EfficientNet-B0  
  - DenseNet121  
- Majority voting for final prediction

---

### 🔄 Multimodal Fusion
- Combines predictions from all modalities
- Logic:
  - If ≥2 modalities detect phishing → final output = phishing  
- Supports partial inputs (e.g., only URL + Email)

---

## 🧠 Federated Learning

<p align="center">
Simulates decentralized training where models are trained locally and aggregated globally using <b>Flower (FedAvg)</b>.
</p>

- Local training per client (email, URL, image)
- Aggregation via federated averaging
- Enhances privacy by avoiding centralized data sharing

---

## 🧰 Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,tensorflow,pytorch,docker" />
</p>

<p align="center">
<b>ML/DL:</b> Scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch  
<b>NLP:</b> BERT, HuggingFace, TF-IDF  
<b>CNN:</b> ResNet, DenseNet, EfficientNet  
<b>FL:</b> Flower  
<b>Visualization:</b> Matplotlib, Seaborn  
</p>

---

## 🧱 System Architecture

<p align="center">
<b>Email / URL / Image → Individual Models → Fusion Layer → Final Prediction</b>
</p>

---

## 📂 Project Structure

```bash
multimodal-phishing-detection/
│
├── email_model/
├── url_model/
├── image_model/
├── fusion/
├── fl_prototype/
├── notebooks/
└── README.md
