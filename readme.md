# 🛡️ Multimodal Phishing Detection with Federated Learning

This project presents a **research-grade phishing detection system** that leverages **text, URLs, and images** with deep learning, traditional ML, and **Federated Learning (FL)**. It features local model training and simulation of decentralized training for enhanced privacy and robustness.

---

<img width="1900" height="1272" alt="image" src="https://github.com/user-attachments/assets/5a50cd97-8297-4324-9981-aecde519782d" />


---

## 💡 Project Highlights

- **Multimodal Detection**: Processes phishing attacks via **Email Text**, **URLs**, and **Phishing Images**
- **Federated Learning**: Simulated client-server training to ensure privacy
- **Fusion Model**: Combines predictions using soft/hard voting
- **Modular Design**: Clean separation between local models and FL logic

---

## 🔍 Module Details

### 1. 📩 Email Phishing Detection
- 📘 `Email_msgs_prediction.ipynb`
- Techniques:
  - Text preprocessing → TF-IDF + ML models: Logistic Regression, SVM, Naive Bayes, XGBoost
  - BERT fine-tuning (HuggingFace)
- Fusion via F1-score weighted soft voting

---

### 2. 🌐 URL Phishing Detection
- 📘 `Url_Prediction.ipynb`
- Feature Engineering:
  - URL length, digits, domain tokens, entropy, etc.
- Models:
  - Logistic Regression, Random Forest, XGBoost, LightGBM
- Performance comparison using F1-score and confusion matrix

---

### 3. 🖼️ Image Phishing Detection
- 📘 `Image_prediction.ipynb`
- Dataset: Phishing brand logos vs. legitimate ones
- CNN Models:
  - ResNet50
  - EfficientNet-B0
  - DenseNet121
- Fusion: Majority voting among CNN outputs

---

### 4. 🔄 Multimodal Fusion
- 📘 `Multimodal.ipynb`
- Inputs: Output probabilities from email, URL, and image models
- Logic:
  - If 2 out of 3 modalities predict phishing → final label = phishing
  - Can run with partial inputs (e.g., email + URL only)

---

## 🧠 Federated Learning (FL)

- 📁 `FL Prototype/client_1/`
- 🔧 `fl_client.py`:
  - Simulates a federated client that trains on local data (email, URL, or image)
  - Uses Flower (FedAvg)
- 📦 `requirements.txt`: Dependencies for running FL simulation locally

> Future extension: Connect multiple clients and build real-time FL aggregator via Flower or PySyft

---

## 🛠️ Tech Stack

| Component       | Tools Used                                             |
|------------------|--------------------------------------------------------|
| ML/DL            | scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch  |
| NLP              | BERT, HuggingFace, TF-IDF                             |
| CNN              | ResNet, DenseNet, EfficientNet                        |
| Fusion           | Soft & Hard Voting (F1-weighted)                      |
| FL Framework     | Flower                                                |
| Utilities        | Matplotlib, Seaborn, Confusion Matrix                 |

---

## 🌐 Future Work

- 🔐 Integrate **Homomorphic Encryption (HE)** (e.g., via TenSEAL)
- 🌍 Run **real FL** with cloud-hosted clients
- 🧠 Use **Multimodal Transformers** (e.g., CLIP, BLIP) for unified modeling
- 🕵️ Implement automatic **suspicious word extraction** from URLs and text

---


