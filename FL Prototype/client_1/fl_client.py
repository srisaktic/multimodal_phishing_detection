# FL and model management
import flwr as fl
import torch
import joblib
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Hugging Face Transformers for BERT
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# PyTorch dataset handling
from torch.utils.data import Dataset, DataLoader

# For training CNNs on phishing images
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Your model architectures
from models_arch.resnet_model import get_resnet_model
from models_arch.densenet_model import get_densenet_model
from models_arch.efficientnet_model import get_efficientnet_model
from models_arch.image_fusion_model import ImageFusionModel

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ✅ Already preprocessed as integers outside
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)  # ✅ just one label
        return item
    

def train_cnn(model, dataloader, device, epochs=1):
    """
    Trains a given CNN model on image phishing dataset for 1 epoch.
    """
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model

class PhishingClient(fl.client.NumPyClient):
    def __init__(self):
        print("\n🚀 Initializing client...")

        from glob import glob

        # === Load latest Email BERT global model ===
        email_model_base = "model_store/bert_email"
        self.email_model = AutoModelForSequenceClassification.from_pretrained(email_model_base)

        email_global_checkpoints = sorted(glob("model_store/global_email_bert/round_*.pt"))
        if email_global_checkpoints:
            print(f"📥 Loading global email BERT from {email_global_checkpoints[-1]}")
            state_dict = torch.load(email_global_checkpoints[-1], map_location="cpu")
            self.email_model.load_state_dict({k: torch.tensor(v) for k, v in zip(self.email_model.state_dict().keys(), state_dict)})

        # === Load global URL logistic regression if exists
        url_model_path = "model_store/global_url_logreg.pkl"
        if os.path.exists(url_model_path):
            print("📥 Loading global URL LogReg model...")
            url_data = joblib.load(url_model_path)
            from sklearn.linear_model import LogisticRegression
            self.url_model = LogisticRegression()
            self.url_model.coef_ = url_data["coef_"]
            self.url_model.intercept_ = url_data["intercept_"]
        else:
            print("📦 Loading base URL LogReg model...")
            self.url_model = joblib.load("model_store/logistic_regression_url.pkl")

        # === Image Fusion CNN ===
        if os.path.exists("model_store/global_image_fusion.pkl"):
            print("📥 Loading global CNN fusion model from round 1...")
            self.resnet = get_resnet_model()
            self.densenet = get_densenet_model()
            self.efficientnet = get_efficientnet_model()
            self.image_fusion = ImageFusionModel(
                self.resnet, self.densenet, self.efficientnet,
                weights=[0.3286, 0.3294, 0.3420]
            )
            state_dict = torch.load("model_store/global_image_fusion.pkl", map_location="cpu")
            model_keys = list(self.image_fusion.state_dict().keys())
            new_state_dict = {k: torch.tensor(v) for k, v in zip(model_keys, state_dict)}
            self.image_fusion.load_state_dict(new_state_dict)
        else:
            print("📦 Loading pretrained individual CNN models...")
            self.resnet = get_resnet_model()
            self.resnet.load_state_dict(torch.load("model_store/resnet50_image.pth", map_location="cpu"))

            self.densenet = get_densenet_model()
            self.densenet.load_state_dict(torch.load("model_store/densenet121_image.pth", map_location="cpu"))

            self.efficientnet = get_efficientnet_model()
            self.efficientnet.load_state_dict(torch.load("model_store/efficientnet_b0_image.pth", map_location="cpu"))

            self.image_fusion = ImageFusionModel(
                self.resnet, self.densenet, self.efficientnet,
                weights=[0.3286, 0.3294, 0.3420]
            )

        print("✅ All models initialized.\n")

    def get_parameters(self, config=None):
        print("📤 Sending local model parameters to server...")

        # === Email BERT parameters
        email_params = [val.cpu().numpy() for val in self.email_model.state_dict().values()]

        # === URL Logistic Regression parameters
        url_params = [
            np.array(self.url_model.coef_).flatten(),
            np.array(self.url_model.intercept_).flatten()
            ]

        # === Image Fusion CNN parameters
        image_params = [val.cpu().numpy() for val in self.image_fusion.state_dict().values()]

        # Return as list of 3 parts
        return email_params + url_params + image_params
    

    def fit(self, parameters, config):
        import torch
        device = torch.device("cpu")
        print("\n🔁 Starting local training for all models...\n")

        # ============================
        # 1️⃣ EMAIL BERT TRAINING
        # ============================
        print("📚 Training Email BERT...")
        # === 1. Load and prepare dataset using your original column names
        df = pd.read_csv("data/email.csv")

        # Drop rows with missing values in the actual column names
        df = df.dropna(subset=["Email Text", "Email Type"])

        # Convert labels: Safe Email → 0, Phishing Email → 1
        df = df[df["Email Type"].isin(["Safe Email", "Phishing Email"])]
        df["label"] = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1}).astype(int)  

        # Extract raw email texts and binary labels
        texts = df["Email Text"].tolist()
        labels = df["label"].tolist()

        # Tokenize the texts using your BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained("model_store/bert_email")
        dataset = EmailDataset(texts, labels, tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=8,
            num_train_epochs=1,
            logging_dir="./logs",
            disable_tqdm=False,
            no_cuda=True  # ✅ Disable CUDA/MPS, force CPU
        )

        trainer = Trainer(model=self.email_model, args=training_args, train_dataset=dataset)
        trainer.train()
        print("✅ Email BERT training done.\n")

        # ============================
        # 2️⃣ URL LOGISTIC REGRESSION
        # ============================
        print("🔗 Training URL Logistic Regression...")

        url_df = pd.read_csv("data/url.csv")
        url_df = url_df.dropna(subset=["status"])
        url_df['status'] = url_df['status'].map({"legitimate": 0, "phishing": 1})

        X = url_df.drop(columns=["status", "url"], errors="ignore")  # Keep only features
        y = url_df['status'].values

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        self.url_model = model

        print("✅ URL LogReg training done.\n")

        # ============================
        # 3️⃣ CNN FUSION MODEL TRAINING
        # ============================
        print("🖼️ Training CNN fusion model...")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        image_dataset = ImageFolder(root="data/image", transform=transform)
        image_loader = DataLoader(image_dataset, batch_size=16, shuffle=True)

      

        self.resnet = train_cnn(self.resnet, image_loader, device)
        self.densenet = train_cnn(self.densenet, image_loader, device)
        self.efficientnet = train_cnn(self.efficientnet, image_loader, device)

        # Rebuild fusion model with updated CNNs
        self.image_fusion = ImageFusionModel(
            self.resnet, self.densenet, self.efficientnet,
            weights=[0.3286, 0.3294, 0.3420]
        )

        print("✅ CNN fusion training done.\n")

        # ============================
        # 💾 Save Trained Models Locally
        # ============================
        torch.save(self.email_model.state_dict(), "results/bert_email_trained.pt")
        joblib.dump(self.url_model, "results/logistic_regression_url_trained.pkl")
        torch.save(self.image_fusion.state_dict(), "results/image_fusion_trained.pth")

        print("✅ All models saved locally.\n")

        # Return updated model weights
        return self.get_parameters(), len(dataset), {}
    
    def evaluate(self, parameters, config):
        """
        This function can be extended for real validation,
        but for now we just return dummy values.
        """
        print("🧪 [Optional] Evaluation skipped.")
        return 0.0, 0, {}  # loss, num_examples, metrics
    

if __name__ == "__main__":
    fl.client.start_client(
        server_address="localhost:8080",
        client=PhishingClient().to_client(),  # ✅ Convert to Flower client interface
        grpc_max_message_length=1024 * 1024 * 1024  # 1GB
    )







