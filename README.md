# 🩺 RadiantTrust AI: Medical Imaging Diagnostics Engine

An end-to-end medical application designed to analyze chest X-ray scans, classify potential anomalies, and eliminate the deep learning "black box" problem using visual attention mapping.

## 🔗 Live Application Profile
You can access and interact with the deployed cloud instance of this application here:
👉 **[RadiantTrust AI Live Dashboard](https://anuragn2107-radianttrust-ai.hf.space/)**

---

## 🛠️ The Core Technical Stack
* **Deep Learning Engine:** PyTorch & Torchvision
* **Neural Architecture:** ResNet18 (Leveraging Transfer Learning pre-trained on ImageNet)
* **Explainability Interface:** Custom Grad-CAM (Gradient-weighted Class Activation Mapping)
* **Application Frontend:** Streamlit User Interface
* **Environment Blueprint:** Python 3.10-slim Docker Container

---

## 🧠 System Methodology & Design

### 1. Architectural Strategy
Standard Convolutional Neural Networks (CNNs) built from scratch struggle to generalize on subtle pathological variations. This system adapts **ResNet18** via Transfer Learning. The lower convolutional layers are locked to retain rich edge and structural feature extractions, while the final fully connected classification tier is modified to handle binary outputs (**Normal** vs. **Pneumonia**).

### 2. The Interpretability Engine (Grad-CAM)
To establish clinical trust, doctors must know *why* a model made a decision. This pipeline registers hooks into the final convolutional layer (`layer4`) of the ResNet model. 

When an image is processed, the system tracks the gradients flowing back into that layer during inference. It applies **Global Average Pooling** to those gradients to calculate importance weights for each feature map, creating a thermodynamic spatial heatmap. Red clusters isolate areas that heavily influenced a positive pneumonia diagnosis.

---

## 📂 Project Repository Structure
* **`app.py`** - Core Streamlit logic handling image processing, inference pipelines, and heatmap transformations.
* **`Dockerfile`** - The environment recipe that configures the underlying Linux server, handles port mapping, and boots the application.
* **`requirements.txt`** - Clean package manifest specifying precise library versions (including headless computer vision builds).
* **`pneumonia_resnet18.pth`** - Pre-trained state dictionary model weights generated during the training phase.
* **`.gitignore`** - Instructs the version control system to skip tracking temporary operating system files and local python caches.

---

## 🚀 Local Installation & Container Rebuild
If you want to run this application container locally on your own machine using Docker:

1. Clone the repository files into a directory.
2. Ensure your model weights file (`pneumonia_resnet18.pth`) is placed inside that folder.
3. Open your terminal in that folder and build the image:
   ```bash
   docker build -t radianttrust-ai .
