import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Set up page configurations
st.set_page_config(page_title="RadiantTrust AI", layout="centered")
st.title("🩺 RadiantTrust AI: Medical Diagnostics Engine")
st.markdown("---")

# 1. Load the model and map to CPU (since Hugging Face free tier runs on CPU)
@st.cache_resource
def load_diagnostic_model():
    device = torch.device("cpu")
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    # Load weights with map_location to avoid GPU-requirement errors
    model.load_state_dict(torch.load("pneumonia_resnet18.pth", map_location=device))
    model.eval()
    return model, device

try:
    model, device = load_diagnostic_model()
    st.sidebar.success("Diagnostic Weights Loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# 2. Production Grad-CAM Engine
class ProdGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))
        
    def get_map(self, tensor, idx):
        self.model.zero_grad()
        out = self.model(tensor)
        out[0][idx].backward()
        weights = torch.mean(self.gradients, dim=[2, 3])[0]
        heatmap = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            heatmap += w * self.activations[0, i, :, :]
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        if np.max(heatmap) != 0: 
            heatmap /= np.max(heatmap)
        return heatmap, out

# 3. File Upload Interface
uploaded_file = st.file_uploader("Upload Clinical Image (.png, .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Transformation pipeline tailored for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(raw_img).unsqueeze(0).to(device)
    
    # Run Inference & Grad-CAM (Target class 1 = Pneumonia)
    cam_processor = ProdGradCAM(model, model.layer4)
    heatmap, outputs = cam_processor.get_map(input_tensor, idx=1)
    
    probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
    classes = ["Normal", "Pneumonia"]
    predicted_class = classes[np.argmax(probabilities)]
    confidence = probabilities[np.argmax(probabilities)] * 100
    
    # Visual Post-Processing for the Heatmap Overlay
    cv_img = cv2.resize(np.array(raw_img), (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend raw image with the colored heatmap overlay (60% original image, 40% heatmap)
    overlayed_img = cv2.addWeighted(cv_img, 0.6, heatmap_colored, 0.4, 0)
    
    # Render Output Results
    st.subheader(f"Diagnostic Finding: **{predicted_class}** ({confidence:.2f}% Confidence)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv_img, caption="Uploaded Scan Profile")
    with col2:
        st.image(overlayed_img, caption="Grad-CAM Clinical Validation Heatmap")
        
    st.info("💡 Interpretability Guide: Red/hot clusters highlight spatial pixel regions heavily dictating a Pneumonia finding.")