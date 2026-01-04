# =========================================================
# Image Classification Web App (CPU-based)
# Using PyTorch + Streamlit + ResNet18
# =========================================================

# -----------------------------
# Step 1 & 2: Import libraries
# -----------------------------
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Page configuration
# -----------------------------
st.set_page_config(
    page_title="CPU Image Classification App",
    layout="centered"
)

st.title("Image Classification using ResNet18 (CPU)")
st.write("This web application classifies images using a pre-trained ResNet18 model.")

# -----------------------------
# Step 3: Configure CPU only
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Step 4: Load pre-trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model

model = load_model()

# -----------------------------
# Step 5: Image preprocessing
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load ImageNet labels
# -----------------------------
@st.cache_data
def load_labels():
    labels = pd.read_csv(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        header=None
    )
    return labels[0].tolist()

labels = load_labels()

# -----------------------------
# Step 6: Image upload UI
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Step 7–10: Inference & display
# -----------------------------
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Model inference (no gradient)
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = F.softmax(outputs[0], dim=0)

    # Step 8: Top-5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    results_df = pd.DataFrame({
        "Class": [labels[i] for i in top5_idx],
        "Probability": [float(p) for p in top5_prob]
    })

    st.subheader("Top-5 Predicted Classes")
    st.table(results_df)

    # Step 9: Bar chart visualization
    fig, ax = plt.subplots()
    ax.barh(results_df["Class"], results_df["Probability"])
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 Prediction Probabilities")

    st.pyplot(fig)

    # Step 10: Discussion (brief display)
    st.markdown("### Process Path")
    st.markdown("""
    1. User uploads an image  
    2. Image is resized, cropped, normalized, and converted to tensor  
    3. Tensor is passed into ResNet18 model  
    4. Model outputs prediction scores  
    5. Softmax converts scores into probabilities  
    6. Top-5 predictions are displayed in table and bar chart  
    """)

else:
    st.info("👆 Please upload an image to start classification.")
