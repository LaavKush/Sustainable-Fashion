import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os, requests

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Fabric Sustainability Classifier 🧵", page_icon="🧶", layout="wide")
st.title("🧵 Fabric Sustainability Classifier Dashboard")

# ------------------ CONSTANTS ------------------
class_names = ['Sustainable', 'Unsustainable', 'Neutral']

# ✅ Replace these links with your actual working file links
# Google Drive direct download for image model (make sure 'Anyone with link' is enabled)
IMAGE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1EHDLdhj7dXE-0Z9VlVal_d4DsM4osBR9"
TEXT_MODEL_DIR = "./bert_text_model"
# If your BERT model files are hosted on Hugging Face or Drive, replace this with that link:
TEXT_MODEL_URL = "https://huggingface.co/YOUR_USERNAME/bert_text_model/resolve/main/pytorch_model.bin"

# ------------------ HELPER: Download model if missing ------------------
def download_file(url, dest_path):
    """Downloads a file from a given URL if not already present."""
    if not os.path.exists(dest_path):
        with st.spinner(f"📥 Downloading {os.path.basename(dest_path)}..."):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success(f"{os.path.basename(dest_path)} downloaded successfully ✅")
            else:
                st.error(f"Failed to download {os.path.basename(dest_path)}")

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_image_model():
    model_path = "fabric_sustainability_image_model.h5"
    if not os.path.exists(model_path):
        download_file(IMAGE_MODEL_URL, model_path)

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading image model: {e}")
        return None

@st.cache_resource
def load_text_model():
    os.makedirs(TEXT_MODEL_DIR, exist_ok=True)
    model_file = os.path.join(TEXT_MODEL_DIR, "pytorch_model.bin")
    
    # Auto-download if missing
    if not os.path.exists(model_file):
        download_file(TEXT_MODEL_URL, model_file)
    
    try:
        tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None

# ------------------ LOAD ALL MODELS ------------------
with st.spinner("⚙️ Loading AI models..."):
    image_model = load_image_model()
    tokenizer, text_model = load_text_model()
st.success("✅ Models loaded successfully!")

# ------------------ SIDEBAR ------------------
st.sidebar.title("🧶 Fabric Sustainability Dashboard")
tab = st.sidebar.radio("Navigate to:", ["🏠 Predict", "📊 Insights", "ℹ️ About Project"])

# ------------------ TAB 1: PREDICTION ------------------
if tab == "🏠 Predict":
    st.header("🧩 Predict Fabric Sustainability")
    st.write("Upload a fabric image or enter its material description to check sustainability level.")

    col1, col2 = st.columns(2)

    # ---- IMAGE MODEL ----
    with col1:
        uploaded_file = st.file_uploader("📸 Upload Fabric Image", type=["jpg", "jpeg", "png"])
        if uploaded_file and image_model:
            image = Image.open(uploaded_file).convert('RGB').resize((224, 224))
            st.image(image, caption="Uploaded Fabric", use_container_width=True)

            img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
            prediction = image_model.predict(img_array)
            pred_idx = np.argmax(prediction)
            pred_class = class_names[pred_idx]
            confidence = np.max(prediction)

            if pred_class == "Sustainable":
                st.success(f"🖼️ **Image Model Prediction:** {pred_class} ({confidence:.2f})")
            elif pred_class == "Unsustainable":
                st.error(f"🖼️ **Image Model Prediction:** {pred_class} ({confidence:.2f})")
            else:
                st.info(f"🖼️ **Image Model Prediction:** {pred_class} ({confidence:.2f})")

    # ---- TEXT MODEL ----
    with col2:
        fabric_text = st.text_area("🧵 Enter fabric description or material composition:")
        if fabric_text.strip() and text_model and tokenizer:
            with st.spinner("Analyzing text with BERT..."):
                inputs = tokenizer(fabric_text, return_tensors="pt", truncation=True, padding=True, max_length=64)
                with torch.no_grad():
                    outputs = text_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_idx].item()

            pred_class = class_names[pred_idx]

            if pred_class == "Sustainable":
                st.success(f"🌱 **Text Model Prediction:** {pred_class} ({confidence:.2f})")
            elif pred_class == "Unsustainable":
                st.error(f"⚠️ **Text Model Prediction:** {pred_class} ({confidence:.2f})")
            else:
                st.info(f"🧶 **Text Model Prediction:** {pred_class} ({confidence:.2f})")

# ------------------ TAB 2: INSIGHTS ------------------
elif tab == "📊 Insights":
    st.header("📊 Sustainability Insights Dashboard")
    st.write("Visual overview of sustainable, unsustainable, and neutral fabrics from your dataset.")

    # Google Sheet CSV link
    sheet_url = "https://docs.google.com/spreadsheets/d/1_1lIVc_YgWODOEmBXKqUxrTa3CYPZybrL4ZLbzGZGZE/export?format=csv"
    
    try:
        df = pd.read_csv(sheet_url)

        if "sustainability_label" in df.columns:
            label_counts = df["sustainability_label"].value_counts().reset_index()
            label_counts.columns = ["Label", "Count"]

            fig = px.pie(
                label_counts,
                values="Count",
                names="Label",
                title="Fabric Sustainability Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)

            if "material" in df.columns:
                st.subheader("📈 Top 5 Most Common Materials by Sustainability")
                top_materials = (
                    df.groupby("sustainability_label")["material"]
                    .count()
                    .sort_values(ascending=False)
                    .head(5)
                )
                st.bar_chart(top_materials)
        else:
            st.warning("⚠️ 'sustainability_label' column not found in your dataset.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# ------------------ TAB 3: ABOUT ------------------
elif tab == "ℹ️ About Project":
    st.header("ℹ️ About the Project")
    st.write("""
    **Fabric Sustainability Classifier** combines both *image* and *text* AI models 
    to predict how environmentally sustainable a fabric is.  

    ### 🚀 Features
    - 🖼️ CNN Image Classification (TensorFlow/Keras)
    - 🧵 Text Classification (Fine-tuned BERT)
    - 📊 Interactive Sustainability Dashboard
    - 🌈 Future Upgrade: Grad-CAM Interpretability

    ### 🧠 Tech Stack
    - TensorFlow / Keras  
    - HuggingFace Transformers  
    - Streamlit  
    - Google Sheets API  
    - Plotly Visualization

    ---
    👩‍💻 **Developed with ❤️ by Sania Verma and Laavanya Kushwaha**
    """)
