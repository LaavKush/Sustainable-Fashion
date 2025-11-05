# 🧵 Fabric Sustainability Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project%20Status-Active-brightgreen)
![Model Accuracy](https://img.shields.io/badge/Image%20Model%20Accuracy-88%25-blue)


> 🌱 A Machine Learning–powered dashboard that classifies fabrics as **Sustainable** or **Unsustainable** using **text and image analysis**, helping the fashion industry move towards eco-friendly choices.

---

## 📖 Table of Contents
1. [Overview](#overview)  
2. [Motivation](#motivation)  
3. [Key Features](#key-features)   
4. [Tech Stack](#tech-stack)
5. [Dataset](#dataset)  
6. [Model Training](#model-training)  
7. [Streamlit Dashboard](#streamlit-dashboard)  
8. [Results](#results)  
9. [How to Run](#how-to-run)  
10. [Future Enhancements](#future-enhancements)  
11. [Acknowledgements](#acknowledgements)  

---

## 🌍 Overview
This project is a part of my work on **sustainable fashion technology** — an intersection of AI and sustainability.  
It uses **Machine Learning** to predict the sustainability of fabrics based on:

- 🧵 **Text classification:** fabric names and material descriptions  
- 🖼️ **Image classification:** visual properties of fabric textures  

A **Streamlit dashboard** allows users to upload fabric images or input fabric materials and receive instant sustainability predictions with explainable AI (Grad-CAM visualizations).

---

## 💡 Motivation
The global fashion industry contributes up to **10% of global carbon emissions**.  
Materials like *polyester, nylon, and acrylic* have high environmental costs, whereas *organic cotton, linen, hemp, and livaeco* are eco-friendly alternatives.

👉 This project aims to:
- Support **transparent sustainability decisions** for designers and retailers  
- Educate users on **eco-conscious fabric selection**  
- Combine **AI explainability** with **user-friendly design**

---

## 🚀 Key Features

| Feature | Description |
|----------|--------------|
| 🧠 **Dual-Mode Classification** | Predict sustainability from both image and text |
| 📷 **Image Model (EfficientNetB0)** | Trained on fabric texture and color patterns |
| ✍️ **Text Model** | Trained on fabric names and materials |
| 🔥 **Explainable AI (Grad-CAM)** | Highlights image regions influencing predictions |
| 📊 **Dashboard Analytics** | Pie charts & stats of sustainable vs unsustainable materials |
| 💬 **Google Sheets Integration** | Dataset dynamically fetched from Google Sheets |
| 💎 **Interactive Streamlit UI** | Upload, visualize, and analyze fabrics with ease |

---

---

## 🛠️ Tech Stack

**Languages:** Python, HTML (via Streamlit)  
**Libraries:**  
- TensorFlow / Keras  
- Pandas, NumPy  
- Matplotlib, Plotly  
- Streamlit  
- SHAP, Grad-CAM  

---

## 📂 Dataset

- **Source:** Custom dataset created from a [Google Sheet](https://docs.google.com/spreadsheets) containing:
  - Fabric name  
  - Material description  
  - Image URL  
  - Label → `Sustainable` / `Unsustainable`

| Fabric Name | Material | Label |
|--------------|-----------|-------|
| Organic Cotton | Natural | Sustainable |
| Polyester Fleece | Synthetic | Unsustainable |

---

## 🧮 Model Training

### 🔹 Image Classification
- **Architecture:** EfficientNetB0 (transfer learning)
- **Input size:** 224×224×3
- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 12 (fine-tuned)  
- **Explainability:** Grad-CAM visualizations  

### 🔹 Text Classification
- **Vectorizer:** TF-IDF  
- **Model:** Logistic Regression / Naive Bayes  
- **Goal:** Detect sustainability from material names

---

## 🖥️ Streamlit Dashboard

### 🧩 Tabs

| Tab | Description |
|------|-------------|
| 🏠 **Predict** | Upload an image or enter fabric name to check sustainability |
| 📊 **Insights** | View dataset trends (sustainable vs unsustainable) |
| ℹ️ **About Project** | Learn about the purpose and tech stack |

### Example Workflow
1. Upload a fabric image (e.g., *cotton.jpg*)  
2. Dashboard shows:
   - Prediction: **Sustainable 🌿**
   - Confidence: *0.92*  
   - Grad-CAM highlighting regions influencing the decision  

---

## 📈 Results

| Metric | Image Model | Text Model |
|---------|--------------|-------------|
| Accuracy | ~88% | ~83% |
| Precision | 0.89 | 0.81 |
| Recall | 0.87 | 0.84 |
| F1-score | 0.88 | 0.82 |

### Grad-CAM Visualization
> Highlights the texture and color regions contributing most to the sustainability decision.

---

## ⚙️ How to Run

### 1️⃣ Access the dashboard

Open the local URL or public URL shown in the terminal 

## Future Enhancements

| Feature                              | Description                                        |
| ------------------------------------ | -------------------------------------------------- |
| 🧾 **Add sustainability scoring**    | Assign environmental impact scores (1–10)          |
| 🧠 **LLM Integration**               | Use GPT/LLM to describe fabric eco-impact          |
| 🌏 **More fabric types**             | Expand to blended and regional textiles            |
| 🔍 **Visual Search**                 | Find eco-friendly alternatives to uploaded fabrics |
| 🔧 **Deploy on Hugging Face Spaces** | Make the dashboard publicly accessible             |


## Acknowledgment
- TensorFlow Team for transfer learning models
- Google Sheets API for easy dataset integration
- Streamlit Community for open-source support
- Mentors at ADTC & SheFi for guidance in sustainability tech

## Authors
- Sania Verma (B.Tech IT)
- Laavanya Kushwaha (B.Tech IT)

