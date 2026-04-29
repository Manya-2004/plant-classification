import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import glob
from pathlib import Path
import cv2

st.set_page_config(page_title="Plant Identifier", page_icon="🌱", layout="wide")

@st.cache_resource
def load_app_data():
    """Load everything needed for the app"""
    try:
        # Load models
        plant_model = tf.keras.models.load_model("models/plant_classifier.h5")
        disease_model = tf.keras.models.load_model("models/disease_classifier.h5")
        
        # Get input shapes
        plant_shape = plant_model.input_shape[1:3]
        disease_shape = disease_model.input_shape[1:3]
        
        # Get classes from dataset (fallback to model output size)
        plant_classes = get_dataset_classes("data/plant_metadata/Healthy")
        disease_classes = get_dataset_classes("data/plant_metadata/diseases")
        
        return {
            'plant_model': plant_model,
            'disease_model': disease_model,
            'plant_shape': plant_shape,
            'disease_shape': disease_shape,
            'plant_classes': plant_classes[:plant_model.output_shape[-1]],
            'disease_classes': disease_classes[:disease_model.output_shape[-1]]
        }
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

def get_dataset_classes(folder_path):
    """Get class names from folder structure"""
    classes = []
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                clean_name = item.replace('_', ' ').replace('-', ' ').title()
                classes.append(clean_name)
    return classes

def preprocess_image(image, target_size):
    """Safe image preprocessing"""
    image = np.array(image)
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = image[:,:,:3]
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, 0)

def main():
    st.title("🌱 Plant Disease & Species Identifier")
    
    # Load data
    with st.spinner("Loading models..."):
        data = load_app_data()
    
    st.success(f"✅ Ready! {len(data['plant_classes'])} plants, {len(data['disease_classes'])} diseases")
    
    # Tabs
    tab1, tab2 = st.tabs(["📸 Image Analysis", "🌿 Plant Gallery"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload plant image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded", use_column_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🌿 Identify Plant"):
                    img = preprocess_image(image, data['plant_shape'])
                    pred = data['plant_model'].predict(img, verbose=0)
                    plant_idx = np.argmax(pred[0])
                    confidence = pred[0][plant_idx]
                    
                    st.success(f"**{data['plant_classes'][plant_idx]}**")
                    st.info(f"Confidence: {confidence:.1%}")
            
            with col2:
                if st.button("🦠 Detect Disease"):
                    img = preprocess_image(image, data['disease_shape'])
                    pred = data['disease_model'].predict(img, verbose=0)
                    disease_idx = np.argmax(pred[0])
                    confidence = pred[0][disease_idx]
                    
                    st.success(f"**{data['disease_classes'][disease_idx]}**")
                    st.info(f"Confidence: {confidence:.1%}")
    
    with tab2:
        st.subheader("Browse Plants")
        search = st.text_input("Search plants")
        
        if search:
            matches = [p for p in data['plant_classes'] if search.lower() in p.lower()]
            if matches:
                selected = st.selectbox("Select:", matches)
                
                # Show sample images
                folder_name = selected.replace(' ', '_')
                img_folder = f"data/plant_metadata/Healthy/{folder_name}"
                
                if os.path.exists(img_folder):
                    imgs = glob.glob(f"{img_folder}/*.jpg")[:4]
                    if imgs:
                        cols = st.columns(4)
                        for i, img_path in enumerate(imgs):
                            with cols[i]:
                                st.image(img_path, use_column_width=True)

if __name__ == "__main__":
    main()