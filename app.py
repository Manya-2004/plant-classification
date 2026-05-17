import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
from serpapi import GoogleSearch

from ragfile import PlantRAG

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Plant AI Assistant",
    layout="wide"
)

# =====================================================
# BACKGROUND IMAGE
# =====================================================

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20241210/pngtree-fresh-green-nature-blur-web-banner-image_16745501.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

[data-testid="stSidebar"]{
background: rgba(255,255,255,0.1);
backdrop-filter: blur(10px);
}

.block-container{
background: rgba(255,255,255,0.75);
padding: 2rem;
border-radius: 20px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# =====================================================
# LOAD RAG
# =====================================================

rag = PlantRAG()

# =====================================================
# SERP API KEY
# =====================================================

SERP_API_KEY = os.getenv("SERP_API_KEY")

# =========================
# LOAD MODELS
# =========================

@st.cache_resource

def load_models():

    models = {

        "level1": tf.keras.models.load_model(
            "models/level1_model.keras"
            compile=False, safe_mode=False ),

        "trees_fruits": tf.keras.models.load_model(
            "models/trees_model.keras"
            compile=False, safe_mode=False ),

        "herbs_medicinal": tf.keras.models.load_model(
            "models/herbs_model.keras"
            compile=False, safe_mode=False
        ),

        "vegetables": tf.keras.models.load_model(
            "models/vegetables_model.keras"
            compile=False, safe_mode=False
        ),

        "flowers": tf.keras.models.load_model(
            "models/flowers_model.keras"
            compile=False, safe_mode=False
        ),

        "weeds": tf.keras.models.load_model(
            "models/weeds_model.keras"
            compile=False, safe_mode=False
        ),

        "climbers": tf.keras.models.load_model(
            "models/climbers_model.keras"
            compile=False, safe_mode=False
        ),

        "disease": tf.keras.models.load_model(
            "models/diseases_model.keras"
            compile=False, safe_mode=False
        )
    }

    return models


models = load_models()

# =========================
# CLASS MAP BUILDER
# =========================


def build_class_map(data_dir):

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    return {
        i: name
        for i, name in enumerate(classes)
    }


class_maps = {

    "level1": build_class_map("data/level1"),

    "trees_fruits": build_class_map("data/trees_fruits"),

    "herbs_medicinal": build_class_map("data/herbs_medicinal"),

    "vegetables": build_class_map("data/vegetables"),

    "flowers": build_class_map("data/flowers"),

    "weeds": build_class_map("data/weeds"),

    "climbers": build_class_map("data/climbers"),

    "disease": build_class_map("data/diseases")
}
# =====================================================
# CLASS LABELS
# =====================================================

plant_classes = list(rag.class_mapping.keys())

disease_classes = [
    "leaf_spot",
    "powdery_mildew",
    "rust",
    "healthy",
    "blight",
    "mosaic_virus"
]

# =====================================================
# IMAGE PREPROCESS
# =====================================================

IMG_SIZE = 224

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0

    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)

    image = np.expand_dims(image, axis=0)

    return image

# =====================================================
# PREDICT PLANT
# =====================================================

def predict_plant(image):

    img = preprocess_image(image)

    prediction = models["level1"].predict(
        img,
        verbose=0
        )

    idx = np.argmax(prediction)

    plant_name = plant_classes[idx % len(plant_classes)]

    return plant_name

# =====================================================
# PREDICT DISEASE
# =====================================================

def predict_disease(image):

    img = preprocess_image(image)

    prediction = models["disease"].predict(
        img,
        verbose=0
        )

    idx = np.argmax(prediction)

    disease_name = disease_classes[idx]

    return disease_name


# =====================================================
# FETCH PLANT IMAGES
# =====================================================

@st.cache_data(ttl=86400)
def fetch_plant_images(plant_name, count=3):

    try:

        params = {

            "engine": "google_images",

            "q": f"{plant_name} plant",

            "api_key": SERP_API_KEY
        }

        search = GoogleSearch(params)

        results = search.get_dict()

        image_results = results.get(
            "images_results",
            []
        )

        image_urls = []

        for img in image_results[:count]:

            if "original" in img:

                image_urls.append(
                    img["original"]
                )

        return image_urls

    except Exception as e:

        st.error(
            f"Image Fetch Error: {e}"
        )

        return []

# =====================================================
# MODULE 1
# =====================================================

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌿 Plant AI Assistant")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Plant + Disease Identify",
        "Plant Information",
        "Health Assistant"
    ]
)

# =====================================================
# PLANT + DISEASE IDENTIFICATION
# =====================================================

if menu == "Plant + Disease Identify":

    st.title("🌱 Plant + Disease Identification")

    uploaded_file = st.file_uploader(
        "Upload Plant Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )

        with st.spinner("Analyzing Image..."):
            
            # =====================================================
            # AI PLANT PREDICTION
            # =====================================================

            ai_prediction = predict_plant(image)

            plant_name = ai_prediction

            # =====================================================
            # DISEASE PREDICTION
            # =====================================================

            disease_name = predict_disease(image)

            # =====================================================
            # RAG INFORMATION
            # =====================================================

            plant_info = rag.get_plant_info(
                plant_name
            )

            disease_info = rag.get_disease_info(
                disease_name
            )

            # =====================================================
            # FETCH IMAGES
            # =====================================================

            image_urls = fetch_plant_images(
                plant_name
            )
            
        st.success("Prediction Complete")

        # =====================================================
        # SHOW PREDICTION
        # =====================================================

        st.subheader("🌿 Plant Name")

        st.write(plant_name)

        # =====================================================
        # SHOW PLANT IMAGES
        # =====================================================

        st.subheader("🖼 Similar Plant Images")

        if image_urls:

            cols = st.columns(len(image_urls))

            for idx, url in enumerate(image_urls):

                try:

                    cols[idx].image(
                        url,
                        caption=plant_name,
                        use_container_width=True
                )

                except:

                    cols[idx].warning(
                        "Image could not load"
                )

        else:

            st.warning("No images found")

        # =====================================================
        # PLANT INFO
        # =====================================================

        st.subheader("📖 Plant Information")

        st.text(plant_info)

        # =====================================================
        # DISEASE INFO
        # =====================================================

        st.subheader("🦠 Disease Name")

        st.write(disease_name)

        st.subheader("💊 Disease Information")

        st.text(disease_info)

# =====================================================
# MODULE 2
# =====================================================

elif menu == "Plant Information":

    st.title("🌿 Plant Information")

    plant_input = st.text_input("Enter Plant Name")

    if st.button("Get Information"):

        if plant_input:

            info = rag.get_plant_info(plant_input)

            st.subheader("📖 Plant Details")
            st.text(info)

            st.subheader("🖼 Plant Images")

            image_urls = fetch_plant_images(plant_input)
            
            if image_urls:
            
                cols = st.columns(len(image_urls))
            
                for idx, url in enumerate(image_urls):
            
                    try:
            
                        cols[idx].image(
            
                            url,
                            caption=plant_input,
                            use_container_width=True
                        )
            
                    except:
            
                        cols[idx].warning("Image could not load")
            
            else:
            
                st.warning("No images found")

# =====================================================
# MODULE 3
# =====================================================

elif menu == "Health Assistant":

    st.title("🩺 Health Assistant")

    health_issues = st.text_area(
        "Enter health issues separated by comma"
    )

    if st.button("Recommend Plants"):

        if health_issues:

            result = rag.recommend_plants(health_issues)

            st.subheader("🌿 Recommended Plants")
            st.text(result)

            plants = []

            lines = result.split("\n")

            for line in lines:

                if "." in line and "(" in line:

                    name = line.split(".")[1].split("(")[0].strip()

                    plants.append(name)

            for plant in plants:

                st.markdown(f"## 🌱 {plant}")

                info = rag.get_plant_info(plant)

                st.text(info)

                image_urls = fetch_plant_images(plant)

                if image_urls:
                    
                    cols = st.columns(len(image_urls))
                    
                    for idx, url in enumerate(image_urls):
                        
                        try:
                            
                            cols[idx].image(
                                url,
                                caption=plant,
                                use_container_width=True
                            )
                        except:
                            cols[idx].warning("Image could not load")
                            
                else:
                    
                    st.warning("No images found")
