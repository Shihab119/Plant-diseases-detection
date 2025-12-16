# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import json
# from PIL import Image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # -----------------------------
# # CONFIG
# # -----------------------------
# MODEL_PATH = "plant_disease_classifier.h5"
# JSON_PATH = "plant_disease_classes_FIXED.json"
# IMG_SIZE = 224
# TOP_K = 5
# IMAGE_DISPLAY_WIDTH = 700

# # -----------------------------
# # LOAD MODEL & CLASSES
# # -----------------------------
# @st.cache_resource
# def load_model_and_classes():
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)

#     with open(JSON_PATH, "r") as f:
#         class_info = json.load(f)

#     return model, class_info["classes"]

# model, class_names = load_model_and_classes()

# # -----------------------------
# # PREPROCESS (EXACT TRAINING LOGIC)
# # -----------------------------
# def preprocess_img(img: Image.Image):
#     img = img.convert("RGB")
#     img = img.resize((IMG_SIZE, IMG_SIZE))
#     arr = np.array(img).astype("float32")
#     arr = np.expand_dims(arr, axis=0)
#     arr = preprocess_input(arr)
#     return arr

# # -----------------------------
# # SAFE SOFTMAX HANDLER
# # -----------------------------
# def normalize_preds(preds):
#     """
#     Apply softmax ONLY if model outputs logits.
#     Prevents double-softmax (0–10% bug).
#     """
#     if preds.max() > 1:
#         preds = tf.nn.softmax(preds, axis=-1).numpy()
#     return preds

# # -----------------------------
# # TOP-K PREDICTION
# # -----------------------------
# def predict_topk(img_array):
#     preds = model.predict(img_array)[0]
#     preds = normalize_preds(preds)

#     top_idx = preds.argsort()[-TOP_K:][::-1]
#     return [(class_names[i], float(preds[i] * 100)) for i in top_idx]

# # -----------------------------
# # TTA PREDICTION (FLIP AVERAGING)
# # -----------------------------
# def predict_tta(img: Image.Image):
#     img1 = img.resize((IMG_SIZE, IMG_SIZE))
#     img2 = img1.transpose(Image.FLIP_LEFT_RIGHT)

#     batch = np.stack([np.array(img1), np.array(img2)], axis=0).astype("float32")
#     batch = preprocess_input(batch)

#     preds = model.predict(batch)
#     preds = normalize_preds(preds)

#     avg_preds = preds.mean(axis=0)
#     top_idx = avg_preds.argsort()[-TOP_K:][::-1]

#     return [(class_names[i], float(avg_preds[i] * 100)) for i in top_idx]

# # -----------------------------
# # SIDEBAR
# # -----------------------------
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox(
#     "Select Page", ["Home", "About", "Disease Recognition"]
# )

# # -----------------------------
# # HOME
# # -----------------------------
# if app_mode == "Home":
#     st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
#     st.image("home_page.jpeg", width=IMAGE_DISPLAY_WIDTH)

#     st.markdown("""
#     **MobileNetV2 Plant Disease Classifier**

#     - Proper confidence scaling (0–100%)
#     - Top-5 predictions
#     - Test-Time Augmentation (TTA)
#     - Supports JPG / PNG / JFIF
#     """)

# # -----------------------------
# # ABOUT
# # -----------------------------
# elif app_mode == "About":
#     st.header("📘 About")
#     st.markdown("""
#     - **Model**: MobileNetV2
#     - **Classes**: 27
#     - **Inference**: Safe softmax handling
#     - **TTA**: Horizontal flip averaging
#     """)

# # -----------------------------
# # PREDICTION
# # -----------------------------
# elif app_mode == "Disease Recognition":
#     st.header("🔍 Disease Recognition")

#     uploaded_image = st.file_uploader(
#         "Upload Leaf Image",
#         type=["jpg", "jpeg", "png", "jfif"]
#     )

#     use_tta = st.checkbox("Use Test-Time Augmentation (Recommended)", value=True)

#     if uploaded_image is not None:
#         img = Image.open(uploaded_image)
#         st.image(img, width=IMAGE_DISPLAY_WIDTH)

#         if st.button("Predict"):
#             with st.spinner("Running inference..."):

#                 if use_tta:
#                     results = predict_tta(img)
#                 else:
#                     img_array = preprocess_img(img)
#                     results = predict_topk(img_array)

#             st.success("✅ Prediction Completed")

#             st.subheader("🔝 Top-5 Predictions")
#             for i, (cls, prob) in enumerate(results, 1):
#                 st.write(f"**{i}. {cls}** — {prob:.2f}%")

#             # Final prediction
#             final_class, final_conf = results[0]
#             st.markdown("---")
#             st.info(f"🌱 **Final Prediction:** {final_class}")
#             st.info(f"🎯 **Confidence:** {final_conf:.2f}%")
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import json
# import cv2
# from PIL import Image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # -----------------------------
# # CONFIG
# # -----------------------------
# MODEL_PATH = "plant_disease_classifier.h5"
# JSON_PATH = "plant_disease_classes_FIXED.json"
# IMG_SIZE = 224
# TOP_K = 5
# IMAGE_DISPLAY_WIDTH = 700

# # -----------------------------
# # LOAD MODEL & CLASSES
# # -----------------------------
# @st.cache_resource
# def load_model_and_classes():
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#     with open(JSON_PATH, "r") as f:
#         class_info = json.load(f)
#     return model, class_info["classes"]

# model, class_names = load_model_and_classes()

# # -----------------------------
# # BRIGHTNESS NORMALIZATION (CLAHE)
# # -----------------------------
# def normalize_brightness(img: Image.Image):
#     """
#     Fix over/under exposed images using CLAHE
#     """
#     img_np = np.array(img)

#     lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     l = clahe.apply(l)

#     lab = cv2.merge((l, a, b))
#     fixed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

#     return Image.fromarray(fixed)

# # -----------------------------
# # SAFE SOFTMAX HANDLER
# # -----------------------------
# def normalize_preds(preds):
#     """
#     Apply softmax ONLY if model outputs logits
#     """
#     if preds.max() > 1:
#         preds = tf.nn.softmax(preds, axis=-1).numpy()
#     return preds

# # -----------------------------
# # PREPROCESS (MATCHES TEST SCRIPT)
# # -----------------------------
# def preprocess_img(img: Image.Image):
#     img = img.convert("RGB")
#     img = normalize_brightness(img)
#     img = img.resize((IMG_SIZE, IMG_SIZE))

#     arr = np.array(img).astype("float32")
#     arr = np.expand_dims(arr, axis=0)
#     arr = preprocess_input(arr)

#     return arr

# # -----------------------------
# # TOP-K PREDICTION
# # -----------------------------
# def predict_topk(img_array):
#     preds = model.predict(img_array)[0]
#     preds = normalize_preds(preds)

#     top_idx = preds.argsort()[-TOP_K:][::-1]
#     return [(class_names[i], float(preds[i] * 100)) for i in top_idx]

# # -----------------------------
# # TTA PREDICTION (FLIP AVERAGING)
# # -----------------------------
# def predict_tta(img: Image.Image):
#     img = img.convert("RGB")
#     img = normalize_brightness(img)

#     img1 = img.resize((IMG_SIZE, IMG_SIZE))
#     img2 = img1.transpose(Image.FLIP_LEFT_RIGHT)

#     batch = np.stack(
#         [np.array(img1), np.array(img2)],
#         axis=0
#     ).astype("float32")

#     batch = preprocess_input(batch)

#     preds = model.predict(batch)
#     preds = normalize_preds(preds)

#     avg_preds = preds.mean(axis=0)
#     top_idx = avg_preds.argsort()[-TOP_K:][::-1]

#     return [(class_names[i], float(avg_preds[i] * 100)) for i in top_idx]

# # -----------------------------
# # BRIGHTNESS WARNING
# # -----------------------------
# def brightness_score(img: Image.Image):
#     gray = np.array(img.convert("L"))
#     return gray.mean()

# # -----------------------------
# # SIDEBAR
# # -----------------------------
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox(
#     "Select Page", ["Home", "About", "Disease Recognition"]
# )

# # -----------------------------
# # HOME
# # -----------------------------
# if app_mode == "Home":
#     st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
#     st.image("home_page.jpeg", width=IMAGE_DISPLAY_WIDTH)

#     st.markdown("""
#     **MobileNetV2 Plant Disease Classifier**

#     ✔ Robust to bright images  
#     ✔ CLAHE contrast normalization  
#     ✔ Top-5 predictions  
#     ✔ Test-Time Augmentation (TTA)  
#     ✔ Correct 0–100% confidence  
#     """)

# # -----------------------------
# # ABOUT
# # -----------------------------
# elif app_mode == "About":
#     st.header("📘 About")
#     st.markdown("""
#     - **Model**: MobileNetV2  
#     - **Classes**: 27  
#     - **Brightness Fix**: CLAHE  
#     - **Inference**: Safe Softmax + TTA  
#     """)

# # -----------------------------
# # PREDICTION
# # -----------------------------
# elif app_mode == "Disease Recognition":
#     st.header("🔍 Disease Recognition")

#     uploaded_image = st.file_uploader(
#         "Upload Leaf Image",
#         type=["jpg", "jpeg", "png", "jfif"]
#     )

#     use_tta = st.checkbox(
#         "Use Test-Time Augmentation (Recommended)",
#         value=True
#     )

#     if uploaded_image is not None:
#         img = Image.open(uploaded_image)
#         st.image(img, width=IMAGE_DISPLAY_WIDTH)

#         score = brightness_score(img)
#         if score > 210:
#             st.warning("⚠ Image is very bright. Brightness normalization applied.")

#         if st.button("Predict"):
#             with st.spinner("Running inference..."):
#                 if use_tta:
#                     results = predict_tta(img)
#                 else:
#                     img_array = preprocess_img(img)
#                     results = predict_topk(img_array)

#             st.success("✅ Prediction Completed")

#             st.subheader("🔝 Top-5 Predictions")
#             for i, (cls, prob) in enumerate(results, 1):
#                 st.write(f"**{i}. {cls}** — {prob:.2f}%")

#             final_class, final_conf = results[0]
#             st.markdown("---")
#             st.info(f"🌱 **Final Prediction:** {final_class}")
#             st.info(f"🎯 **Confidence:** {final_conf:.2f}%")
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import json
# import cv2
# from PIL import Image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# MODEL_PATH = "plant_disease_classifier.h5"
# JSON_PATH = "plant_disease_classes_FIXED.json"
# # JSON_PATH = "plant_disease_classes.json"
# IMG_SIZE = 224
# TOP_K = 5
# IMAGE_DISPLAY_WIDTH = 700


# @st.cache_resource
# def load_model_and_classes():
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#     with open(JSON_PATH, "r") as f:
#         class_info = json.load(f)
#     return model, class_info["classes"]

# model, class_names = load_model_and_classes()


# DISEASE_INFO = {
#     "Apple Scab Leaf": {
#         "description": "Apple scab is a fungal disease causing dark scabby lesions on apple leaves.",
#         "treatment": ["Apply captan or myclobutanil", "Remove fallen infected leaves"],
#         "prevention": ["Use resistant varieties", "Improve air circulation"]
#     },
#     "Apple leaf": {
#         "description": "Healthy apple leaf with no disease symptoms.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Regular orchard maintenance"]
#     },
#     "Apple rust leaf": {
#         "description": "Rust disease causes orange-yellow spots on leaves.",
#         "treatment": ["Apply sulfur fungicide"],
#         "prevention": ["Avoid nearby juniper plants"]
#     },
#     "Bell_pepper leaf": {
#         "description": "Healthy bell pepper leaf.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Balanced fertilization"]
#     },
#     "Bell_pepper leaf spot": {
#         "description": "Bacterial leaf spot causes dark lesions.",
#         "treatment": ["Copper-based sprays"],
#         "prevention": ["Avoid overhead watering"]
#     },
#     "Blueberry leaf": {
#         "description": "Healthy blueberry leaf.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Maintain acidic soil"]
#     },
#     "Cherry leaf": {
#         "description": "Healthy cherry leaf.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Proper pruning"]
#     },
#     "Corn Gray leaf spot": {
#         "description": "Gray rectangular fungal lesions on corn leaves.",
#         "treatment": ["Apply fungicides"],
#         "prevention": ["Crop rotation"]
#     },
#     "Corn leaf blight": {
#         "description": "Blight causes long gray-green lesions.",
#         "treatment": ["Use foliar fungicides"],
#         "prevention": ["Resistant hybrids"]
#     },
#     "Corn rust leaf": {
#         "description": "Rust disease shows reddish-brown pustules.",
#         "treatment": ["Apply fungicide if severe"],
#         "prevention": ["Use resistant varieties"]
#     },
#     "Peach leaf": {
#         "description": "Healthy peach leaf.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Routine orchard care"]
#     },
#     "Potato leaf early blight": {
#         "description": "Brown concentric rings appear on leaves.",
#         "treatment": ["Apply mancozeb"],
#         "prevention": ["Crop rotation"]
#     },
#     "Potato leaf late blight": {
#         "description": "Rapid leaf decay caused by fungus.",
#         "treatment": ["Systemic fungicides"],
#         "prevention": ["Avoid wet conditions"]
#     },
#     "Raspberry leaf": {
#         "description": "Healthy raspberry leaf.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Good pruning"]
#     },
#     "Soyabean leaf": {
#         "description": "Healthy soybean leaf.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Balanced nutrition"]
#     },
#     "Squash Powdery mildew leaf": {
#         "description": "White powdery fungal growth on leaves.",
#         "treatment": ["Use sulfur or neem oil"],
#         "prevention": ["Ensure airflow"]
#     },
#     "Strawberry leaf": {
#         "description": "Healthy strawberry leaf.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Proper irrigation"]
#     },
#     "Tomato Early blight leaf": {
#         "description": "Dark concentric lesions on tomato leaves.",
#         "treatment": ["Copper fungicide"],
#         "prevention": ["Avoid overhead watering"]
#     },
#     "Tomato Septoria leaf spot": {
#         "description": "Small dark spots with gray centers.",
#         "treatment": ["Chlorothalonil"],
#         "prevention": ["Mulching"]
#     },
#     "Tomato leaf": {
#         "description": "Healthy tomato leaf.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Proper fertilization"]
#     },
#     "Tomato leaf bacterial spot": {
#         "description": "Dark water-soaked spots on leaves.",
#         "treatment": ["Copper sprays"],
#         "prevention": ["Use certified seeds"]
#     },
#     "Tomato leaf late blight": {
#         "description": "Rapid browning and collapse of leaves.",
#         "treatment": ["Systemic fungicides"],
#         "prevention": ["Reduce humidity"]
#     },
#     "Tomato leaf mosaic virus": {
#         "description": "Mottled leaves and stunted growth.",
#         "treatment": ["Remove infected plants"],
#         "prevention": ["Disinfect tools"]
#     },
#     "Tomato leaf yellow virus": {
#         "description": "Yellowing and curling of leaves.",
#         "treatment": ["Control whiteflies"],
#         "prevention": ["Use insect nets"]
#     },
#     "Tomato mold leaf": {
#         "description": "Gray mold growth under leaves.",
#         "treatment": ["Apply fungicides"],
#         "prevention": ["Improve ventilation"]
#     },
#     "grape leaf": {
#         "description": "Healthy grape leaf.",
#         "treatment": ["No treatment required"],
#         "prevention": ["Regular vineyard monitoring"]
#     },
#     "grape leaf black rot": {
#         "description": "Circular brown lesions on grape leaves.",
#         "treatment": ["Myclobutanil fungicide"],
#         "prevention": ["Prune infected vines"]
#     }
# }


# def normalize_brightness(img: Image.Image):
#     img_np = np.array(img)
#     lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(2.0, (8, 8))
#     l = clahe.apply(l)
#     lab = cv2.merge((l, a, b))
#     return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

# def normalize_preds(preds):
#     if preds.max() > 1:
#         preds = tf.nn.softmax(preds, axis=-1).numpy()
#     return preds


# def preprocess_img(img: Image.Image):
#     img = normalize_brightness(img.convert("RGB"))
#     img = img.resize((IMG_SIZE, IMG_SIZE))
#     arr = np.expand_dims(np.array(img).astype("float32"), axis=0)
#     return preprocess_input(arr)


# def predict_topk(img_array):
#     preds = normalize_preds(model.predict(img_array)[0])
#     top_idx = preds.argsort()[-TOP_K:][::-1]
#     return [(class_names[i], preds[i] * 100) for i in top_idx]



# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


# if app_mode == "Home":
#     st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
#     st.image("home_page.jpeg", width=IMAGE_DISPLAY_WIDTH)


# elif app_mode == "About":
#     st.header("📘 About This Project")

#     st.markdown("""
#     ### 🔬 Model Details
#     - **Architecture:** MobileNetV2 (Transfer Learning)
#     - **Framework:** TensorFlow + Keras
#     - **Input Image Size:** 224 × 224
#     - **Total Classes:** 27
#     - **Image Preprocessing:** CLAHE + MobileNetV2 normalization
#     """)

#     st.markdown("### 🗂 Class Labels (Training Order)")
#     for idx, name in enumerate(class_names):
#         st.write(f"**{idx:02d} → {name}**")

#     st.markdown("""
#     ### 🎯 Purpose
#     This system helps identify plant leaf diseases using deep learning and
#     provides AI-based explanations and treatment suggestions for farmers and researchers.
#     """)


# elif app_mode == "Disease Recognition":
#     st.header("🔍 Disease Recognition")

#     uploaded_image = st.file_uploader(
#         "Upload Leaf Image",
#         type=["jpg", "jpeg", "png", "jfif"]
#     )

#     if uploaded_image is not None:
#         img = Image.open(uploaded_image)
#         st.image(img, width=IMAGE_DISPLAY_WIDTH)

#         if st.button("Predict"):
#             with st.spinner("Running inference..."):
#                 img_array = preprocess_img(img)
#                 results = predict_topk(img_array)

#             st.success("✅ Prediction Completed")

#             st.subheader("🔝 Top-5 Predictions")
#             for i, (cls, prob) in enumerate(results, 1):
#                 st.write(f"**{i}. {cls}** — {prob:.2f}%")

#             final_class, final_conf = results[0]
#             st.markdown("---")
#             st.info(f"🌱 **Final Prediction:** {final_class}")
#             st.info(f"🎯 **Confidence:** {final_conf:.2f}%")

            
#             st.markdown("---")
#             st.subheader(" AI Assistant")

#             info = DISEASE_INFO.get(final_class)
#             if info:
#                 st.markdown("### 🦠 What is this?")
#                 st.write(info["description"])

#                 st.markdown("### 💊 Treatment Suggestions")
#                 for t in info["treatment"]:
#                     st.write(f"- {t}")

#                 st.markdown("### 🛡 Prevention Tips")
#                 for p in info["prevention"]:
#                     st.write(f"- {p}")
#             else:
#                 st.write("Information not available for this disease.")



import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "plant_disease_classifier.h5"
JSON_PATH = "plant_disease_classes_FIXED.json"
IMG_SIZE = 224
TOP_K = 5
IMAGE_DISPLAY_WIDTH = 700

# -----------------------------
# LOAD MODEL & CLASSES
# -----------------------------
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(JSON_PATH, "r") as f:
        class_info = json.load(f)
    return model, class_info["classes"]

model, class_names = load_model_and_classes()

# -----------------------------
# DISEASE + MEDICINE INFO (ALL 27)
# -----------------------------
DISEASE_INFO = {

    "Apple Scab Leaf": {
        "description": "Fungal disease causing dark scabby lesions on apple leaves.",
        "medicine": ["Mancozeb", "Captan"],
        "treatment": ["Spray fungicide every 7–10 days"],
        "prevention": ["Use resistant varieties", "Improve air circulation"]
    },

    "Apple rust leaf": {
        "description": "Rust disease causing orange-yellow spots on leaves.",
        "medicine": ["Sulfur"],
        "treatment": ["Apply sulfur fungicide"],
        "prevention": ["Remove nearby juniper plants"]
    },

    "Apple leaf": {
        "description": "Healthy apple leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Regular orchard monitoring"]
    },

    "Bell_pepper leaf spot": {
        "description": "Bacterial disease causing dark lesions.",
        "medicine": ["Copper fungicide"],
        "treatment": ["Spray copper-based fungicide"],
        "prevention": ["Avoid overhead watering"]
    },

    "Bell_pepper leaf": {
        "description": "Healthy bell pepper leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Balanced fertilization"]
    },

    "Blueberry leaf": {
        "description": "Healthy blueberry leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Maintain acidic soil"]
    },

    "Cherry leaf": {
        "description": "Healthy cherry leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Proper pruning"]
    },

    "Corn Gray leaf spot": {
        "description": "Gray rectangular fungal lesions on corn leaves.",
        "medicine": ["Mancozeb"],
        "treatment": ["Apply foliar fungicide"],
        "prevention": ["Crop rotation"]
    },

    "Corn leaf blight": {
        "description": "Long gray-green lesions on corn leaves.",
        "medicine": ["Mancozeb"],
        "treatment": ["Spray fungicide early"],
        "prevention": ["Use resistant hybrids"]
    },

    "Corn rust leaf": {
        "description": "Reddish-brown pustules on leaves.",
        "medicine": ["Propiconazole"],
        "treatment": ["Apply fungicide if severe"],
        "prevention": ["Resistant varieties"]
    },

    "grape leaf black rot": {
        "description": "Brown circular lesions on grape leaves.",
        "medicine": ["Myclobutanil"],
        "treatment": ["Spray fungicide before flowering"],
        "prevention": ["Prune infected vines"]
    },

    "grape leaf": {
        "description": "Healthy grape leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Regular vineyard inspection"]
    },

    "Peach leaf": {
        "description": "Healthy peach leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Routine orchard care"]
    },

    "Potato leaf early blight": {
        "description": "Brown concentric rings on potato leaves.",
        "medicine": ["Mancozeb"],
        "treatment": ["Spray fungicide weekly"],
        "prevention": ["Crop rotation"]
    },

    "Potato leaf late blight": {
        "description": "Rapid leaf decay caused by fungus.",
        "medicine": ["Metalaxyl"],
        "treatment": ["Apply systemic fungicide immediately"],
        "prevention": ["Avoid wet conditions"]
    },

    "Raspberry leaf": {
        "description": "Healthy raspberry leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Good pruning"]
    },

    "Soyabean leaf": {
        "description": "Healthy soybean leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Balanced nutrition"]
    },

    "Squash Powdery mildew leaf": {
        "description": "White powdery fungal growth on leaf surface.",
        "medicine": ["Sulfur"],
        "treatment": ["Apply sulfur or neem oil"],
        "prevention": ["Ensure airflow"]
    },

    "Strawberry leaf": {
        "description": "Healthy strawberry leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Proper irrigation"]
    },

    "Tomato Early blight leaf": {
        "description": "Dark concentric lesions on tomato leaves.",
        "medicine": ["Mancozeb"],
        "treatment": ["Apply fungicide weekly"],
        "prevention": ["Avoid overhead watering"]
    },

    "Tomato Septoria leaf spot": {
        "description": "Small dark spots with gray centers.",
        "medicine": ["Chlorothalonil"],
        "treatment": ["Spray fungicide regularly"],
        "prevention": ["Remove infected leaves"]
    },

    "Tomato leaf bacterial spot": {
        "description": "Water-soaked bacterial lesions.",
        "medicine": ["Copper fungicide"],
        "treatment": ["Apply copper spray"],
        "prevention": ["Use certified seeds"]
    },

    "Tomato leaf late blight": {
        "description": "Rapid browning and collapse of leaves.",
        "medicine": ["Metalaxyl"],
        "treatment": ["Apply systemic fungicide"],
        "prevention": ["Reduce humidity"]
    },

    "Tomato leaf mosaic virus": {
        "description": "Mottled leaves and stunted growth.",
        "medicine": ["No chemical cure"],
        "treatment": ["Remove infected plants"],
        "prevention": ["Disinfect tools"]
    },

    "Tomato leaf yellow virus": {
        "description": "Yellowing and curling of leaves.",
        "medicine": ["Imidacloprid"],
        "treatment": ["Control whiteflies"],
        "prevention": ["Use insect nets"]
    },

    "Tomato mold leaf": {
        "description": "Gray mold on underside of leaves.",
        "medicine": ["Chlorothalonil"],
        "treatment": ["Apply fungicide"],
        "prevention": ["Improve ventilation"]
    },

    "Tomato leaf": {
        "description": "Healthy tomato leaf.",
        "medicine": ["Not required"],
        "treatment": ["No treatment required"],
        "prevention": ["Proper fertilization"]
    }
}

# -----------------------------
# MEDICINE IMAGES
# -----------------------------
MEDICINE_IMAGES = {
    "Mancozeb": "data:image/jpeg;base64,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",
    "Captan": "data:image/jpeg;base64,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",
    "Metalaxyl": "data:image/jpeg;base64,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",
    "Sulfur": "data:image/jpeg;base64,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",
    "Copper fungicide": "data:image/jpeg;base64,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",
    "Chlorothalonil": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEhUSEhIVFRUVFRAPFRUPDw8PEA8VFRIWFhUVFRUYHSggGBolHRUVITEhJSkrLjAuFx8zODMsNygtLisBCgoKDg0OGxAQGi0dHx0tLS0rLS4tLSsrLS0tLS0tLS0vLSstLS0rLS0tLS0tLS0rLSstLS0tLSstKy0tNy0tLf/AABEIAOEA4QMBIgACEQEDEQH/xAAcAAABBQEBAQAAAAAAAAAAAAADAAIEBQYBBwj/xAA+EAABAwIEAggDBgUCBwAAAAABAAIDBBEFEiExQVEGEyJhcYGRoTJSwRRCkrHR8AcjYqLhQ4IVFiRywtLx/8QAGgEAAgMBAQAAAAAAAAAAAAAAAgMAAQQFBv/EAC0RAAICAgECBAUDBQAAAAAAAAABAhEDIRIEMSIyQVETcZHh8GGBsQUUI6HB/9oADAMBAAIRAxEAPwDFNQZmqbDHdcnh0XDUWmXyVlS9ifCxOe1ca7VNd0aVJNEoNXQLLgeuApIjirDsKkxS2UK6G+RFHYbyKBdNnHNTcPq9bcToFknVZCm9G6ovqoG85GX8jf6J+ODtC8nUpqj1uDAG/ekdfS9mgfVShgUWxc8+bR9EcS6lElk1HkusscfYzNIC3o1T85Pxj9EOfotSnT+YPB4/RWMUqUz9fNH8KHsBxXsVMnQGkd9+UeD2f+qiTfw5pDp1s4/3RH/xWobKuucp8KHsVxRjz/C6nGoqJvMRH6KPi/R8UbWObIXtcS05gAQbXG3mt3nKzvT0/wDRud8jo3f3W+qHJCKi2iS0rKWjqFcU9SsDSYnZXlDiYPFY450Xjypm0pqi6sYnrO4fNmV9TjRPUrNUdkxpRQhxtUhrUQVDQ1OATg1OAV2Q6xqO0IbUTMqZB1kk26SolHzTC1EkboobKuyIKm6xJJmRp2BkjugOhUrOmt3QSiaYS0BEZT8imxxInUApbxgynXYrHoEhVrPTKDJTlXHGxMpNlbIrjoRFeuh7i53owqO2jWi6F0WWpDuTX+9h9VqxxfJCkm2ejsf+aPK/X0URiO8cV0UOZJjlRJ3fmFFYFKkZcbjgiBCNkRs2iisClxNBG6hDudU/TBmajmH9F/RwP0VqAouNR5oJG843j2QyVpkfY8fbTlJkjmHRXYo1HqKHiuPODRjlBp6NJ0YqLgXW3pNliOjzbWW2ojon4Xo6eHyljEFIaEGMomZaB1D00uQ3SKHUVgHFXZVE4zLgqQsjiPSNjDYG55KXhlQ99jz1QOa7FrbNN1qShWKSlhUfMEjihsqCpU8KjCFZbSMyZPgfdSmhRKVisur0VoGToJC+ymU+pVTI/KjUtZql86YmUy5dACgtpL8E+GpuFJikWrE4suMkwBolcYBBlkv3EKK6S6n4STn8loSSYaNKxw5I7XjkPdRWogKciMlxvb8o9Sjte35R6lV7XorXoiiwZI35R7ozZR8o91XteitcoUTM45BAq33a4WGoI27lwOQpjofAqMhnTRKPU0mivMqY6G6yShYNWUuHnKbLU0VVoqqSi4pXLUiuI/FcdGphqQnvqgsg7GA3cqrxHpY1ul1TzD3OK7mwr8Va0XJXnfSTpiSSyPU7aKkxbHpJjlaTrpopnRzo4XODni/FDychEszk6iTuiWEySv62Ykk62OwXqlDTAAKswmhDLaLQRNTIQo0QXFUdyJIi6m0WfMtS1RMikzFDaNVyVJsyx7EuijVs2BRMPYrhkei2Y+wmUtmZxTQqtjmVzjLFmydUMo2KbtF3T1tlZQVizMcinU0iXycRKm4M0sdQr3o8+7z4fVZGmlV9grwH66aW5LTiy3JGjHl5M24CeAqxhdzPqjNe75j7LoKRoonAIjQq4SO+Y+gS+0O+b2CvkVRbtCK1U7Kl3M+36IrZ3cz7for5FUWwTJjofAquMjjxPqugX39yqbJQViMwITAiMKSVEO2MFQayFTBIgVD7pUlZoTVGMxqnN9O9Y6ooHudxXp1TACg0+FAm9vZZvhuwJQUzLYFgBBBIXoOGUOUDRPpKEDgraGKyfCFDIY1ENTtU1qjsRMyZQwLmXUHMkoWfNT1wBOcmgrjRMl6LbD3WVuZxlWbZNZEfXaLXjZjk9ixaRZqTdWlbUXVW5Gy4oTSp1O9QLI8L0uatAZI2i7pZFf4ccxA56LJwyLYdDYc7y87N0Hif8fmpgXiorAvFRq6aC1v1UtsSdEzVSmNC6qR0GRm06BUREbK0YEOVoujoGyrDX/sIrGPU0NCKAFOJdkJsbualwQ80+4T2vsqpIq7G2SJTpXapiWxfYG9yC96NIFFelsnJoJGy6sqaAWVfTvAVrBKLIUh+OVh2RowQmPRLo6NA4PTs6DdOurIPzpId0lCz5yc9N6xAe9KJy5aho53LRIdIotRMURyjytTYIWlsE9xKQCM2NOdEiYZHsnMXXMXAhYMlom069R6LUPVwM5uGc+LtfysvL6Q6i/MfmvboqWzRl2sLIulj4myulj4mwbHWKIHlNfA7kU0tPG/ot6ZtDddYb+X6oZeU+CEHi0eLrFMlFj+hujTBFnTg5BBRGq7KCgpzCmtb+7IrIHH7p9LKMg699EZsaTYCCLqU2NA0A1shSRqvqDZW1Ros9ictkuYEtHWVGqsKepWRfW6p0OK2O6R8RICOSmbyGZSmvWYoMRBG6uIKsFNjKzbCaZZArqDHKiZkaGpnbpJudJWXZ81OT4wuBqlQRLnWc1ugVkJ6nvjUKdtkcSkNDkQG6hOenRT2ROJZIkCHZOL7ojI0tqgZMUPJe4YdiUWRrS7K4NaCJOydAF4tCzUHkQfdezTPY9rRlBItmaWBzrW4XsT5JnTp26Cwpu2i0a4HYg+BBSIVCIGBziwZS1kjh2pAbgH7rhtqOKyOCfxCe0Bszcw013/z+a2Jyra+g+5LuvoekliHM5jBd3OwABc5x4AAakqswzpTTTDR4B5H96eaNiVy6KRkoY0Z2l+VsjWl2W17mwBsRfv71alF9i1JMPR4pHIbAOaczo+01ti9vxNuCRm7jqrJpWWpWPeMomJJnJ6vICTllJdKSdWtOjuVwAL3V9WYrDFcvkAtwBuf8IuSSt6JJpdyUXnOBzBN/Aj9UdzgNSQPE2Xn2NfxDjbcQi52vv77fmn4LiklRAyWSB0xL5Acj3DLd+VosAb7HVVylVxX119wbk/Kvro2sFYx8mRrg4gEnLqBqOKnuKo8Bzhzs8bI9LhjCCWjQHMRv5q1lkUV1sm/Uh10lgsjjFUACrzFJ1hsZnuSs+aVGfLMrqmt1Uf7co0kRKNTYeSscmIi0XGGYmRotVQV+yy9JhZ5K8o6YtR42xkZNdjWUlTdWcZus7hxV3E5a4s6GOVolZUkPOki5DT51japcTrIF0F09lzkrOdKLJskihzaqOalcdMtEYhRiBkCEE9zkIHVMoZxJtLuFe0sAcNVSUh1C1GHMvZBJCJoZ/w3iF6bSNGVoJ07JtK0PZlAN8ruGmqyMLNFrsPBygjNtkJjIcAdD2ozsfDmiwRpsPp41Y6NvZfZlm9XIbslLo9W3HZP5rwqLYeH0XunXRgPe58YaIpM7sropGgDXTawBC8oqujltaeeKdvANkY2X8JOvl6LZCcYumbENZ0ZnuOrfG538kkMkc18QmbmjLrtFr9xK7QYrXRZS0SkPF25o5DnFr9kgXdouy9I6hk8ZkDw2Lq/5DnOY09XF1YOo8/NSv8Am8Bga2JzbAgHrB2SaV0Fwct/vZtSeSZKCmtpMjipd1Y+o6R1rg5jYnMygGQNifdve4AC3moOJYVV2a6TM8ua2QMHWOc1rmucCRaw0aSf/qlw9MABHeN5dEYXtImyiR7IBF/N07TdL27yhw9Ky0sIi7TDC9hMhtnjgMTSQBqLuLreSqOKMdpFLGo7SKOrpnxmz2lp1Fja4sbEEcF6b0Ea77DFaN7wXT3DZTEwXksC4/5WOpejNTWSOlEfUse5z7yl1hmNzlB7Ttz+q9MwjBWU8EUNs4aHuzySCNty4u+Eb2NvJVPLCXhT2FJ6LHDiQdcg7JBZF2mg33LuJtZFqZUyIWF7t1H+mLMHhzUOsmSmzLNlVi0+hWKqpLkrRYvPoVj6qaxWPLtmTIrJkDQStNhtGNFlMLku5bjDXCwSIxti8cbZY09KBwRvst0SA3VlFGFrjA2qJAggyqfGEUQpr2WTOIyOjt0kHVJSg+Z89zSWUOR6I7VJsV1mhEUCYLrr2qbDToc7E2yctlc56EXJ7hquBiuw7JFHNY6rVYXWBZSONW2HtN0DETZu6R9wtXh7tGg2uRs85Hi9xZrho4aLF4Q4rXUWaNoBswAB5LmF8LtSSQd2HbuTMQeD1JssIlY6GTPlfE9hDmjbYkSjiQPdYp2AMIGYTDLpZ8MU4F7E6gXO262lBE9pJdHlFn6xzF0TgRcHIdjwQo9kHUZJQao0owzcFLRl+1F0YNslXRPkYPC+rR3ghcPROkl0ZMGSG1hFndG6407L9R6rdldCz/3E12YRhqH+HQveWe45RMykjvJv+S1eF9HaWnsY4m5vmf23+rtvJWITggyZ8k+7I2ET5z2W3EWXLcmV3w9s65eI280MIlQ4ANOZjbNFy5ud9s5tZtudk3o/O/kAwMspG5J5EtDGeDG8lT11SpldwOurd36PcRp8PAcfNUdYwrczNkeyqxCa91lsQOqvqwm6pKtl0icQGtHMMnsVscKq1iadlitTg+rgEuMdlQirN3hWqu4wqzC47AK5hatKRoQVrNEGYKWNlGnCIJkbKklddVgnzW0qbStuqtj1a0ZWdIponZAAoFSp0h0UGQXKEWQWQp/2dT6eFS3U+iFyFTyUVDYlZ4cNVEmCl4cbFVyK5Wa7DRYL0CjqGthDnHs2F+Pdb6LzSkqFvMNj7LbHKSG3OXMx3aG44OT8RowepIw3EesD43dmVocXMIALATpoO4j1HNRozoj07CZHSOZGCWPYHxPN3A6jM0jU9k68LHmo8eyz9Z3RqQ5dXAuhw577d6xljwnhAp6ljxma4EXLbg6XBII9QQpDVCDgny1IBDOsa13Vh1mx55i3tk24W008E0LtQ1xItIGNEYLhHHmmdcm4zn4RqPda+j8z+QDIlY0l1iLEDXMbyPsSA53IED3VdVM0VtSRDWwsCL3c7O92vxOdxKgV7bXW9mbL3MfibbXVI4XVtjUoBKo433KTJgrsGpYLlazAKbtXWcgNitThFQAAgj3ItG0otArKFyzlPWCysqSrum2NUi4BTJhomRSXXZnaIwyFdJMukoAfMsR1V1QhU8LVbUZISS5lhINFFaLlHlfoo7HWKEXRaUkIUiZosq+KosEn1aXKNmacLIVUNU6meos0uqfE5B2F00aXDTdel0JswdojRpuwbcA1w48dV5jhBXpmHfCNyQG/DbOBYWuPvDcrRgZs6Z3ZIpY7ue+8brsLczCQ/s8C3b07lCMga0uN7AEmwJOg4Abq0icDexaTlcfhyyAWFrj8/EKkrqpkbC57so2BsSbnYADc9yT1ndGxFLj+LXbEYrlrpqfttvYjrBpf2sg1Uwgf1ozGlL+21v8AoPuDmaOLCTqBsVWVFJNI9jqaF4aJBM/r3CGOoex2ZpEZ1aeF9L8VOwer61zzIQZ8sjZIpAGdQzUljGHcba8eKzONL391+exda2RcDqnTwtghLg/PNJJJ92JrpXOFvmeRsFeUNTkqOqhDiGwXsTfM7rdXm+5N9SqLAjGMPY57hGOskf1gIbI1zXkAt4uNtLeSK1lVK+Oomhk6rqzC/wCz2ZLMzNmDnR3uAdCQNVco3J1pfyW1vRvKSobI3MwhwuW3btcaGx46qcWm4sCewODiOR3OUe6pMAqI3sIjeDlNsmUxmIW0aWHVuytK1sji0NzEZGdkZi25vqRoOG5J8E/o/MxbCTMy63Go4a6cNeKzGM1NrrR4o8gNaRY5dQ3UBZnE4Lhb5GXL5jB4k9znIMLCr6TD9dvqgS0Dhsx3k0lZ2UR44b6hTKeoLd12Clf8jvwuRpqJxHwu/CVRRNZium6tMExLMd1kPsUt9GP8mOP0V50dge12rHDxa4Kk9kpnotHJojSO0VfSONtj6FHkcbbH0WhMekweZJR7P+U+hSVWFxMe2CMf6Y/DZFZC3cNaNL7AqQInX4d9m3J0sixU7r6nlbQaeu68zfHf/Tt6ZHDW3tlB/wBt0RtLmOkY/CB68lNbR63vf0At6p3VciT4WCiyK9/n+iuPsRHUrW6FhHg3N7hMMMZ0sTpc5mFuniQB5Ky6i/Pzdf2JQjCb6X/fmj5w9L+v2B4v9Pp9yCykhvbKBbmGlP8AskXANt3hS/spPD1JN/dOFN/TfxH6lKlJe7CUf0ANp4+AH4SrCm1+E3sACBo9jbbs4m6Taf8ApH4W6INJCBpbUXIt2Xb7tPLmF0/6W7lLfsZerVJFnA8HNr912j2lshPHU8BoNuCqKmQtYS1pcRsGgE+hIVpFUPJLbg3a+zXtLZLgeiqmVNjkkBjdydsfA7Faus9DFyS0zHR4oIppTJmaXATR9c18MbZvhLXFw2+F1/FKfDqV8Tz1rHSNbJN17JmdbJIRc7G+U7WWrxwSdUTGGkj7rwLH14LFYjhlPHBJeNr5cr5DKLt7Z17IGgbyCzwavVr5BKkLonQ0v2brZy1ziZGBkkjQ2MbXa0nc73U04s0Rx05nzkSMjzQu60mDcl2W5zAXZ37ql6PYbTvpnulbd5LmtcLgtPC2titBgl2xtidE1pbma4jsmUWuHXA46ehTJpOTu3v9i5NWa3Bnhwc8B4zHeWPqy4AaWG9vFaun+BvgPyXnGHOc6RgZncQSRlBIy8AdgPGy0lbWvjqIsxfljiu6Nly1z3Ai172BFuI4iyb0cVGToVyTdIa+eV0zhILDJG5tgRfNfNfvuEXqWncA+YSpag1H8ywb8QbqHXaHW3HfdSRAfHv/AGVn6qV5n+38G7Aqgvz1GMgZtlHkEVtMOXsF2OK2pPuB9VI6vTQ/v1S4tjGkCFOB90eib1A+X2CJlPG/oU5zD3+n+UxWCCEI4N/tRGxAfd9kVkSJk/eqO2QC23AeiI0ogZ4eicGdyJWQHZJEyLqslmQaWp+iiBxT8685RuJTbHinDKNggNCc6M81CEhsjeS71g5BRWxlSGRqygocOSQcOSZ1ab1ZVpr2JsNomYZUMfdjtw5wF+NjpZcGiqA2z3/9zvzuup/S5eOXyMnVrwo1UNPldcONuIdr4WO4RZ6djxle0OHeFQU+ISN43HJ2qmx4z8zPQ/qu5aapnOaOPwEWIjcbfJJ22f4WU6R9H5xFK1rHXcLAR5Sw/otvT4rGfmHiF12JxfN/aUl9NDvHQHBryujzbol0fqOryvjcLOLmAmzQTbtFbHD+iOt5pC6+pDTv4nirf/isPz/2uSONRcMx8G/qqXTxu5O/4I4uT8TJ1HRxxNyxtDR3DU+J4quqsC6yo68yOADcga0DiBe5O3wjZcfjnys/EfoFElrJJNzYcm6BP0lSCUa7ExpjDiyPZg4cyddeJ0UoFVVFHlJ8AprXlcrqH/lZ0cK8CJPZ5LocOAUfMutclqw2iSuNeeSDdOa5MVghjIniRAzBdDwmJMFh+sXc6FmTg5HRQ7OV1MzJIizF5intXBGnhi83RuHtRQ5DDURrFOJB4cnArmRcMaLiUOzrpKY1ifkV8CWcyqBVx5X5uDrA9xViGrror7+6fgm8U+SF5IKcaIDSngJzqS3wnyO3qmFrhu0+Wo9l3MfU45rvRzpYZR9ArGocjEzrh4eOi4ZxzWi0LaOiNEZGg/aBzCc2UnYE+AJUckiqbJbGhclmDfy03QmU8juGUf1HX0Clw0YbrueZ4eA4JGTqYR7bY6GCT76JFILN13Op7u5FzoAaU0grntuTtm1JJUiXdIOQGIrAiooOHBLMuNYn5ExAsanApAJwajToFnAUVpTQ1PaEdlHUl2ySosyTU8JJLzhtHhEauJJi7FD1xJJWQQRQkkjQJ1q65JJEQanNSSRoEUiiu3SSTYAyDQqTwXUlciI41dSSURY5qY9dSTECzgRWJJI0Cw7U4pJI0UzgTwkkiQI4J4SSRoh1JJJWQ//Z"
}

# -----------------------------
# MEDICINE INFO PAGES
# -----------------------------
MEDICINE_PAGES = {
    "Mancozeb": "https://www.daraz.com.bd/products/mancer-75wp-fungicide-carbendazim-12mancozeb-63-100-gm-i155946764-s1085366401.html?c=&channelLpJumpArgs=&clickTrackInfo=query%253Amancozeb%252Bfungicide%253Bnid%253A155946764%253Bsrc%253ALazadaMainSrp%253Brn%253A1bb8d0e17c530fb31d2d7983a557d86a%253Bregion%253Abd%253Bsku%253A155946764_BD%253Bprice%253A199%253Bclient%253Adesktop%253Bsupplier_id%253A1025968%253Bbiz_source%253Ah5_external%253Bslot%253A0%253Butlog_bucket_id%253A470687%253Basc_category_id%253A10000723%253Bitem_id%253A155946764%253Bsku_id%253A1085366401%253Bshop_id%253A32125%253BtemplateInfo%253A-1_A3_C%25231124_L%2523&freeshipping=0&fs_ab=1&fuse_fs=&lang=en&location=Rajshahi&price=199&priceCompare=skuId%3A1085366401%3Bsource%3Alazada-search-voucher%3Bsn%3A1bb8d0e17c530fb31d2d7983a557d86a%3BoriginPrice%3A19900%3BdisplayPrice%3A19900%3BsinglePromotionId%3A50000047336003%3BsingleToolCode%3ApromPrice%3BvoucherPricePlugin%3A0%3Btimestamp%3A1765908744985&ratingscore=4.813528336380256&request_id=1bb8d0e17c530fb31d2d7983a557d86a&review=547&sale=2814&search=1&source=search&spm=a2a0e.searchlist.list.0&stock=1",
    "Captan": "https://www.daraz.com.bd/products/captan-fungicide-80-wp-broad-spectrum-plant-disease-control-i529231950.html",
    "Metalaxyl": "https://www.zashopbd.com/product/metataf-25wp-50gm/",
    "Sulfur": "https://www.daraz.com.bd/products/aci-epsom-salt-magnesium-95-sulphur-125-1kg-i243097942-s1187201721.html?c=&channelLpJumpArgs=&clickTrackInfo=query%253Asulphur%253Bnid%253A243097942%253Bsrc%253ALazadaMainSrp%253Brn%253A8e4d37472bae0ec9413eaa4f7faa2949%253Bregion%253Abd%253Bsku%253A243097942_BD%253Bprice%253A100%253Bclient%253Adesktop%253Bsupplier_id%253A700513032660%253Bbiz_source%253Ah5_external%253Bslot%253A0%253Butlog_bucket_id%253A470687%253Basc_category_id%253A10000723%253Bitem_id%253A243097942%253Bsku_id%253A1187201721%253Bshop_id%253A242358%253BtemplateInfo%253A-1_A3_C%25231124_L%2523&freeshipping=0&fs_ab=1&fuse_fs=&lang=en&location=Rangpur&price=1E%202&priceCompare=skuId%3A1187201721%3Bsource%3Alazada-search-voucher%3Bsn%3A8e4d37472bae0ec9413eaa4f7faa2949%3BoriginPrice%3A10000%3BdisplayPrice%3A10000%3BsinglePromotionId%3A-1%3BsingleToolCode%3AmockedSalePrice%3BvoucherPricePlugin%3A0%3Btimestamp%3A1765910566071&ratingscore=4.8&request_id=8e4d37472bae0ec9413eaa4f7faa2949&review=5&sale=67&search=1&source=search&spm=a2a0e.searchlist.list.0&stock=1",
    "Copper fungicide": "https://www.daraz.com.bd/products/fungicide-oxicob-100-i293117459-s1300442396.html?c=&channelLpJumpArgs=&clickTrackInfo=query%253Ablue%252Bcopper%252Bfungicide%253Bnid%253A293117459%253Bsrc%253ALazadaMainSrp%253Brn%253Aac3fd9fe2d23a632cc25cc81ba88dc55%253Bregion%253Abd%253Bsku%253A293117459_BD%253Bprice%253A140%253Bclient%253Adesktop%253Bsupplier_id%253A700517088266%253Bbiz_source%253Ah5_external%253Bslot%253A0%253Butlog_bucket_id%253A470687%253Basc_category_id%253A10000723%253Bitem_id%253A293117459%253Bsku_id%253A1300442396%253Bshop_id%253A255983%253BtemplateInfo%253A-1_A3_C%25231124_L%2523&freeshipping=0&fs_ab=1&fuse_fs=&lang=en&location=Rajshahi&price=1.4E%202&priceCompare=skuId%3A1300442396%3Bsource%3Alazada-search-voucher%3Bsn%3Aac3fd9fe2d23a632cc25cc81ba88dc55%3BoriginPrice%3A14000%3BdisplayPrice%3A14000%3BsinglePromotionId%3A-1%3BsingleToolCode%3AmockedSalePrice%3BvoucherPricePlugin%3A0%3Btimestamp%3A1765910677701&ratingscore=4.75&request_id=ac3fd9fe2d23a632cc25cc81ba88dc55&review=20&sale=128&search=1&source=search&spm=a2a0e.searchlist.list.0&stock=1",
    "Chlorothalonil": "https://www.daraz.lk/products/daconil-chlorothalonil-100ml-fungicide-i308623087.html"
    # "Imidacloprid": "https://www.daraz.com.bd/products/imidacloprid-1kg-i529231954.html"
}

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def normalize_brightness(img):
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

def preprocess_img(img):
    img = normalize_brightness(img.convert("RGB"))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img).astype("float32"), axis=0)
    return preprocess_input(arr)

def predict_topk(img_array):
    preds = tf.nn.softmax(model.predict(img_array)[0]).numpy()
    top_idx = preds.argsort()[-TOP_K:][::-1]
    return [(class_names[i], preds[i] * 100) for i in top_idx]

# -----------------------------
# UI
# -----------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", width=IMAGE_DISPLAY_WIDTH)

elif app_mode == "About":
    st.header("📘 About This Project")
    st.write("MobileNetV2 based plant disease recognition with medicine recommendations.")

elif app_mode == "Disease Recognition":
    st.header("🔍 Disease Recognition")

    uploaded_image = st.file_uploader("Upload Leaf Image", ["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, width=IMAGE_DISPLAY_WIDTH)

        if st.button("Predict"):
            results = predict_topk(preprocess_img(img))
            final_class, final_conf = results[0]

            st.success(f"🌱 {final_class} ({final_conf:.2f}%)")
            info = DISEASE_INFO[final_class]

            st.subheader("🦠 Disease Description")
            st.write(info["description"])

            st.subheader("💊 Recommended Medicine")
            for med in info["medicine"]:
                st.markdown(f"### {med}")
                if med in MEDICINE_IMAGES:
                    st.image(MEDICINE_IMAGES[med], width=300)
                if med in MEDICINE_PAGES:
                    st.link_button("📘 Buy Now", MEDICINE_PAGES[med])

            st.subheader("🧪 Treatment")
            for t in info["treatment"]:
                st.write(f"- {t}")

            st.subheader("🛡 Prevention")
            for p in info["prevention"]:
                st.write(f"- {p}")
