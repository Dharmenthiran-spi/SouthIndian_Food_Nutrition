import os
import glob
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="South Indian Food Detection",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING
# ==========================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #FF5722;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #607D8B;
        margin-bottom: 1.5rem;
    }
    .food-card {
        background-color: #F8F9FA;
        border-left: 5px solid #FF5722;
        padding: 12px 18px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 1.15rem;
        font-weight: 600;
        color: #212121;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #FFE0B2;
        color: #E65100;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR - SETTINGS & MODEL
# ==========================================
st.sidebar.title("⚙️ Model Settings")

# Model paths to search
candidate_models = [
    "Food_Detection/Models/Food Detection/Version1/best.pt",
    "Food_Detection/Models/Food Detection/Version1/last.pt",
    "/content/drive/MyDrive/SouthIndian_Food_Nutrition/Food_Detection/Models/Food Detection/Version1/best.pt",
    "runs/detect/Food_Detection_V1/yolov8s_31classes/weights/best.pt",
    "yolov8s.pt",
    "yolov8n.pt"
]

available_models = [m for m in candidate_models if os.path.exists(m)]
if not available_models:
    available_models = ["yolov8s.pt"]

selected_model_path = st.sidebar.selectbox(
    "Select Model Weights:",
    options=available_models,
    index=0,
    help="Select the trained best.pt model or a pretrained baseline."
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold:",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Minimum confidence score to accept a food detection."
)

iou_threshold = st.sidebar.slider(
    "IoU (Overlap) Threshold:",
    min_value=0.10,
    max_value=1.00,
    value=0.45,
    step=0.05,
    help="Intersection-over-Union threshold for Non-Maximum Suppression."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "**South Indian Food Nutrition & Detection**\n\n"
    "Upload any food plate image to detect dishes, predict bounding boxes, and display food names."
)

# ==========================================
# MODEL LOADER (CACHED)
# ==========================================
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

try:
    model = load_yolo_model(selected_model_path)
    st.sidebar.success(f"✅ Loaded: `{os.path.basename(selected_model_path)}`")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-title">🍛 South Indian Food Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a food plate image to detect and identify food items instantly.</div>', unsafe_allow_html=True)

# Sample Images Selector (Optional convenience)
sample_dir = "Food_Detection/DataSet/Version1 Used Data/images/test"
sample_files = glob.glob(os.path.join(sample_dir, "*.jpg"))[:8] if os.path.exists(sample_dir) else []

selected_sample = None
if sample_files:
    sample_names = ["-- Or choose a sample image --"] + [os.path.basename(f) for f in sample_files]
    choice = st.selectbox("Quick test with sample dataset images:", sample_names, index=0)
    if choice != "-- Or choose a sample image --":
        selected_sample = os.path.join(sample_dir, choice)

# Main File Uploader
uploaded_file = st.file_uploader(
    "📤 Upload a Food Image (JPG, PNG, WebP):",
    type=["jpg", "jpeg", "png", "webp"]
)

# Determine image source
image_to_process = None
if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
elif selected_sample is not None:
    image_to_process = Image.open(selected_sample).convert("RGB")

# ==========================================
# PREDICTION & RESULT DISPLAY
# ==========================================
if image_to_process is not None:
    st.markdown("---")
    
    # Run Prediction
    with st.spinner("🔍 Analyzing image and detecting food items..."):
        results = model.predict(
            source=image_to_process,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
    
    first_res = results[0]
    boxes = first_res.boxes
    names = first_res.names
    
    # Render annotated image
    annotated_bgr = first_res.plot()
    annotated_rgb = annotated_bgr[..., ::-1] # BGR to RGB
    
    # Display images side by side
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📷 Original Image")
        st.image(image_to_process, use_container_width=True)
    with col2:
        st.subheader("🎯 Detected Food Items")
        st.image(annotated_rgb, use_container_width=True)

    # Extract detected food names and confidences
    st.markdown("---")
    st.subheader("🍽️ Detection Results")
    
    if len(boxes) > 0:
        detected_items = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0]) * 100
            food_name = names.get(cls_id, f"Class {cls_id}")
            detected_items.append((food_name, conf_val))
        
        # Display list of food names as distinct cards
        unique_foods = sorted(list(set([item[0] for item in detected_items])))
        
        st.success(f"**Identified {len(detected_items)} food item(s):** {', '.join(unique_foods)}")
        
        cols = st.columns(min(3, len(detected_items)))
        for idx, (food, conf) in enumerate(detected_items):
            col = cols[idx % len(cols)]
            with col:
                st.markdown(
                    f'<div class="food-card">🍽️ {food}<span class="badge">{conf:.1f}%</span></div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning(
            "⚠️ No food items detected with confidence ≥ "
            f"{conf_threshold*100:.0f}%. Try lowering the confidence threshold in the sidebar."
        )

else:
    st.info("👆 Please upload a food photo above to begin detection.")
