import os
import glob
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Import Ingredient & Nutrition Module
try:
    from Incredient_Identification.ingredient_predictor import (
        get_ingredient_nutrition,
        predict_with_model,
        get_predictor,
        get_home_hotel_recipe,
        calculate_custom_nutrition,
        calculate_hotel_proportional
    )
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Incredient_Identification.ingredient_predictor import (
        get_ingredient_nutrition,
        predict_with_model,
        get_predictor,
        get_home_hotel_recipe,
        calculate_custom_nutrition,
        calculate_hotel_proportional
    )

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="South Indian Food & Nutrition AI",
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
    .ing-chip {
        display: inline-block;
        background: #E8F5E9;
        color: #2E7D32;
        padding: 5px 12px;
        margin: 4px;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid #C8E6C9;
    }
    .ing-weight {
        font-weight: 700;
        color: #1B5E20;
        background: #C8E6C9;
        padding: 1px 7px;
        margin-left: 5px;
        border-radius: 10px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR - SETTINGS & MODEL
# ==========================================
st.sidebar.title("⚙️ AI Model Settings")

# Model paths to search (prioritize Version 2 - 221 classes)
candidate_models = [
    "Food_Detection/Models/Food Detection/Version2/food_detection_v2_best.pt",
    "Food_Detection/Models/food_detection_v2_best.pt",
    "Food_Detection/Models/Food Detection/Version1/food_detection_v1_best.pt",
    "Food_Detection/Models/Food Detection/Version1/best.pt",
    "Food_Detection/Models/Food Detection/Version1/food_detection_v1_last.pt",
    "runs/detect/Food_Detection_V1/yolov8s_31classes/weights/best.pt",
    "yolov8s.pt",
    "yolov8n.pt"
]

available_models = [m for m in candidate_models if os.path.exists(m)]
if not available_models:
    available_models = ["yolov8s.pt"]

selected_model_path = st.sidebar.selectbox(
    "Select Model 1 (YOLO) Weights:",
    options=available_models,
    index=0,
    help="Select the trained Food Detection model."
)

# Standard internal detection parameters (fixed for optimal food recognition)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

st.sidebar.markdown("---")
st.sidebar.markdown("### 🥗 Model 2 Checkpoint Status")
candidate_m2 = [
    "Incredient_Identification/Models/Ingredient Identification/Version2/best.pt",
    "Incredient_Identification/Models/Ingredient Identification/Version2/ingredient_identification_v2_best.pt",
    "Incredient_Identification/Models/Ingredient Identification/Version1/ingredient_identification_v1_best.pt"
]
loaded_m2 = next((p for p in candidate_m2 if os.path.exists(p)), None)
if loaded_m2:
    v_tag = "Version 2 (221 classes)" if "Version2" in loaded_m2 else "Version 1 (31 classes)"
    st.sidebar.success(f"✅ Model 2 Loaded:\n`{os.path.basename(loaded_m2)}` ({v_tag})")
else:
    st.sidebar.info("ℹ️ Model 2 using Knowledge Base lookup.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "**South Indian Food Recognition & Nutrition AI**\n\n"
    "• **Model 1**: Detects food items on plates.\n"
    "• **Model 2**: Predicts ingredients & detailed nutrition facts."
)

# ==========================================
# HOME VS HOTEL COMPONENT HELPER
# ==========================================
COMMON_EXTRA_INGREDIENTS = [
    "-- Select extra ingredient to add --",
    "Extra Pure Desi Ghee",
    "Extra Cooking Oil",
    "Butter",
    "Fried Cashews",
    "Fresh Grated Coconut",
    "Boiled Egg",
    "Chicken Pieces",
    "Mutton Pieces",
    "Paneer Cubes",
    "Curd / Yogurt",
    "White Sugar",
    "Jaggery",
    "Milk",
    "Fried Onion"
]

def render_micronutrients_section(nut, scale_factor=1.0):
    if not nut:
        return
    micro_keys = ['calcium_mg', 'iron_mg', 'vitamin_c_mg', 'sodium_mg', 'potassium_mg', 'vitamin_a_mcg']
    if any(k in nut for k in micro_keys):
        with st.expander("🔬 View Detailed Micronutrients, Vitamins & Minerals (28 Parameters)", expanded=False):
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("##### 🪨 Essential Minerals")
                st.write(f"• **Calcium:** `{round(nut.get('calcium_mg', 0) * scale_factor, 1)} mg`")
                st.write(f"• **Iron:** `{round(nut.get('iron_mg', 0) * scale_factor, 1)} mg`")
                st.write(f"• **Potassium:** `{round(nut.get('potassium_mg', 0) * scale_factor, 1)} mg`")
                st.write(f"• **Sodium:** `{round(nut.get('sodium_mg', 0) * scale_factor, 1)} mg`")
                st.write(f"• **Magnesium:** `{round(nut.get('magnesium_mg', 0) * scale_factor, 1)} mg`")
                st.write(f"• **Zinc:** `{round(nut.get('zinc_mg', 0) * scale_factor, 1)} mg`")
                st.write(f"• **Phosphorus:** `{round(nut.get('phosphorus_mg', 0) * scale_factor, 1)} mg`")
            with m2:
                st.markdown("##### 🍊 Vital Vitamins")
                st.write(f"• **Vitamin A:** `{round(nut.get('vitamin_a_mcg', 0) * scale_factor, 1)} mcg`")
                st.write(f"• **Vitamin C:** `{round(nut.get('vitamin_c_mg', 0) * scale_factor, 1)} mg`")
                st.write(f"• **Vitamin D:** `{round(nut.get('vitamin_d_mcg', 0) * scale_factor, 1)} mcg`")
                st.write(f"• **Vitamin E:** `{round(nut.get('vitamin_e_mg', 0) * scale_factor, 1)} mg`")
                st.write(f"• **Vitamin B1 (Thiamine):** `{round(nut.get('vitamin_b1_thiamine_mg', 0) * scale_factor, 2)} mg`")
                st.write(f"• **Vitamin B6:** `{round(nut.get('vitamin_b6_mg', 0) * scale_factor, 2)} mg`")
                st.write(f"• **Vitamin B9 (Folate):** `{round(nut.get('vitamin_b9_folate_mcg', 0) * scale_factor, 1)} mcg`")
                st.write(f"• **Vitamin B12:** `{round(nut.get('vitamin_b12_mcg', 0) * scale_factor, 2)} mcg`")
            with m3:
                st.markdown("##### 💧 Fats, Sugars & Moisture")
                st.write(f"• **Saturated Fat:** `{round(nut.get('saturated_fat_g', 0) * scale_factor, 1)} g`")
                st.write(f"• **Monounsaturated:** `{round(nut.get('monounsaturated_fat_g', 0) * scale_factor, 1)} g`")
                st.write(f"• **Polyunsaturated:** `{round(nut.get('polyunsaturated_fat_g', 0) * scale_factor, 1)} g`")
                st.write(f"• **Trans Fat:** `{round(nut.get('trans_fat_g', 0) * scale_factor, 2)} g`")
                st.write(f"• **Cholesterol:** `{round(nut.get('cholesterol_mg', 0) * scale_factor, 1)} mg`")
                st.write(f"• **Free Sugar:** `{round(nut.get('sugar_g', 0) * scale_factor, 1)} g`")
                st.write(f"• **Moisture / Water:** `{round(nut.get('water_moisture_g', 0) * scale_factor, 1)} g`")

def render_home_hotel_section(food_name, key_prefix="text"):
    recipe = get_home_hotel_recipe(food_name)
    if not recipe:
        return
    
    st.markdown("---")
    st.markdown("### ⚖️ Preparation Style: 🏠 Home Made vs 🏨 Hotel / Restaurant")
    
    col_sel1, col_sel2 = st.columns([1.2, 1])
    with col_sel1:
        prep_mode = st.radio(
            f"Select Preparation Source for **{food_name}**:",
            ["🏠 Home Made Style", "🏨 Hotel / Restaurant Style"],
            key=f"{key_prefix}_prep_{food_name}",
            horizontal=True
        )
        
    if "🏨 Hotel" in prep_mode:
        hotel = recipe["hotel"]
        base_hotel_wt = float(hotel["total_weight_g"])
        
        with col_sel2:
            typed_hotel_weight = st.number_input(
                f"⚖️ Type Total Hotel Serving Weight (grams):",
                min_value=20.0,
                max_value=2500.0,
                value=base_hotel_wt,
                step=10.0,
                key=f"{key_prefix}_hotel_wt_{food_name}",
                help="Type overall serving weight in grams. The system will proportionally split ingredient weights and calculate nutrition."
            )
            
        # Calculate proportional hotel split
        hotel_calc = calculate_hotel_proportional(food_name, typed_hotel_weight)
        
        banner_scale_text = f" (Scaled {hotel_calc['scale_factor']}x from standard {hotel['total_weight_g']}g)" if hotel_calc['scale_factor'] != 1.0 else ""
        st.markdown(
            f'<div style="background-color: #EBF3FB; border-left: 5px solid #1E88E5; padding: 14px 18px; border-radius: 6px; margin: 12px 0;">'
            f'<h4 style="color: #0D47A1; margin: 0 0 6px 0;">🏨 Commercial Restaurant / Hotel Analysis</h4>'
            f'<p style="margin: 0; color: #37474F; font-size: 0.95rem;">'
            f'<b>Standard Benchmark:</b> {hotel["portion_desc"]} &nbsp;|&nbsp; '
            f'<b>Overall Serving Weight:</b> <span style="background:#BBDEFB; padding:3px 10px; border-radius:12px; font-weight:bold; color:#0D47A1;">{hotel_calc["total_weight_g"]} g</span>'
            f'{banner_scale_text}'
            f'</p>'
            f'<p style="margin: 8px 0 0 0; color: #455A64; font-size: 0.9rem;"><i>💡 {hotel["note"]}</i></p>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown(f"**Split Ingredients for {int(hotel_calc['total_weight_g'])}g Hotel Serving:**")
        cols = st.columns(min(4, max(1, len(hotel_calc['split_ingredients']))))
        for idx, (ing, wt) in enumerate(hotel_calc['split_ingredients'].items()):
            c = cols[idx % len(cols)]
            with c:
                st.caption(f"🌿 **{ing.replace('_', ' ').title()}**: `{wt} g`")
                
        st.markdown(f"#### 📊 Hotel Nutritional Breakdown ({int(hotel_calc['total_weight_g'])}g)")
        h_nut = hotel_calc["nutrition"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🔥 Calories", f"{h_nut['calories_kcal']} kcal")
        c2.metric("🍗 Protein", f"{h_nut['protein_g']} g")
        c3.metric("🍚 Carbs", f"{h_nut['carbs_g']} g")
        c4.metric("🥑 Total Fat", f"{h_nut['fat_g']} g")
        c5.metric("🌾 Fiber", f"{h_nut['fiber_g']} g")
        
    else:
        home = recipe["home"]
        with col_sel2:
            st.info(f"📌 **Standard Home Portion:** {home['portion_desc']}\n*Baseline Cooked Weight: {home['total_weight_g']}g*")
            
        st.markdown(
            f'<div style="background-color: #E8F5E9; border-left: 5px solid #43A047; padding: 14px 18px; border-radius: 6px; margin: 12px 0;">'
            f'<h4 style="color: #1B5E20; margin: 0 0 6px 0;">🏠 Traditional Home Cooked Recipe (Customizable)</h4>'
            f'<p style="margin: 0; color: #2E7D32; font-size: 0.95rem;">'
            f'<b>Standard Recipe:</b> {home["portion_desc"]} &nbsp;|&nbsp; '
            f'<i>Type exact ingredient weights below or add additional ingredients.</i>'
            f'</p>'
            f'<p style="margin: 6px 0 0 0; color: #388E3C; font-size: 0.9rem;"><i>💡 {home["note"]}</i></p>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        with st.expander(f"🎛️ Type Ingredient Weights & Add Extra Ingredients ({food_name})", expanded=True):
            st.write("Type your exact ingredient weights in grams below:")
            custom_typed_dict = {}
            ing_cols = st.columns(min(3, max(1, len(home['ingredients']))))
            
            for idx, (ing, base_wt) in enumerate(home['ingredients'].items()):
                c = ing_cols[idx % len(ing_cols)]
                with c:
                    custom_typed_dict[ing] = st.number_input(
                        f"🌱 {ing.replace('_', ' ').title()} (g):",
                        min_value=0.0,
                        max_value=1500.0,
                        value=float(base_wt),
                        step=5.0,
                        key=f"{key_prefix}_type_{food_name}_{ing}"
                    )
            
            st.markdown("---")
            st.markdown("➕ **Add Additional Ingredients (Optional):**")
            col_extra1, col_extra2 = st.columns([1.5, 1])
            with col_extra1:
                selected_extra = st.selectbox(
                    "Choose an additional ingredient to add:",
                    options=COMMON_EXTRA_INGREDIENTS,
                    index=0,
                    key=f"{key_prefix}_extra_sel_{food_name}"
                )
            with col_extra2:
                extra_weight = st.number_input(
                    "Additional Weight (grams):",
                    min_value=0.0,
                    max_value=500.0,
                    value=0.0,
                    step=5.0,
                    key=f"{key_prefix}_extra_wt_{food_name}"
                )
                
            extra_dict = {}
            if selected_extra != "-- Select extra ingredient to add --" and extra_weight > 0:
                extra_dict[selected_extra] = extra_weight
                st.success(f"Added extra: **{selected_extra}** (+{extra_weight}g)")
                
            # Real-time calculation
            custom_res = calculate_custom_nutrition(food_name, custom_typed_dict, extra_dict)
            
            st.markdown("---")
            st.markdown("#### 🥗 Customized Home Nutritional Profile")
            st.write(f"⚖️ **Calculated Overall Cooked Weight:** <span style='background:#C8E6C9; padding:4px 12px; border-radius:12px; font-weight:bold; color:#1B5E20;'>{custom_res['total_weight_g']} g</span> *(Baseline was {home['total_weight_g']} g)*", unsafe_allow_html=True)
            
            c_nut = custom_res["nutrition"]
            c1, c2, c3, c4, c5 = st.columns(5)
            cal_delta = round(c_nut['calories_kcal'] - home['nutrition']['calories_kcal'], 1)
            prot_delta = round(c_nut['protein_g'] - home['nutrition']['protein_g'], 1)
            carb_delta = round(c_nut['carbs_g'] - home['nutrition']['carbs_g'], 1)
            fat_delta = round(c_nut['fat_g'] - home['nutrition']['fat_g'], 1)
            fib_delta = round(c_nut['fiber_g'] - home['nutrition']['fiber_g'], 1)
            
            c1.metric("🔥 Calories", f"{c_nut['calories_kcal']} kcal", delta=f"{cal_delta:+g} kcal" if cal_delta != 0 else None)
            c2.metric("🍗 Protein", f"{c_nut['protein_g']} g", delta=f"{prot_delta:+g} g" if prot_delta != 0 else None)
            c3.metric("🍚 Carbs", f"{c_nut['carbs_g']} g", delta=f"{carb_delta:+g} g" if carb_delta != 0 else None)
            c4.metric("🥑 Total Fat", f"{c_nut['fat_g']} g", delta=f"{fat_delta:+g} g" if fat_delta != 0 else None)
            c5.metric("🌾 Fiber", f"{c_nut['fiber_g']} g", delta=f"{fib_delta:+g} g" if fib_delta != 0 else None)

# ==========================================
# MAIN INTERFACE (TABS)
# ==========================================
st.markdown('<div class="main-title">🍛 South Indian Food & Nutrition AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Food Detection, Ingredient Prediction & Automated Nutrition Analysis</div>', unsafe_allow_html=True)

tab_image, tab_text = st.tabs([
    "📸 Food Plate Image Detection (Upload & Analyze)",
    "🥗 Quick Text Test (Food Name ➔ Ingredients & Nutrition)"
])

# 31 Canonical Dishes List for quick selection
POPULAR_DISHES = [
    "Idly", "Dosa", "Ven Pongal", "Chicken briyani", "Sambar", "Lemon Rice",
    "Chicken 65", "Appam", "Mutton Briyani", "Uzhuntha vadai", "Parupu vada",
    "Koozh", "Kali", "Nei satham", "Sambar satham", "Kaara chutney",
    "Thengai chutney", "Puthina Chutney", "Nandu masala", "Prawn thokku",
    "Beetroot poriyal", "Carrot poriyal", "Boiled Egg", "Paneer briyani",
    "Panner masala", "Veg briyani", "Mushroom briyani", "Paal kolukattai",
    "Poorna kolukattai", "Pidi kolukattai", "Satham"
]

# Dynamic Dishes List: Prioritize all 221 classes from Version 2
v2_dish_map = "Incredient_Identification/DataSet/Version2 Used Data/dishes_ingredients_map_v2.json"
if os.path.exists(v2_dish_map):
    try:
        import json
        with open(v2_dish_map, "r", encoding="utf-8") as f:
            v2_dict = json.load(f)
        ALL_DISHES = sorted(list(v2_dict.keys()))
    except Exception:
        ALL_DISHES = POPULAR_DISHES
else:
    ALL_DISHES = POPULAR_DISHES

# ==========================================
# TAB 1: END-TO-END IMAGE DETECTION
# ==========================================
with tab_image:
    st.header("📸 Step 1: Upload Food Image ➔ Detect Dish & Analyze Nutrition")
    st.write("Upload a food plate image or pick a test sample below. Model 1 identifies the dishes, and Model 2 analyzes Home/Hotel ingredients and nutrition:")

    # Model loader (cached)
    @st.cache_resource
    def load_yolo_model(model_path):
        return YOLO(model_path)

    try:
        yolo_model = load_yolo_model(selected_model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        yolo_model = None

    # Sample Images Selector (Supports Version 2 with 1,105 test images and Version 1)
    sample_dirs = [
        "Food_Detection/DataSet/Version2 Used Data/images/test",
        "Food_Detection/DataSet/Version1 Used Data/images/test"
    ]
    sample_files = []
    active_sample_dir = None
    for sd in sample_dirs:
        if os.path.exists(sd):
            f_list = glob.glob(os.path.join(sd, "*.jpg"))
            if f_list:
                sample_files = f_list[:25]
                active_sample_dir = sd
                break

    selected_sample = None
    if sample_files:
        sample_names = ["-- Or choose a sample image --"] + [os.path.basename(f) for f in sample_files]
        choice = st.selectbox(f"Quick test with sample dataset images ({os.path.basename(os.path.dirname(os.path.dirname(active_sample_dir)))}):", sample_names, index=0)
        if choice != "-- Or choose a sample image --":
            selected_sample = os.path.join(active_sample_dir, choice)

    # Main File Uploader
    uploaded_file = st.file_uploader(
        "📤 Upload a Food Image (JPG, PNG, WebP):",
        type=["jpg", "jpeg", "png", "webp"]
    )

    image_to_process = None
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
    elif selected_sample is not None:
        image_to_process = Image.open(selected_sample).convert("RGB")

    if image_to_process is not None and yolo_model is not None:
        st.markdown("---")
        with st.spinner("🔍 Analyzing image and detecting food items..."):
            results = yolo_model.predict(
                source=image_to_process,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )
        
        first_res = results[0]
        boxes = first_res.boxes
        names = first_res.names

        annotated_bgr = first_res.plot()
        annotated_rgb = annotated_bgr[..., ::-1]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📷 Original Image")
            st.image(image_to_process, use_container_width=True)
        with col2:
            st.subheader("🎯 Detected Food Items")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("---")
        st.subheader("🍽️ Detection Results")

        if len(boxes) > 0:
            detected_items = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0]) * 100
                food_name = names.get(cls_id, f"Class {cls_id}")
                detected_items.append((food_name, conf_val))

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

            # Ingredient & Nutrition breakdown for detected items
            st.markdown("---")
            st.subheader("🥗 Step 2 & 3: Ingredient Breakdown & Home vs. Hotel Analysis")
            for food in unique_foods:
                pred_res = predict_with_model(food)
                if pred_res:
                    weight_badge = pred_res.get('weight_range_desc', f"~{pred_res.get('estimated_weight_g', 200)}g")
                    reg_disp = pred_res.get('regional_name')
                    reg_title = f" — {reg_disp}" if reg_disp and reg_disp != food else ""
                    with st.expander(f"🍲 **{food}**{reg_title} | ⚖️ Standard Portion Weight: {weight_badge}", expanded=True):
                        
                        # Typeable Portion Weight Control
                        base_std_wt = float(pred_res.get('estimated_weight_g', 200))
                        col_w1, col_w2 = st.columns([1.2, 1.8])
                        with col_w1:
                            typed_std_wt = st.number_input(
                                f"⚖️ Type Portion Weight (grams) for **{food}**:",
                                min_value=10.0,
                                max_value=2500.0,
                                value=base_std_wt,
                                step=10.0,
                                key=f"type_std_wt_tab1_{food}",
                                help="Type any overall portion weight. Ingredients and nutrition recalculate proportionally."
                            )
                        with col_w2:
                            scale_factor = typed_std_wt / base_std_wt if base_std_wt > 0 else 1.0
                            if scale_factor != 1.0:
                                st.info(f"⚡ Proportional Scaling Active: **{scale_factor:.2f}x** from standard baseline ({int(base_std_wt)}g)")

                        st.markdown(f"**🌿 Predicted Ingredients & Gram Weights ({int(typed_std_wt)}g):**")
                        chips_html = "".join([
                            f'<span class="ing-chip">🌱 {ing[0]} <span class="ing-weight">{round(ing[2] * scale_factor, 1)}g</span></span>' if len(ing) >= 3 and ing[2] > 0
                            else f'<span class="ing-chip">🌱 {ing[0]}</span>'
                            for ing in pred_res["predicted_ingredients"]
                        ])
                        st.markdown(chips_html, unsafe_allow_html=True)

                        # Scaled Nutrition
                        nut = pred_res["nutrition"]
                        scaled_cal = round(nut['calories_kcal'] * scale_factor, 1)
                        scaled_prot = round(nut['protein_g'] * scale_factor, 1)
                        scaled_carb = round(nut['carbs_g'] * scale_factor, 1)
                        scaled_fat = round(nut['fat_g'] * scale_factor, 1)
                        scaled_fib = round(nut['fiber_g'] * scale_factor, 1)

                        c1, c2, c3, c4, c5 = st.columns(5)
                        cal_diff = round(scaled_cal - nut['calories_kcal'], 1)
                        prot_diff = round(scaled_prot - nut['protein_g'], 1)
                        carb_diff = round(scaled_carb - nut['carbs_g'], 1)
                        fat_diff = round(scaled_fat - nut['fat_g'], 1)
                        fib_diff = round(scaled_fib - nut['fiber_g'], 1)

                        with c1:
                            st.metric("🔥 Calories", f"{scaled_cal} kcal", delta=f"{cal_diff:+g} kcal" if cal_diff != 0 else None)
                        with c2:
                            st.metric("🍗 Protein", f"{scaled_prot} g", delta=f"{prot_diff:+g} g" if prot_diff != 0 else None)
                        with c3:
                            st.metric("🍚 Carbs", f"{scaled_carb} g", delta=f"{carb_diff:+g} g" if carb_diff != 0 else None)
                        with c4:
                            st.metric("🥑 Fat", f"{scaled_fat} g", delta=f"{fat_diff:+g} g" if fat_diff != 0 else None)
                        with c5:
                            st.metric("🌾 Fiber", f"{scaled_fib} g", delta=f"{fib_diff:+g} g" if fib_diff != 0 else None)

                        st.info(f"💡 **Dietary Profile:** {pred_res['health_benefits']}")

                        # Extended Micronutrient & Vitamin Breakdown
                        render_micronutrients_section(nut, scale_factor=scale_factor)

                        # Home vs Hotel Analysis with typed weights & extra ingredients
                        render_home_hotel_section(food, key_prefix=f"tab_img_{food}")
        else:
            st.warning("⚠️ No food items detected in the image. Please try a clearer or better-lit food plate photo.")

# ==========================================
# TAB 2: MODEL 2 TEXT TEST
# ==========================================
with tab_text:
    st.header("🥗 Test Model 2: Food Name ➔ Ingredients & Nutrition")
    st.write("Type any South Indian dish name or pick from the list below to test **Model 2** directly:")

    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        typed_food = st.text_input(
            "⌨️ Enter Food Name:",
            value="",
            placeholder="e.g. Masala dosa, Idli, Chicken briyani, Appam, Paneer butter masala...",
            help="Type any food name from the 221 food categories."
        )
    with col_input2:
        selected_dish = st.selectbox(
            f"📋 Or Choose from {len(ALL_DISHES)} Dishes:",
            options=["-- Select a dish --"] + ALL_DISHES,
            index=0
        )

    # Determine input food
    target_food = typed_food.strip() if typed_food.strip() else (selected_dish if selected_dish != "-- Select a dish --" else None)

    col_btn, col_mode = st.columns([1, 2])
    with col_btn:
        run_prediction = st.button("🔍 Predict Ingredients & Nutrition", type="primary", use_container_width=True)

    with col_mode:
        model_mode = st.radio(
            "Inference Engine:",
            ["🧠 PyTorch Model (ingredient_identification_v1_best.pt)", "📚 Ground Truth Database"],
            horizontal=True
        )

    if (run_prediction or target_food) and target_food:
        st.markdown("---")
        
        with st.spinner(f"Analyzing {target_food}..."):
            if "PyTorch" in model_mode:
                result = predict_with_model(target_food)
            else:
                raw_info = get_ingredient_nutrition(target_food)
                if raw_info:
                    result = {
                        "source": "Ground Truth Database",
                        "food_name": target_food,
                        "regional_name": raw_info.get("regional_name", target_food),
                        "category": raw_info.get("category", "South Indian"),
                        "portion_size": raw_info.get("portion_size", "~150g"),
                        "predicted_ingredients": [(ing.replace('_', ' ').title(), 1.0) for ing in raw_info.get("main_ingredients", [])],
                        "nutrition": raw_info.get("nutrition", {}),
                        "health_benefits": raw_info.get("health_benefits", f"{raw_info.get('diet_type', '')} | {raw_info.get('fitness_goal', '')}"),
                        "cooking_method": raw_info.get("cooking_method", raw_info.get("food_state", "Traditional Preparation"))
                    }
                else:
                    result = None

        if result:
            weight_badge = result.get('weight_range_desc', result.get('portion_size', ''))
            reg_disp = result.get('regional_name')
            reg_text = f" — *{reg_disp}*" if reg_disp and reg_disp != result['food_name'] else ""
            st.success(f"**Identified Dish:** {result['food_name']}{reg_text} ({result.get('category', 'Food')})")
            st.caption(f"Engine: `{result['source']}` | ⚖️ Standard Serving Weight: `{weight_badge}`")

            # Typeable Portion Weight Control
            base_std_wt = float(result.get('estimated_weight_g', 200))
            col_tw1, col_tw2 = st.columns([1.2, 1.8])
            with col_tw1:
                typed_std_wt = st.number_input(
                    f"⚖️ Type Portion Weight (grams) for **{result['food_name']}**:",
                    min_value=10.0,
                    max_value=2500.0,
                    value=base_std_wt,
                    step=10.0,
                    key=f"type_std_wt_tab2_{result['food_name']}",
                    help="Type any overall portion weight. Ingredients and nutrition recalculate proportionally."
                )
            with col_tw2:
                scale_factor = typed_std_wt / base_std_wt if base_std_wt > 0 else 1.0
                if scale_factor != 1.0:
                    st.info(f"⚡ Proportional Scaling Active: **{scale_factor:.2f}x** from standard baseline ({int(base_std_wt)}g)")

            # 1. Ingredients section
            st.subheader(f"🌿 Predicted Ingredients & Gram Weights ({int(typed_std_wt)}g)")
            chips_html = "".join([
                f'<span class="ing-chip">🌱 {ing[0]} <span class="ing-weight">{round(ing[2] * scale_factor, 1)}g</span></span>' if len(ing) >= 3 and ing[2] > 0
                else f'<span class="ing-chip">🌱 {ing[0]}</span>'
                for ing in result["predicted_ingredients"]
            ])
            st.markdown(chips_html, unsafe_allow_html=True)

            # 2. Nutrition Facts Cards
            st.markdown("---")
            st.subheader(f"📊 Nutritional Profile ({int(typed_std_wt)}g serving)")
            nut = result["nutrition"]
            scaled_cal = round(nut['calories_kcal'] * scale_factor, 1)
            scaled_prot = round(nut['protein_g'] * scale_factor, 1)
            scaled_carb = round(nut['carbs_g'] * scale_factor, 1)
            scaled_fat = round(nut['fat_g'] * scale_factor, 1)
            scaled_fib = round(nut['fiber_g'] * scale_factor, 1)

            cal_diff = round(scaled_cal - nut['calories_kcal'], 1)
            prot_diff = round(scaled_prot - nut['protein_g'], 1)
            carb_diff = round(scaled_carb - nut['carbs_g'], 1)
            fat_diff = round(scaled_fat - nut['fat_g'], 1)
            fib_diff = round(scaled_fib - nut['fiber_g'], 1)

            n1, n2, n3, n4, n5 = st.columns(5)
            with n1:
                st.metric("🔥 Calories", f"{scaled_cal} kcal", delta=f"{cal_diff:+g} kcal" if cal_diff != 0 else None)
            with n2:
                st.metric("🍗 Protein", f"{scaled_prot} g", delta=f"{prot_diff:+g} g" if prot_diff != 0 else None)
            with n3:
                st.metric("🍚 Carbohydrates", f"{scaled_carb} g", delta=f"{carb_diff:+g} g" if carb_diff != 0 else None)
            with n4:
                st.metric("🥑 Total Fat", f"{scaled_fat} g", delta=f"{fat_diff:+g} g" if fat_diff != 0 else None)
            with n5:
                st.metric("🌾 Dietary Fiber", f"{scaled_fib} g", delta=f"{fib_diff:+g} g" if fib_diff != 0 else None)

            # 3. Dietary & Health Profile
            st.markdown("---")
            col_h1, col_h2 = st.columns([1, 1])
            with col_h1:
                st.info(f"💡 **Dietary & Health Profile:**\n\n{result['health_benefits']}")
            with col_h2:
                st.info(f"🍳 **Cooking Method:**\n\n{result['cooking_method']}")

            # Extended Micronutrient & Vitamin Breakdown
            render_micronutrients_section(nut, scale_factor=scale_factor)

            # 4. Home vs Hotel Analysis & Dynamic Customization
            render_home_hotel_section(result["food_name"], key_prefix="tab_txt")
        else:
            st.error(f"❌ Could not find ingredient profile for '{target_food}'. Please check spelling or select from the dishes dropdown.")
