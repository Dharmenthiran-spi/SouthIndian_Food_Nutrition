"""
Khana Raw Dataset -> YOLO Classification Format Converter
1. Adds proper image extensions (.jpg, .png) based on magic byte detection.
2. Removes macOS junk files (._*, .DS_Store).
3. Translates 80 Hindi dish names into clean English labels and filenames.
4. Splits data into 80% train and 20% validation sets.
5. Generates dataset.yaml and labels.txt.
"""

import os
import random
import yaml

translation_map = {
    'aloo gobi': 'potato_cauliflower',
    'aloo methi': 'potato_fenugreek',
    'aloo mutter': 'potato_peas',
    'aloo paratha': 'potato_stuffed_flatbread',
    'amritsari kulcha': 'stuffed_leavened_flatbread',
    'anda curry': 'egg_curry',
    'balushahi': 'glazed_flaky_doughnut',
    'banana chips': 'banana_chips',
    'besan laddu': 'chickpea_flour_sweet_ball',
    'bhindi masala': 'spiced_okra',
    'biryani': 'spiced_rice',
    'boondi laddu': 'sweet_gram_pearl_ball',
    'chaas': 'buttermilk',
    'chana masala': 'chickpea_curry',
    'chapati': 'wheat_flatbread',
    'chicken pizza': 'chicken_pizza',
    'chicken wings': 'chicken_wings',
    'chikki': 'peanut_brittle',
    'chivda': 'savory_flattened_rice_mix',
    'chole bhature': 'chickpeas_with_fried_bread',
    'dabeli': 'spiced_potato_burger',
    'dal khichdi': 'lentil_rice_porridge',
    'dhokla': 'steamed_chickpea_cake',
    'falooda': 'sweet_vermicelli_dessert',
    'fish curry': 'fish_curry',
    'gajar ka halwa': 'carrot_pudding',
    'garlic bread': 'garlic_bread',
    'garlic naan': 'garlic_flatbread',
    'ghevar': 'honeycomb_sweet_cake',
    'grilled sandwich': 'grilled_sandwich',
    'gujhia': 'sweet_fried_dumpling',
    'gulab jamun': 'sweet_milk_balls_in_syrup',
    'hara bhara kabab': 'spinach_pea_cutlet',
    'idiyappam': 'steamed_rice_noodles',
    'idli': 'steamed_rice_cake',
    'jalebi': 'crispy_sweet_spirals',
    'kaju katli': 'cashew_fudge',
    'khakhra': 'crispy_thin_flatbread',
    'kheer': 'sweet_rice_pudding',
    'kulfi': 'condensed_milk_ice_cream',
    'margherita pizza': 'margherita_pizza',
    'masala dosa': 'spiced_potato_crepe',
    'masala papad': 'spiced_lentil_cracker',
    'medu vada': 'fried_lentil_donut',
    'misal pav': 'spicy_sprouts_with_bread_roll',
    'modak': 'sweet_rice_dumpling',
    'moong dal halwa': 'yellow_lentil_pudding',
    'murukku': 'crispy_spiral_snack',
    'mysore pak': 'gram_flour_ghee_fudge',
    'navratan korma': 'mixed_vegetable_nut_curry',
    'neer dosa': 'delicate_rice_crepe',
    'onion pakoda': 'onion_fritters',
    'palak paneer': 'spinach_cottage_cheese',
    'paneer masala': 'cottage_cheese_curry',
    'paneer pizza': 'cottage_cheese_pizza',
    'pani puri': 'crisp_water_balls',
    'paniyaram': 'steamed_rice_dumpling_balls',
    'papdi chaat': 'crisp_wafer_yogurt_snack',
    'patrode': 'taro_leaf_rolls',
    'pav bhaji': 'vegetable_curry_with_bread_roll',
    'pepperoni pizza': 'pepperoni_pizza',
    'phirni': 'ground_rice_pudding',
    'poha': 'flattened_rice_stir_fry',
    'pongal': 'savory_rice_lentil_porridge',
    'puri bhaji': 'puffed_bread_with_potato_curry',
    'rajma chawal': 'kidney_bean_curry_with_rice',
    'rasgulla': 'cottage_cheese_balls_in_syrup',
    'rava dosa': 'semolina_crepe',
    'sabudana khichdi': 'tapioca_pearl_pilaf',
    'sabudana vada': 'tapioca_pearl_fritters',
    'samosa': 'savory_stuffed_pastry',
    'seekh kebab': 'grilled_minced_skewers',
    'set dosa': 'fluffy_rice_crepe',
    'sev puri': 'crisp_wafer_potato_snack',
    'solkadhi': 'coconut_kokum_drink',
    'steamed momo': 'steamed_dumplings',
    'thali': 'assorted_meal_platter',
    'thukpa': 'noodle_soup',
    'uttapam': 'savory_rice_pancake',
    'vada pav': 'potato_fritter_in_bun'
}

def get_image_ext(filepath):
    try:
        with open(filepath, 'rb') as f:
            h = f.read(16)
            if h.startswith(b'\xff\xd8\xff'):
                return '.jpg'
            elif h.startswith(b'\x89PNG\r\n\x1a\n'):
                return '.png'
            elif h.startswith(b'GIF87a') or h.startswith(b'GIF89a'):
                return '.gif'
            elif h.startswith(b'RIFF') and len(h) >= 12 and h[8:12] == b'WEBP':
                return '.webp'
            elif h.startswith(b'BM'):
                return '.bmp'
    except Exception:
        pass
    return '.jpg'

