# 🌐 Raw Data Sources, Repositories & Citation Links

This document provides all **official links, DOI citations, repository URLs, and access portals** for the raw datasets used to construct **Model 2 (Ingredient Identification & Nutrition)**.

---

## 1. 🔬 ICMR - National Institute of Nutrition (NIN)
### Indian Food Composition Tables (IFCT 2017)
- **Official ICMR-NIN Portal**: [https://www.nin.res.in/](https://www.nin.res.in/)
- **Direct Official eBook (PDF)**: [https://www.nin.res.in/ebooks/IFCT2017.pdf](https://www.nin.res.in/ebooks/IFCT2017.pdf)
- **Searchable Open Database Portal**: [https://ifct2017.github.io/](https://ifct2017.github.io/)
- **Open-Source Developer API & Dataset**: [https://github.com/ifct2017/ifct2017](https://github.com/ifct2017/ifct2017)
- **Publisher**: National Institute of Nutrition, Indian Council of Medical Research (ICMR), Department of Health Research, Ministry of Health & Family Welfare, Government of India.
- **Scope**: Contains comprehensive macro- and micro-nutrient profiles for **528 key Indian raw foods & ingredients** analyzed using HPLC, GC-MS, and ICP-MS.
- **Local File in this Directory**: [`raw_icmr_nin_ingredients_composition.csv`](raw_icmr_nin_ingredients_composition.csv) (Nutritional values per 100g for all 68 South Indian ingredients).
- **Citation**:
  > Longvah, T., Ananthan, R., Bhaskarachary, K., & Venkaiah, K. (2017). *Indian Food Composition Tables*. National Institute of Nutrition, ICMR, Hyderabad. ISBN: 978-93-5267-938-2.

---

## 2. 🍲 Archana's Kitchen Indian Food Recipes Dataset
- **Kaggle Dataset Link**: [https://www.kaggle.com/datasets/kanishk307/archanas-kitchen-dataset](https://www.kaggle.com/datasets/kanishk307/archanas-kitchen-dataset)
- **Source Website**: [https://www.archanaskitchen.com/](https://www.archanaskitchen.com/)
- **Alternative Mirror**: [https://github.com/archanaskitchen/](https://github.com/archanaskitchen/)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (**CC BY-SA 4.0**).
- **Scope**: Contains over 6,000 authentic Indian home-cooked and regional recipes with translated Tamil/Hindi dish names, preparation times, ingredient measurements, and step-by-step cooking procedures.
- **Local File in this Directory**: [`raw_archana_kitchen_south_indian_recipes.csv`](raw_archana_kitchen_south_indian_recipes.csv)

---

## 3. 🧬 RecipeDB: Center for Computational Gastronomy (IIIT-Delhi)
- **Official Research Portal**: [https://cosylab.iiitd.edu.in/recipedb/](https://cosylab.iiitd.edu.in/recipedb/)
- **FlavorDB Link**: [https://cosylab.iiitd.edu.in/flavordb/](https://cosylab.iiitd.edu.in/flavordb/)
- **Research Group**: Complex Systems Laboratory (CosyLab), Department of Computer Science & Engineering, Indraprastha Institute of Information Technology Delhi (IIIT-Delhi).
- **Scope**: Computational database of culinary recipes, canonical ingredient ontologies, and flavor networks across Indian cuisines.
- **Citation**:
  > Batra, D., Niranjan, N., Sammany, R., & Bagler, G. (2020). *RecipeDB: A Resource for Exploring Recipes Across the Globe*. Database, Volume 2020, baaa077. DOI: [10.1093/database/baaa077](https://doi.org/10.1093/database/baaa077).

---

## 4. 🍛 Indian Food 101 Dataset (Kaggle)
- **Kaggle Link**: [https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-101](https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-101)
- **License**: Open Database License (ODbL).
- **Scope**: Categorization of traditional Indian foods by state (Tamil Nadu, Kerala, Andhra Pradesh, Karnataka), course, diet, and flavor profile.

---

## 5. 🌐 USDA FoodData Central & Open Food Facts
- **USDA FoodData Central**: [https://fdc.nal.usda.gov/](https://fdc.nal.usda.gov/)
  - Used for secondary validation of raw macro energy factors (Atwater system: 4 kcal/g carbs, 4 kcal/g protein, 9 kcal/g fat).
- **Open Food Facts (India)**: [https://in.openfoodfacts.org/](https://in.openfoodfacts.org/)
  - Open collaborative database of commercial food products and ingredient label transparency in India.

---

## 📋 Summary of Raw Files Available in this Folder

| File Name | Records | Data Description | Source Authority |
|:---|:---|:---|:---|
| [`raw_icmr_nin_ingredients_composition.csv`](raw_icmr_nin_ingredients_composition.csv) | 48 raw ingredients | Exact macro & mineral composition per 100g raw weight | ICMR-NIN IFCT 2017 |
| [`raw_archana_kitchen_south_indian_recipes.csv`](raw_archana_kitchen_south_indian_recipes.csv) | South Indian recipes | Original recipe names, ingredients text, step-by-step methods & source URLs | Archana's Kitchen (CC BY-SA 4.0) |
| [`DATA_SOURCES_AND_LINKS.md`](DATA_SOURCES_AND_LINKS.md) | Full Documentation | Complete links, DOI references, and license terms | Official Publication |