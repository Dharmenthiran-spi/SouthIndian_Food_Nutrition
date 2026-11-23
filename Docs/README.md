# 📚 South Indian Food Inspection AI: Technical Documentation Index

Welcome to the technical documentation for the **South Indian Food Inspection & Nutrition AI System**. 

---

## 📑 Documentation Directory Map

| Document | Description | Target Audience |
|:---|:---|:---|
| [**01_PROJECT_ARCHITECTURE_AND_USABILITY.md**](file:///D:/Yolo_Food_Inspection/Docs/01_PROJECT_ARCHITECTURE_AND_USABILITY.md) | High-level system architecture, complete folder hierarchy, 31 South Indian food categories, and Streamlit user interface walkthrough. | Developers, Product Managers, End-Users |
| [**02_MODEL_USAGE_AND_API_INTEGRATION.md**](file:///D:/Yolo_Food_Inspection/Docs/02_MODEL_USAGE_AND_API_INTEGRATION.md) | How to use both models: exact input parameters, output data shapes, Python code samples, FastAPI microservice code, and ONNX mobile/edge export guide. | Backend Engineers, App Developers, Data Scientists |
| [**03_TRAINING_AND_EVALUATION_GUIDE.md**](file:///D:/Yolo_Food_Inspection/Docs/03_TRAINING_AND_EVALUATION_GUIDE.md) | Complete training pipelines for Model 1 (YOLOv8) and Model 2 (PyTorch Multi-Task NN): dataset splits, loss functions, hyperparameters, metrics, and Colab sync. | ML Engineers, Researchers, Model Trainers |
| [**WORKFLOWS.md**](file:///D:/Yolo_Food_Inspection/Docs/WORKFLOWS.md) *(or [04_WORKFLOWS_AND_CALCULATIONS.md](file:///D:/Yolo_Food_Inspection/Docs/04_WORKFLOWS_AND_CALCULATIONS.md))* | Complete end-to-end execution flowcharts, sequence diagrams, inter-module call mappings (`app.py` ➔ `ingredient_predictor.py`), and mathematical formulations for standard scaling, hotel splits, and home component-level nutrition. | Developers, Dietitians, Reviewers |

---

## ⚡ Quick Links & Primary Entry Points

- **🚀 Main Application**: [app.py](file:///D:/Yolo_Food_Inspection/app.py) (`streamlit run app.py`)
- **🍛 Model 1 Training Notebook**: [food_detection_v1_training.ipynb](file:///D:/Yolo_Food_Inspection/Food_Detection/NoteBooks/Food%20Detection/Version1/food_detection_v1_training.ipynb)
- **🌿 Model 2 Training Notebook**: [ingredient_identification_v1_training.ipynb](file:///D:/Yolo_Food_Inspection/Incredient_Identification/NoteBooks/Ingredient%20Identification/Version1/ingredient_identification_v1_training.ipynb)
- **⚙️ Inference Engine**: [ingredient_predictor.py](file:///D:/Yolo_Food_Inspection/Incredient_Identification/ingredient_predictor.py)
- **🗃️ Home vs Hotel Recipe Database**: [home_hotel_recipes.json](file:///D:/Yolo_Food_Inspection/Incredient_Identification/DataSet/Version1%20Used%20Data/home_hotel_recipes.json)

