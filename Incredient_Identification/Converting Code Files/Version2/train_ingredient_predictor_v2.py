"""
Ingredient Identification Version 2 - PyTorch Dual-Head Model Training Pipeline
Trains:
1. PyTorch Multi-Task Deep Neural Network (Food ID -> Multi-Label Ingredients + Comprehensive Nutrition Regression)
   Saved at: Incredient_Identification/Models/Ingredient Identification/Version2/best.pt
"""

import os
import sys
import json
import csv
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
USED_DIR = os.path.join(BASE_DIR, "DataSet", "Version2 Used Data")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
V2_MODELS_DIR = os.path.join(MODELS_DIR, "Ingredient Identification", "Version2")

TRAIN_CSV = os.path.join(USED_DIR, "train.csv")
VAL_CSV = os.path.join(USED_DIR, "val.csv")
MASTER_CSV = os.path.join(USED_DIR, "ingredient_nutrition_training_v2.csv")
VOCAB_CSV = os.path.join(USED_DIR, "ingredients_vocabulary_v2.csv")
CLASSES_CSV = os.path.join(USED_DIR, "food_and_nutrition_master_v2.csv")

class SouthIndianIngredientNutritionModel(nn.Module):
    def __init__(self, num_foods=221, num_ingredients=324, embed_dim=64, num_nutrition=11):
        super().__init__()
        self.food_embedding = nn.Embedding(num_foods, embed_dim)
        self.shared_trunk = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.ingredient_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, num_ingredients)
        )
        self.nutrition_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_nutrition)
        )

    def forward(self, food_ids):
        embed = self.food_embedding(food_ids)
        shared = self.shared_trunk(embed)
        return self.ingredient_head(shared), self.nutrition_head(shared)

def train_pytorch_model(epochs=50, batch_size=32, lr=0.002):
    print("=" * 65)
    print("🚀 TRAINING PYTORCH DUAL-HEAD NEURAL NETWORK (VERSION 2 - 221 CLASSES)")
    print("=" * 65)
    
    os.makedirs(V2_MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    vocab_df = pd.read_csv(VOCAB_CSV)
    classes_df = pd.read_csv(CLASSES_CSV)
    
    num_foods = len(classes_df)
    num_ingredients = len(vocab_df)
    
    food2idx = {r['food_name']: int(r['food_id']) for _, r in classes_df.iterrows()}
    ing2idx = {r['ingredient_key'].lower(): int(r['ingredient_id']) for _, r in vocab_df.iterrows()}
    
    nut_cols = [
        'calories_kcal', 'protein_g', 'carbohydrates_g', 'fat_g', 'fiber_g',
        'sugar_g', 'sodium_mg', 'potassium_mg', 'calcium_mg', 'iron_mg', 'vitamin_c_mg'
    ]
    
    class RecipeDataset(Dataset):
        def __init__(self, df):
            self.food_ids = torch.tensor(df['food_id'].values, dtype=torch.long)
            
            # Multi-hot targets for 324 ingredients
            labels = np.zeros((len(df), num_ingredients), dtype=np.float32)
            for idx, tokens in enumerate(df['ingredients_tokens']):
                for t in str(tokens).split(';'):
                    t_clean = t.strip()
                    if t_clean in ing2idx:
                        labels[idx, ing2idx[t_clean]] = 1.0
            self.ing_targets = torch.tensor(labels, dtype=torch.float32)
            
            # Nutrition targets (normalized)
            nut_vals = df[nut_cols].values.astype(np.float32)
            self.nut_mean = torch.tensor(nut_vals.mean(axis=0), dtype=torch.float32)
            self.nut_std = torch.tensor(nut_vals.std(axis=0) + 1e-6, dtype=torch.float32)
            self.nut_targets = (torch.tensor(nut_vals, dtype=torch.float32) - self.nut_mean) / self.nut_std
            
        def __len__(self):
            return len(self.food_ids)
            
        def __getitem__(self, idx):
            return self.food_ids[idx], self.ing_targets[idx], self.nut_targets[idx]
            
    train_ds = RecipeDataset(df_train)
    val_ds = RecipeDataset(df_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = SouthIndianIngredientNutritionModel(num_foods, num_ingredients, embed_dim=64, num_nutrition=len(nut_cols)).to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_pt_path = os.path.join(V2_MODELS_DIR, "best.pt")
    
    print(f"Training on {len(train_ds)} train samples, validating on {len(val_ds)} samples across {num_foods} classes...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for f_ids, ing_tg, nut_tg in train_loader:
            f_ids, ing_tg, nut_tg = f_ids.to(device), ing_tg.to(device), nut_tg.to(device)
            optimizer.zero_grad()
            pred_ing, pred_nut = model(f_ids)
            loss = bce_loss(pred_ing, ing_tg) * 2.0 + mse_loss(pred_nut, nut_tg)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(f_ids)
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for f_ids, ing_tg, nut_tg in val_loader:
                f_ids, ing_tg, nut_tg = f_ids.to(device), ing_tg.to(device), nut_tg.to(device)
                pred_ing, pred_nut = model(f_ids)
                loss = bce_loss(pred_ing, ing_tg) * 2.0 + mse_loss(pred_nut, nut_tg)
                val_loss += loss.item() * len(f_ids)
        val_loss /= len(val_ds)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'food2idx': food2idx,
                'ing2idx': ing2idx,
                'nut_mean': train_ds.nut_mean.tolist(),
                'nut_std': train_ds.nut_std.tolist(),
                'num_foods': num_foods,
                'num_ingredients': num_ingredients,
                'nut_cols': nut_cols,
                'epoch': epoch,
                'val_loss': best_val_loss
            }, best_pt_path)
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:2d}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
    print(f"\n✅ PyTorch Model successfully trained and saved to: {best_pt_path}")
    return best_pt_path

def main():
    train_pytorch_model(epochs=40)

if __name__ == "__main__":
    main()
