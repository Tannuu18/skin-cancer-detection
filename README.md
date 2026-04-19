# 🧠 Skin Cancer Classification & Stage Prediction using Fuzzy CNN + Sakaguchi Loss

A machine learning and deep learning pipeline for **multi-class skin lesion classification and stage prediction** using the HAM10000 dataset.

🚀 **Best Accuracy: 77.78%**  
🔥 **Key Innovation:**  
- Bipolar Fuzzy Sets (uncertainty modeling)  
- Custom Sakaguchi Loss (inter-class similarity learning)  
- Extended to **Stage Prediction for clinical relevance**

---

## 💡 Key Idea

Skin lesion classification is inherently uncertain — many conditions share similar visual patterns.

Instead of forcing rigid predictions:
- Model learns **membership AND non-membership**
- Understands **similarity between diseases**
- Extends to **stage prediction (severity awareness)**

👉 Result: More realistic, clinically meaningful predictions

---

## 📊 Dataset

- **Dataset:** HAM10000 (Human Against Machine with 10,000 images)
- **Classes:** 7 types of skin lesions
- Includes metadata:
  - Age
  - Sex
  - Lesion location

### ⚠️ Challenges
- Severe class imbalance (nv dominates)
- High visual similarity between classes
- Medical ambiguity in diagnosis

---

## 🛠️ Project Pipeline

Image → Preprocessing → Feature Learning (CNN)  
→ Membership + Non-membership (Fuzzy Output)  
→ Sakaguchi Loss (Similarity-Aware Learning)  
→ Classification + Stage Prediction  

---

## 🔬 Model Development Journey

### 1. Baseline Models (Raw Pixels)
- Models: Naive Bayes, Decision Tree, KNN, Logistic Regression, Random Forest  
- Input: 32×32 flattened images  
- Best Accuracy: **71.14% (Random Forest)**

---

### 2. Feature Engineering + Data Balancing
- HOG (texture & edges)
- LBP (local texture)
- Color statistics
- PCA for dimensionality reduction
- Oversampling for class balance  

📈 Improved Accuracy: **76.46% (Random Forest)**

---

### 3. Standard CNN
- Architecture: Conv2D + MaxPooling + Dense + Dropout  
- Input: 64×64 images  
- Techniques:
  - Class weighting
  - Early stopping  

📉 Accuracy: **58.41%**

---

### 4. 🚀 Advanced CNN (Core Contribution)

#### 🔹 Transfer Learning
- Base model: MobileNetV2

#### 🔹 Bipolar Fuzzy Output
- Predicts:
  - Membership (belongs to class)
  - Non-membership (does NOT belong)

#### 🔹 Sakaguchi Loss
- Uses similarity matrix between classes
- Penalizes predictions based on biological closeness

---

### 5. Fine-Tuning
- Unfroze top layers of MobileNetV2  
- Reduced learning rate  
- Improved feature adaptation  

---

## 🏆 Final Performance

| Model | Accuracy |
|------|--------|
| Random Forest (Raw) | 71.14% |
| Random Forest (Engineered) | 76.46% |
| CNN (Baseline) | 58.41% |
| **Fuzzy CNN + Sakaguchi** | **77.78%** |

---

## 🧠 Stage Prediction (NEW 🚀)

We extended the system beyond classification to include **stage prediction**.

### What’s New:
- Predicts **disease severity/stage**
- Moves from detection → **clinical understanding**

### Why It Matters:
- Helps in **treatment planning**
- Improves real-world applicability
- Handles ambiguity better than traditional models

👉 Transforms the system into a **decision-support tool**

---

## 🔍 Key Insights

- Feature engineering significantly boosts traditional ML models  
- Class imbalance handling is critical  
- Naive CNNs underperform without careful design  
- Modeling **uncertainty** improves medical AI  
- Learning **class relationships** improves predictions  

---

## ⚙️ Setup & Usage

```bash
git clone <your-repo-link>
cd project
pip install -r requirements.txt
python train.py
```

---

## 🔮 Future Work

- Dynamic similarity matrix learning  
- Ensemble methods  
- Higher resolution training (128×128+)  
- Model interpretability  
- Advanced data augmentation  

---

## 🏥 Impact

This project moves toward real-world medical AI by:

- Handling uncertainty explicitly  
- Learning relationships between diseases  
- Providing stage-aware predictions  

---

## 🧾 Conclusion

Moving from rigid classification to uncertainty-aware, stage-aware intelligent diagnosis
