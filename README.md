
# 🫀 Heart Disease Prediction using Artificial Neural Networks (ANN)

## 🔍 Overview
This project aims to predict the presence of heart disease using patient health data and a deep learning model built with Keras (TensorFlow). Early and accurate prediction of heart disease can aid in preventive healthcare and clinical decision-making.

## 📊 Dataset
- **Source**: UCI Heart Disease Dataset (CSV format)
- **Features**: 13 patient attributes (e.g., age, cholesterol, blood pressure, etc.)
- **Target**: Binary classification  
  - `1` = Heart disease present  
  - `0` = No heart disease

## 🧠 Model: Artificial Neural Network
- **Architecture**:
  - Input layer: 32 neurons, ReLU
  - Hidden layer: 16 neurons, ReLU
  - Output layer: 1 neuron, Sigmoid
- **Techniques Used**:
  - Feature Standardization
  - Dropout Regularization
  - EarlyStopping for training efficiency
  - Train-validation-test split for robust evaluation

## 📈 Results
- **Test Accuracy**: ~86–90% (depending on train-test split)
- **Evaluation Metrics**:
  - Confusion Matrix
  - Precision, Recall, F1-score
- **Key Insight**: The model generalizes well and captures most true heart disease cases with high recall.

## 💡 Why It Matters
- **Real-world relevance**: Assists healthcare professionals in risk screening and early detection
- **ML Value**: Showcases full model development pipeline on structured data using deep learning
- **Extensible**: Can be improved with hyperparameter tuning, feature selection, or explainability tools

## 🔧 Tech Stack
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

## 🧠 Model Architecture
![Model Architecture](images/nn_diagram.png)

## 📈 Training Metrics
![Accuracy Curve](https://github.com/user-attachments/assets/6bed786b-1548-4dd9-95cd-ab3dc947114c)

## 🔍 Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/ec93c95f-f3d6-46b6-a0d4-ecf598191f7e)

## 📈 Model Performance

### 🔹 Accuracy Curve
<img src = https://github.com/user-attachments/assets/ba752af8-0f55-4cdd-b6a8-5d73434b6074 width = 700 height = 350>



