# 🫀 Heart Disease Classification Using Machine Learning (R)

This project implements various machine learning models to classify heart disease using a structured pipeline in R. It includes data preprocessing, exploratory data analysis (EDA), model building, evaluation, and comparison of model performance.

---

## 📁 Dataset

The dataset `kalp.csv` contains clinical parameters related to heart disease. It is loaded from a local directory and includes features such as age, sex, chest pain type, cholesterol levels, and more.

---

## 📊 Data Exploration & Visualization

The initial steps include:
- Checking structure and summary statistics
- Counting missing values
- Visualizing distributions with:
  - **Histograms** (age, cholesterol, etc.)
  - **Bar plots** (sex, fasting blood sugar)
  - **Pie charts** (chest pain type, ECG, slope, thal)

---

## 🧼 Data Cleaning

### ✅ Missing Values:
Missing values (introduced in the `age` column for testing) are replaced with the **most frequent (mode)** value.

### ✅ Outlier Handling:
Outliers are detected using **Z-scores** and replaced with the **maximum non-outlier** value in the feature.

### ✅ Feature Scaling:
Two normalization techniques were applied:
- **Min-Max Normalization** (0–1 scaling)
- **Z-score Standardization** (mean=0, sd=1)

---

## 🤖 Machine Learning Models

### 1. **K-Nearest Neighbors (KNN)**
- Optimal k determined using validation set
- Evaluated with **Accuracy**, **Sensitivity**, **Specificity**, **F1-score**

### 2. **Neural Network (NN)**
- Built with `neuralnet` library
- Architecture: 3 layers (5, 3, 2 neurons)
- Evaluated with correlation and classification metrics

### 3. **Random Forest (RF)**
- Trained using 500 trees
- High accuracy and robust performance

### 4. **Support Vector Machine (SVM)**
- Linear and Radial basis function kernels tested
- Used `e1071` library
- Metrics evaluated on both kernels

### 5. **Logistic Regression (GLM)**
- Built with `glm()` using binomial family
- Classic baseline classifier

---

## 📈 Model Evaluation

Each model is evaluated using:

- **Accuracy**
- **Sensitivity (Recall)**
- **Specificity**
- **Precision**
- **F1-Score**

A bar plot is created to visually compare the accuracies of each model.

---

## 🏆 Best Model Selection & Saving

- All models are compared based on their test set accuracy.
- The model with the highest accuracy is automatically saved using `saveRDS()`.
- Supported models for saving: `KNN`, `NN`, `RF`, `SVM`, `GLM`.

---

## 🧪 New Data Prediction

- The saved best model is loaded using `readRDS()`.
- A new sample patient data is passed to the model for prediction.
- The model outputs whether the person is likely to have heart disease.

---

## 🧰 Requirements

Install the required R packages:

```r
install.packages(c(
  "ggplot2", "ggthemes", "plotly", "dplyr", "psych", 
  "corrplot", "class", "tidyverse", "caret", "zoo", 
  "e1071", "rstatix", "rpart", "neuralnet", "randomForest"
))
```

---

## 👤 Author

This project was developed as an applied machine learning study in R to detect and predict heart disease.

---

