🩺 Diabetes Prediction Using Machine Learning
📌 Project Overview

This project predicts whether a patient is likely to have diabetes based on medical diagnostic measurements. Multiple machine learning algorithms were trained and compared to identify the best-performing model.

The project includes:

Data preprocessing
Feature scaling
Model training and evaluation
Hyperparameter tuning
Model persistence using Joblib
📂 Dataset

The dataset contains several medical attributes such as:

Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age

Target Variable:

Outcome
0 → Non-Diabetic
1 → Diabetic
🛠️ Technologies Used
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
XGBoost
Joblib
🚀 Machine Learning Workflow
1. Data Preprocessing
Loaded dataset using Pandas
Checked for:
Missing values
Duplicate records
Data statistics
2. Train-Test Split
Training Data: 85%
Testing Data: 15%
3. Feature Scaling

Used StandardScaler to normalize feature values.

4. Model Training

The following models were trained:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)
XGBoost Classifier
5. Model Evaluation

Models were evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
📊 Models Compared
Model	Evaluation
Logistic Regression	✅
Decision Tree	✅
Random Forest	✅
Support Vector Machine	✅
XGBoost	✅

The model with the best performance was further optimized using GridSearchCV.

⚙️ Hyperparameter Tuning

Logistic Regression was optimized using Grid Search with:

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

Cross-validation was performed to identify the best parameter combination.

💾 Model Saving

The trained model and scaler were saved for future deployment.

joblib.dump(best_lr, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")

Saved files:

loan_model.pkl
scaler.pkl
📁 Project Structure
Diabetes-Prediction/
│
├── diabetes.csv
├── Diabetes_Prediction.ipynb
├── loan_model.pkl
├── scaler.pkl
├── README.md
│
└── requirements.txt
📈 Results

The project compares multiple machine learning algorithms and selects the most accurate model after hyperparameter tuning.

Key achievements:

Data preprocessing and scaling
Performance comparison across models
Hyperparameter optimization
Model serialization for deployment
🔮 Future Improvements
Build a Streamlit web application
Deploy on Render or Hugging Face Spaces
Add advanced feature engineering
Perform cross-validation on all models
Create an interactive prediction dashboard

You can add a **Sample Input & Output** section like this in your README:

## 🎯 Sample Prediction

### Sample Input

| Feature                    | Value |
| -------------------------- | ----- |
| Pregnancies                | 6     |
| Glucose                    | 148   |
| Blood Pressure             | 72    |
| Skin Thickness             | 35    |
| Insulin                    | 0     |
| BMI                        | 33.6  |
| Diabetes Pedigree Function | 0.627 |
| Age                        | 50    |

```python
sample = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
```

### Model Prediction

```python
prediction = model.predict(sample)
print(prediction)
```

### Output

```text
[1]
```

### Interpretation

✅ **Patient is likely to have Diabetes**

---

### Another Example

#### Input

| Feature                    | Value |
| -------------------------- | ----- |
| Pregnancies                | 1     |
| Glucose                    | 85    |
| Blood Pressure             | 66    |
| Skin Thickness             | 29    |
| Insulin                    | 0     |
| BMI                        | 26.6  |
| Diabetes Pedigree Function | 0.351 |
| Age                        | 31    |

```python
sample = [[1, 85, 66, 29, 0, 26.6, 0.351, 31]]
```

#### Output

```text
[0]
```

#### Interpretation

✅ **Patient is not likely to have Diabetes**

---

## 📌 Prediction Labels

| Output | Meaning      |
| ------ | ------------ |
| 0      | Non-Diabetic |
| 1      | Diabetic     |

This section makes the repository look more complete and helps recruiters immediately understand what your model does.

👨‍💻 Author

P. Purandhar

Aspiring Data Scientist
Machine Learning Enthusiast
Kaggle Competitor
