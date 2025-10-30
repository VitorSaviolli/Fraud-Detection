# 🔍 Credit Card Fraud Detection System

Complete Machine Learning system to detect fraud in credit card transactions, implemented from exploratory analysis to production model.
Note that the system was created for learning Machine Learning, data analysis, and other technologies, meaning the entire code was written from scratch without "copy and paste".

## 🎯 Objective

Develop a robust fraud detection system focusing on:
- **High Recall (>85%)**: Capture maximum possible frauds
- **Balanced Precision**: Minimize false positives
- **Production Ready**: Modular and reusable code
- **Performance**: Real-time predictions

## 📊 Dataset

- **Source**: European credit card transactions
- **Size**: 284,807 transactions
- **Imbalance**: 99.83% legitimate vs 0.17% fraud (492 frauds)
- **Features**: 
  - 28 anonymous features (V1-V28) transformed by PCA
  - `Time`: seconds since first transaction
  - `Amount`: transaction amount
  - `Class`: 0 (legitimate) or 1 (fraud)

## 🛠️ Tech Stack

### Core
- **Python 3.14+**
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Preprocessing and metrics
- **XGBoost**: Gradient boosting model

### Balancing & Processing
- **SMOTE (imblearn)**: Synthetic oversampling
- **StandardScaler**: Feature normalization
- **LabelEncoder**: Category encoding

### Visualization & Analysis
- **Matplotlib & Seaborn**: Graphs and visual analysis
- **Jupyter Notebook**: Interactive exploration

### Persistence
- **Pickle**: Model serialization

## 🔄 ML Pipeline

### 1. **Exploratory Data Analysis (EDA)**
```python
notebooks/exploracao_dados.ipynb
```
- Temporal distribution of frauds (0-6h, 6-12h, etc.)
- Feature correlation with frauds (V14, V17 most relevant)
- Transaction amount analysis (frauds tend to low amounts)
- Pattern and outlier visualizations

### 2. **Feature Engineering**
```python
src/fraud_detector.py → create_features()
```

**Numerical Features:**
- `Amount_log`: `np.log1p(Amount)` - reduces outlier impact
- `Amount_sqrt`: `√Amount` - smooths distribution
- `Time_hours`: conversion from seconds to hours

**Cyclic Temporal Features:**
- `Time_sin`: `sin(2π × Time_hours / 24)` - daily circular pattern
- `Time_cos`: `cos(2π × Time_hours / 24)` - sine complement

**Interaction Features:**
- `V14_V17`: multiplication of important features
- `Amount_V14`: amount × V14 interaction

**Categorical Features:**
- `Amount_category`: [Very low, Low, Medium, High, Very High]

### 3. **Preprocessing**
```python
src/fraud_detector.py → preprocess()
```

**Sequence:**
1. **Feature Engineering** (create_features)
2. **Label Encoding** (Amount_category → numbers)
3. **Removal** of non-numeric features
4. **Split** train/test (80/20 with stratify)
5. **SMOTE** (227k legitimate → 227k synthetic frauds)
6. **StandardScaler** (mean=0, std=1)

### 4. **Modeling**

**XGBoost Architecture:**
```python
XGBClassifier(
    n_estimators=50,      # 50 sequential trees
    max_depth=6,          # Maximum depth
    learning_rate=0.1,    # Learning rate (10%)
    random_state=42,      # Reproducibility
    eval_metric='logloss' # Evaluation metric
)
```

**Why XGBoost?**
- Ensemble of trees that correct previous errors
- Built-in regularization (avoids overfitting)
- Excellent for imbalanced data
- Fast and efficient

## 📈 Results

### Final Metrics
| Metric | Value |
|---------|-------|
| **Recall** | ~85-90% |
| **Precision** | ~28-35% |
| **F1-Score** | ~42-48% |

### Interpretation
- ✅ **High Recall**: Detects most frauds (priority)
- ⚠️ **Low Precision**: Some false positives (acceptable for fraud)
- 💡 **Trade-off**: Better to block legitimate fraud than let real fraud pass

### Confusion Matrix (Example)
```
                Pred Legitimate  Pred Fraud
Real Legitimate      56,800         62
Real Fraud              15          83
```
- 83/98 frauds detected (84.7% recall)
- 62 honest customers blocked (false positives)

## 📁 Project Structure

```
fraudcard/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # Original dataset
│   └── processed/                  # (empty - for processed data)
│
├── notebooks/
│   └── exploracao_dados.ipynb      # Complete EDA + analysis
│
├── src/
│   ├── __init__.py                 # Makes src a module
│   ├── fraud_detector.py           # Main model class
│   └── sample_transaction.py       # Sample transaction
│
├── models/
│   └── fraud_detector.pkl          # Saved trained model
│
├── main.py                         # Execution script
├── requirements.txt                # Dependencies
├── README.md                       # Portuguese README
└── README_EN.md                    # This file
```

## 🚀 How to Run

### 1. **Installation**

```bash
# Clone the repository
git clone https://github.com/VitorSaviolli/Fraud-Detection.git
cd Fraud-Detection

# Install dependencies
pip install -r requirements.txt
```

### 2. **Training**

```bash
# Trains model, saves, and tests prediction
python main.py
```

**Expected output:**
```
Fraud Detection System:
Loading data...
Creating features...
Separating X and y...
Splitting train and test...
Applying SMOTE...
Normalizing...
Training XGBOOST...
Evaluating...
Recall 0.857 (85.7%)
Precision 0.286 (28.6%)
Model saved in models/fraud_detector.pkl

Testing prediction...
Result: LEGITIMATE
Fraud Probability: 0.084
Legitimate Probability: 0.916
```

### 3. **Using the Model**

```python
from src.fraud_detector import FraudDetector

# Load trained model
detector = FraudDetector()
detector.load_model('models/fraud_detector.pkl')

# New transaction
transaction = {
    'Time': 406,
    'V1': -1.359807,
    'V2': -0.072781,
    # ... V3-V28
    'Amount': 149.62
}

# Prediction
result = detector.predict(transaction)
print(result)
# {'prediction': 'LEGITIMATE', 
#  'probability_fraud': 0.084, 
#  'probability_legit': 0.916}
```

## 🔍 Technical Details

### FraudDetector Class

```python
class FraudDetector:
    def __init__(self)
    def create_features(df)      # Feature engineering
    def preprocess(df, fit)       # Preprocessing
    def train(csv_path)           # Train model
    def predict(transaction)      # Predict fraud
    def save_model(filepath)      # Save model
    def load_model(filepath)      # Load model
```

### Prediction Flow

```
New Transaction (dict)
    ↓
create_features()  → Feature engineering
    ↓
preprocess()       → Label encoding + selection
    ↓
scaler.transform() → Normalization
    ↓
model.predict()    → Class (0 or 1)
    ↓
model.predict_proba() → Probabilities
    ↓
return {prediction, probability_fraud, probability_legit}
```

## 📚 Applied Concepts

### SMOTE (Synthetic Minority Over-sampling)
- Creates synthetic frauds by interpolating between neighbors
- Balances dataset (50/50) without overfitting
- Improves minority class detection

### StandardScaler
- Normalizes features (mean=0, std=1)
- Avoids dominance of high-value features
- Essential for distance-based algorithms

### Stratified Split
- Maintains class proportion in train/test
- Guarantees statistical representativeness
- Essential for imbalanced data

### Temporal Feature Engineering
- `sin/cos` capture cyclic patterns (schedules)
- Better than linear features for time
- Model understands 11pm is close to 12am

## 🎓 Learnings

**Technical:**
- Handling extremely imbalanced data
- Creative feature engineering with PCA
- Trade-offs between Recall and Precision
- ML model serialization and deployment

**Conceptual:**
- Importance of Recall in fraud detection
- Cost of false positives vs false negatives
- Interpretability vs Performance
- Complete ML pipeline (EDA → Production)

## 📋 Future Improvements

- [ ] Hyperparameter tuning with GridSearch/Optuna
- [ ] Threshold optimization to maximize Recall
- [ ] Feature importance analysis
- [ ] REST API with FastAPI
- [ ] Data drift monitoring
- [ ] Unit and integration tests
- [ ] CI/CD pipeline
- [ ] Docker containerization

## 📄 License

This project is open-source for educational purposes.

---

⭐ **If this project helped you, leave a star on the repository!**
