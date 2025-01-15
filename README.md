# Customer Churn Analysis Project

## Overview
This project implements comprehensive churn analysis using multiple machine learning approaches, comparing Linear Regression, Logistic Regression, and Generalized Additive Models (GAM) to predict customer churn in a telecommunications company.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Model Comparison](#model-comparison)
- [Results](#results)
- [Usage](#usage)

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
pygam
scipy
```

## Installation
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn sklearn statsmodels pygam scipy

# Clone the repository
git clone https://github.com/calicartels/Interpretable_ML.git
```

## Dataset
The analysis uses the Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) with features including:
- Customer services (phone, internet, streaming)
- Account information (tenure, charges)
- Demographics (gender, senior citizen status)
- Contract details

## Project Structure
```
project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_Linear_Regression.ipynb
│   ├── 3_Logistic_Regression.ipynb
│   └── 4_GAM_Analysis.ipynb
│
└── README.md
```

## Implementation Details

### 1. Data Preprocessing
```python
# Label encoding for binary features
binary_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

# One-hot encoding for categorical features
categorical_features = ['InternetService', 'Contract', 'PaymentMethod']

# Standardization for numerical features
numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
```

### 2. Feature Selection Methods
- Linear Regression: Recursive Feature Elimination (RFE)
- Logistic Regression: LassoCV
- GAM: Automatic feature significance testing

### 3. Model Implementation
```python
# Linear Regression with RFE
rfe = RFE(estimator=LinearRegression(), n_features_to_select=13)
linear_model = LinearRegression()

# Logistic Regression with LassoCV
lasso = LassoCV(cv=10)
logistic_model = LogisticRegression(max_iter=1000)

# GAM
gam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4))
```

## Model Comparison

### Linear Regression
- **Pros**: Simple implementation, interpretable coefficients
- **Cons**: Not suitable for binary classification
- **Performance**: 
  - Accuracy: 81.7%
  - Precision: 68.5%
  - Recall: 57.3%
  - F1 Score: 62.4%

### Logistic Regression
- **Pros**: Better suited for binary classification, interpretable
- **Cons**: Assumes linear relationship between features
- **Performance**: Similar to Linear Regression with slightly better metrics

### GAM
- **Pros**: Captures non-linear relationships
- **Cons**: More complex interpretation
- **Performance**: Comparable to other models with added flexibility

## Results

### Key Findings
1. Most influential features for churn:
   - Positive correlation:
     - Fiber optic service
     - Electronic check payment
     - Monthly charges
   - Negative correlation:
     - Contract length
     - Tenure
     - Phone service

2. Model Assumptions:
   - Homoscedasticity: Violated (requires weighted least squares)
   - Normality: Partial violation with outliers
   - Autocorrelation: No significant issues (Durbin-Watson near 2)

3. Feature Importance:
```python
# Top positive coefficients (Logistic Regression)
- TotalCharges: 0.654931
- InternetService_Fiber: 0.505281
- PaymentMethod_Electronic: 0.362013

# Top negative coefficients
- Contract_Two_year: -1.393828
- Tenure: -1.356694
- PhoneService: -0.737503
```

## Usage

### Basic Implementation
```python
# Load and preprocess data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = preprocess_data(df)

# Split data
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
evaluate_model(model, X_test, y_test)
```

### Model Selection Recommendations
1. For quick deployment: Use Logistic Regression
2. For complex patterns: Consider GAM
3. Avoid Linear Regression for this binary classification task

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License.
