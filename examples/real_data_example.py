# Real Data Example Using Pima Indians Diabetes Dataset

This example demonstrates how pymars can be applied to real health data using the publicly available Pima Indians Diabetes dataset. This dataset is commonly used in machine learning for predicting the onset of diabetes based on diagnostic measurements and provides an excellent example for showcasing pymars capabilities in health economic outcomes research.

## Dataset Information

The dataset contains 768 observations with the following features:
- Number of times pregnant
- Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- Diastolic blood pressure (mm Hg)
- Triceps skin fold thickness (mm)
- 2-Hour serum insulin (mu U/ml)
- Body mass index (weight in kg/(height in m)^2)
- Diabetes pedigree function
- Age (years)
- Class variable (0 or 1) representing diabetes onset within 5 years

## Loading and Preparing the Data

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pymars as earth

# Load the dataset
column_names = [
    'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome'
]

# Load data from public repository
data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv', 
                   names=column_names)

print(f"Dataset shape: {data.shape}")
print(f"Diabetes prevalence: {data['outcome'].mean()*100:.2f}%")

# Handle zero values as missing data (except for pregnancies and outcome)
data_processed = data.copy()
cols_with_zeros = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']

# Replace 0s with NaN for these columns
for col in cols_with_zeros:
    data_processed[col] = data_processed[col].replace(0, np.nan)

# Separate features and target
X = data_processed.drop('outcome', axis=1)
y = data_processed['outcome']

print(f"Missing data after preprocessing:")
print(data_processed.isnull().sum())
```

## Fitting MARS Models

```python
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit MARS model for regression (predicting probability)
mars_reg = earth.Earth(
    max_degree=2,      # Allow two-way interactions
    penalty=3.0,       # GCV penalty
    max_terms=21,      # Max number of terms
    minspan_alpha=0.05,  # Minimum span control
    endspan_alpha=0.05,  # End span control
    allow_linear=True,   # Allow linear terms
    allow_missing=True,  # Handle missing values
    feature_importance_type='gcv'  # Calculate feature importance
)

# Fit the model
mars_reg.fit(X_train.values, y_train.values)

# Model evaluation
train_r2 = mars_reg.score(X_train.values, y_train.values)
test_r2 = mars_reg.score(X_test.values, y_test.values)

print(f"MARS Regression Model Summary:")
print(f"Number of basis functions: {len(mars_reg.basis_) - 1}")
print(f"Training R-squared: {train_r2:.3f}")
print(f"Test R-squared: {test_r2:.3f}")
print(f"GCV Score: {mars_reg.gcv_:.3f}")

# Feature importance
feature_names = list(X.columns)
importances = mars_reg.feature_importances_
print(f"\\nFeature Importances:")
for name, imp in zip(feature_names, importances):
    print(f"  {name}: {imp:.4f}")
```

## Health Economics Application

We can also use pymars for health economic modeling by predicting healthcare costs based on patient characteristics:

```python
# Simulate healthcare costs based on the diabetes outcome and other factors
def simulate_healthcare_costs(row):
    """Simulate healthcare costs based on diabetes status and other health factors"""
    base_cost = 2000  # Base annual healthcare cost
    
    # Cost associated with diabetes
    diabetes_cost = 8000 * row['outcome']  # Extra $8000 for diabetic patients
    
    # Cost associated with BMI (obesity-related costs)
    bmi_cost = 0
    if not np.isnan(row['bmi']):
        if row['bmi'] >= 30:  # Obese
            bmi_cost = 3000
        elif row['bmi'] >= 25:  # Overweight
            bmi_cost = 1500
    
    # Cost associated with age
    age_cost = 50 * max(0, row['age'] - 40)  # Increasing costs after age 40
    
    # Cost associated with blood pressure
    bp_cost = 0
    if not np.isnan(row['blood_pressure']):
        if row['blood_pressure'] >= 90:  # Hypertensive
            bp_cost = 2000
    
    # Random variation
    random_variation = np.random.normal(0, 1000)
    
    # Total cost
    total_cost = base_cost + diabetes_cost + bmi_cost + age_cost + bp_cost + random_variation
    
    return max(0, total_cost)  # Ensure non-negative costs

# Apply the cost simulation function
healthcare_costs = data.apply(simulate_healthcare_costs, axis=1)

# Combine original features with the outcome for cost prediction
X_cost = data.drop('outcome', axis=1).copy()
X_cost['diabetes'] = data['outcome']  # Include diabetes status as a predictor
y_cost = healthcare_costs

# Split the data
X_cost_train, X_cost_test, y_cost_train, y_cost_test = train_test_split(
    X_cost, y_cost, test_size=0.2, random_state=42
)

# Fit MARS model for cost prediction
mars_cost = earth.Earth(
    max_degree=2,         # Allow two-way interactions
    penalty=3.0,          # GCV penalty
    max_terms=21,        # Max number of terms
    minspan_alpha=0.05,  # Minimum span control
    endspan_alpha=0.05,  # End span control
    allow_linear=True,   # Allow linear terms
    allow_missing=True,   # Handle missing values
    feature_importance_type='gcv'  # Calculate feature importance
)

# Fit the model
mars_cost.fit(X_cost_train.values, y_cost_train.values)

# Model evaluation
train_r2_cost = mars_cost.score(X_cost_train.values, y_cost_train.values)
test_r2_cost = mars_cost.score(X_cost_test.values, y_cost_test.values)

print(f"\\nHealthcare Cost Prediction Model Summary:")
print(f"Number of basis functions: {len(mars_cost.basis_) - 1}")
print(f"Training R-squared: {train_r2_cost:.3f}")
print(f"Test R-squared: {test_r2_cost:.3f}")
print(f"GCV Score: {mars_cost.gcv_:.3f}")

# Feature importance for cost prediction
cost_feature_names = list(X_cost.columns)
cost_importances = mars_cost.feature_importances_

print(f"\\nFeature Importances for Healthcare Cost Prediction:")
for name, imp in zip(cost_feature_names, cost_importances):
    print(f"  {name}: {imp:.4f}")
```

## Conclusion

This real-world example demonstrates how pymars can be valuable for health economic outcomes research, where understanding complex relationships between patient characteristics, health outcomes, and costs is crucial for evidence-based policy making. The Pima Indians Diabetes dataset showcases several key advantages of pymars:

1. **Missing Data Handling**: pymars can handle the missing data patterns common in health records
2. **Non-linear Relationships**: MARS automatically detects and models complex non-linearities between health factors
3. **Interaction Detection**: The algorithm identifies important interaction effects between variables
4. **Interpretability**: The explicit basis functions provide clear insights into how different factors affect outcomes
5. **Scalability**: Works effectively with datasets of typical size in health research

The ability to model both clinical outcomes and health economic costs using the same flexible, interpretable framework makes pymars particularly valuable for health services research and policy analysis.