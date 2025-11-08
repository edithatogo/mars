# Healthcare-focused Examples and Use Cases for pymars

This document provides detailed examples of using pymars in real-world health economic scenarios. The examples will be more comprehensive and realistic than the basic examples in the paper.

## Australian Healthcare Example: Hospital Length of Stay Prediction

This example demonstrates how pymars can be used to predict hospital length of stay based on patient characteristics, which is crucial for hospital resource planning and cost management.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pymars as earth
from pymars.explain import plot_partial_dependence

# Set random seed for reproducibility
np.random.seed(12345)

# Generate synthetic Australian hospital data
n_patients = 5000

# Demographics
age = np.random.gamma(2, 10, n_patients)  # Patients aged 0 to ~80
gender = np.random.binomial(1, 0.52, n_patients)  # ~52% female, as in Australia
postcode_area = np.random.choice([1, 2, 3, 4], n_patients, p=[0.6, 0.2, 0.15, 0.05])  # 4 regions

# Clinical factors
comorbidity_score = np.random.gamma(2, 1.5, n_patients)  # Charlson Comorbidity Index
emergency_admission = np.random.binomial(1, 0.65, n_patients)  # 65% emergency admissions
primary_diagnosis = np.random.choice([1, 2, 3, 4, 5], n_patients, p=[0.3, 0.25, 0.2, 0.15, 0.1])

# Socioeconomic factors
ses_quintile = np.random.choice([1, 2, 3, 4, 5], n_patients, p=[0.2, 0.2, 0.2, 0.2, 0.2])  # SEIFA quintiles

# Simulate length of stay with realistic non-linear relationships
los = (
    2.0  # Base LOS
    + 0.02 * age  # Effect of age
    + 0.0003 * age**2  # Quadratic age effect
    + 0.5 * gender  # Gender effect
    + 0.8 * comorbidity_score  # Comorbidity effect
    + 1.2 * emergency_admission  # Emergency admission effect
    + 0.3 * ses_quintile  # Socioeconomic effect (higher SES = slightly longer stay)
    + np.where(age > 65, 1.5 * comorbidity_score, 0.5 * comorbidity_score)  # Age-comorbidity interaction
    + np.where(postcode_area == 4, 0.5, 0)  # Rural area effect
    + np.random.exponential(1, n_patients)  # Random variation
)

# Add some diagnostic-specific effects
los = np.where(primary_diagnosis == 1, los + 1.0, los)  # Diagnosis 1: longer stay
los = np.where(primary_diagnosis == 2, los + 0.5, los)  # Diagnosis 2: moderate increase
los = np.where(primary_diagnosis == 5, los + 2.0, los)  # Diagnosis 5: much longer stay

# Ensure realistic range (0.5 to 30 days)
los = np.clip(los, 0.5, 30)

# Create feature matrix
X = pd.DataFrame({
    'age': age,
    'gender': gender,
    'ses_quintile': ses_quintile,
    'comorbidity_score': comorbidity_score,
    'emergency_admission': emergency_admission,
    'postcode_area': postcode_area,
    'primary_diagnosis': primary_diagnosis
})

# Convert to dummy variables for categorical features
X = pd.get_dummies(X, columns=['postcode_area', 'primary_diagnosis'], prefix=['region', 'diag'])

# Prepare target variable
y = los

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit MARS model
print("Fitting MARS model for hospital length of stay prediction...")
model_los = earth.Earth(
    max_degree=2,           # Allow two-way interactions
    penalty=3.0,            # GCV penalty
    max_terms=21,           # Max number of terms (rule of thumb: 2*n_features + 1)
    minspan_alpha=0.05,     # Minimum span control
    endspan_alpha=0.05,     # End span control
    allow_linear=True,      # Allow linear terms
    feature_importance_type='gcv'  # Calculate feature importance
)

# Fit the model
model_los.fit(X_train.values, y_train)

# Model evaluation
train_r2 = model_los.score(X_train.values, y_train)
test_r2 = model_los.score(X_test.values, y_test)

print(f"\nHospital LOS Model Summary:")
print(f"Number of basis functions: {len(model_los.basis_) - 1}")  # Subtract 1 for intercept
print(f"Training R-squared: {train_r2:.3f}")
print(f"Test R-squared: {test_r2:.3f}")
print(f"GCV Score: {model_los.gcv_:.3f}")

# Feature importance
print(f"\nFeature Importances:")
print(model_los.summary_feature_importances())

# Predictions
y_pred_train = model_los.predict(X_train.values)
y_pred_test = model_los.predict(X_test.values)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Training set
axes[0].scatter(y_train, y_pred_train, alpha=0.5)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Length of Stay')
axes[0].set_ylabel('Predicted Length of Stay')
axes[0].set_title(f'Training Set (R² = {train_r2:.3f})')

# Test set
axes[1].scatter(y_test, y_pred_test, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Length of Stay')
axes[1].set_ylabel('Predicted Length of Stay')
axes[1].set_title(f'Test Set (R² = {test_r2:.3f})')

plt.tight_layout()
plt.show()

# Generate partial dependence plots for the top 3 most important features
feature_names = list(X.columns)
top_features = np.argsort(model_los.feature_importances_)[-3:][::-1]

fig, axes = plot_partial_dependence(
    model_los, 
    X_train.values, 
    top_features, 
    feature_names=feature_names, 
    n_cols=3, 
    figsize=(18, 5)
)
plt.suptitle("Partial Dependence Plots for Top 3 Features", fontsize=16)
plt.tight_layout()
plt.show()
```

## New Zealand Health Equity Example: Māori Health Disparities

This example demonstrates how pymars can be used to analyze health disparities, specifically focusing on factors affecting health outcomes for Māori compared to non-Māori populations in New Zealand.

```python
# Set random seed for reproducibility
np.random.seed(67890)

# Generate synthetic New Zealand health data with Māori health disparities
n_individuals = 4000

# Demographics
age = np.random.normal(45, 16, n_individuals)
age = np.clip(age, 0, 90)  # Realistic age range
gender = np.random.binomial(1, 0.51, n_individuals)  # Slightly more females in NZ

# Ethnicity indicators (mimicking NZ demographics)
maori = np.random.binomial(1, 0.15, n_individuals)  # ~15% Māori
pacific = np.random.binomial(1, 0.08, n_individuals)  # ~8% Pacific peoples
asian = np.random.binomial(1, 0.15, n_individuals)  # ~15% Asian
european_other = 1 - (maori + pacific + asian)  # Remaining ~62%

# Socioeconomic factors
deprivation_quintile = np.random.choice([1, 2, 3, 4, 5], n_individuals, p=[0.2, 0.2, 0.2, 0.2, 0.2])  # NZDep quintiles
employment_status = np.random.binomial(1, 0.65, n_individuals)  # 65% employed
education_level = np.random.choice([1, 2, 3], n_individuals, p=[0.3, 0.4, 0.3])  # 1=low, 2=medium, 3=high

# Geographic factors
rurality = np.random.binomial(1, 0.14, n_individuals)  # ~14% rural in NZ
region = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n_individuals)  # 8 DHB regions

# Healthcare access
primary_care_access_score = np.random.beta(2, 1, n_individuals)  # 0-1 scale
specialist_access_score = np.random.beta(1.5, 2, n_individuals)  # 0-1 scale

# Simulate health outcome score with disparities
health_score = (
    50  # Base score
    - 0.3 * age  # Deterioration with age
    - 0.002 * age**2  # Accelerating effect
    + 2 * gender  # Gender effect (female=1, male=0)
    - 3 * deprivation_quintile  # Worse outcomes with higher deprivation
    + 10 * employment_status  # Better outcomes with employment
    + 8 * education_level  # Better outcomes with higher education
    - 5 * rurality  # Rural disadvantage
    - 8 * maori  # Māori health disparities
    - 5 * pacific  # Pacific health disparities  
    - 3 * asian  # Asian health disparities (simplified)
    + 15 * primary_care_access_score  # Better access = better outcomes
    + 10 * specialist_access_score  # Better specialist access = better outcomes
    # Important interaction: Māori and deprivation effect
    + np.where(deprivation_quintile >= 4, -5 * maori, 0)
    # Additional interaction: Age and employment
    + np.where(age > 65, -3 * (1 - employment_status), 0)
    + np.random.normal(0, 5, n_individuals)  # Random noise
)

# Ensure realistic range (0-100 scale)
health_score = np.clip(health_score, 0, 100)

# Create feature matrix
X_nz = pd.DataFrame({
    'age': age,
    'gender': gender,
    'deprivation_quintile': deprivation_quintile,
    'employment_status': employment_status,
    'education_level': education_level,
    'rurality': rurality,
    'primary_care_access_score': primary_care_access_score,
    'specialist_access_score': specialist_access_score,
    'maori': maori,
    'pacific': pacific,
    'asian': asian
})

# Prepare target variable
y_nz = health_score

# Split the data
X_nz_train, X_nz_test, y_nz_train, y_nz_test = train_test_split(X_nz, y_nz, test_size=0.2, random_state=42)

# Fit MARS model to detect disparities
print("\nFitting MARS model for New Zealand health disparities analysis...")
model_disparities = earth.Earth(
    max_degree=2,           # Allow two-way interactions
    penalty=3.0,            # GCV penalty
    max_terms=21,           # Max number of terms
    minspan_alpha=0.05,     # Minimum span control
    endspan_alpha=0.05,     # End span control
    allow_linear=True,      # Allow linear terms
    feature_importance_type='nb_subsets'  # Calculate feature importance
)

# Fit the model
model_disparities.fit(X_nz_train.values, y_nz_train)

# Model evaluation
train_r2_nz = model_disparities.score(X_nz_train.values, y_nz_train)
test_r2_nz = model_disparities.score(X_nz_test.values, y_nz_test)

print(f"\nNZ Health Disparities Model Summary:")
print(f"Number of basis functions: {len(model_disparities.basis_) - 1}")  # Subtract 1 for intercept
print(f"Training R-squared: {train_r2_nz:.3f}")
print(f"Test R-squared: {test_r2_nz:.3f}")
print(f"GCV Score: {model_disparities.gcv_:.3f}")

# Feature importance
print(f"\nFeature Importances:")
feature_names_nz = list(X_nz.columns)
print(model_disparities.summary_feature_importances())

# Show the top 5 most important features and their role
importance_indices = np.argsort(model_disparities.feature_importances_)[-5:][::-1]
print(f"\nTop 5 Most Important Features:")
for idx in importance_indices:
    print(f"  {feature_names_nz[idx]}: {model_disparities.feature_importances_[idx]:.3f}")

# Analyze disparities by ethnicity
ethnicity_cols = ['maori', 'pacific', 'asian']
for eth_col in ethnicity_cols:
    eth_mask = X_nz_test[eth_col] == 1
    non_eth_mask = X_nz_test[eth_col] == 0
    
    if eth_mask.sum() > 0:  # Check if any individuals of this ethnicity exist in test set
        eth_pred = model_disparities.predict(X_nz_test.values[eth_mask])
        non_eth_pred = model_disparities.predict(X_nz_test.values[non_eth_mask])
        
        eth_actual = y_nz_test[eth_mask]
        non_eth_actual = y_nz_test[non_eth_mask]
        
        print(f"\n{eth_col.capitalize()} Health Disparities Analysis:")
        print(f"  {eth_col.capitalize()} Mean Predicted Score: {eth_pred.mean():.2f}")
        print(f"  Non-{eth_col.capitalize()} Mean Predicted Score: {non_eth_pred.mean():.2f}")
        print(f"  {eth_col.capitalize()} Actual Score: {eth_actual.mean():.2f}")
        print(f"  Non-{eth_col.capitalize()} Actual Score: {non_eth_actual.mean():.2f}")
        print(f"  Predicted Disparity: {non_eth_pred.mean() - eth_pred.mean():.2f}")
        print(f"  Actual Disparity: {non_eth_actual.mean() - eth_actual.mean():.2f}")

# Generate partial dependence plots for ethnicity-related interactions
# Specifically looking at the interaction between ethnicity and deprivation
top_features_nz = np.argsort(model_disparities.feature_importances_)[-4:][::-1]

fig_nz, axes_nz = plot_partial_dependence(
    model_disparities, 
    X_nz_train.values, 
    top_features_nz, 
    feature_names=feature_names_nz, 
    n_cols=2, 
    figsize=(12, 10)
)
plt.suptitle("Partial Dependence Plots for NZ Health Disparities", fontsize=16)
plt.tight_layout()
plt.show()
```

## Cost-Effectiveness Analysis Example

This example demonstrates how pymars can be applied to cost-effectiveness analysis in healthcare settings, showing the non-linear relationship between intervention intensity and health outcomes.

```python
# Set random seed for reproducibility
np.random.seed(98765)

# Generate synthetic cost-effectiveness data
n_interventions = 3000

# Intervention characteristics
intensity_level = np.random.uniform(0, 10, n_interventions)  # 0-10 scale intervention intensity
duration_months = np.random.uniform(1, 12, n_interventions)  # 1-12 months duration
target_population_size = np.random.lognormal(8, 0.8, n_interventions)  # Population size
target_population_risk = np.random.beta(2, 5, n_interventions)  # Baseline risk (0-1)

# Patient characteristics that might affect intervention effectiveness
avg_age = np.random.normal(55, 12, n_interventions)
comorbidity_burden = np.random.gamma(1.5, 1.2, n_interventions)

# Simulate health outcomes with non-linear intervention effects
# Using a saturating function to model diminishing returns at high intensity
effectiveness_score = (
    10 * target_population_risk  # Baseline effectiveness based on risk
    + 5 * (1 - np.exp(-0.3 * intensity_level))  # Saturating effect of intervention intensity
    + 2 * (1 - np.exp(-0.1 * duration_months))  # Saturating effect of duration
    - 0.05 * avg_age  # Effectiveness decreases with age
    - 1.5 * comorbidity_burden  # Effectiveness decreases with comorbidity
    + np.where(intensity_level > 7, 3 * target_population_risk, 0)  # High intensity benefit for high-risk
    + np.random.normal(0, 2, n_interventions)  # Random noise
)

# Calculate costs with economies of scale and diminishing returns
# Higher intensity interventions are more expensive but with diminishing returns to scale
cost_per_person = (
    100  # Base cost
    + 20 * intensity_level  # Cost increases with intensity
    + 5 * intensity_level**1.2  # Diminishing returns effect
    + 30 * duration_months  # Cost increases with duration
    + 100 / (1 + target_population_size / 10000)  # Economies of scale
    + 50 * comorbidity_burden  # Higher cost for complex patients
    + np.random.normal(0, 10, n_interventions)  # Random cost variation
)

# Total cost for the intervention
total_cost = cost_per_person * target_population_size

# Calculate QALYs gained (Quality Adjusted Life Years)
# Non-linear relationship with intervention characteristics
qalys_gained = (
    0.1 * target_population_risk  # Base QALYs based on risk
    + 0.05 * (1 - np.exp(-0.4 * intensity_level))  # Saturating QALYs with intensity
    + 0.02 * (1 - np.exp(-0.15 * duration_months))  # Saturating QALYs with duration
    + 0.01 * target_population_size / 1000  # Small effect for larger populations
    - 0.005 * avg_age / 10  # Slightly lower QALYs for older populations
    - 0.03 * comorbidity_burden  # Lower QALYs for complex patients
    + np.random.normal(0, 0.01, n_interventions)  # Random variation
) * target_population_size

# Calculate cost-effectiveness ratio (Cost per QALY)
cost_per_qaly = total_cost / (qalys_gained + 0.01)  # Add small value to avoid division by zero

# Create feature matrix for cost-effectiveness prediction
X_ce = pd.DataFrame({
    'intensity_level': intensity_level,
    'duration_months': duration_months,
    'target_population_size': target_population_size,
    'target_population_risk': target_population_risk,
    'avg_age': avg_age,
    'comorbidity_burden': comorbidity_burden
})

# Target variable: cost per QALY (lower is better)
y_ce = cost_per_qaly

# Remove any infinite or NaN values that might have resulted from the calculation
valid_mask = (np.isfinite(y_ce)) & (y_ce > 0)
X_ce = X_ce[valid_mask]
y_ce = y_ce[valid_mask]

# Take a sample for computational efficiency
sample_size = min(2000, len(X_ce))
sample_indices = np.random.choice(len(X_ce), sample_size, replace=False)
X_ce = X_ce.iloc[sample_indices]
y_ce = y_ce.iloc[sample_indices]

# Split the data
X_ce_train, X_ce_test, y_ce_train, y_ce_test = train_test_split(X_ce, y_ce, test_size=0.2, random_state=42)

# Fit MARS model for cost-effectiveness prediction
print("\nFitting MARS model for cost-effectiveness analysis...")
model_ce = earth.Earth(
    max_degree=2,           # Allow two-way interactions
    penalty=3.0,            # GCV penalty
    max_terms=21,           # Max number of terms
    minspan_alpha=0.05,     # Minimum span control
    endspan_alpha=0.05,     # End span control
    allow_linear=True,      # Allow linear terms
    feature_importance_type='rss'  # Calculate feature importance
)

# Fit the model
model_ce.fit(X_ce_train.values, y_ce_train)

# Model evaluation
train_r2_ce = model_ce.score(X_ce_train.values, y_ce_train)
test_r2_ce = model_ce.score(X_ce_test.values, y_ce_test)

print(f"\nCost-Effectiveness Model Summary:")
print(f"Number of basis functions: {len(model_ce.basis_) - 1}")  # Subtract 1 for intercept
print(f"Training R-squared: {train_r2_ce:.3f}")
print(f"Test R-squared: {test_r2_ce:.3f}")
print(f"GCV Score: {model_ce.gcv_:.3f}")

# Feature importance for cost-effectiveness
print(f"\nFeature Importances for Cost-Effectiveness:")
feature_names_ce = list(X_ce.columns)
print(model_ce.summary_feature_importances())

# Show selected basis functions and their coefficients
print(f"\nSelected Basis Functions (Top 10 by absolute coefficient):")
# Sort basis functions by absolute coefficient value
coef_indices = np.argsort(np.abs(model_ce.coef_))[::-1]
for i in coef_indices[:10]:
    print(f"  {model_ce.basis_[i]}: coefficient = {model_ce.coef_[i]:.3f}")

# Identify most cost-effective interventions
pred_costs_per_qaly = model_ce.predict(X_ce_test.values)
most_cost_effective = np.argsort(pred_costs_per_qaly)  # Lower is better

print(f"\nAnalysis of Most vs Least Cost-Effective Interventions:")
print(f"  Most cost-effective (top 5): Average cost/QALY = {pred_costs_per_qaly[most_cost_effective[:5]].mean():.2f}")
print(f"  Least cost-effective (bottom 5): Average cost/QALY = {pred_costs_per_qaly[most_cost_effective[-5:]].mean():.2f}")

# Analyze the relationship between intensity and cost-effectiveness
intensity_test = X_ce_test['intensity_level'].values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(intensity_test, pred_costs_per_qaly, alpha=0.6)
plt.xlabel('Intervention Intensity Level')
plt.ylabel('Predicted Cost per QALY')
plt.title('Intervention Intensity vs Cost-Effectiveness')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Show the non-linear relationship by plotting actual vs predicted
plt.scatter(y_ce_test, pred_costs_per_qaly, alpha=0.6)
plt.plot([y_ce_test.min(), y_ce_test.max()], [y_ce_test.min(), y_ce_test.max()], 'r--', lw=2)
plt.xlabel('Actual Cost per QALY')
plt.ylabel('Predicted Cost per QALY')
plt.title(f'Actual vs Predicted Cost-Effectiveness (R² = {test_r2_ce:.3f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Policy Impact Simulation

Finally, let's demonstrate how pymars can be used to simulate policy impacts, which is important for health economic modeling:

```python
# Set random seed for reproducibility
np.random.seed(11223)

# Generate synthetic population data for policy simulation
n_individuals = 6000

# Baseline characteristics
age = np.random.normal(40, 15, n_individuals)
age = np.clip(age, 0, 90)
gender = np.random.binomial(1, 0.51, n_individuals)  # Slight female majority
ethnicity = np.random.choice([0, 1, 2], n_individuals, p=[0.68, 0.15, 0.17])  # Pakeha/Maori/Other NZ demographics
income_log = np.random.normal(10, 0.8, n_individuals)  # Log income
income = np.exp(income_log)  # Actual income
education = np.random.choice([1, 2, 3], n_individuals, p=[0.25, 0.5, 0.25])  # Low/Medium/High
deprivation = np.random.choice([1, 2, 3, 4, 5], n_individuals, p=[0.2, 0.2, 0.2, 0.2, 0.2])  # NZDep quintiles

# Health system factors
rurality = np.random.binomial(1, 0.14, n_individuals)  # ~14% rural
access_to_primary_care = np.random.beta(2, 1, n_individuals)  # 0-1 scale
access_to_specialist = np.random.beta(1.5, 2, n_individuals)  # 0-1 scale

# Simulate baseline health outcome
baseline_health = (
    70  # Base health score
    - 0.3 * age  # Deterioration with age
    - 0.001 * age**2  # Accelerating effect
    + 2 * gender  # Gender difference
    + 2 * education  # Education effect
    - 5 * (deprivation - 1)  # Deprivation effect
    + 15 * access_to_primary_care  # Better access = better health
    + 8 * access_to_specialist  # Better specialist access = better health
    - 10 * (ethnicity == 1)  # Māori health disparities
    - 8 * (ethnicity == 2)  # Other ethnicity disparities
    + 0.000001 * income  # Income effect
    + np.random.normal(0, 8, n_individuals)  # Random variation
)

# Simulate policy intervention (e.g., health initiative targeting high-risk groups)
# Policy increases access to care for high-risk individuals
high_risk = (deprivation >= 4) | (age > 65) | (baseline_health < 40)
policy_access_impact = np.where(high_risk, 0.3, 0.1)  # Greater impact on high-risk
policy_education_impact = np.where(high_risk, 0.5, 0.2)  # Education improvement

# Simulate health outcome after policy implementation
post_policy_health = (
    baseline_health
    + 8 * policy_access_impact  # Health gain from improved access
    + 5 * policy_education_impact  # Health gain from education
    + np.random.normal(0, 5, n_individuals)  # Random variation post-policy
)

# Create feature matrix (before policy)
X_policy = pd.DataFrame({
    'age': age,
    'gender': gender,
    'education': education,
    'deprivation': deprivation,
    'access_to_primary_care': access_to_primary_care,
    'access_to_specialist': access_to_specialist,
    'ethnicity_maori': (ethnicity == 1).astype(int),
    'ethnicity_other': (ethnicity == 2).astype(int),
    'income_log': income_log,
    'rurality': rurality
})

# Target variable: health improvement (post-policy - baseline)
y_policy = post_policy_health - baseline_health

# Split the data
X_pol_train, X_pol_test, y_pol_train, y_pol_test = train_test_split(X_policy, y_policy, test_size=0.2, random_state=42)

# Fit MARS model to predict policy impact
print("\nFitting MARS model for policy impact simulation...")
model_policy = earth.Earth(
    max_degree=2,           # Allow two-way interactions
    penalty=3.0,            # GCV penalty
    max_terms=21,           # Max number of terms
    minspan_alpha=0.05,     # Minimum span control
    endspan_alpha=0.05,     # End span control
    allow_linear=True,      # Allow linear terms
    feature_importance_type='gcv'  # Calculate feature importance
)

# Fit the model
model_policy.fit(X_pol_train.values, y_pol_train)

# Model evaluation
train_r2_pol = model_policy.score(X_pol_train.values, y_pol_train)
test_r2_pol = model_policy.score(X_pol_test.values, y_pol_test)

print(f"\nPolicy Impact Model Summary:")
print(f"Number of basis functions: {len(model_policy.basis_) - 1}")  # Subtract 1 for intercept
print(f"Training R-squared: {train_r2_pol:.3f}")
print(f"Test R-squared: {test_r2_pol:.3f}")
print(f"GCV Score: {model_policy.gcv_:.3f}")

# Feature importance
print(f"\nFeature Importances for Policy Impact:")
feature_names_pol = list(X_policy.columns)
print(model_policy.summary_feature_importances())

# Analyze policy impact by demographic groups
print(f"\nPolicy Impact Analysis:")
ethnicity_labels = ['Pakeha', 'Maori', 'Other']
for i, eth_label in enumerate(ethnicity_labels):
    eth_mask = (X_pol_test['ethnicity_maori'] == (i == 1)) & (X_pol_test['ethnicity_other'] == (i == 2))
    if eth_mask.any():
        eth_pred_impact = model_policy.predict(X_pol_test.values[eth_mask])
        print(f"  {eth_label}: Average predicted health improvement = {eth_pred_impact.mean():.2f}")

# Analyze by deprivation quintiles
for quint in range(1, 6):
    dep_mask = X_pol_test['deprivation'] == quint
    if dep_mask.any():
        dep_pred_impact = model_policy.predict(X_pol_test.values[dep_mask])
        print(f"  Deprivation Q{quint}: Average predicted health improvement = {dep_pred_impact.mean():.2f}")

# Plot actual vs predicted policy impact
plt.figure(figsize=(8, 6))
plt.scatter(y_pol_test, model_policy.predict(X_pol_test.values), alpha=0.6)
plt.plot([y_pol_test.min(), y_pol_test.max()], [y_pol_test.min(), y_pol_test.max()], 'r--', lw=2)
plt.xlabel('Actual Health Improvement')
plt.ylabel('Predicted Health Improvement')
plt.title(f'Policy Impact: Actual vs Predicted (R² = {test_r2_pol:.3f})')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nDemonstration of pymars in health economic applications completed!")
print(f"Examples covered:")
print(f"  1. Hospital length of stay prediction")
print(f"  2. Health equity analysis (NZ Māori disparities)")
print(f"  3. Cost-effectiveness analysis")
print(f"  4. Policy impact simulation")

print(f"\nThese examples demonstrate how pymars can:")
print(f"  - Model complex non-linear relationships in health data")
print(f"  - Capture interaction effects between variables")
print(f"  - Identify important factors in health outcomes and costs")
print(f"  - Support evidence-based decision making in healthcare")
```