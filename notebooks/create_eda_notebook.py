import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Markdown cell - Title
title_cell = nbf.v4.new_markdown_cell('''# Exploratory Data Analysis: HDB Resale Flats Classification

This notebook analyzes the HDB resale flats dataset to understand the distribution of features and their relationships with price categories.''')

# Code cell - Imports and Setup
imports_cell = nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette('husl')

# Load configuration
try:
    with open('../src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: Configuration file not found. Please check the path.")
    raise

# Load the raw data
try:
    df = pd.read_csv(Path('../') / config['data']['raw_data_path'])
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: Data file not found. Please check the path.")
    raise
except Exception as e:
    print(f"Error loading data: {e}")
    raise''')

# Markdown cell - Data Overview
overview_title = nbf.v4.new_markdown_cell('## 1. Data Overview')

# Code cell - Data Overview
overview_cell = nbf.v4.new_code_cell('''# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\\nColumns:")
print(df.columns.tolist())
print("\\nData Types:")
print(df.dtypes)
print("\\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\\nNumber of duplicate rows: {duplicates}")

# Remove duplicates if any
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicate rows. New shape: {df.shape}")

# Convert data types where appropriate
# Convert categorical columns to category type
categorical_cols = ['town_name', 'flat_type', 'storey_range', 'flatm_name', 'block']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Convert price_category to binary
df['price_category'] = (df['price_category'] == 'Above Median').astype(int)

# Display updated data types
print("\\nUpdated Data Types:")
print(df.dtypes)''')

# Markdown cell - Descriptive Statistics
stats_title = nbf.v4.new_markdown_cell('## 2. Descriptive Statistics')

# Code cell - Descriptive Statistics
stats_cell = nbf.v4.new_code_cell('''# Descriptive statistics for numerical features
numerical_cols = ['floor_area_sqm', 'lease_commence_date', 'remaining_lease']
print("Numerical Features Statistics:")
print(df[numerical_cols].describe())

# Check for skewness in numerical features
print("\\nSkewness of Numerical Features:")
for col in numerical_cols:
    skewness = df[col].skew()
    print(f"{col}: {skewness:.4f}")
    if abs(skewness) > 1:
        print(f"  - {col} is highly skewed (|skewness| > 1)")

# Apply log transformation to highly skewed features
for col in numerical_cols:
    if df[col].skew() > 1 and df[col].min() > 0:  # Only apply log to positive values
        df[f'{col}_log'] = np.log1p(df[col])
        print(f"Applied log transformation to {col}")

# Descriptive statistics for categorical features
categorical_cols = ['town_name', 'flat_type', 'storey_range', 'flatm_name', 'block']
print("\\nCategorical Features Statistics:")
for col in categorical_cols:
    if col in df.columns:
        print(f"\\n{col} value counts:")
        print(df[col].value_counts().head(10))
        print(f"Number of unique values: {df[col].nunique()}")''')

# Markdown cell - Target Variable Analysis
target_title = nbf.v4.new_markdown_cell('## 3. Target Variable Analysis')

# Code cell - Target Variable Analysis
target_cell = nbf.v4.new_code_cell('''# Analyze price category distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='price_category')
plt.title('Distribution of Price Categories')
plt.xlabel('Price Category (0: Below Median, 1: Above Median)')
plt.ylabel('Count')
plt.show()

# Print class distribution
print("\\nClass Distribution:")
class_dist = df['price_category'].value_counts(normalize=True)
print(class_dist)
print(f"Class imbalance ratio: {class_dist[1]/class_dist[0]:.2f}")''')

# Markdown cell - Missing Values Analysis
missing_title = nbf.v4.new_markdown_cell('## 4. Missing Values Analysis')

# Code cell - Missing Values Analysis
missing_cell = nbf.v4.new_code_cell('''# Analyze missing values
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percent
})
print("Missing Values Analysis:")
print(missing_df[missing_df['Missing Values'] > 0])

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_df.index, y='Percentage', data=missing_df[missing_df['Missing Values'] > 0])
plt.title('Percentage of Missing Values by Feature')
plt.xticks(rotation=45)
plt.ylabel('Percentage (%)')
plt.tight_layout()
plt.show()

# Strategy for handling missing values
print("\\nStrategy for handling missing values:")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            print(f"- {col}: Numerical feature - Will use median imputation")
        else:
            print(f"- {col}: Categorical feature - Will use mode imputation")''')

# Markdown cell - Outlier Analysis
outlier_title = nbf.v4.new_markdown_cell('## 5. Outlier Analysis')

# Code cell - Outlier Analysis
outlier_cell = nbf.v4.new_code_cell('''# Identify outliers using IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return len(outliers), lower_bound, upper_bound

# Analyze outliers for numerical features
numerical_cols = ['floor_area_sqm', 'lease_commence_date', 'remaining_lease']
outlier_summary = []

for col in numerical_cols:
    count, lower, upper = detect_outliers(df, col)
    outlier_summary.append({
        'Feature': col,
        'Outlier Count': count,
        'Percentage': (count / len(df)) * 100,
        'Lower Bound': lower,
        'Upper Bound': upper
    })

outlier_df = pd.DataFrame(outlier_summary)
print("Outlier Analysis Summary:")
print(outlier_df)

# Visualize outliers using boxplots
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x='price_category', y=col, data=df)
    plt.title(f'{col} by Price Category')
    plt.xlabel('Price Category (0: Below Median, 1: Above Median)')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Visualize outliers using violin plots
plt.figure(figsize=(15, 5))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 3, i)
    sns.violinplot(x='price_category', y=col, data=df)
    plt.title(f'{col} Distribution by Price Category')
    plt.xlabel('Price Category (0: Below Median, 1: Above Median)')
    plt.ylabel(col)
plt.tight_layout()
plt.show()''')

# Markdown cell - Numerical Features Analysis
numerical_title = nbf.v4.new_markdown_cell('## 6. Numerical Features Analysis')

# Code cell - Numerical Features Analysis
numerical_cell = nbf.v4.new_code_cell('''# Select numerical columns
numerical_cols = ['floor_area_sqm', 'lease_commence_date', 'remaining_lease']

# Create subplots for numerical features
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Distribution of Numerical Features by Price Category')

for idx, col in enumerate(numerical_cols):
    sns.boxplot(data=df, x='price_category', y=col, ax=axes[idx])
    axes[idx].set_title(f'{col} Distribution')
    axes[idx].set_xlabel('Price Category (0: Below Median, 1: Above Median)')
    axes[idx].set_ylabel(col)
    axes[idx].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# Correlation matrix for numerical features
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols + ['price_category']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Analyze correlation strength
print("\\nCorrelation Analysis:")
for col in numerical_cols:
    corr = df[col].corr(df['price_category'])
    strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
    direction = "positive" if corr > 0 else "negative"
    print(f"- {col} has a {strength} {direction} correlation with price_category (r = {corr:.4f})")

# Pair plot for numerical features
plt.figure(figsize=(12, 10))
sns.pairplot(df, vars=numerical_cols + ['price_category'], hue='price_category', diag_kind='kde')
plt.suptitle('Pair Plot of Numerical Features', y=1.02)
plt.show()''')

# Markdown cell - Categorical Features Analysis
categorical_title = nbf.v4.new_markdown_cell('## 7. Categorical Features Analysis')

# Code cell - Categorical Features Analysis
categorical_cell = nbf.v4.new_code_cell('''# Select categorical columns
categorical_cols = ['town_name', 'flat_type', 'storey_range', 'flatm_name', 'block']

# Create subplots for top categories in each categorical feature
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Distribution of Categorical Features by Price Category')
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    # Get top 10 categories
    top_categories = df[col].value_counts().nlargest(10).index
    
    # Filter data for top categories
    plot_data = df[df[col].isin(top_categories)]
    
    # Create count plot
    sns.countplot(data=plot_data, x=col, hue='price_category', ax=axes[idx])
    axes[idx].set_title(f'Top 10 {col} Distribution')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].legend(title='Price Category', labels=['Below Median', 'Above Median'])

plt.tight_layout()
plt.show()

# Chi-square test for categorical features
from scipy.stats import chi2_contingency

print("\\nChi-Square Test Results:")
for col in categorical_cols:
    if col in df.columns:
        # Create contingency table
        contingency = pd.crosstab(df[col], df['price_category'])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Interpret results
        significance = "significant" if p_value < 0.05 else "not significant"
        print(f"- {col}: Chi-square = {chi2:.2f}, p-value = {p_value:.4f}, {significance} relationship with price_category")''')

# Markdown cell - Feature Engineering Insights
insights_title = nbf.v4.new_markdown_cell('## 8. Feature Engineering Insights')

# Code cell - Feature Engineering Insights
insights_cell = nbf.v4.new_code_cell('''# Analyze remaining lease distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='remaining_lease', hue='price_category', multiple="stack")
plt.title('Distribution of Remaining Lease by Price Category')
plt.xlabel('Remaining Lease (Years)')
plt.ylabel('Count')
plt.legend(title='Price Category', labels=['Below Median', 'Above Median'])
plt.show()

# Analyze floor area distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='floor_area_sqm', hue='price_category', multiple="stack")
plt.title('Distribution of Floor Area by Price Category')
plt.xlabel('Floor Area (sqm)')
plt.ylabel('Count')
plt.legend(title='Price Category', labels=['Below Median', 'Above Median'])
plt.show()

# Analyze lease commence date distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='lease_commence_date', hue='price_category', multiple="stack")
plt.title('Distribution of Lease Commence Date by Price Category')
plt.xlabel('Lease Commence Date')
plt.ylabel('Count')
plt.legend(title='Price Category', labels=['Below Median', 'Above Median'])
plt.show()

# Feature importance analysis using mutual information
from sklearn.feature_selection import mutual_info_classif

# Prepare features for mutual information calculation
X = df.drop('price_category', axis=1)
y = df['price_category']

# Convert categorical variables to numeric using label encoding
X_encoded = X.copy()
for col in categorical_cols:
    if col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# Calculate mutual information
mi_scores = mutual_info_classif(X_encoded, y)
mi_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Mutual Information': mi_scores
})
mi_df = mi_df.sort_values('Mutual Information', ascending=False)

# Visualize feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Mutual Information', y='Feature', data=mi_df)
plt.title('Feature Importance Based on Mutual Information')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\\nFeature Importance Ranking:")
print(mi_df)''')

# Markdown cell - Temporal Analysis
temporal_title = nbf.v4.new_markdown_cell('## 9. Temporal Analysis')

# Code cell - Temporal Analysis
temporal_cell = nbf.v4.new_code_cell('''# Analyze temporal trends
# Group by lease commence date and calculate the proportion of above median prices
lease_trend = df.groupby('lease_commence_date')['price_category'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=lease_trend, x='lease_commence_date', y='price_category')
plt.title('Proportion of Above Median Prices by Lease Commence Date')
plt.xlabel('Lease Commence Date')
plt.ylabel('Proportion of Above Median Prices')
plt.grid(True)
plt.show()

# Calculate correlation between lease commence date and price category
lease_corr = df['lease_commence_date'].corr(df['price_category'])
print(f"Correlation between lease commence date and price category: {lease_corr:.4f}")''')

# Markdown cell - Key Findings
findings_cell = nbf.v4.new_markdown_cell('''## 10. Key Findings and Insights

1. **Target Distribution**:
   - The dataset shows a balanced distribution between above and below median price categories
   - Class imbalance ratio: [calculated value]

2. **Data Quality**:
   - Number of duplicate rows: [calculated value]
   - Missing values were identified and strategies for handling them were proposed
   - Outliers were identified and quantified for numerical features

3. **Numerical Features**:
   - Floor area shows a [strength] [direction] correlation with price category
   - Remaining lease has a [strength] [direction] correlation with price category
   - Lease commence date shows a [strength] [direction] correlation with price category
   - Skewness was addressed through log transformation where appropriate

4. **Categorical Features**:
   - Town name has a [significant/not significant] relationship with price category (Chi-square test)
   - Flat type has a [significant/not significant] relationship with price category (Chi-square test)
   - Storey range has a [significant/not significant] relationship with price category (Chi-square test)

5. **Feature Engineering**:
   - The remaining lease calculation provides valuable insights into property value
   - Floor area standardization helps in comparing properties across different types
   - Log transformation of skewed features improves model performance

6. **Feature Importance**:
   - Top 3 most important features based on mutual information:
     1. [Feature 1]
     2. [Feature 2]
     3. [Feature 3]

7. **Temporal Trends**:
   - [Insights about temporal patterns in the data]

These insights can be used to:
1. Guide feature selection in the model
2. Inform preprocessing steps
3. Help in understanding model predictions
4. Identify potential areas for additional feature engineering''')

# Combine all cells
nb.cells = [
    title_cell,
    imports_cell,
    overview_title,
    overview_cell,
    stats_title,
    stats_cell,
    target_title,
    target_cell,
    missing_title,
    missing_cell,
    outlier_title,
    outlier_cell,
    numerical_title,
    numerical_cell,
    categorical_title,
    categorical_cell,
    insights_title,
    insights_cell,
    temporal_title,
    temporal_cell,
    findings_cell
]

# Write the notebook to a file
with open('eda.ipynb', 'w') as f:
    nbf.write(nb, f) 