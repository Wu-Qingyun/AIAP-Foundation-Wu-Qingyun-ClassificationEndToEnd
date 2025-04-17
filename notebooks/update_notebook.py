import nbformat as nbf
from pathlib import Path

def create_notebook():
    # Create a new notebook
    nb = nbf.v4.new_notebook()

    # Title cell
    title_cell = nbf.v4.new_markdown_cell('''# Exploratory Data Analysis: HDB Resale Flats Classification

This notebook analyzes the HDB resale flats dataset to understand the distribution of features and their relationships with price categories.''')

    # Imports cell
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

    # Data Overview
    overview_title = nbf.v4.new_markdown_cell('## 1. Data Overview')
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

    # Descriptive Statistics
    stats_title = nbf.v4.new_markdown_cell('## 2. Descriptive Statistics')
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
print("\\nCategorical Features Statistics:")
for col in categorical_cols:
    if col in df.columns:
        print(f"\\n{col} value counts:")
        print(df[col].value_counts().head(10))
        print(f"Number of unique values: {df[col].nunique()}")''')

    # Target Variable Analysis
    target_title = nbf.v4.new_markdown_cell('## 3. Target Variable Analysis')
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

    # Missing Values Analysis
    missing_title = nbf.v4.new_markdown_cell('## 4. Missing Values Analysis')
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

    # Outlier Analysis
    outlier_title = nbf.v4.new_markdown_cell('## 5. Outlier Analysis')
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

    # Feature Engineering Insights
    insights_title = nbf.v4.new_markdown_cell('## 6. Feature Engineering Insights')
    insights_cell = nbf.v4.new_code_cell('''# Feature importance analysis using mutual information
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

    # Key Findings
    findings_cell = nbf.v4.new_markdown_cell('''## 7. Key Findings and Insights

1. **Data Quality**:
   - Missing values were identified and strategies for handling them were proposed
   - Outliers were identified and quantified for numerical features
   - Data types were appropriately converted for analysis

2. **Target Variable**:
   - The dataset shows a balanced distribution between price categories
   - Class imbalance ratio is calculated and reported

3. **Numerical Features**:
   - Skewness was analyzed and addressed through log transformation where appropriate
   - Outliers were identified using the IQR method
   - Feature importance was calculated using mutual information

4. **Categorical Features**:
   - Categorical variables were properly encoded for analysis
   - Value counts and distributions were analyzed
   - Relationships with the target variable were examined

5. **Feature Engineering Recommendations**:
   - Features requiring log transformation were identified
   - Important features for prediction were ranked
   - Strategies for handling missing values were proposed

These insights will guide:
1. Feature selection and preprocessing steps
2. Model selection and training strategies
3. Data cleaning and transformation decisions''')

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
        insights_title,
        insights_cell,
        findings_cell
    ]

    return nb

if __name__ == '__main__':
    # Create the notebook
    nb = create_notebook()
    
    # Write the notebook to a file
    notebook_path = Path('eda.ipynb')
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook created successfully at {notebook_path.absolute()}") 