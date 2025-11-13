import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = 'Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# Step 1: Handle missing values using mean/median/imputation
# For numerical columns: impute using mean (if not specified)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer_mean = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer_mean.fit_transform(df[numeric_cols])

# For categorical columns: impute using most frequent (mode)
categorical_cols = df.select_dtypes(include=['object']).columns
imputer_mode = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])

# Step 3: Convert categorical features into numerical using encoding
# Label Encoding for binary/category columns, OneHotEncoding for others
le = LabelEncoder()
df_encoded = df.copy()

for col in categorical_cols:
    if df[col].nunique() == 2:
        # Binary columns
        df_encoded[col] = le.fit_transform(df[col])
    else:
        # More than two categories, use OneHotEncoding
        onehot = pd.get_dummies(df[col], prefix=col)
        df_encoded = pd.concat([df_encoded.drop(col, axis=1), onehot], axis=1)

# Step 4: Normalize/standardize numerical features
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

# Step 5: Visualize outliers using boxplots and remove them
outlier_report = {}
for col in numeric_cols:
    # Visualize boxplot
    plt.figure()
    sns.boxplot(df_encoded[col])
    plt.title(f'Boxplot of {col}')
    plt.savefig(f'{col}_boxplot.png')
    plt.close()
    # Remove outliers using IQR
    Q1 = df_encoded[col].quantile(0.25)
    Q3 = df_encoded[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    removed = (df_encoded[col] < lower) | (df_encoded[col] > upper)
    outlier_report[col] = int(removed.sum())
    df_encoded = df_encoded[~removed]

# Save clean dataset
df_encoded.to_csv('Titanic-Cleaned.csv', index=False)

# Prepare outlier report
outlier_report_csv = pd.DataFrame.from_dict(outlier_report, orient='index', columns=['Outliers Removed'])
outlier_report_csv.to_csv('Outlier-Report.csv')
'Finished cleaning and encoding data, generating boxplots and outlier report.'
# List generated output files
import os
output_files = [f for f in os.listdir('.') if f.endswith('.csv') or f.endswith('.png')]
output_files[:5]  # show a sample of generated files for confirmation

# This code saves: Titanic-Cleaned.csv (final cleaned dataset), Outlier-Report.csv (summary of outliers removed), and .png boxplot images for each numerical feature. The cleaned dataset is ready for submission.