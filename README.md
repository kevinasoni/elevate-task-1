# Elevate-task-1
# Titanic Dataset Preprocessing

This repository contains the cleaned and preprocessed Titanic dataset along with the necessary visualizations and reports generated for an internship task.

## Dataset

- **Original dataset:** Titanic-Dataset.csv (raw data)
- **Processed dataset:** Titanic-Cleaned.csv (data with missing values handled, categorical features encoded, numerical features standardized, and outliers removed)

## Preprocessing Steps

1. **Handling Missing Values:**
   - Numerical columns imputed using mean values.
   - Categorical columns imputed using the most frequent values.

2. **Encoding Categorical Features:**
   - Binary categorical columns converted using Label Encoding.
   - Multi-category columns converted using One-Hot Encoding.

3. **Normalization/Standardization:**
   - Numerical features were standardized using `StandardScaler`.

4. **Outlier Handling:**
   - Boxplots were generated to visualize outliers for all numerical columns.
   - Outliers were removed using the Interquartile Range (IQR) method.
   - Outlier removal summary is available in Outlier-Report.csv.

## Files Included

- `Titanic-Cleaned.csv`: Cleaned and preprocessed data ready for modeling or analysis.
- `Outlier-Report.csv`: Number of outliers removed per numerical feature.
- Boxplot PNG images (`*.png`): Visual representations of outliers for each numerical feature.
