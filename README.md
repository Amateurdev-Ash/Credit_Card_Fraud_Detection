# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used contains transactions made by European cardholders in September 2013, and the challenge is to identify frauds from a highly imbalanced dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Modeling Process](#modeling-process)
- [Evaluation](#evaluation)
- [How to Run](#how-to-run)

## Project Overview
Credit card fraud detection is a critical problem in the financial sector. With increasing volumes of transactions, it is essential to have systems that can detect fraud with high accuracy. This project demonstrates a machine learning approach to predict fraudulent transactions using the **Credit Card Fraud Detection** dataset, which consists of anonymized features and labels indicating whether a transaction is fraudulent or not.

The key challenge of this task is handling the **imbalanced dataset** where fraudulent transactions make up a very small portion of the overall data.

## Dataset
The dataset used in this project is provided by Kaggle, which consists of 284,807 transactions, each represented by 31 columns:
- **Time**: Seconds elapsed between this transaction and the first transaction in the dataset.
- **V1 to V28**: Anonymized features resulting from a PCA transformation.
- **Amount**: Transaction amount.
- **Class**: Target label where 1 = fraud and 0 = non-fraud.

The dataset is highly imbalanced, with only ~0.172% fraudulent transactions.

## Modeling Process
The project uses the following steps for fraud detection:

1. **Data Preprocessing**:
   - Handle missing values (if any).
   - Scale numerical features like `Amount` using standard scaling techniques.
   - Handle the class imbalance using techniques like SMOTE (Synthetic Minority Over-sampling Technique) or undersampling of the majority class.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizations and statistical summaries to understand the distribution of data and the correlation between different features.
   - Analyze the distribution of fraud vs. non-fraud transactions.

3. **Modeling**:
   - We trained a **Logistic Regression** model as a baseline classifier for fraud detection. 
   - The model predicts the probability of a transaction being fraudulent.

4. **Threshold Tuning**:
   - Since fraud detection is highly imbalanced, tuning the decision threshold of the classifier (rather than using the default threshold of 0.5) helps in maximizing sensitivity (recall) while controlling false positives.
   - **ROC Curve** and **Confusion Matrix** were used to evaluate various thresholds and their impact on model performance.

5. **Model Evaluation**:
   - Performance metrics:
     - **Accuracy**: Proportion of correct predictions (not as useful due to imbalanced data).
     - **Precision**: How many predicted frauds were actual frauds.
     - **Recall (Sensitivity)**: How many actual frauds were detected.
     - **ROC-AUC**: Evaluates model performance across all classification thresholds.
     - **F1 Score**: Harmonic mean of precision and recall.
   - **Confusion Matrix**: Breakdown of true positives, true negatives, false positives, and false negatives.

## Evaluation
The model is evaluated based on:
- **ROC Curve**: Plots True Positive Rate (Sensitivity) against False Positive Rate (1 - Specificity).
- **Confusion Matrix**: Helps in understanding the trade-off between detecting fraud and avoiding false positives.
- **AUC (Area Under Curve)**: Measures the overall performance of the model across various thresholds.
  
## How to Run
### Prerequisites:
- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, statsmodels
