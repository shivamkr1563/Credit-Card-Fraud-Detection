# Dataset Information

## Kaggle Credit Card Fraud Detection Dataset

You need to download the dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Steps to Download:
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it in this folder and rename it to `raw_transactions.csv`

### Alternative (Manual Dataset Creation):
If you don't have access to Kaggle, you can:
1. Use any CSV file with transaction data
2. Ensure it has a 'Class' column (0 = legitimate, 1 = fraud)
3. Name it `raw_transactions.csv` and place it here

### Dataset Structure Expected:
- Time: Time elapsed between transactions
- V1-V28: Anonymized features (PCA transformed)
- Amount: Transaction amount
- Class: Target variable (0 = legitimate, 1 = fraud)
