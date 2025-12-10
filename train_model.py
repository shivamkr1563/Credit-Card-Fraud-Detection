"""
train_model.py
Trains a fraud detection model using RandomForest on the Credit Card Fraud dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    auc
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime


def load_data(filepath='C:\\Users\\shiva\\Desktop\\Cyber_Fraud_Detection\\data\\raw_transactions.csv'):
    """Load the credit card fraud dataset."""
    print(f"Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}\n"
            "Please download the Kaggle Credit Card Fraud Detection dataset\n"
            "and place it in the data/ folder as 'raw_transactions.csv'"
        )
    
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_data(df):
    """Perform basic exploratory data analysis."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nClass Distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print(f"\nFraud Rate: {(class_counts[1] / len(df)) * 100:.2f}%")
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum().sum())
    
    return df


def prepare_data(df):
    """Prepare features and target for training."""
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Standardize the 'Amount' feature (Time and V1-V28 are already scaled)
    if 'Amount' in X.columns:
        scaler = StandardScaler()
        X['Amount'] = scaler.fit_transform(X[['Amount']])
        
        # Save the scaler for later use
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/amount_scaler.pkl')
        print("Amount scaler saved to models/amount_scaler.pkl")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training fraud cases: {y_train.sum()}")
    print(f"Test fraud cases: {y_test.sum()}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    print("\nTraining RandomForest Classifier...")
    print("This may take a few minutes...")
    
    # Initialize the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        class_weight='balanced',  # Handle class imbalance
        verbose=1
    )
    
    # Train the model
    start_time = datetime.now()
    model.fit(X_train, y_train)
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives: {cm[1, 1]}")
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    
    # Feature Importance (Top 10)
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    print(feature_importance.to_string(index=False))
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm
    }


def save_model(model, filepath='models/fraud_model.pkl'):
    """Save the trained model."""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\nModel saved successfully to {filepath}")
    
    # Get model size
    model_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    print(f"Model size: {model_size:.2f} MB")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load data
        df = load_data()
        
        # Explore data
        df = explore_data(df)
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model
        save_model(model)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nModel Performance Summary:")
        print(f"  - ROC-AUC Score: {metrics['roc_auc']:.4f}")
        print(f"  - PR-AUC Score: {metrics['pr_auc']:.4f}")
        print(f"\nModel saved to: models/fraud_model.pkl")
        print(f"Scaler saved to: models/amount_scaler.pkl")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nERROR during training: {e}")
        raise


if __name__ == "__main__":
    main()
