"""
ModelAgent
Responsible for loading the trained ML model and making fraud predictions
"""

import numpy as np
import joblib
import os
from typing import Dict, Any, List, Union
import pandas as pd


class ModelAgent:
    """
    Agent that loads the trained fraud detection model and makes predictions.
    Returns fraud probabilities for transactions.
    """
    
    def __init__(self, model_path: str = 'models/fraud_model.pkl', verbose: bool = True):
        """
        Initialize the Model Agent.
        
        Args:
            model_path: Path to the saved trained model
            verbose: Whether to print model information
        """
        self.verbose = verbose
        self.model = None
        self.model_path = model_path
        self.predictions_made = 0
        self.model_info = {}
        
        # Load model
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load the trained fraud detection model.
        
        Args:
            model_path: Path to the saved model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        try:
            self.model = joblib.load(model_path)
            self.model_path = model_path
            
            # Extract model information
            self._extract_model_info()
            
            if self.verbose:
                print(f"[ModelAgent] Model loaded successfully from {model_path}")
                print(f"[ModelAgent] Model type: {self.model_info['model_type']}")
                print(f"[ModelAgent] Number of estimators: {self.model_info.get('n_estimators', 'N/A')}")
                
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def _extract_model_info(self):
        """Extract information about the loaded model."""
        self.model_info = {
            'model_type': type(self.model).__name__,
            'model_loaded': True
        }
        
        # Get RandomForest specific info
        if hasattr(self.model, 'n_estimators'):
            self.model_info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            self.model_info['max_depth'] = self.model.max_depth
        
        if hasattr(self.model, 'n_features_in_'):
            self.model_info['n_features'] = self.model.n_features_in_
    
    def predict_fraud_probability(self, features: np.ndarray) -> float:
        """
        Predict the probability that a transaction is fraudulent.
        
        Args:
            features: Numpy array of features (1D or 2D)
            
        Returns:
            Probability of fraud (0.0 to 1.0)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        try:
            # Get probability of fraud (class 1)
            probabilities = self.model.predict_proba(features)
            fraud_probability = probabilities[0][1]
            
            self.predictions_made += 1
            
            return float(fraud_probability)
            
        except Exception as e:
            raise Exception(f"Error making prediction: {e}")
    
    def predict_fraud_probability_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities for multiple transactions.
        
        Args:
            features: Numpy array of features (n_samples, n_features)
            
        Returns:
            Array of fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")
        
        try:
            # Get probabilities for all samples
            probabilities = self.model.predict_proba(features)
            fraud_probabilities = probabilities[:, 1]
            
            self.predictions_made += len(fraud_probabilities)
            
            return fraud_probabilities
            
        except Exception as e:
            raise Exception(f"Error making batch predictions: {e}")
    
    def predict(self, features: np.ndarray) -> tuple:
        """
        Predict fraud for a transaction and return both label and probability.
        
        This is the main prediction method that returns:
        - label: 0 (legitimate) or 1 (fraud)
        - probability: fraud probability (0.0 to 1.0)
        
        Args:
            features: Numpy array of features shaped (1, n_features)
            
        Returns:
            Tuple of (label, probability) where:
                - label is int (0 or 1)
                - probability is float (0.0 to 1.0) representing fraud probability
                
        Example:
            >>> agent = ModelAgent('models/fraud_model.pkl')
            >>> features = np.array([[0.5, 1.2, ...]])  # shape (1, 30)
            >>> label, probability = agent.predict(features)
            >>> print(f"Predicted: {label}, Fraud probability: {probability:.4f}")
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        try:
            # Get prediction label (0 or 1)
            label = int(self.model.predict(features)[0])
            
            # Get fraud probability (probability of class 1)
            probability = float(self.model.predict_proba(features)[0][1])
            
            self.predictions_made += 1
            
            return (label, probability)
            
        except Exception as e:
            raise Exception(f"Error making prediction: {e}")
    
    def predict_class(self, features: np.ndarray) -> int:
        """
        Predict the class (0=legitimate, 1=fraud) for a transaction.
        
        Args:
            features: Numpy array of features (1D or 2D)
            
        Returns:
            Predicted class (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        try:
            prediction = self.model.predict(features)
            return int(prediction[0])
            
        except Exception as e:
            raise Exception(f"Error making class prediction: {e}")
    
    def predict_with_details(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make a prediction with detailed information.
        
        Args:
            features: Numpy array of features (1D or 2D)
            
        Returns:
            Dictionary with prediction details
        """
        fraud_prob = self.predict_fraud_probability(features)
        predicted_class = self.predict_class(features)
        
        return {
            'fraud_probability': fraud_prob,
            'legitimate_probability': 1.0 - fraud_prob,
            'predicted_class': predicted_class,
            'predicted_label': 'FRAUD' if predicted_class == 1 else 'LEGITIMATE',
            'confidence': max(fraud_prob, 1.0 - fraud_prob)
        }
    
    def predict_batch_with_details(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple transactions with detailed information.
        
        Args:
            features: Numpy array of features (n_samples, n_features)
            
        Returns:
            List of dictionaries with prediction details
        """
        fraud_probs = self.predict_fraud_probability_batch(features)
        predictions = self.model.predict(features)
        
        results = []
        for i in range(len(fraud_probs)):
            fraud_prob = float(fraud_probs[i])
            predicted_class = int(predictions[i])
            
            results.append({
                'fraud_probability': fraud_prob,
                'legitimate_probability': 1.0 - fraud_prob,
                'predicted_class': predicted_class,
                'predicted_label': 'FRAUD' if predicted_class == 1 else 'LEGITIMATE',
                'confidence': max(fraud_prob, 1.0 - fraud_prob)
            })
        
        return results
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importance from the model (if available).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature names and importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        
        # Create feature names (V1-V28, Time, Amount)
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Sort by importance
        importance_dict = dict(zip(feature_names, importances))
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        return dict(sorted_features[:top_n])
    
    def validate_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Validate that features are in the correct format for the model.
        
        Args:
            features: Feature array to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': []
        }
        
        # Check if model is loaded
        if self.model is None:
            validation['is_valid'] = False
            validation['errors'].append("Model not loaded")
            return validation
        
        # Check feature shape
        expected_features = self.model_info.get('n_features')
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if expected_features and features.shape[1] != expected_features:
            validation['is_valid'] = False
            validation['errors'].append(
                f"Expected {expected_features} features, got {features.shape[1]}"
            )
        
        # Check for NaN or infinite values
        if np.isnan(features).any():
            validation['is_valid'] = False
            validation['errors'].append("Features contain NaN values")
        
        if np.isinf(features).any():
            validation['is_valid'] = False
            validation['errors'].append("Features contain infinite values")
        
        return validation
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's prediction activity.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'predictions_made': self.predictions_made,
            'model_loaded': self.model is not None,
            'model_path': self.model_path
        }
        stats.update(self.model_info)
        
        return stats
    
    def reset_statistics(self):
        """Reset prediction counter."""
        self.predictions_made = 0
        
        if self.verbose:
            print("[ModelAgent] Statistics reset")


if __name__ == "__main__":
    # Test the ModelAgent
    print("Testing ModelAgent...")
    
    # Initialize agent
    agent = ModelAgent(model_path='models/fraud_model.pkl', verbose=True)
    
    # Create sample features (30 features: Time, V1-V28, Amount)
    np.random.seed(42)
    sample_features = np.random.randn(30).reshape(1, -1)
    
    print("\n1. Testing main predict() method...")
    label, probability = agent.predict(sample_features)
    print(f"   Returned: (label={label}, probability={probability:.4f})")
    print(f"   Label type: {type(label)}")
    print(f"   Probability type: {type(probability)}")
    print(f"   Prediction: {'FRAUD' if label == 1 else 'LEGITIMATE'}")
    
    print("\n2. Making single probability prediction...")
    fraud_prob = agent.predict_fraud_probability(sample_features)
    print(f"   Fraud probability: {fraud_prob:.4f}")
    
    print("\n3. Making detailed prediction...")
    details = agent.predict_with_details(sample_features)
    print(f"   Predicted: {details['predicted_label']}")
    print(f"   Confidence: {details['confidence']:.4f}")
    
    print("\n4. Validating features...")
    validation = agent.validate_features(sample_features)
    print(f"   Valid: {validation['is_valid']}")
    
    print("\n5. Getting feature importance...")
    importance = agent.get_feature_importance(top_n=5)
    print("   Top 5 features:")
    for feature, score in importance.items():
        print(f"     {feature}: {score:.4f}")
    
    print("\n6. Agent statistics...")
    stats = agent.get_statistics()
    print(f"   Predictions made: {stats['predictions_made']}")
    print(f"   Model type: {stats['model_type']}")
    
    print("\n7. Testing multiple transactions...")
    for i in range(3):
        test_features = np.random.randn(30).reshape(1, -1)
        label, prob = agent.predict(test_features)
        print(f"   Transaction {i+1}: Label={label}, Probability={prob:.4f}")
    
    print("\n✓ ModelAgent test complete!")
