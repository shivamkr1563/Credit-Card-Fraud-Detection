"""
AnalysisAgent
Responsible for preprocessing transactions and extracting features
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import joblib
import os


class AnalysisAgent:
    """
    Agent that analyzes and preprocesses transactions.
    Extracts features and prepares data for model prediction.
    """
    
    def __init__(self, feature_columns: List[str], scaler_path: str = 'models/amount_scaler.pkl', verbose: bool = True):
        """
        Initialize the Analysis Agent.
        
        Args:
            feature_columns: List of feature column names to extract from transactions
            scaler_path: Path to the saved StandardScaler for Amount feature (optional)
            verbose: Whether to print analysis information
        """
        self.feature_columns = feature_columns
        self.verbose = verbose
        self.scaler = None
        self.scaler_path = scaler_path
        self.expected_features = None
        self.transactions_analyzed = 0
        
        # Load scaler if available
        if os.path.exists(scaler_path):
            self.load_scaler(scaler_path)
        elif self.verbose:
            print(f"[AnalysisAgent] Warning: Scaler not found at {scaler_path}")
    
    def load_scaler(self, scaler_path: str):
        """
        Load the StandardScaler for the Amount feature.
        
        Args:
            scaler_path: Path to the saved scaler
        """
        try:
            self.scaler = joblib.load(scaler_path)
            if self.verbose:
                print(f"[AnalysisAgent] Scaler loaded from {scaler_path}")
        except Exception as e:
            if self.verbose:
                print(f"[AnalysisAgent] Error loading scaler: {e}")
            self.scaler = None
    
    def preprocess_transaction(self, transaction_row: pd.Series) -> np.ndarray:
        """
        Preprocess a single transaction row and extract features as numpy array.
        
        This method:
        1. Selects only the specified feature columns from the transaction row
        2. Applies scaling to the Amount feature if scaler is available
        3. Converts to a 2D numpy array shaped (1, n_features) ready for model prediction
        
        Args:
            transaction_row: A pandas Series representing a single transaction
            
        Returns:
            Numpy array shaped (1, n_features) containing the preprocessed features
            
        Example:
            >>> agent = AnalysisAgent(feature_columns=['Time', 'V1', 'V2', 'Amount'])
            >>> row = df.iloc[0]  # Get first row from DataFrame
            >>> features = agent.preprocess_transaction(row)
            >>> features.shape  # (1, 4)
        """
        # Create a copy to avoid modifying original
        processed_row = transaction_row.copy()
        
        # Scale the Amount feature if scaler is available
        if 'Amount' in processed_row.index and self.scaler is not None:
            try:
                # Scaler expects 2D array
                amount_scaled = self.scaler.transform([[processed_row['Amount']]])[0][0]
                processed_row['Amount'] = amount_scaled
            except Exception as e:
                if self.verbose:
                    print(f"[AnalysisAgent] Warning: Could not scale amount: {e}")
        
        # Select only the feature columns in the correct order
        features = processed_row[self.feature_columns].values
        
        # Reshape to 2D array (1, n_features) for model input
        features_2d = features.reshape(1, -1)
        
        self.transactions_analyzed += 1
        
        return features_2d
    
    def preprocess_transaction_dict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        # Convert to dict if it's a Series
        if isinstance(transaction, pd.Series):
            transaction = transaction.to_dict()
        
        # Preprocess the transaction
        processed = self.preprocess_transaction_dict(transaction)
            Dictionary with preprocessed transaction data
        """
        # Create a copy to avoid modifying original
        processed = transaction.copy()
        
        # Scale the Amount feature if scaler is available
        if 'Amount' in processed and self.scaler is not None:
            try:
                # Scaler expects 2D array
                amount_scaled = self.scaler.transform([[processed['Amount']]])[0][0]
                processed['Amount'] = amount_scaled
                processed['amount_scaled'] = True
            except Exception as e:
                if self.verbose:
                    print(f"[AnalysisAgent] Warning: Could not scale amount: {e}")
                processed['amount_scaled'] = False
        else:
            processed['amount_scaled'] = False
        
        self.transactions_analyzed += 1
        
        return processed
    
    def extract_features(
        self, 
        transaction: Union[Dict[str, Any], pd.Series],
        remove_label: bool = True
    ) -> np.ndarray:
        """
        Extract features from a transaction for model prediction.
        
        Args:
            transaction: Transaction data (dict or pandas Series)
            remove_label: Whether to remove the 'Class' label if present
            
        Returns:
            Numpy array of features ready for model input
        """
        # Convert to dict if it's a Series
        if isinstance(transaction, pd.Series):
            transaction = transaction.to_dict()
        
        # Preprocess the transaction
        processed = self.preprocess_transaction(transaction)
        
        # Remove the label if present and requested
        if remove_label and 'Class' in processed:
            processed.pop('Class')
        
        # Remove metadata fields that aren't features
        metadata_fields = ['amount_scaled', 'transaction_id', 'transaction_number']
        for field in metadata_fields:
            processed.pop(field, None)
        
        # Convert to numpy array maintaining feature order
        if self.expected_features is None:
            # First time - establish feature order
            self.expected_features = sorted(processed.keys())
        
        # Extract features in expected order
        features = [processed[key] for key in self.expected_features]
        
        return np.array(features).reshape(1, -1)
    
    def extract_features_batch(
        self, 
        transactions: pd.DataFrame,
        remove_label: bool = True
    ) -> np.ndarray:
        """
        Extract features from multiple transactions.
        
        Args:
            transactions: DataFrame containing multiple transactions
            remove_label: Whether to remove the 'Class' label if present
            
        Returns:
            Numpy array of features (n_samples, n_features)
        """
        # Create a copy
        df = transactions.copy()
        
        # Scale Amount if scaler is available
        if 'Amount' in df.columns and self.scaler is not None:
            try:
                df['Amount'] = self.scaler.transform(df[['Amount']])
            except Exception as e:
                if self.verbose:
                    print(f"[AnalysisAgent] Warning: Could not scale amounts: {e}")
        
        # Remove label if requested
        if remove_label and 'Class' in df.columns:
            df = df.drop('Class', axis=1)
        
        self.transactions_analyzed += len(df)
        
        return df.values
    
    def analyze_transaction_patterns(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns and characteristics of a transaction.
        
        Args:
            transaction: Dictionary containing transaction data
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'has_amount': 'Amount' in transaction,
            'has_time': 'Time' in transaction,
            'num_features': len(transaction),
            'features_present': list(transaction.keys())
        }
        
        # Analyze amount if present
        if 'Amount' in transaction:
            amount = transaction['Amount']
            analysis['amount'] = float(amount)
            analysis['amount_category'] = self._categorize_amount(amount)
        
        # Analyze time if present
        if 'Time' in transaction:
            time = transaction['Time']
            analysis['time'] = float(time)
            analysis['time_category'] = self._categorize_time(time)
        
        # Count V features (PCA components)
        v_features = [k for k in transaction.keys() if k.startswith('V')]
        analysis['num_v_features'] = len(v_features)
        
        # Check for extreme values in V features
        if v_features:
            v_values = [abs(transaction[k]) for k in v_features]
            analysis['max_v_value'] = max(v_values)
            analysis['has_extreme_v'] = any(v > 10 for v in v_values)
        
        return analysis
    
    def _categorize_amount(self, amount: float) -> str:
        """Categorize transaction amount."""
        if amount < 10:
            return "very_low"
        elif amount < 50:
            return "low"
        elif amount < 200:
            return "medium"
        elif amount < 1000:
            return "high"
        else:
            return "very_high"
    
    def _categorize_time(self, time: float) -> str:
        """Categorize transaction time (assuming seconds from start)."""
        # Convert to hours
        hours = time / 3600
        
        if hours < 6:
            return "night"
        elif hours < 12:
            return "morning"
        elif hours < 18:
            return "afternoon"
        else:
            return "evening"
    
    def validate_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that a transaction has all required features.
        
        Args:
            transaction: Dictionary containing transaction data
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for Amount
        if 'Amount' not in transaction:
            validation['errors'].append("Missing 'Amount' feature")
            validation['is_valid'] = False
        elif not isinstance(transaction['Amount'], (int, float)):
            validation['errors'].append("'Amount' must be numeric")
            validation['is_valid'] = False
        
        # Check for Time
        if 'Time' not in transaction:
            validation['warnings'].append("Missing 'Time' feature")
        
        # Check for V features
        v_features = [k for k in transaction.keys() if k.startswith('V')]
        if len(v_features) < 28:
            validation['warnings'].append(f"Expected 28 V features, found {len(v_features)}")
        
        # Check for NaN values
        for key, value in transaction.items():
            if pd.isna(value):
                validation['errors'].append(f"NaN value in feature '{key}'")
                validation['is_valid'] = False
        
        return validation
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's analysis activity.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'transactions_analyzed': self.transactions_analyzed,
            'scaler_loaded': self.scaler is not None,
            'expected_features': self.expected_features,
            'num_features': len(self.expected_features) if self.expected_features else 0
        }
    
    def reset(self):
        """Reset the agent's state."""
        self.transactions_analyzed = 0
        self.expected_features = None
        
        if self.verbose:
            print("[AnalysisAgent] Agent reset")


if __name__ == "__main__":
    # Test the AnalysisAgent
    print("Testing AnalysisAgent...")
    
    # Define feature columns (excluding 'Class' label)
    feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'Amount']
    
    # Initialize agent with feature columns
    agent = AnalysisAgent(feature_columns=feature_cols, scaler_path='models/amount_scaler.pkl', verbose=True)
    
    # Create sample transaction as pandas Series
    sample_data = {
        'Time': 1000.0,
        'V1': -1.5,
        'V2': 2.3,
        'V3': 0.5,
        'V4': 1.2,
        'V5': -0.8,
        'Amount': 150.0,
        'Class': 0  # Will be ignored since not in feature_columns
    }
    sample_row = pd.Series(sample_data)
    
    print("\n1. Testing preprocess_transaction (main method)...")
    features = agent.preprocess_transaction(sample_row)
    print(f"   Input: pandas Series with {len(sample_row)} fields")
    print(f"   Output shape: {features.shape}")
    print(f"   Output type: {type(features)}")
    print(f"   Features extracted: {agent.feature_columns}")
    print(f"   Feature values: {features}")
    
    print("\n2. Testing with DataFrame row...")
    df = pd.DataFrame([sample_data])
    row = df.iloc[0]
    features_from_df = agent.preprocess_transaction(row)
    print(f"   Shape: {features_from_df.shape}")
    print(f"   Ready for model.predict(): {features_from_df.ndim == 2}")
    
    print("\n3. Validating transaction (bonus feature)...")
    validation = agent.validate_transaction(sample_data)
    print(f"   Valid: {validation['is_valid']}")
    print(f"   Warnings: {validation['warnings']}")
    
    print("\n4. Agent statistics...")
    stats = agent.get_statistics()
    print(f"   Transactions analyzed: {stats['transactions_analyzed']}")
    print(f"   Scaler loaded: {stats['scaler_loaded']}")
    print(f"   Feature columns: {len(agent.feature_columns)}")
    
    print("\n✓ AnalysisAgent test complete!")
