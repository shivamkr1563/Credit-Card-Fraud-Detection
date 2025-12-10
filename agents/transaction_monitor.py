"""
TransactionMonitoringAgent
Responsible for reading and streaming transactions from CSV or DataFrame
"""

import pandas as pd
from typing import Generator, Union, Dict, Any
import os


class TransactionMonitoringAgent:
    """
    Agent that monitors and streams transactions for processing.
    Can read from CSV files or accept DataFrames directly.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the Transaction Monitoring Agent.
        
        Args:
            verbose: Whether to print monitoring information
        """
        self.verbose = verbose
        self.transaction_count = 0
        self.data_source = None
        
    def load_transactions_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load transactions from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame containing the transactions
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Transaction file not found: {filepath}")
        
        if self.verbose:
            print(f"[TransactionMonitor] Loading transactions from {filepath}...")
        
        df = pd.read_csv(filepath)
        self.data_source = filepath
        
        if self.verbose:
            print(f"[TransactionMonitor] Loaded {len(df)} transactions")
        
        return df
    
    def load_transactions_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load transactions from a DataFrame.
        
        Args:
            df: DataFrame containing the transactions
            
        Returns:
            The input DataFrame
        """
        if self.verbose:
            print(f"[TransactionMonitor] Received DataFrame with {len(df)} transactions")
        
        self.data_source = "DataFrame"
        return df
    
    def stream_transactions(
        self, 
        data_source: Union[str, pd.DataFrame],
        batch_size: int = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream transactions one by one or in batches.
        
        Args:
            data_source: Either a filepath (str) or a DataFrame
            batch_size: If specified, yield batches of transactions instead of individual ones
            
        Yields:
            Dictionary containing transaction data and metadata
        """
        # Load data based on source type
        if isinstance(data_source, str):
            df = self.load_transactions_from_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            df = self.load_transactions_from_dataframe(data_source)
        else:
            raise ValueError("data_source must be a filepath (str) or DataFrame")
        
        # Reset transaction count
        self.transaction_count = 0
        
        if self.verbose:
            print(f"[TransactionMonitor] Starting transaction stream...")
        
        # Stream transactions
        if batch_size:
            # Batch streaming
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                self.transaction_count += len(batch)
                
                yield {
                    'batch_id': i // batch_size + 1,
                    'batch_size': len(batch),
                    'transactions': batch,
                    'total_processed': self.transaction_count
                }
        else:
            # Individual transaction streaming
            for idx, row in df.iterrows():
                self.transaction_count += 1
                
                # Convert row to dictionary
                transaction_data = row.to_dict()
                
                yield {
                    'transaction_id': idx,
                    'transaction_number': self.transaction_count,
                    'data': transaction_data,
                    'has_label': 'Class' in transaction_data
                }
        
        if self.verbose:
            print(f"[TransactionMonitor] Completed streaming {self.transaction_count} transactions")
    
    def get_transaction_by_id(self, df: pd.DataFrame, transaction_id: int) -> Dict[str, Any]:
        """
        Get a specific transaction by its ID.
        
        Args:
            df: DataFrame containing transactions
            transaction_id: The index/ID of the transaction
            
        Returns:
            Dictionary containing the transaction data
        """
        if transaction_id not in df.index:
            raise ValueError(f"Transaction ID {transaction_id} not found")
        
        row = df.loc[transaction_id]
        return {
            'transaction_id': transaction_id,
            'data': row.to_dict()
        }
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the transactions being monitored.
        
        Args:
            df: DataFrame containing transactions
            
        Returns:
            Dictionary with transaction statistics
        """
        stats = {
            'total_transactions': len(df),
            'features': list(df.columns),
            'num_features': len(df.columns)
        }
        
        # Add class distribution if available
        if 'Class' in df.columns:
            class_dist = df['Class'].value_counts().to_dict()
            stats['class_distribution'] = class_dist
            stats['fraud_rate'] = (class_dist.get(1, 0) / len(df)) * 100
        
        # Add amount statistics if available
        if 'Amount' in df.columns:
            stats['amount_stats'] = {
                'min': float(df['Amount'].min()),
                'max': float(df['Amount'].max()),
                'mean': float(df['Amount'].mean()),
                'median': float(df['Amount'].median())
            }
        
        return stats
    
    def reset(self):
        """Reset the agent's state."""
        self.transaction_count = 0
        self.data_source = None
        
        if self.verbose:
            print("[TransactionMonitor] Agent reset")


if __name__ == "__main__":
    # Test the TransactionMonitoringAgent
    print("Testing TransactionMonitoringAgent...")
    
    # Initialize agent
    agent = TransactionMonitoringAgent(verbose=True)
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'Time': [0, 1, 2],
        'V1': [1.0, 2.0, 3.0],
        'V2': [0.5, 1.5, 2.5],
        'Amount': [100.0, 200.0, 300.0],
        'Class': [0, 1, 0]
    })
    
    # Test statistics
    stats = agent.get_statistics(sample_data)
    print("\nStatistics:")
    print(stats)
    
    # Test streaming
    print("\nStreaming transactions:")
    for transaction in agent.stream_transactions(sample_data):
        print(f"  Transaction #{transaction['transaction_number']}: Amount = ${transaction['data']['Amount']}")
    
    print("\n✓ TransactionMonitoringAgent test complete!")
