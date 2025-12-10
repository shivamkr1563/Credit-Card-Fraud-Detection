"""
AlertAgent
Responsible for logging suspicious transactions and managing alerts
"""

import pandas as pd
import os
from typing import Dict, Any
from datetime import datetime


class AlertAgent:
    """
    Agent that manages alerts for suspicious transactions.
    Logs suspicious activities to CSV and provides alert retrieval.
    """
    
    def __init__(
        self,
        alert_log_path: str = 'alerts_log.csv',
        verbose: bool = True
    ):
        """
        Initialize the Alert Agent.
        
        Args:
            alert_log_path: Path to the CSV file for logging alerts (default: 'alerts_log.csv')
            verbose: Whether to print alert information
        """
        self.verbose = verbose
        self.alert_log_path = alert_log_path
        self.alerts_logged = 0
        self.current_session_alerts = []
        
        # Ensure directory exists if path contains directories
        alert_dir = os.path.dirname(alert_log_path)
        if alert_dir:
            os.makedirs(alert_dir, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not os.path.exists(alert_log_path):
            self._initialize_log_file()
        
        if self.verbose:
            print(f"[AlertAgent] Initialized with log file: {alert_log_path}")
    
    def _initialize_log_file(self):
        """Create a new alert log file with headers."""
        headers = [
            'timestamp',
            'transaction_id',
            'fraud_probability',
            'risk_level',
            'decision',
            'amount',
            'time',
            'additional_info'
        ]
        
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.alert_log_path, index=False)
        
        if self.verbose:
            print(f"[AlertAgent] Created new alert log file: {self.alert_log_path}")
    
    def log_alert(
        self,
        index: int,
        transaction_row: pd.Series,
        fraud_proba: float,
        risk_level: str,
        decision: str
    ):
        """
        Log a suspicious transaction alert to CSV with timestamp.
        
        This method appends a new row to the alerts log file with:
        - Current timestamp
        - Transaction index
        - Fraud probability
        - Risk level and decision
        - Transaction details (Amount, Time)
        
        Args:
            index: Transaction index/ID
            transaction_row: Pandas Series containing the transaction data
            fraud_proba: Fraud probability (0.0 to 1.0)
            risk_level: Risk level string ("LOW", "MEDIUM", "HIGH")
            decision: Decision string ("allow", "review", "flag")
            
        Example:
            >>> agent = AlertAgent(alert_log_path="alerts_log.csv")
            >>> row = df.iloc[5]  # Get a transaction row
            >>> agent.log_alert(5, row, 0.75, "HIGH", "flag")
        """
        # Create alert record
        alert = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'transaction_id': index,
            'fraud_probability': fraud_proba,
            'risk_level': risk_level,
            'decision': decision,
            'amount': transaction_row.get('Amount', 'N/A'),
            'time': transaction_row.get('Time', 'N/A'),
            'additional_info': ''
        }
        
        # Add additional context if available
        if 'Amount' in transaction_row.index and transaction_row['Amount'] > 1000:
            alert['additional_info'] = f"High-value transaction: ${transaction_row['Amount']:.2f}"
        
        # Append to CSV log
        try:
            df = pd.DataFrame([alert])
            df.to_csv(self.alert_log_path, mode='a', header=False, index=False)
            
            # Add to current session
            self.current_session_alerts.append(alert)
            self.alerts_logged += 1
            
            if self.verbose:
                print(f"[AlertAgent] Alert logged for transaction {index} - Risk: {risk_level}, Decision: {decision}")
            
        except Exception as e:
            if self.verbose:
                print(f"[AlertAgent] Error logging alert: {e}")
            raise
    
    def get_alerts(self) -> pd.DataFrame:
        """
        Retrieve all logged alerts as a pandas DataFrame.
        
        Returns:
            DataFrame containing all alerts from the log file.
            Returns empty DataFrame if no alerts exist.
            
        Example:
            >>> agent = AlertAgent()
            >>> alerts_df = agent.get_alerts()
            >>> print(f"Total alerts: {len(alerts_df)}")
            >>> print(alerts_df[['timestamp', 'risk_level', 'decision']])
        """
        try:
            if not os.path.exists(self.alert_log_path):
                return pd.DataFrame()
            
            df = pd.read_csv(self.alert_log_path)
            
            if self.verbose:
                print(f"[AlertAgent] Retrieved {len(df)} alerts from log")
            
            return df
            
        except Exception as e:
            if self.verbose:
                print(f"[AlertAgent] Error reading alerts: {e}")
            return pd.DataFrame()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged alerts."""
        alerts_df = self.get_alerts()
        
        stats = {
            'alerts_logged': self.alerts_logged,
            'total_alerts_in_file': len(alerts_df),
            'session_alerts': len(self.current_session_alerts)
        }
        
        if not alerts_df.empty:
            stats['risk_distribution'] = alerts_df['risk_level'].value_counts().to_dict()
            stats['decision_distribution'] = alerts_df['decision'].value_counts().to_dict()
        
        return stats


if __name__ == "__main__":
    # Test the AlertAgent
    print("Testing AlertAgent...")
    
    # Initialize agent with default log path
    agent = AlertAgent(alert_log_path='alerts_log.csv', verbose=True)
    
    print("\n1. Testing log_alert() method...")
    # Create sample transaction rows
    sample_data = {
        'Time': 1000.0,
        'V1': -1.5,
        'V2': 2.3,
        'Amount': 2500.0,
        'Class': 1
    }
    transaction_row = pd.Series(sample_data)
    
    # Log alerts
    agent.log_alert(
        index=12345,
        transaction_row=transaction_row,
        fraud_proba=0.85,
        risk_level="HIGH",
        decision="flag"
    )
    
    agent.log_alert(
        index=12346,
        transaction_row=transaction_row,
        fraud_proba=0.95,
        risk_level="HIGH",
        decision="flag"
    )
    
    print("\n2. Testing get_alerts() method...")
    alerts_df = agent.get_alerts()
    print(f"   Total alerts retrieved: {len(alerts_df)}")
    if not alerts_df.empty:
        print(f"   Columns: {list(alerts_df.columns)}")
        print(f"\n   Recent alerts:")
        print(alerts_df.tail())
    
    print("\n3. Testing with different risk levels...")
    test_cases = [
        (100, 0.25, "LOW", "allow"),
        (101, 0.45, "MEDIUM", "review"),
        (102, 0.75, "HIGH", "flag"),
    ]
    
    for idx, prob, risk, dec in test_cases:
        agent.log_alert(idx, transaction_row, prob, risk, dec)
        print(f"   Logged: Transaction {idx} - {risk}/{dec}")
    
    print("\n4. Final statistics...")
    stats = agent.get_statistics()
    print(f"   Total alerts in log: {stats['total_alerts_in_file']}")
    print(f"   Alerts this session: {stats['alerts_logged']}")
    if 'risk_distribution' in stats:
        print(f"   Risk distribution: {stats['risk_distribution']}")
    if 'decision_distribution' in stats:
        print(f"   Decision distribution: {stats['decision_distribution']}")
    
    print("\n✓ AlertAgent test complete!")
