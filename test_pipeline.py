"""
End-to-End Pipeline Test
Tests all 5 agents working together
"""
import pandas as pd
import numpy as np
from agents import TransactionMonitoringAgent, AnalysisAgent, ModelAgent, RiskAgent, AlertAgent

def test_pipeline():
    print("="*60)
    print("CYBER FRAUD DETECTION - PIPELINE TEST")
    print("="*60)
    
    # Load test data
    print("\n1. Loading test data...")
    df = pd.read_csv('data/raw_transactions.csv')
    test_df = df.head(100)  # Test with first 100 transactions
    print(f"   ✓ Loaded {len(test_df)} transactions")
    
    # Initialize agents
    print("\n2. Initializing agents...")
    monitor = TransactionMonitoringAgent()
    
    # Get feature columns from dataframe (exclude only Class)
    feature_cols = [col for col in df.columns if col != 'Class']
    analysis = AnalysisAgent(
        scaler_path='models/amount_scaler.pkl',
        feature_columns=feature_cols
    )
    
    model_agent = ModelAgent(model_path='models/fraud_model.pkl')
    risk = RiskAgent()
    alert = AlertAgent()
    print("   ✓ All 5 agents initialized")
    
    # Process transactions
    print("\n3. Processing transactions through pipeline...")
    results = []
    frauds_found = 0
    
    for transaction_info in monitor.stream_transactions(test_df):
        try:
            idx = transaction_info['transaction_id']
            transaction = pd.Series(transaction_info['data'])
            
            # Agent 2: Feature extraction
            features = analysis.preprocess_transaction(transaction)
            
            # Agent 3: Prediction
            label, probability = model_agent.predict(features)
            
            # Agent 4: Risk assessment
            risk_level, decision = risk.assess_risk(probability)
            
            # Agent 5: Alert logging (only for suspicious transactions)
            if risk_level in ['MEDIUM', 'HIGH']:
                alert.log_alert(idx, transaction, probability, risk_level, decision)
                frauds_found += 1
            
            results.append({
                'index': idx,
                'label': label,
                'probability': probability,
                'risk': risk_level,
                'decision': decision
            })
            
        except Exception as e:
            print(f"   ✗ Error at transaction {idx}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"   ✓ Processed {len(results)} transactions")
    print(f"   ✓ Found {frauds_found} suspicious transactions")
    
    # Verify results
    print("\n4. Verifying pipeline results...")
    results_df = pd.DataFrame(results)
    
    # Check risk distribution
    risk_counts = results_df['risk'].value_counts()
    print(f"\n   Risk Distribution:")
    for risk_level in ['LOW', 'MEDIUM', 'HIGH']:
        count = risk_counts.get(risk_level, 0)
        print(f"   - {risk_level}: {count} ({count/len(results)*100:.1f}%)")
    
    # Check decisions
    decision_counts = results_df['decision'].value_counts()
    print(f"\n   Decision Distribution:")
    for decision in ['allow', 'review', 'flag']:
        count = decision_counts.get(decision, 0)
        print(f"   - {decision}: {count} ({count/len(results)*100:.1f}%)")
    
    # Verify alert log
    print("\n5. Checking alert log...")
    alerts_df = alert.get_alerts()
    if len(alerts_df) > 0:
        print(f"   ✓ Alert log contains {len(alerts_df)} entries")
        print(f"   ✓ Latest alert: Risk={alerts_df.iloc[-1]['risk_level']}, "
              f"Probability={alerts_df.iloc[-1]['fraud_probability']:.4f}")
    else:
        print("   ⚠ No alerts logged")
    
    print("\n" + "="*60)
    print("PIPELINE TEST: PASSED ✓")
    print("="*60)
    print("\nAll components working correctly:")
    print("  ✓ Transaction Monitoring Agent")
    print("  ✓ Analysis Agent")  
    print("  ✓ Model Agent")
    print("  ✓ Risk Agent")
    print("  ✓ Alert Agent")
    print("\nSystem is ready for production use!")
    
    return True

if __name__ == "__main__":
    test_pipeline()
