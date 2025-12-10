"""
Cyber Fraud Detection - Multi-Agent System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from agents.transaction_monitor import TransactionMonitoringAgent
from agents.analysis_agent import AnalysisAgent
from agents.model_agent import ModelAgent
from agents.risk_agent import RiskAgent
from agents.alert_agent import AlertAgent


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Cyber Fraud Detection – Multi-Agent System",
        page_icon="🔒",
        layout="wide"
    )
    
    # Title and description
    st.title("🔒 Cyber Fraud Detection – Multi-Agent System")
    st.markdown("""
    ### Welcome to the AI-Powered Fraud Detection System
    
    This application leverages a sophisticated **multi-agent architecture** to identify fraudulent transactions 
    in real-time. Each transaction passes through a pipeline of specialized AI agents:
    
    - 📊 **Transaction Monitor** streams and validates transactions
    - 🔍 **Analysis Agent** preprocesses and extracts features
    - 🤖 **Model Agent** uses Machine Learning to predict fraud probability
    - ⚠️ **Risk Agent** assesses risk levels and recommends actions
    - 🚨 **Alert Agent** logs suspicious activities for investigation
    
    **Get started by uploading your transaction data below.**
    """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    st.sidebar.markdown("""
    ### Agent Pipeline:
    1. 📊 **Transaction Monitoring Agent** - Streams transactions
    2. 🔍 **Analysis Agent** - Preprocesses features
    3. 🤖 **Model Agent** - Predicts fraud probability
    4. ⚠️ **Risk Agent** - Assesses risk level
    5. 🚨 **Alert Agent** - Logs suspicious activity
    """)
    
    # File uploader
    st.header("1. Upload Transaction Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with the same structure as the training data"
    )
    
    if uploaded_file is not None:
        try:
            # Load the uploaded data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} transactions loaded.")
            
            # Show dataset info
            with st.expander("📋 View Dataset Information"):
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.write(f"**Columns:** {list(df.columns)}")
                st.dataframe(df.head(10))
            
            # Determine feature columns
            # All numeric columns except 'Class' (if present)
            if 'Class' in df.columns:
                feature_columns = [col for col in df.columns if col != 'Class']
                st.info(f"ℹ️ Detected {len(feature_columns)} feature columns (excluding 'Class')")
            else:
                feature_columns = list(df.columns)
                st.info(f"ℹ️ Detected {len(feature_columns)} feature columns (no 'Class' column found)")
            
            # Initialize agents
            st.header("2. Initialize Multi-Agent System")
            
            with st.spinner("Initializing agents..."):
                try:
                    transaction_agent = TransactionMonitoringAgent(verbose=False)
                    analysis_agent = AnalysisAgent(
                        scaler_path='models/amount_scaler.pkl',
                        feature_columns=feature_columns,
                        verbose=False
                    )
                    model_agent = ModelAgent(
                        model_path='models/fraud_model.pkl',
                        verbose=False
                    )
                    risk_agent = RiskAgent(verbose=False)
                    alert_agent = AlertAgent(
                        alert_log_path='alerts_log.csv',
                        verbose=False
                    )
                    
                    st.success("✅ All agents initialized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error initializing agents: {e}")
                    st.stop()
            
            # Process transactions
            st.header("3. Process Transactions Through Agent Pipeline")
            
            if st.button("🚀 Start Analysis", type="primary"):
                
                # Results storage
                results = []
                suspicious_count = 0
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each transaction
                for idx, row in df.iterrows():
                    # Update progress
                    progress = (idx + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing transaction {idx + 1}/{len(df)}...")
                    
                    try:
                        # Step 1: Get transaction row as Series
                        transaction_row = row
                        
                        # Step 2: Preprocess with AnalysisAgent
                        features = analysis_agent.preprocess_transaction(transaction_row)
                        
                        # Step 3: Predict with ModelAgent
                        prediction, fraud_proba = model_agent.predict(features)
                        
                        # Step 4: Assess risk with RiskAgent
                        risk_level, decision = risk_agent.assess_risk(fraud_proba)
                        
                        # Step 5: Log suspicious transactions with AlertAgent
                        if decision in ['review', 'flag']:
                            alert_agent.log_alert(
                                index=idx,
                                transaction_row=transaction_row,
                                fraud_proba=fraud_proba,
                                risk_level=risk_level,
                                decision=decision
                            )
                            suspicious_count += 1
                        
                        # Store results
                        result = {
                            'Transaction_ID': idx,
                            'Amount': transaction_row.get('Amount', 'N/A'),
                            'Time': transaction_row.get('Time', 'N/A'),
                            'Prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
                            'Fraud_Probability': round(fraud_proba, 4),
                            'Risk_Level': risk_level,
                            'Decision': decision.upper(),
                            'Actual_Class': transaction_row.get('Class', 'N/A') if 'Class' in df.columns else 'N/A'
                        }
                        results.append(result)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error processing transaction {idx}: {e}")
                        continue
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Display summary
                st.success("✅ Analysis Complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(results_df))
                with col2:
                    fraud_count = len(results_df[results_df['Prediction'] == 'FRAUD'])
                    st.metric("Predicted Frauds", fraud_count)
                with col3:
                    st.metric("Suspicious Transactions", suspicious_count)
                with col4:
                    fraud_rate = (fraud_count / len(results_df) * 100) if len(results_df) > 0 else 0
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                
                # Display all results
                st.header("4. Analysis Results")
                st.markdown("---")
                
                st.subheader("📊 All Transactions")
                st.markdown("""
                Complete analysis of all transactions including predictions, fraud probabilities, 
                risk assessments, and recommended actions.
                """)
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    height=400
                )
                
                # Download button for all results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Results",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )
                
                # Display suspicious transactions
                st.markdown("---")
                st.subheader("🚨 Suspicious Transactions (Review/Flag)")
                st.markdown("""
                Transactions flagged as suspicious by the Risk Agent. These require manual review 
                or immediate action based on their risk level.
                """)
                suspicious_df = results_df[results_df['Decision'].isin(['REVIEW', 'FLAG'])]
                
                if len(suspicious_df) > 0:
                    st.dataframe(
                        suspicious_df,
                        use_container_width=True,
                        height=300
                    )
                    
                    # Download button for suspicious transactions
                    suspicious_csv = suspicious_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Suspicious Transactions",
                        data=suspicious_csv,
                        file_name="suspicious_transactions.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("✅ **Great news!** No suspicious transactions detected in this batch.")
                    st.markdown("""
                    All transactions have been classified as low-risk and approved for processing. 
                    The system didn't identify any patterns requiring manual review.
                    """)
                
                # Display risk distribution
                st.markdown("---")
                st.subheader("📈 Risk Distribution")
                st.markdown("""
                Visual breakdown of transactions by risk level and decision type.
                """)
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_dist = results_df['Risk_Level'].value_counts()
                    st.bar_chart(risk_dist)
                    st.caption("Distribution by Risk Level")
                
                with col2:
                    decision_dist = results_df['Decision'].value_counts()
                    st.bar_chart(decision_dist)
                    st.caption("Distribution by Decision")
                
                # Display alert log
                st.markdown("---")
                st.header("5. Alert Log")
                st.markdown("""
                Historical log of all suspicious transactions with timestamps and details.
                """)
                
                if st.button("📋 Show Full Alert Log"):
                    try:
                        alerts_df = alert_agent.get_alerts()
                        
                        if not alerts_df.empty:
                            st.success(f"📋 Alert Log: {len(alerts_df)} total alerts")
                            st.dataframe(
                                alerts_df,
                                use_container_width=True,
                                height=400
                            )
                            
                            # Download alert log
                            alert_csv = alerts_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Alert Log",
                                data=alert_csv,
                                file_name="alert_log.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("ℹ️ No alerts in the log.")
                            
                    except Exception as e:
                        st.error(f"❌ Error loading alert log: {e}")
                
                # Performance metrics (if actual class is available)
                if 'Class' in df.columns and 'Actual_Class' in results_df.columns:
                    st.markdown("---")
                    st.header("6. Model Performance")
                    st.markdown("""
                    Evaluation metrics comparing model predictions against actual labels.
                    """)
                    
                    actual = results_df['Actual_Class'].values
                    predicted = (results_df['Prediction'] == 'FRAUD').astype(int).values
                    
                    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
                    
                    # Confusion matrix
                    cm = confusion_matrix(actual, predicted)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Confusion Matrix:**")
                        cm_df = pd.DataFrame(
                            cm,
                            columns=['Predicted Legitimate', 'Predicted Fraud'],
                            index=['Actual Legitimate', 'Actual Fraud']
                        )
                        st.dataframe(cm_df)
                    
                    with col2:
                        # Calculate metrics
                        tn, fp, fn, tp = cm.ravel()
                        accuracy = (tp + tn) / (tp + tn + fp + fn)
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        st.write("**Metrics:**")
                        st.write(f"- **Accuracy:** {accuracy:.4f}")
                        st.write(f"- **Precision:** {precision:.4f}")
                        st.write(f"- **Recall:** {recall:.4f}")
                        st.write(f"- **F1-Score:** {f1:.4f}")
                        
                        # ROC AUC
                        try:
                            probabilities = results_df['Fraud_Probability'].values
                            roc_auc = roc_auc_score(actual, probabilities)
                            st.write(f"- **ROC-AUC:** {roc_auc:.4f}")
                        except:
                            pass
        
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            st.stop()
    
    else:
        # Instructions when no file is uploaded
        st.info("""
        ### 📁 Getting Started
        
        1. **Upload a CSV file** with transaction data
        2. The file should have the same structure as the training data
        3. Required columns: `Time`, `V1-V28`, `Amount`
        4. Optional column: `Class` (for performance evaluation)
        
        ### 📊 Example Data Format
        
        | Time | V1 | V2 | ... | V28 | Amount | Class |
        |------|----|----|-----|-----|--------|-------|
        | 0 | -1.35 | 1.19 | ... | 0.14 | 149.62 | 0 |
        | 1 | 1.19 | 0.26 | ... | -0.01 | 2.69 | 0 |
        
        ### 🔗 Sample Dataset
        
        Use the training dataset from: `data/raw_transactions.csv`
        """)
        
        st.markdown("---")
        st.markdown("**Developed with ❤️ using Streamlit and Multi-Agent AI**")


if __name__ == "__main__":
    main()
