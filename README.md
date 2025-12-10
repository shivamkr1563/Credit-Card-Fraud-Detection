# 🔒 Cyber Fraud Detection Multi-Agent System

A sophisticated AI-powered fraud detection system that leverages a multi-agent architecture to identify and classify fraudulent transactions in real-time. Built with Machine Learning and Streamlit for an interactive web interface.

## 📄 Abstract

The exponential growth of digital financial transactions has led to a corresponding increase in cyber fraud, necessitating advanced detection mechanisms. This project presents a novel AI-based multi-agent system for real-time fraud detection in credit card transactions. The system employs a collaborative architecture consisting of five specialized agents: Transaction Monitoring, Analysis, Model, Risk Assessment, and Alert Management. Utilizing a Random Forest classifier trained on 284,807 transactions, the system achieves a ROC-AUC score of 0.9786 and fraud detection recall of 83.7%. Each agent performs a distinct function in the processing pipeline, from data validation and feature preprocessing to probabilistic classification and risk stratification. The system implements a three-tier risk assessment framework (LOW/MEDIUM/HIGH) with corresponding action recommendations. Deployed through an interactive Streamlit interface, the solution provides real-time analytics, comprehensive alert logging, and performance visualization. This multi-agent approach demonstrates superior modularity, scalability, and interpretability compared to monolithic fraud detection systems, making it suitable for deployment in financial institutions requiring robust and transparent fraud prevention mechanisms.

## 📋 Overview

This project implements an intelligent fraud detection system using a **multi-agent approach** where specialized AI agents work together in a pipeline to analyze transactions, predict fraud probability, assess risk levels, and log suspicious activities. The system achieves high accuracy (ROC-AUC: 0.9786) using a Random Forest classifier trained on credit card transaction data.

## 🏗️ Architecture

The system consists of **5 specialized agents** that process transactions through a coordinated pipeline:

### 1. 📊 Transaction Monitoring Agent
- Streams transactions from CSV files or DataFrames
- Validates transaction data structure
- Tracks processing statistics

### 2. 🔍 Analysis Agent
- Preprocesses raw transaction data
- Extracts and scales features
- Prepares data for ML model inference

### 3. 🤖 Model Agent
- Loads trained Random Forest classifier
- Predicts fraud probability for each transaction
- Returns binary classification (fraud/legitimate) with confidence scores

### 4. ⚠️ Risk Agent
- Assesses risk levels based on fraud probability
  - **LOW**: < 30% probability → Allow transaction
  - **MEDIUM**: 30-60% probability → Review required
  - **HIGH**: ≥ 60% probability → Flag for immediate action
- Recommends appropriate actions

### 5. 🚨 Alert Agent
- Logs suspicious transactions to CSV with timestamps
- Maintains historical alert database
- Provides alert retrieval and analytics

## 📊 Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle:

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 features (Time, V1-V28 PCA components, Amount)
- **Target**: Class (0 = Legitimate, 1 = Fraud)
- **Imbalance**: Only 0.17% fraudulent transactions

### Download Instructions
1. Visit the Kaggle dataset page
2. Download `creditcard.csv`
3. Place it in the `data/` folder as `raw_transactions.csv`

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Cyber_Fraud_Detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- pandas
- numpy
- scikit-learn
- joblib
- streamlit
- matplotlib
- seaborn

### Step 3: Prepare Dataset
Download the dataset from Kaggle and place it in `data/raw_transactions.csv`

### Step 4: Train the Model
```bash
python train_model.py
```

This will:
- Load and explore the dataset
- Preprocess features (scale Amount)
- Train a Random Forest classifier
- Evaluate model performance
- Save trained model to `models/fraud_model.pkl`
- Save scaler to `models/amount_scaler.pkl`

**Expected Output:**
```
ROC-AUC Score: 0.9786
Precision-Recall AUC: 0.8422
Fraud Detection Recall: 83.7%
```

### Step 5: Launch the Web Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## ✨ Features

### 🎯 Core Functionality
- **Real-time Fraud Detection**: Analyze transactions instantly through ML pipeline
- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Risk Assessment**: Three-tier risk classification (LOW/MEDIUM/HIGH)
- **Alert Management**: Automatic logging of suspicious transactions
- **Interactive Dashboard**: User-friendly Streamlit interface

### 📈 Analytics & Visualization
- Transaction analysis summary with key metrics
- Risk distribution charts
- Decision distribution visualization
- Confusion matrix and performance metrics
- Historical alert log with timestamps

### 💾 Data Management
- CSV file upload support
- Download analysis results
- Export suspicious transactions
- Alert log export functionality

### 🔍 Model Performance
- **ROC-AUC**: 0.9786
- **Precision-Recall AUC**: 0.8422
- **Fraud Recall**: 83.7%
- **False Positive Rate**: Low due to balanced class weights

## 🛠️ Tech Stack

### Machine Learning
- **scikit-learn**: Random Forest Classifier
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **joblib**: Model serialization

### Web Application
- **Streamlit**: Interactive web interface
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualizations

### Development
- **Python 3.10**: Core programming language
- **Git**: Version control

## 📁 Project Structure

```
Cyber_Fraud_Detection/
│
├── agents/
│   ├── __init__.py                    # Package initialization
│   ├── transaction_monitor.py         # Transaction Monitoring Agent
│   ├── analysis_agent.py              # Analysis Agent
│   ├── model_agent.py                 # Model Agent
│   ├── risk_agent.py                  # Risk Agent
│   └── alert_agent.py                 # Alert Agent
│
├── models/
│   ├── fraud_model.pkl                # Trained Random Forest model
│   └── amount_scaler.pkl              # Feature scaler
│
├── data/
│   ├── raw_transactions.csv           # Training dataset
│   └── README.md                      # Dataset instructions
│
├── app.py                             # Streamlit web application
├── train_model.py                     # Model training script
├── requirements.txt                   # Python dependencies
├── alerts_log.csv                     # Alert history (generated)
└── README.md                          # Project documentation
```

## 🎯 How It Works

### Transaction Processing Pipeline

```
Upload CSV → Transaction Monitor → Analysis Agent → Model Agent → Risk Agent → Alert Agent
                                                         ↓
                                            Prediction + Risk Assessment
                                                         ↓
                                         Display Results + Log Alerts
```

### Workflow
1. **Upload**: User uploads CSV file with transactions
2. **Stream**: Transaction Monitor validates and streams data
3. **Preprocess**: Analysis Agent extracts and scales features
4. **Predict**: Model Agent predicts fraud probability
5. **Assess**: Risk Agent determines risk level and action
6. **Log**: Alert Agent logs suspicious transactions
7. **Display**: Results shown in interactive dashboard

## 📊 Model Training Details

### Algorithm
- **Random Forest Classifier**
- 100 estimators
- Max depth: 10
- Balanced class weights (handles imbalanced data)
- Random state: 42 (reproducible results)

### Data Split
- Training: 80% (227,845 transactions)
- Testing: 20% (56,962 transactions)

### Feature Engineering
- StandardScaler applied to `Amount` feature
- V1-V28 features (PCA components) used as-is
- Time feature included in analysis

## 🔐 Security Considerations

- **Data Privacy**: No transaction data is stored permanently
- **Local Processing**: All computations done locally
- **Audit Trail**: Complete alert logging for compliance
- **Risk Thresholds**: Configurable for different risk appetites

## 🚀 Future Scope

The current implementation provides a robust foundation for fraud detection, with significant opportunities for enhancement and extension:

### 🔬 Advanced Machine Learning
- **Deep Learning Integration**: Implement LSTM, GRU, and Transformer architectures to capture temporal patterns and sequential dependencies in transaction histories
- **Ensemble Learning**: Combine multiple algorithms (XGBoost, LightGBM, Neural Networks) with stacking or boosting techniques for improved prediction accuracy
- **Autoencoder-based Anomaly Detection**: Deploy unsupervised learning models to identify novel fraud patterns not present in training data

### 🌐 Real-Time Processing & Scalability
- **Streaming Data Integration**: Connect to Kafka or Apache Flink for real-time transaction processing with microsecond latency
- **Distributed Computing**: Implement Apache Spark for parallel processing of millions of transactions
- **Cloud Deployment**: Migrate to AWS/Azure/GCP with auto-scaling capabilities and containerization using Docker and Kubernetes

### 🧠 Explainable AI & Interpretability
- **SHAP Values Integration**: Provide feature importance explanations for each prediction to ensure regulatory compliance
- **LIME Framework**: Generate local interpretable model-agnostic explanations for individual transaction decisions
- **Counterfactual Explanations**: Show users what changes would flip a fraud decision, enhancing transparency

### 📱 Communication & Integration
- **Multi-Channel Alerting**: Implement email, SMS, and push notifications for high-risk transactions with customizable alert rules
- **API Gateway Development**: Create RESTful and GraphQL APIs for seamless integration with banking systems and payment gateways
- **Webhook Support**: Enable real-time callbacks to external systems for automated response workflows

### 🔐 Security & Compliance
- **Federated Learning**: Train models across multiple institutions without sharing sensitive data, preserving privacy
- **Blockchain Integration**: Implement immutable audit trails for all fraud detection decisions and model updates
- **GDPR Compliance Module**: Add data anonymization, right-to-explanation, and automated data deletion capabilities

### 📊 Enhanced Analytics & User Experience
- **Advanced Dashboards**: Develop interactive visualizations using Plotly and D3.js with drill-down capabilities
- **A/B Testing Framework**: Compare multiple model versions in production with statistical significance testing
- **User Behavioral Analytics**: Incorporate device fingerprinting, geolocation analysis, and user profiling for context-aware detection

## 👨‍💻 Author

**Shivam Kumar**

- Project: Cyber Fraud Detection Multi-Agent System
- Year: 2025
- Technology: AI/ML, Python, Streamlit

## 📝 License

This project is developed for educational and research purposes.

## 🙏 Acknowledgments

- **Kaggle**: For providing the Credit Card Fraud Detection dataset
- **scikit-learn**: For robust ML algorithms
- **Streamlit**: For the excellent web framework
- **Python Community**: For amazing open-source tools

## 📞 Support

For questions, issues, or contributions, please open an issue in the repository.

---

**Built with ❤️ using Python, Machine Learning, and Multi-Agent Architecture**
