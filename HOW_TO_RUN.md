# 🚀 How to Run This Project

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit App
```bash
streamlit run app.py
```

### Step 3: Open Browser
The app will automatically open at: **http://localhost:8501**

---

## Detailed Instructions

### Prerequisites
- Python 3.8 or higher installed
- pip package manager

### Installation

1. **Open Terminal/Command Prompt** in project folder:
   ```bash
   cd C:\Users\shiva\Desktop\Cyber_Fraud_Detection
   ```

2. **Create Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux/Mac**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Option 1: Run Streamlit App (Recommended)**
```bash
streamlit run app.py
```

**Option 2: Train Model First (if models not present)**
```bash
python train_model.py
```
Then run the app:
```bash
streamlit run app.py
```

**Option 3: Test Pipeline**
```bash
python test_pipeline.py
```

---

## Usage Guide

### After Starting the App:

1. **Browser Opens Automatically** at http://localhost:8501

2. **Upload Dataset**:
   - Click "Browse files" button
   - Select `data/sample_1000.csv` (for quick testing)
   - Or upload your own CSV file

3. **View Results**:
   - Transaction processing progress bar
   - Risk distribution charts
   - Suspicious transactions table
   - Alert logs

4. **Export Results**:
   - Download alerts as CSV
   - View detailed statistics

---

## Testing with Sample Data

### Quick Test (1,000 transactions):
```bash
streamlit run app.py
```
Then upload: `data/sample_1000.csv`

### Full Dataset Test:
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place as `data/raw_transactions.csv`
3. Upload in Streamlit app

---

## Troubleshooting

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "streamlit: command not found"
```bash
pip install streamlit
# Then run:
python -m streamlit run app.py
```

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Error: "Model file not found"
```bash
python train_model.py
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

---

## Available Scripts

### 1. Main Application
```bash
streamlit run app.py
```
Launches web interface for fraud detection

### 2. Train Model
```bash
python train_model.py
```
Trains Random Forest model on dataset

### 3. Test Pipeline
```bash
python test_pipeline.py
```
Validates all 5 agents working correctly

### 4. Create Sample Dataset
```bash
python create_sample.py
```
Generates small 1,000 transaction sample

### 5. Check Data
```bash
python check_data.py
```
Validates dataset structure

---

## Command Reference

### Windows (PowerShell)
```powershell
# Navigate to project
cd C:\Users\shiva\Desktop\Cyber_Fraud_Detection

# Activate venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Windows (CMD)
```cmd
cd C:\Users\shiva\Desktop\Cyber_Fraud_Detection
venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

### Linux/Mac
```bash
cd ~/Cyber_Fraud_Detection
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Expected Output

When you run `streamlit run app.py`, you should see:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Browser will open automatically showing:
- 🔒 **Title**: Cyber Fraud Detection System
- 📤 **Upload button** for CSV files
- 📊 **Dashboard** with charts and tables

---

## Project Structure

```
Cyber_Fraud_Detection/
├── app.py                 ← Main Streamlit application (RUN THIS)
├── train_model.py         ← Model training script
├── test_pipeline.py       ← Pipeline testing
├── agents/                ← 5 AI agents
├── models/                ← Trained models
├── data/                  ← Dataset folder
│   └── sample_1000.csv   ← Sample data (upload this)
└── requirements.txt       ← Dependencies
```

---

## Performance Notes

- **Sample Dataset (1,000)**: ~2-3 seconds processing
- **Full Dataset (284,807)**: ~1-2 minutes processing
- **Model Loading**: ~1 second
- **Memory Usage**: ~500 MB RAM

---

## Next Steps After Running

1. ✅ Upload sample dataset
2. ✅ View fraud detection results
3. ✅ Export alerts to CSV
4. ✅ Try with your own data
5. ✅ Explore risk distributions

---

## Need Help?

- Check `README.md` for detailed documentation
- See `PROJECT_HEALTH_REPORT.md` for project status
- Review `GITHUB_UPLOAD_GUIDE.md` for sharing

**Happy Fraud Detection! 🔒**
