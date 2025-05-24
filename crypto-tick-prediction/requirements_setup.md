# BTC/USDT Tick-Level Micro-Prediction Pipeline - Requirements & Setup

## 📋 System Requirements

### Hardware Requirements (Minimum)
- **RAM**: 8GB (16GB+ recommended for large datasets)
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: 5GB free space for data and results
- **GPU**: Optional (for faster XGBoost training)

### Hardware Requirements (Recommended)
- **RAM**: 32GB+ for processing large tick datasets
- **CPU**: 8+ cores with high clock speed
- **Storage**: SSD with 20GB+ free space
- **GPU**: NVIDIA GPU with CUDA support

## 🐍 Python Environment

### Python Version
- **Python 3.8+** (3.9 or 3.10 recommended)

### Required Python Packages

```bash
# Core data processing
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.1.0
xgboost>=1.6.0
shap>=0.41.0

# Data visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Progress bars and utilities
tqdm>=4.64.0
jupyter>=1.0.0
notebook>=6.4.0

# File handling
pyarrow>=8.0.0  # For parquet files
openpyxl>=3.0.0  # For Excel files (optional)

# Date/time handling
python-dateutil>=2.8.0
pytz>=2022.1
```

## 📦 Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Create virtual environment
python -m venv btc_prediction_env
source btc_prediction_env/bin/activate  # On Windows: btc_prediction_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install packages
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib seaborn plotly tqdm jupyter notebook pyarrow python-dateutil pytz

# Start Jupyter
jupyter notebook
```

### Method 2: Using conda

```bash
# Create conda environment
conda create -n btc_prediction python=3.9
conda activate btc_prediction

# Install packages
conda install pandas numpy scipy scikit-learn matplotlib seaborn jupyter notebook
pip install xgboost shap plotly tqdm pyarrow

# Start Jupyter
jupyter notebook
```

### Method 3: Using requirements.txt

Create a `requirements.txt` file:

```txt
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.1.0
xgboost>=1.6.0
shap>=0.41.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
tqdm>=4.64.0
jupyter>=1.0.0
notebook>=6.4.0
pyarrow>=8.0.0
python-dateutil>=2.8.0
pytz>=2022.1
```

Then install:

```bash
pip install -r requirements.txt
```

## 📁 Data Requirements

### Expected Data Format

The pipeline expects tick-level trade data with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `timestamp` | datetime/int64 | Trade timestamp (UTC) | `2024-01-01 12:00:00.123` |
| `price` | float64 | Trade price | `45000.50` |
| `amount` | float64 | Trade quantity/size | `0.01234` |
| `side` | int/string | Buy/Sell indicator | `1`/`0` or `buy`/`sell` |

### Supported File Formats
- **CSV**: `.csv`
- **Compressed CSV**: `.csv.gz`
- **Parquet**: `.parquet` (recommended for large datasets)

### Data Sources

#### Tardis.dev (Primary)
1. Sign up at [tardis.dev](https://tardis.dev)
2. Download BTC/USDT trades data from Binance
3. Choose appropriate date range and format

#### Alternative Sources
- **Binance API**: Historical data via REST API
- **CryptoCompare**: Tick data API
- **Kaiko**: Professional crypto data
- **CoinAPI**: Real-time and historical data

### Sample Data Structure

```csv
timestamp,price,amount,side
2024-01-01T00:00:01.123Z,45000.50,0.01234,buy
2024-01-01T00:00:01.456Z,45000.25,0.05678,sell
2024-01-01T00:00:02.789Z,45000.75,0.02345,buy
...
```

## ⚙️ Configuration

Before running the notebook, update the CONFIG section:

```python
CONFIG = {
    # Update this path to your actual data file
    'DATA_PATH': './data/btcusdt_trades.csv.gz',
    'OUTPUT_DIR': './results/',
    
    # Adjust these parameters as needed
    'CANDLE_INTERVAL': '1T',  # 1-minute candles
    'IMBALANCE_WINDOW': 10,   # seconds
    'VOLATILITY_WINDOW': 5,   # minutes
    'TEST_SIZE': 0.1,         # 10% for testing
    'RANDOM_STATE': 42,
}
```

## 🚀 Performance Optimization

### For Large Datasets (1M+ ticks)

```python
# Add these optimizations to CONFIG
CONFIG.update({
    'CHUNK_SIZE': 100000,     # Process data in chunks
    'SAMPLE_RATE': 0.1,       # Use 10% sample for development
    'PARALLEL_JOBS': -1,      # Use all CPU cores
    'MEMORY_LIMIT': '8GB',    # XGBoost memory limit
})
```

### Memory Management

```python
# Add memory optimization functions
import gc
import psutil

def check_memory():
    """Monitor memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")

def optimize_memory():
    """Force garbage collection"""
    gc.collect()
    print("Memory optimized")
```

## 🧪 Testing the Setup

Create a test script `test_setup.py`:

```python
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("✅ All packages imported successfully!")

# Test data generation
np.random.seed(42)
test_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1s'),
    'price': 45000 + np.cumsum(np.random.normal(0, 1, 1000)),
    'amount': np.random.exponential(0.1, 1000),
    'side': np.random.choice([0, 1], 1000)
})

print(f"✅ Test data generated: {len(test_data)} rows")

# Test XGBoost
X = np.random.random((100, 5))
y = np.random.randint(0, 2, 100)
model = xgb.XGBClassifier()
model.fit(X, y)
print("✅ XGBoost test passed!")

# Test SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[:10])
print("✅ SHAP test passed!")

print("\n🎉 Setup verification complete!")
```

Run the test:
```bash
python test_setup.py
```

## 🐛 Common Issues & Solutions

### 1. Memory Errors
```python
# Solution: Process data in chunks
def process_in_chunks(df, chunk_size=50000):
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        yield process_chunk(chunk)
```

### 2. SHAP Installation Issues
```bash
# If SHAP installation fails
pip install shap --no-use-pep517
# Or use conda
conda install -c conda-forge shap
```

### 3. XGBoost GPU Issues
```bash
# For GPU support
pip install xgboost[gpu]
# Or compile from source with CUDA
```

### 4. Jupyter Kernel Issues
```bash
# Install kernel
python -m ipykernel install --user --name=btc_prediction
# Select kernel in Jupyter: Kernel > Change Kernel > btc_prediction
```

## 📊 Expected Resource Usage

### Small Dataset (< 100k ticks)
- **Runtime**: 5-15 minutes
- **Memory**: 2-4 GB
- **Output**: ~50 MB

### Medium Dataset (100k - 1M ticks)
- **Runtime**: 30-60 minutes
- **Memory**: 8-16 GB
- **Output**: ~200 MB

### Large Dataset (> 1M ticks)
- **Runtime**: 2-4 hours
- **Memory**: 16-32 GB
- **Output**: ~1 GB

## 🔧 Troubleshooting

### Performance Issues
1. **Reduce sample size** for initial testing
2. **Use chunked processing** for large datasets
3. **Enable parallel processing** where possible
4. **Monitor memory usage** and optimize

### Data Issues
1. **Check timestamp format** and timezone
2. **Verify column names** match expectations
3. **Handle missing values** appropriately
4. **Validate price/amount ranges**

### Model Issues
1. **Adjust hyperparameters** for your dataset
2. **Check feature scaling** if needed
3. **Validate train/test split** chronologically
4. **Monitor overfitting** with cross-validation

## 📞 Support

If you encounter issues:
1. Check the error message carefully
2. Verify all requirements are installed
3. Test with synthetic data first
4. Check data format and structure
5. Monitor system resources during execution

Happy trading! 🚀📈
