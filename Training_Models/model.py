import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load and preprocess data
# (Make sure to replace this with your actual data loading code)
df = pd.read_csv('orderbook_data_BTC_USDT_SWAP.csv', header=None)

columns = [
    'timestamp', 'bid_price', 'ask_price', 'spread', 'mid_price',
    'volume_imbalance', 'trade_volume', 'metric_7', 'metric_8',
    'maker_volume', 'taker_volume', 'time_left'
]

df = pd.DataFrame(data, columns=columns)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Feature Engineering
df['price_change'] = df['mid_price'].pct_change().fillna(0)
df['volatility'] = df['price_change'].rolling(5).std().fillna(0)  # Reduced window for small dataset

# =============================================================================
# 1. Almgren-Chriss Model Implementation (FIXED SYNTAX ERROR)
# =============================================================================

def estimate_almgren_chriss(df):
    """Estimate model parameters for temporary and permanent market impact"""
    X = df[['trade_volume']].values  # Trade size
    y = df['mid_price'].diff().shift(-1).fillna(0).values  # Future price changes
    
    # Permanent impact (gamma)
    reg = LinearRegression().fit(X, y)
    gamma = reg.coef_[0]
    
    # Temporary impact (eta) - estimated from variance
    returns = df['mid_price'].pct_change().dropna()
    sigma = returns.std() * np.sqrt(252)  # Annualized volatility
    eta = sigma / (2 * np.sqrt(np.abs(df['trade_volume'].mean())))  # FIXED: Added missing parenthesis
    
    return {'gamma': gamma, 'eta': eta, 'sigma': sigma}

ac_params = estimate_almgren_chriss(df)
print("Almgren-Chriss Parameters:", ac_params)

# =============================================================================
# 2. Slippage Regression Model
# =============================================================================

# Assuming metric_7 represents slippage (adjust as needed)
df['slippage'] = df['metric_7']

# Select features and target
X_slip = df[['trade_volume', 'spread', 'volatility', 'volume_imbalance']].fillna(0)
y_slip = df['slippage'].fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_slip, y_slip, test_size=0.2, random_state=42)

# Train model
slip_model = LinearRegression()
slip_model.fit(X_train, y_train)

# Evaluate
preds = slip_model.predict(X_test)
print(f"\nSlippage Model RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
print("Slippage Coefficients:", dict(zip(X_slip.columns, slip_model.coef_)))

# =============================================================================
# 3. Maker/Taker Proportion Prediction
# =============================================================================

# Calculate maker/taker proportion
df['total_volume'] = df['maker_volume'] + df['taker_volume']
df['maker_proportion'] = df['maker_volume'] / np.where(df['total_volume'] > 0, df['total_volume'], 1)

# Select features and target
X_mt = df[['spread', 'volatility', 'volume_imbalance', 'time_left']].fillna(0)
y_mt = df['maker_proportion'].fillna(0.5)  # Default to 0.5 if undefined

# Train/test split
X_train_mt, X_test_mt, y_train_mt, y_test_mt = train_test_split(X_mt, y_mt, test_size=0.2, random_state=42)

# Train model
mt_model = LinearRegression()
mt_model.fit(X_train_mt, y_train_mt)

# Evaluate
preds_mt = mt_model.predict(X_test_mt)
print(f"\nMaker Proportion RMSE: {np.sqrt(mean_squared_error(y_test_mt, preds_mt)):.4f}")
print("Maker/Taker Coefficients:", dict(zip(X_mt.columns, mt_model.coef_)))
