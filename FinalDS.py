import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. CONFIG
# =========================
START = "2012-01-01"
END = "2025-01-01"

ASSETS = ["SPY", "GLD", "TLT"]
FEATURES = ["CPI", "GDP", "Rates"]

# =========================
# 2. DATA COLLECTION
# =========================
def get_fred(series):
    return pdr.DataReader(series, "fred", START, END)

print("Loading macro data...")

cpi = get_fred("CPIAUCSL")
gdp = get_fred("GDP")
rates = get_fred("FEDFUNDS")

macro = pd.concat([cpi, gdp, rates], axis=1)
macro.columns = FEATURES
macro = macro.ffill()

print("Loading market data...")

prices = yf.download(ASSETS, start=START, end=END, auto_adjust=True)["Close"]
prices = prices.resample("M").last()
macro = macro.resample("M").last()

data = macro.join(prices).dropna()

print(f"Dataset shape: {data.shape}")

# =========================
# 3. FEATURE ENGINEERING
# =========================

for asset in ASSETS:
    data[f"{asset}_ret"] = data[asset].pct_change().shift(-1)

for f in FEATURES:
    data[f"{f}_lag1"] = data[f].shift(1)

data["inflation_proxy"] = data["CPI"].pct_change()

data = data.dropna()

TARGETS = [f"{a}_ret" for a in ASSETS]
FEATURE_SET = [f"{f}_lag1" for f in FEATURES] + ["inflation_proxy"]

X = data[FEATURE_SET]

# =========================
# 4. TRAIN/TEST SPLIT
# =========================
split = int(len(data) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
data_train, data_test = data.iloc[:split], data.iloc[split:]

# =========================
# 5. SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 6. REGIME DETECTION
# =========================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

train_regimes = kmeans.fit_predict(X_train_scaled)
test_regimes = kmeans.predict(X_test_scaled)

data_train["regime"] = train_regimes
data_test["regime"] = test_regimes

# =========================
# 7. MODEL TRAINING
# =========================
models = {r: {} for r in range(3)}

for r in range(3):
    idx = data_train.index[data_train["regime"] == r]

    if len(idx) < 20:
        continue

    X_r = scaler.transform(data_train.loc[idx, FEATURE_SET])

    for asset in ASSETS:
        y_r = data_train.loc[idx, f"{asset}_ret"]

        model = XGBRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )

        model.fit(X_r, y_r)
        models[r][asset] = model

# =========================
# 8. PREDICTION FUNCTION
# =========================
def predict_assets(x, regime):
    preds = {}
    for asset in ASSETS:
        if regime in models and asset in models[regime]:
            preds[asset] = models[regime][asset].predict(x)[0]
        else:
            preds[asset] = -0.01
    return preds

# =========================
# 9. BACKTEST
# =========================
portfolio_returns = []

for i in range(len(X_test_scaled) - 1):
    regime = test_regimes[i]
    x = X_test_scaled[i].reshape(1, -1)

    preds = predict_assets(x, regime)
    best_asset = max(preds, key=preds.get)

    actual_ret = data_test.iloc[i + 1][f"{best_asset}_ret"]
    portfolio_returns.append(actual_ret)

portfolio_returns = np.array(portfolio_returns)

cumulative_return = np.prod(1 + portfolio_returns) - 1
volatility = np.std(portfolio_returns)
sharpe = np.mean(portfolio_returns) / (volatility + 1e-9)

# =========================
# 10. BASELINE (BUY & HOLD)
# =========================
bh = data_test["SPY_ret"].dropna()

bh_return = np.prod(1 + bh) - 1
bh_vol = np.std(bh)
bh_sharpe = np.mean(bh) / (bh_vol + 1e-9)

# =========================
# 11. FEATURE IMPORTANCE
# =========================
print("\n===== FEATURE IMPORTANCE (Regime 0 - SPY model) =====")
if 0 in models and "SPY" in models[0]:
    importances = models[0]["SPY"].feature_importances_
    for f, imp in zip(FEATURE_SET, importances):
        print(f"{f}: {imp:.4f}")

# =========================
# 12. FINAL PREDICTION
# =========================
latest_x = scaler.transform(X.iloc[[-1]])
latest_regime = kmeans.predict(latest_x)[0]

final_preds = predict_assets(latest_x, latest_regime)
best = max(final_preds, key=final_preds.get)

# =========================
# 13. OUTPUT
# =========================
print("\n===== PREDICTIONS =====")
for k, v in final_preds.items():
    print(f"{k}: {v:.4f}")

print("\nBest Investment Recommendation:")
print(best)

print("\n===== BACKTEST RESULTS =====")
print(f"Cumulative Return: {cumulative_return:.2%}")
print(f"Volatility: {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")

print("\n===== BASELINE (BUY & HOLD SPY) =====")
print(f"Return: {bh_return:.2%}")
print(f"Sharpe: {bh_sharpe:.4f}")

# =========================
# 14. REGIME INTERPRETATION
# =========================
print("\n===== REGIME SUMMARY =====")
for r in range(3):
    subset = data_train[data_train["regime"] == r]

    print(f"\nRegime {r}:")
    print(f"Avg CPI change: {subset['CPI'].pct_change().mean():.4f}")
    print(f"Avg Rates: {subset['Rates'].mean():.2f}")
    print(f"Avg GDP: {subset['GDP'].mean():.2f}")
    print(f"Size: {len(subset)} samples")

# =========================
# 15. MODEL QUALITY WARNING
# =========================
if sharpe < 0.3:
    print("\nWARNING: Low Sharpe Ratio → high noise / macro unpredictability expected.")
    
# =========================
# 16. VISUALIZATIONS
# =========================

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# Portfolio Growth Plot
# -------------------------

strategy_curve = np.cumprod(1 + portfolio_returns)

spy_curve = np.cumprod(1 + bh.values[:len(strategy_curve)])

plt.figure(figsize=(10,6))
plt.plot(strategy_curve, label="Strategy")
plt.plot(spy_curve, label="Buy & Hold SPY")
plt.title("Portfolio Growth Comparison")
plt.xlabel("Time")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True)
plt.show()


# -------------------------
# Feature Importance Plot
# -------------------------

if 0 in models and "SPY" in models[0]:

    importances = models[0]["SPY"].feature_importances_

    plt.figure(figsize=(8,5))
    plt.bar(FEATURE_SET, importances)
    plt.title("Feature Importance (Regime 0 - SPY Model)")
    plt.ylabel("Importance")
    plt.xticks(rotation=20)
    plt.grid(True)
    plt.show()


# -------------------------
# PCA Visualization of Regimes
# -------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8,6))

for r in range(3):
    idx = train_regimes == r

    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=f"Regime {r}",
        alpha=0.7
    )

plt.title("PCA Projection of Economic Regimes")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()


# -------------------------
# Regime Timeline
# -------------------------

plt.figure(figsize=(12,4))

plt.plot(
    data_train.index,
    data_train["regime"],
    marker='o',
    linestyle='-'
)

plt.title("Economic Regimes Over Time")
plt.xlabel("Date")
plt.ylabel("Regime")
plt.grid(True)

plt.show()


# -------------------------
# Correlation Heatmap
# -------------------------

corr = data[FEATURE_SET + TARGETS].corr()

plt.figure(figsize=(10,8))

plt.imshow(corr, cmap='coolwarm', interpolation='nearest')

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()