# CS439 Final Project  
## Macroeconomic Regime-Based Asset Allocation using Machine Learning

---

## 📌 Overview
This project develops a machine learning-based investment strategy that dynamically allocates capital across major asset classes (SPY, GLD, TLT) using macroeconomic indicators. The system combines unsupervised learning (K-Means clustering) to detect economic regimes with supervised learning (XGBoost regression) to predict future asset returns.

---

## 🎯 Objective
Traditional investment strategies often assume static market behavior. This project investigates whether incorporating macroeconomic regime shifts (inflation, interest rates, GDP changes) can improve asset allocation decisions and risk-adjusted returns.

---

## 📊 Dataset
- **Macroeconomic Data:** Federal Reserve Economic Data (FRED)
  - CPI (Inflation)
  - GDP (Economic growth)
  - FED Funds Rate (Interest rates)

- **Market Data:** Yahoo Finance
  - SPY (Equities)
  - GLD (Gold)
  - TLT (Treasury Bonds)

- **Time Period:** 2012–2025 (monthly resampled)

---

## 🧠 Methodology
1. **Data Preprocessing**
   - Monthly resampling of macro + market data
   - Lagged macroeconomic features to prevent data leakage
   - Inflation proxy via CPI percentage change
   - Standardization using StandardScaler

2. **Regime Detection**
   - K-Means clustering (k=3)
   - Identifies macroeconomic regimes:
     - High inflation / high rates
     - Low growth / recession-like
     - Stable economic conditions

3. **Predictive Modeling**
   - XGBoost regression per regime and asset
   - Predict next-month returns for SPY, GLD, TLT

4. **Portfolio Strategy**
   - Select asset with highest predicted return each month
   - Dynamic allocation based on regime classification

---

## 📈 Experiments & Evaluation
The model is evaluated using:

- Cumulative Return
- Volatility
- Sharpe Ratio
- Comparison against Buy & Hold SPY baseline

---

## 📊 Results
- **Strategy Return:** ~31.5%
- **Sharpe Ratio:** ~0.23
- **Baseline SPY Return:** ~47.85%
- **Baseline Sharpe:** ~0.29

### Key Insight
- Interest rates and inflation are the most influential predictors of asset returns.
- The model successfully identifies distinct macroeconomic regimes but is limited by noise in financial data.

---

## 📉 Visualizations
The following figures are included in the report:
- PCA projection of macroeconomic regimes
- Correlation heatmap
- Feature importance (XGBoost)
- Portfolio vs Buy & Hold performance
- Regime distribution over time

---

## ⚠️ Limitations
- Limited dataset size (monthly resolution)
- No transaction costs included
- Simple clustering approach (K-Means)
- Market noise reduces predictive stability

---

## 🚀 Future Work
- Reinforcement learning-based portfolio optimization
- More advanced regime models (HMM, Transformers)
- Inclusion of additional macro + sentiment data
- Transaction cost-aware backtesting

---

## 🛠️ How to Run
```bash
pip install -r requirements.txt