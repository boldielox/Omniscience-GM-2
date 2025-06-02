import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import shap
import warnings

# ========== CONFIG ==========
REQUIRED_COLUMNS = ['Close', 'Outcome']  # Add or edit as needed

# ========== FILE PARSER & VALIDATOR ==========
def parse_uploaded_file(file):
    """
    Attempts to read a file and validate its structure.
    Returns a DataFrame with only useful columns, or None if not usable.
    """
    try:
        df = pd.read_csv(file)
    except Exception as e:
        warnings.warn(f"Could not read file: {e}")
        return None

    # Check for required columns
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        warnings.warn(f"Missing required columns: {REQUIRED_COLUMNS}")
        return None

    # Drop completely empty or constant columns
    df = df.dropna(axis=1, how='all')
    nunique = df.nunique()
    df = df.loc[:, nunique > 1]

    # Keep only numeric and outcome columns
    useful_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) or col == 'Outcome']
    df = df[useful_cols]

    # Remove rows with missing outcome
    df = df.dropna(subset=['Outcome'])
    if df.empty:
        warnings.warn("No usable data after filtering.")
        return None

    return df

# ========== FEATURE ENGINEERING ==========
def add_technical_indicators(df, price_col='Close'):
    """Adds RSI, MACD, and Bollinger Bands to the DataFrame."""
    df['RSI'] = RSIIndicator(df[price_col], window=14).rsi()
    macd = MACD(df[price_col])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = BollingerBands(df[price_col])
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# ========== MODEL TRAINING ==========
def train_omniscience_ensemble(df):
    """
    Trains an XGBoost + LightGBM ensemble on the given DataFrame.
    Assumes 'Outcome' is the target (1=Win, 0=Loss).
    """
    features = [col for col in df.columns if col not in ['Outcome']]
    X = df[features]
    y = df['Outcome'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    lgb_model.fit(X_train, y_train)

    # Ensemble prediction (average)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    ensemble_pred = (xgb_pred + lgb_pred) / 2
    pred_labels = (ensemble_pred > 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels)
    print(f"Ensemble Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    # SHAP explainability (XGBoost as example)
    explainer = shap.Explainer(xgb_model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)  # Remove show=False to display in notebook

    return xgb_model, lgb_model, features

# ========== MAIN PIPELINE ==========
def omniscience_pipeline(file):
    df = parse_uploaded_file(file)
    if df is None:
        print("No usable data found. Skipping file.")
        return None

    df = add_technical_indicators(df)
    if 'Outcome' not in df.columns or df['Outcome'].nunique() < 2:
        print("Not enough outcome classes for training.")
        return None

    xgb_model, lgb_model, features = train_omniscience_ensemble(df)
    print("Model training complete. Ready for live predictions.")
    return xgb_model, lgb_model, features

# ========== USAGE EXAMPLE ==========
# with open('your_data.csv', 'r') as file:
#     omniscience_pipeline(file)

# For Streamlit:
# uploaded_file = st.file_uploader("Upload your CSV file")
# if uploaded_file is not None:
#     omniscience_pipeline(uploaded_file)
