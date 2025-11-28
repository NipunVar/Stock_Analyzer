# streamlit_app/app.py
"""
Streamlit demo — improved folder discovery for notebook sections.

Key fix:
- find_folder_path(folder_name, PROJECT_ROOT) searches recursively for the requested folder name
  (case-insensitive) and returns the actual path. This ensures notebooks are found even if
  they live under Stock-Prediction/ or in another subfolder.
"""
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from datetime import timedelta
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Prediction Project", layout="wide")

# -----------------------------
# Project root detection
# -----------------------------
def find_project_root(cwd="."):
    cwd = os.path.abspath(cwd)
    base = os.path.basename(cwd).lower()
    if base in ("stock-prediction", "stock"):
        return cwd
    for name in os.listdir(cwd):
        if name.lower() in ("stock-prediction", "stock") and os.path.isdir(os.path.join(cwd, name)):
            return os.path.abspath(os.path.join(cwd, name))
    up = os.path.abspath(os.path.join(cwd, ".."))
    for name in os.listdir(up):
        if name.lower() in ("stock-prediction", "stock"):
            return os.path.abspath(os.path.join(up, name))
    return cwd

PROJECT_ROOT = find_project_root(".")

# -----------------------------
# File discovery helpers
# -----------------------------
def list_files_recursive(base_dir, exts=None):
    matches = []
    for root, _, files in os.walk(base_dir):
        # skip common hidden/system dirs
        if ".git" in root or "__pycache__" in root:
            continue
        for f in files:
            if exts:
                if any(f.lower().endswith(e) for e in exts):
                    matches.append(os.path.join(root, f))
            else:
                matches.append(os.path.join(root, f))
    return sorted(matches)

def list_all_images(base_dir=PROJECT_ROOT):
    exts = (".png", ".jpg", ".jpeg", ".gif", ".svg")
    return list_files_recursive(base_dir, exts=exts)

def find_folder_path(folder_name, base_dir=PROJECT_ROOT):
    """
    Search for a folder with name `folder_name` anywhere under base_dir (case-insensitive).
    Return the absolute path of the first matching folder, or None if not found.
    """
    folder_name_lower = folder_name.lower()
    # walk the tree
    for root, dirs, _ in os.walk(base_dir):
        # skip hidden/system dirs
        if ".git" in root or "__pycache__" in root:
            continue
        for d in dirs:
            if d.lower() == folder_name_lower:
                return os.path.join(root, d)
    return None

def list_notebooks_in_folder(folder_name, base_dir=PROJECT_ROOT):
    """
    Return list of notebook filepaths that are inside the folder with name `folder_name`
    anywhere under the project root. If no such folder exists, return [].
    """
    folder_path = find_folder_path(folder_name, base_dir)
    if not folder_path:
        return []
    # list only ipynb files directly under that folder (including nested subfolders if any)
    return list_files_recursive(folder_path, exts=(".ipynb",))

def list_models(base_dir=os.path.join(PROJECT_ROOT, "models")):
    exts = (".pkl", ".joblib", ".h5", ".pt", ".pth", ".csv")
    if not os.path.exists(base_dir):
        return []
    return sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.lower().endswith(exts)])

# -----------------------------
# CSV reading helper
# -----------------------------
def read_stock_csv_obj(obj):
    if isinstance(obj, str):
        df = pd.read_csv(obj)
    else:
        df = pd.read_csv(obj)
    date_cols = [c for c in df.columns if "date" in c.lower()] or [df.columns[0]]
    price_cols = [c for c in df.columns if ("adj" in c.lower() and "close" in c.lower())] or \
                 [c for c in df.columns if c.lower() == "close"] or \
                 [c for c in df.columns if "price" in c.lower()] or \
                 [df.columns[1] if len(df.columns) > 1 else df.columns[0]]
    df = df.rename(columns={date_cols[0]: "Date", price_cols[0]: "Adj Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df[["Date", "Adj Close"]]
    return df

# -----------------------------
# plotting helpers
# -----------------------------
def plot_series(df, title="Series"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Adj Close"], label="Actual", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    return fig

def plot_with_forecast(df, forecast_df, title="Forecast"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Adj Close"], label="Actual")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast")
    if "Lower" in forecast_df.columns and "Upper" in forecast_df.columns:
        ax.fill_between(forecast_df["Date"], forecast_df["Lower"], forecast_df["Upper"], alpha=0.2)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# quick forecast helpers and trainers (unchanged)
# -----------------------------
def persistence_forecast(series, fh):
    return np.repeat(series.iloc[-1], fh)

def moving_average_forecast(series, window, fh):
    last_mean = series.iloc[-window:].mean()
    return np.repeat(last_mean, fh)

def train_arima_forecast(series, order=(1,1,1), seasonal_order=(0,0,0,0), fh=30):
    try:
        import statsmodels.api as sm
    except Exception:
        raise RuntimeError("statsmodels required for ARIMA.")
    model = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                      enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=50)
    pred = res.get_forecast(steps=fh)
    mean = pred.predicted_mean
    ci = pred.conf_int()
    return mean, ci

def train_prophet_forecast(df, fh=30):
    try:
        from prophet import Prophet
    except Exception:
        try:
            from fbprophet import Prophet
        except Exception:
            raise RuntimeError("Prophet not installed.")
    mdf = df.rename(columns={"Date":"ds","Adj Close":"y"})[["ds","y"]]
    m = Prophet(daily_seasonality=False)
    m.fit(mdf)
    future = m.make_future_dataframe(periods=fh)
    forecast = m.predict(future)
    fh_df = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(fh).rename(columns={'ds':'Date','yhat':'Forecast','yhat_lower':'Lower','yhat_upper':'Upper'})
    return fh_df

def train_small_lstm_forecast(df, fh=30, lookback=20, epochs=5):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
    except Exception:
        raise RuntimeError("TensorFlow and scikit-learn required for LSTM.")
    series = df['Adj Close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)
    def create_seq(data, lookback):
        X, y = [], []
        for i in range(len(data)-lookback):
            X.append(data[i:i+lookback, 0])
            y.append(data[i+lookback, 0])
        X = np.array(X); y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y
    X, y = create_seq(series_scaled, lookback)
    if len(X) < 10:
        raise RuntimeError("Not enough data to train LSTM with this lookback.")
    model = Sequential()
    model.add(LSTM(32, input_shape=(lookback,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    last_seq = series_scaled[-lookback:].reshape(1, lookback, 1)
    preds = []
    seq = last_seq.copy()
    for i in range(fh):
        p = model.predict(seq, verbose=0)[0,0]
        preds.append(p)
        seq = np.concatenate([seq[:,1:,:], p.reshape(1,1,1)], axis=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    last_date = df['Date'].max()
    dates = [last_date + timedelta(days=i+1) for i in range(fh)]
    forecast_df = pd.DataFrame({"Date": dates, "Forecast": preds})
    return forecast_df

# -----------------------------
# model loader (same as before)
# -----------------------------
def load_model_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".joblib"):
        try:
            mdl = joblib.load(path)
        except Exception as e:
            raise RuntimeError(f"Could not load pickle/joblib: {e}")
        def predict_wrapper(df, fh=30):
            if hasattr(mdl, "get_forecast"):
                pred = mdl.get_forecast(steps=fh)
                mean = pred.predicted_mean
                ci = pred.conf_int()
                last_date = df["Date"].max()
                dates = [last_date + timedelta(days=i+1) for i in range(fh)]
                fdf = pd.DataFrame({"Date": dates, "Forecast": mean.values})
                if ci is not None:
                    fdf["Lower"] = ci.iloc[:,0].values
                    fdf["Upper"] = ci.iloc[:,1].values
                return fdf
            if hasattr(mdl, "predict"):
                try:
                    last_vals = df["Adj Close"].values.reshape(1, -1)
                    preds = mdl.predict(last_vals)
                    if np.ndim(preds) == 0 or len(preds) == 1:
                        preds = np.repeat(preds, fh)
                    last_date = df["Date"].max()
                    dates = [last_date + timedelta(days=i+1) for i in range(fh)]
                    return pd.DataFrame({"Date": dates, "Forecast": preds[:fh]})
                except Exception:
                    raise RuntimeError("Sklearn model loaded but automatic feature prep failed.")
            if mdl.__class__.__name__.lower().startswith("prophet"):
                future = mdl.make_future_dataframe(periods=fh)
                forecast = mdl.predict(future)
                fh_df = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(fh).rename(columns={'ds':'Date','yhat':'Forecast','yhat_lower':'Lower','yhat_upper':'Upper'})
                return fh_df
            raise RuntimeError("Unrecognized model in .pkl/.joblib")
        return predict_wrapper

    elif ext in (".h5",):
        try:
            from tensorflow.keras.models import load_model
            mdl = load_model(path)
        except Exception as e:
            raise RuntimeError("Could not load keras .h5 model (TensorFlow required).")
        def keras_wrapper(df, fh=30, lookback=20):
            series = df['Adj Close'].values
            if len(series) < lookback:
                raise RuntimeError("Not enough rows for keras wrapper.")
            window = series[-lookback:]
            minv, maxv = window.min(), window.max()
            norm = (window - minv) / (maxv - minv + 1e-9)
            seq = norm.reshape(1, lookback, 1)
            preds = []
            for i in range(fh):
                p = mdl.predict(seq, verbose=0)[0,0]
                preds.append(p)
                seq = np.concatenate([seq[:,1:,:], p.reshape(1,1,1)], axis=1)
            preds = np.array(preds) * (maxv - minv + 1e-9) + minv
            dates = [df['Date'].max() + timedelta(days=i+1) for i in range(fh)]
            return pd.DataFrame({"Date": dates, "Forecast": preds})
        return keras_wrapper

    elif ext in (".pt", ".pth"):
        try:
            import torch
        except Exception:
            raise RuntimeError("PyTorch required to load .pt/.pth")
        torch_model = torch.load(path, map_location=torch.device('cpu'))
        def torch_wrapper(df, fh=30):
            raise RuntimeError("PyTorch model loaded but automatic wrapper not implemented.")
        return torch_wrapper

    elif ext == ".csv":
        def csv_wrapper(df, fh=30):
            ff = pd.read_csv(path, parse_dates=["Date"])
            return ff
        return csv_wrapper

    else:
        raise RuntimeError("Unsupported model extension: " + ext)

# -----------------------------
# Streamlit UI layout (requested)
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Images", "Sentiment Analysis", "Technical Analysis", "Time Series", "Models & Forecast"])



# Overview
# Overview
if page == "Overview":
    st.title("Stock Prediction Project")

    # ---------- Author block ----------
    st.markdown(
        """
**Author:** Nipun Varshneya  
**Email:** varshneya.nipun16@gmail.com  
**Phone:** +91 7303615333  
**GitHub:** https://github.com/NipunVar/Stock_Analyzer
""",
        unsafe_allow_html=False,
    )
    # ----------------------------------

    st.markdown("""
## Project description
The aim of the project is to investigate the performance of various machine learning models to predict stock market movements based on historical time series data and news article sentiment collected using APIs and web scraping. The basic principle would be to buy low and sell high, but the complexity arises in knowing when to buy and sell a stock.

Four types of analysis exist to forecast the markets - fundamental, technical, quantitative and sentiment - each with its own underlying principles, tools, techniques and strategies, and it is likely that understanding the intuition of each and combining complementary approaches is more optimal than relying solely on one. Forecasting strategies will be developed based on predictions and backtested against a benchmark.

The FTSE 100 share index comprises the top 100 blue chip companies by market capitalisation listed on the London Stock Exchange, the primary stock exchange in the UK. As London’s benchmark index it is both representative of the UK’s stock market and an economic bellwether for the global economy given the international exposure of most of its constituents. This study focuses on data from six of the top FTSE 100 companies (AstraZeneca, GlaxoSmithKline, BP, Royal Dutch Shell, HSBC and Unilever) representing a range of sectors including oil, pharmaceuticals, finance and consumer.

## Exploratory Data Analysis
The [yfinance API](https://github.com/ranaroussi/yfinance) will be used to download stock data for opening price (Open), highest and lowest price the stock traded at (High, Low), closing price (Close), number of stocks traded (Volume) and Adjusted Close. For the most part the Adjusted Close price will be selected for prediction purposes to take into account all corporate actions, such as stock splits and dividends, to give a more accurate reflection of the true value of the stock and present a coherent picture of returns.

Data will be transformed to calculate and visualise returns, and covariance and correlation matrices will show strength and direction of the relationship between stocks' returns. These observations could be used to select a portfolio of stocks that complement each other in terms of price movement.

## Technical Analysis
Technical analysis is the use of charts and technical indicators to identify trading signals and price patterns. Various technical strategies will be investigated using the most common leading and lagging trend, momentum, volatility and volume indicators including Moving Averages, Moving Average Convergence Divergence (MACD), Stochastic Oscillator, Relative Strength Index (RSI), Money Flow Index (MFI), Rate of Change (ROC), Bollinger Bands, and On-Balance Volume (OBV).

## Time Series
A time series is basically a series of data points ordered in time and is an important factor in predicting stock market trends. In time series forecasting models, time is the independent variable and the goal is to predict future values based on previously observed values.

Stock prices are often non-stationary and may contain trends or volatility but different transformations can be applied to turn the time series into a stationary process so that it can be modelled.

The Augmented Dickey-Fuller (ADF) test will be used to check for stationarity, and the order of differencing required to make the series stationary will be determined.
Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots will show whether transformations have removed seasonality and any non-stationary behaviours - a necessary step before focusing on autoregressive time series models.
Models to be evaluated will include Moving Averages, Auto-Regressive Integrated Moving Average (ARIMA), Seasonal Auto-Regressive Integrated Moving Average (SARIMA) and Facebook Prophet.

Recurrent Neural Network (RNN) models such as Simple RNN, Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) will also be explored and various machine learning and deep learning models created, trained, tested and optimised.

## Sentiment Analysis
News articles will be collected from [Investing.com ](https://uk.investing.com/) by web scraping using Selenium and Beautiful Soup. Sentiment analysis will then be performed using NLP tools such as NLTK's VADER and TextBlob to find sentiment scores before combining the results with historical stock price data to determine whether news sentiment influences stock price direction.

## Algorithms and techniques
Predicting the stock market will be posed both as a regression problem of price prediction to forecast prices 'n' days in the future, and a classification problem of direction prediction to forecast whether prices will increase or decrease.

The X matrix of features will comprise any additional features engineered from the Adj Close price. For the regression problem, the y vector of the target variable will be the Adjusted Close price offset by however many days in advance we want to predict. For the classification problem it will be Buy and Sell signals, or 1 if the price will increase 'n' days in the future, and 0 if it will decrease, respectively.

To avoid look-ahead bias when splitting time series data into training and test sets sklearn's TimeSeriesSplit() class will be used. Successive training sets are supersets of those that come before them so that the model is not trained on data it has already seen. To use randomised data rather than walk-forward validation would lead to overfitting.

Pipelines will be built, and various Gradient-Descent based, Distance-Based and Tree-Based regression and classifier models spot checked, before selecting the best performing models for optimisation using Grid Search cross-validation, and hyperparameter tuning.

## Data sources

* [Yahoo! Finance](https://uk.finance.yahoo.com/) for historical stock data.
* [Investing.com ](https://uk.investing.com/) for market news articles.

## Python libraries

* Numpy
* Pandas
* Matplotlib
* Mplfinance
* Seaborn
* Plotly
* SciPy
* Statsmodels
* Scikit-learn
* Keras
* TensorFlow
* Yfinance
* Beautiful Soup
* Selenium
* NLTK
* TextBlob
* SpaCy
* Gensim
* BERT
* Hugging Face
* PyTorch
""")

# Images (single page showing selected images only)
elif page == "Images":
    st.title("Images")

    imgs = list_all_images(PROJECT_ROOT)
    if not imgs:
        st.info("No images found. Place PNG/JPG images somewhere under the project folder.")
    else:
        # present filenames relative to project root
        display_names = [os.path.relpath(p, PROJECT_ROOT) for p in imgs]

        # single or multiple selection toggle (kept as requested)
        sel_mode = st.radio("Selection mode", ("Single", "Multiple"), index=0, horizontal=True, key="img_sel_mode_simple")

        if sel_mode == "Single":
            choice = st.selectbox("Select an image to preview", options=display_names, index=0, key="img_select_single_simple")
            if choice:
                path = os.path.join(PROJECT_ROOT, choice)
                try:
                    with open(path, "rb") as f:
                        img = Image.open(f)
                        st.image(img, caption=os.path.basename(path), use_container_width=True)
                        # need to re-open for download since file pointer is at end after Image.open
                        with open(path, "rb") as fh:
                            st.download_button("Download image", data=fh.read(), file_name=os.path.basename(path), mime="image/png")
                except Exception as e:
                    st.error(f"Could not open image {path}: {e}")
        else:
            choices = st.multiselect("Select image(s) to preview", options=display_names, default=None, key="img_select_multi_simple")
            if choices:
                cols = st.columns(2)
                for i, rel in enumerate(choices):
                    path = os.path.join(PROJECT_ROOT, rel)
                    col = cols[i % 2]
                    try:
                        with open(path, "rb") as f:
                            img = Image.open(f)
                            with col:
                                st.image(img, caption=os.path.basename(path), use_container_width=True)
                                with open(path, "rb") as fh:
                                    st.download_button(f"Download {os.path.basename(path)}", data=fh.read(), file_name=os.path.basename(path), mime="image/png")
                    except Exception as e:
                        with col:
                            st.write(f"Could not open image {path}: {e}")

# Sentiment Analysis page
# Sentiment Analysis page (replace existing block with this)
elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    st.markdown("Notebooks inside the `Sentiment_Analysis` folder. Below each filename you will see a short overview of the notebook and a download button.")

    # locate the folder (searches recursively)
    folder_path = find_folder_path("Sentiment_Analysis", PROJECT_ROOT)
    if not folder_path:
        st.info("No folder named `Sentiment_Analysis` found under the project root.")
    else:
        nbs = list_notebooks_in_folder("Sentiment_Analysis", PROJECT_ROOT)
        if not nbs:
            st.info("No notebooks found inside `Sentiment_Analysis`.")
        else:
            # map of filename -> brief overview text (edit as you like)
            overview_map = {
                "BERT_Long_Text_Classification.ipynb":
                    "Overview: Demonstrates fine-tuning and using a BERT-based model for long text classification. "
                    "Includes data preprocessing, tokenization adapted for long documents, model training, and evaluation metrics (accuracy, F1).",
                "NLP_Text_Preprocessing_and_Classification.ipynb":
                    "Overview: Covers standard NLP preprocessing (tokenization, stopword removal, lemmatization), feature extraction (TF-IDF, word embeddings), "
                    "and baseline classifiers (Logistic Regression, SVM). Useful as a preprocessing and baseline modelling reference.",
                "Sentiment_Analysis_and_Classifiers.ipynb":
                    "Overview: End-to-end sentiment analysis pipeline: loads news/text data, creates labels, trains multiple classifiers, "
                    "and compares model performance using confusion matrices and classification reports.",
                "Stock_news_data_collection.ipynb":
                    "Overview: Scripts and examples to collect stock-related news data from web sources and APIs, cleanup raw text, and save datasets for downstream NLP tasks."
            }

            # show each notebook with a short overview and download button
            for fp in nbs:
                fname = os.path.basename(fp)
                st.subheader(fname)
                # show the overview text (use filename mapping; fallback when missing)
                overview_text = overview_map.get(fname, "Overview: (no summary available).")
                st.markdown(overview_text)

                # show metadata optionally (size)
                try:
                    size_bytes = os.path.getsize(fp)
                    size_kb = size_bytes / 1024.0
                    st.caption(f"Size: {size_kb:.1f} KB")
                except Exception:
                    pass

                # download button
                try:
                    with open(fp, "rb") as fh:
                        st.download_button(
                            label=f"Download {fname}",
                            data=fh,
                            file_name=fname,
                            mime="application/x-ipynb+json"
                        )
                except Exception:
                    st.write(" (cannot download — permission or path issue)")
                st.markdown("---")  # separator between notebooks


# Technical Analysis page
# Technical Analysis page (replace existing Technical Analysis block with this)
elif page == "Technical Analysis":
    st.title("Technical Analysis")
    st.markdown("Notebooks inside the `Technical_Analysis` folder. Below each filename you will see a short overview of the notebook and a download button.")

    # locate the folder (searches recursively)
    folder_path = find_folder_path("Technical_Analysis", PROJECT_ROOT)
    if not folder_path:
        st.info("No folder named `Technical_Analysis` found under the project root.")
    else:
        nbs = list_notebooks_in_folder("Technical_Analysis", PROJECT_ROOT)
        if not nbs:
            st.info("No notebooks found inside `Technical_Analysis`.")
        else:
            # brief overviews; edit text as needed
            overview_map = {
                "Chart_patterns_and_technical_indicators.ipynb":
                    "Overview: Explains common chart patterns (head & shoulders, double top/bottom, flags) and computes technical indicators "
                    "(SMA, EMA, RSI, MACD) with plotting examples. Useful for feature engineering for trading strategies.",
                "FTSE100_data_collection_and_EDA.ipynb":
                    "Overview: Demonstrates downloading FTSE100 stock data, performs exploratory data analysis (EDA), visualizations, and basic summary statistics.",
                "Hypothesis_Testing.ipynb":
                    "Overview: Applies statistical hypothesis testing to financial time series — t-tests, A/B style comparisons, and stationarity checks.",
                "Trading_Dashboards.ipynb":
                    "Overview: Shows how to create summary dashboards for trading signals and portfolio metrics (returns, drawdown, position sizing), with plotting examples."
            }

            for fp in nbs:
                fname = os.path.basename(fp)
                st.subheader(fname)
                overview_text = overview_map.get(fname, "Overview: (no summary available).")
                st.markdown(overview_text)
                try:
                    size_bytes = os.path.getsize(fp)
                    size_kb = size_bytes / 1024.0
                    st.caption(f"Size: {size_kb:.1f} KB")
                except Exception:
                    pass
                try:
                    with open(fp, "rb") as fh:
                        st.download_button(
                            label=f"Download {fname}",
                            data=fh,
                            file_name=fname,
                            mime="application/x-ipynb+json"
                        )
                except Exception:
                    st.write(" (cannot download — permission or path issue)")
                st.markdown("---")

# Time Series page
# Time Series page (replace existing Time Series block with this)
elif page == "Time Series":
    st.title("Time Series")
    st.markdown("Notebooks inside the `Time_Series` folder. Below each filename you will see a short overview of the notebook and a download button.")

    folder_path = find_folder_path("Time_Series", PROJECT_ROOT)
    if not folder_path:
        st.info("No folder named `Time_Series` found under the project root.")
    else:
        nbs = list_notebooks_in_folder("Time_Series", PROJECT_ROOT)
        if not nbs:
            st.info("No notebooks found inside `Time_Series`.")
        else:
            overview_map = {
                "ARIMA.ipynb":
                    "Overview: Walkthrough of ARIMA/SARIMA modelling for stock prices — model identification (ACF/PACF), differencing, fitting and forecasting.",
                "Classifier_Models.ipynb":
                    "Overview: Implements classification models used in trading (e.g., predicting up/down moves). Compares algorithms and presents performance metrics.",
                "Facebook_Prophet.ipynb":
                    "Overview: Uses Prophet for time series forecasting, shows changepoint detection, model fitting, and forecast visualizations.",
                "LSTM.ipynb":
                    "Overview: Demonstrates LSTM networks for time series prediction — data windowing, normalization, model architecture, training and evaluation.",
                "Regression_Models.ipynb":
                    "Overview: Applies regression approaches (Linear, Ridge, LASSO) for trend modelling and baseline forecasts. Includes feature engineering examples.",
                "RNN_LSTM_GRU.ipynb":
                    "Overview: Compares recurrent architectures (simple RNN, LSTM, GRU) on sequence prediction tasks and highlights differences in performance and training behaviour.",
                "SARIMA.ipynb":
                    "Overview: Seasonal ARIMA modelling with seasonal decomposition, diagnostics, and multi-step forecasting examples.",
                "Time_Series_Machine_Learning_and_Deep_Learning.ipynb":
                    "Overview: Higher-level summary of ML & deep learning techniques for time series (feature-based learners, tree ensembles, sequence models), with examples and best practices."
            }

            for fp in nbs:
                fname = os.path.basename(fp)
                st.subheader(fname)
                overview_text = overview_map.get(fname, "Overview: (no summary available).")
                st.markdown(overview_text)
                try:
                    size_bytes = os.path.getsize(fp)
                    size_kb = size_bytes / 1024.0
                    st.caption(f"Size: {size_kb:.1f} KB")
                except Exception:
                    pass
                try:
                    with open(fp, "rb") as fh:
                        st.download_button(
                            label=f"Download {fname}",
                            data=fh,
                            file_name=fname,
                            mime="application/x-ipynb+json"
                        )
                except Exception:
                    st.write(" (cannot download — permission or path issue)")
                st.markdown("---")

# Models & Forecast (same as before)
elif page == "Models & Forecast":
    st.title("Models & Forecast")
    st.markdown("Upload CSV (`Date,Adj Close`) or use `data/sample_stock.csv`. Load pretrained models from `models/` or train quick demos here.")

    # -----------------------------
    # 100-entry demo dataset + quick model suite
    # -----------------------------
    st.subheader("Quick demo: create a 100-row sample dataset and run models")
    st.markdown("This will generate a synthetic 100-day price series and run several quick forecasting methods (persistence, moving average, linear regression). You can set forecast horizon below.")

    demo_fh = st.number_input("Demo forecast horizon (days)", min_value=1, max_value=365, value=30, key="demo_fh")
    if st.button("Create 100-row demo dataset & run models", key="create_demo_100"):
        import math
        from sklearn.linear_model import LinearRegression

        # create synthetic 100-row dataset (dates + price signal with trend + seasonality + noise)
        end_date = pd.Timestamp.today().normalize()
        dates = pd.date_range(end=end_date - pd.Timedelta(days=99), periods=100, freq="D")
        np.random.seed(42)
        trend = np.linspace(50, 70, 100)                       # gentle upward trend
        season = 2 * np.sin(np.linspace(0, 6 * math.pi, 100))  # seasonality
        noise = np.random.normal(scale=1.5, size=100)          # noise
        prices = trend + season + noise

        demo_df = pd.DataFrame({"Date": dates, "Adj Close": prices})
        st.success("Demo dataset created (100 rows).")
        st.subheader("Demo data preview")
        st.dataframe(demo_df.tail(10))

        st.subheader("Demo series plot")
        st.pyplot(plot_series(demo_df, title="Synthetic 100-day series"))

        # prepare forecast dates
        last_date = demo_df["Date"].max()
        fh = int(demo_fh)
        future_dates = [last_date + timedelta(days=i+1) for i in range(fh)]

        # 1) Persistence forecast (last value repeated)
        pers_preds = persistence_forecast(demo_df["Adj Close"], fh=fh)
        pers_df = pd.DataFrame({"Date": future_dates, "Forecast": pers_preds})
        # 2) Moving average forecast (window=5)
        ma_preds = moving_average_forecast(demo_df["Adj Close"], window=5, fh=fh)
        ma_df = pd.DataFrame({"Date": future_dates, "Forecast": ma_preds})
        # 3) Linear regression on time index
        try:
            X = (demo_df.index.values).reshape(-1, 1)
            y = demo_df["Adj Close"].values
            lr = LinearRegression()
            lr.fit(X, y)
            next_idx = np.arange(demo_df.index[-1] + 1, demo_df.index[-1] + 1 + fh).reshape(-1, 1)
            lr_preds = lr.predict(next_idx)
            lr_df = pd.DataFrame({"Date": future_dates, "Forecast": lr_preds})
        except Exception as e:
            lr_df = pd.DataFrame({"Date": future_dates, "Forecast": np.repeat(np.nan, fh)})
            st.warning(f"Linear regression failed: {e}")

        # 4) ARIMA (try if statsmodels available)
        arima_df = None
        try:
            mean, ci = train_arima_forecast(demo_df["Adj Close"], order=(1,1,1), fh=fh)
            dates_ar = future_dates
            arima_df = pd.DataFrame({"Date": dates_ar, "Forecast": mean.values})
            if ci is not None:
                arima_df["Lower"] = ci.iloc[:,0].values
                arima_df["Upper"] = ci.iloc[:,1].values
        except Exception as e:
            st.info(f"ARIMA unavailable or failed: {e}")

        # 5) Prophet (try if installed)
        prophet_df = None
        try:
            prophet_df = train_prophet_forecast(demo_df, fh=fh)
        except Exception as e:
            st.info(f"Prophet unavailable or failed: {e}")

        # show results in expanders
        model_results = {
            "Persistence": pers_df,
            "Moving Average (w=5)": ma_df,
            "Linear Regression (time index)": lr_df
        }
        if arima_df is not None:
            model_results["ARIMA (1,1,1)"] = arima_df
        if prophet_df is not None:
            model_results["Prophet"] = prophet_df

        # Combined visual: overlay actual + each forecast (one plot per model)
        for name, fdf in model_results.items():
            with st.expander(f"Model: {name} — show forecast and table", expanded=False):
                st.markdown(f"**Model:** {name}")
                # table
                st.dataframe(fdf.head(fh))
                # plot overlay
                try:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(demo_df["Date"], demo_df["Adj Close"], label="Actual")
                    ax.plot(fdf["Date"], fdf["Forecast"], label="Forecast")
                    if "Lower" in fdf.columns and "Upper" in fdf.columns:
                        ax.fill_between(fdf["Date"], fdf["Lower"], fdf["Upper"], alpha=0.2)
                    ax.set_title(f"{name} forecast")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.write("Plot failed:", e)
                # download button
                csv_data = fdf.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download {name} forecast CSV", csv_data, file_name=f"{name.replace(' ', '_')}_forecast.csv", mime="text/csv")

        # also show a combined CSV (stacked)
        combined = []
        for name, fdf in model_results.items():
            tmp = fdf.copy()
            tmp["Model"] = name
            combined.append(tmp)
        combined_df = pd.concat(combined, ignore_index=True)
        st.subheader("All model outcomes (stacked)")
        st.dataframe(combined_df.head(200))
        st.download_button("Download all outcomes CSV", combined_df.to_csv(index=False).encode("utf-8"), "all_model_outcomes.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload CSV (Date, Adj Close)", type=["csv"])
    sample_path = os.path.join(PROJECT_ROOT, "data", "sample_stock.csv")
    use_sample = False
    if uploaded_file is None and os.path.exists(sample_path):
        if st.button("Use sample_stock.csv"):
            uploaded_file = sample_path
            use_sample = True

    df = None
    if uploaded_file is not None:
        try:
            df = read_stock_csv_obj(uploaded_file)
            st.success("CSV loaded.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

    if df is not None:
        st.subheader("Data preview")
        st.dataframe(df.tail(10))
        st.subheader("Series plot")
        st.pyplot(plot_series(df, title="Adjusted Close"))

        model_files = list_models(os.path.join(PROJECT_ROOT, "models"))
        st.subheader("Pretrained models in `models/`")
        if model_files:
            for m in model_files:
                st.write("- ", os.path.relpath(m, PROJECT_ROOT))
        else:
            st.info("No pretrained model files found in `models/` under project root.")

        if model_files:
            chosen = st.selectbox("Choose pretrained model (or None)", options=["None"] + model_files)
            if chosen and chosen != "None":
                model_path = chosen
                st.write("Loading:", os.path.relpath(model_path, PROJECT_ROOT))
                try:
                    predict_fn = load_model_from_file(model_path)
                    fh = st.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=30)
                    if st.button("Run pretrained model forecast"):
                        with st.spinner("Running model..."):
                            try:
                                pred_df = predict_fn(df, fh=fh)
                                st.subheader("Forecast result")
                                st.dataframe(pred_df.head(20))
                                plot_with_forecast(df, pred_df, title=f"Forecast from {os.path.basename(model_path)}")
                                st.download_button("Download forecast CSV", pred_df.to_csv(index=False).encode("utf-8"), "forecast.csv", "text/csv")
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
                except Exception as e:
                    st.error(f"Could not load model: {e}")

        st.subheader("Train demo models")
        choice = st.selectbox("Train", ["None", "ARIMA", "Prophet", "LSTM"])
        if choice != "None":
            fh = st.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=30, key="fh_ui2")
            if choice == "ARIMA":
                p = st.number_input("p", min_value=0, max_value=5, value=1)
                d = st.number_input("d", min_value=0, max_value=2, value=1)
                q = st.number_input("q", min_value=0, max_value=5, value=1)
                if st.button("Train ARIMA"):
                    with st.spinner("Fitting ARIMA..."):
                        try:
                            mean, ci = train_arima_forecast(df["Adj Close"], order=(p,d,q), fh=fh)
                            last_date = df["Date"].max()
                            dates = [last_date + timedelta(days=i+1) for i in range(fh)]
                            fdf = pd.DataFrame({"Date": dates, "Forecast": mean.values})
                            if ci is not None:
                                fdf["Lower"] = ci.iloc[:,0].values
                                fdf["Upper"] = ci.iloc[:,1].values
                            st.dataframe(fdf.head(20))
                            plot_with_forecast(df, fdf, title=f"ARIMA (p={p},d={d},q={q})")
                            st.download_button("Download ARIMA CSV", fdf.to_csv(index=False).encode("utf-8"), "arima_forecast.csv", "text/csv")
                        except Exception as e:
                            st.error(f"ARIMA failed: {e}")
            elif choice == "Prophet":
                if st.button("Train Prophet"):
                    with st.spinner("Fitting Prophet..."):
                        try:
                            fdf = train_prophet_forecast(df, fh=fh)
                            st.dataframe(fdf.head(20))
                            plot_with_forecast(df, fdf, title="Prophet forecast")
                            st.download_button("Download Prophet CSV", fdf.to_csv(index=False).encode("utf-8"), "prophet_forecast.csv", "text/csv")
                        except Exception as e:
                            st.error(f"Prophet failed: {e}")
            elif choice == "LSTM":
                lookback = st.number_input("LSTM lookback", min_value=3, max_value=60, value=20)
                epochs = st.number_input("Epochs", min_value=1, max_value=50, value=5)
                if st.button("Train LSTM"):
                    with st.spinner("Training LSTM..."):
                        try:
                            fdf = train_small_lstm_forecast(df, fh=fh, lookback=lookback, epochs=epochs)
                            st.dataframe(fdf.head(20))
                            plot_with_forecast(df, fdf, title="LSTM forecast")
                            st.download_button("Download LSTM CSV", fdf.to_csv(index=False).encode("utf-8"), "lstm_forecast.csv", "text/csv")
                        except Exception as e:
                            st.error(f"LSTM failed: {e}")
