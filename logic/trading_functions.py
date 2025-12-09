import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def create_individual_portfolio_charts():
    """Create separate interactive charts for each stock in portfolio"""

    print("📈 Creating individual closing price charts for each stock...")
    colors = px.colors.qualitative.Set3
    successful_charts = 0
    for i, (ticker_symbol, shares) in enumerate(portfolio.items()):
        print(f"📊 Processing {i+1}/{len(portfolio)}: {ticker_symbol.upper()}...", end=" ")
        try:
            # Get stock data
            data = yf.download(
                ticker_symbol.upper(),
                period=period,
                interval=interval,
                progress=False
            )
            if not data.empty:
                # Ensure datetime index and 'Date' column
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]
                data = data.reset_index()
                if 'Date' not in data.columns:
                    # yfinance sometimes uses 'index' as the date column
                    data.rename(columns={data.columns[0]: 'Date'}, inplace=True)
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                print(data[['Date']].head())  # Check the first few dates
                # Create figure
                fig = go.Figure()
                # Add closing price trace
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'].dt.strftime('%Y-%m-%d'),
                        y=data['Close'],
                        name=f"{ticker_symbol.upper()}",
                        line=dict(color=colors[i % len(colors)], width=3),
                        mode='lines',
                        hovertemplate=f"<b>{ticker_symbol.upper()}</b><br>" +
                                      "Date: %{x}<br>" +
                                      "Close: $%{y:.2f}<br>" +
                                      f"Position: {shares} shares<br>" +
                                      "Value: $%{{customdata:.2f}}<br>" +
                                      "<extra></extra>",
                        customdata=data['Close'] * shares
                    )
                )
                fig.update_layout(
                    title=f"{ticker_symbol.upper()} Closing Price",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_white"
                )
                pio.show(fig)
                successful_charts += 1
                print("✅")
            else:
                print("❌ No data")
        except Exception as e:
            print(f"❌ Error: {str(e)[:60]}...")
    print(f"\n📊 Successfully created {successful_charts}/{len(portfolio)} individual charts")

def rsi_calculator(df, period=14):
    """Add RSI column to a DataFrame with 'Close' and 'Open' columns."""
    gain = (df['Close'] - df['Open']).clip(lower=0)
    loss = (df['Open'] - df['Close']).clip(lower=0)
    ema_gain = gain.ewm(span=period, min_periods=period).mean()
    ema_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = ema_gain / ema_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    return df

# Position class for trades opened/closed during the backtest
class Position:
    def __init__(self, open_datetime, open_price, order_type, volume, sl, tp):
        self.open_datetime = open_datetime
        self.open_price = open_price
        self.order_type = order_type
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.close_datetime = None
        self.close_price = None
        self.profit = None
        self.status = 'open'

    def close_position(self, close_datetime, close_price):
        self.close_datetime = close_datetime
        self.close_price = close_price
        if self.order_type == 'buy':
            self.profit = (self.close_price - self.open_price) * self.volume
        else:
            self.profit = (self.open_price - self.close_price) * self.volume
        self.status = 'closed'

    def _asdict(self):
        return {
            'open_datetime': self.open_datetime,
            'open_price': self.open_price,
            'order_type': self.order_type,
            'volume': self.volume,
            'sl': self.sl,
            'tp': self.tp,
            'close_datetime': self.close_datetime,
            'close_price': self.close_price,
            'profit': self.profit,
            'status': self.status,
        }

# Strategy class for backtesting logic
class Strategy:
    def __init__(self, df, starting_balance):
        self.starting_balance = starting_balance
        self.positions = []
        self.data = df

    def get_positions_df(self):
        if not self.positions:
            return pd.DataFrame()

        df = pd.DataFrame([position._asdict() for position in self.positions])

        # Replace None with 0.0 or NaN
        if 'profit' not in df.columns:
            df['profit'] = 0.0
        else:
            df['profit'] = df['profit'].fillna(0.0)
        df['pnl'] = df['profit'].cumsum() + self.starting_balance
        return df


    def add_position(self, position):
        self.positions.append(position)
        return True

    def close_tp_sl(self, data):
        for pos in self.positions:
            if pos.status == 'open':
                # Use 'Close' and 'Date' columns from your DataFrame
                if (pos.sl >= data['Close'] and pos.order_type == 'buy'):
                    pos.close_position(data['Date'], pos.sl)
                elif (pos.sl <= data['Close'] and pos.order_type == 'sell'):
                    pos.close_position(data['Date'], pos.sl)
                elif (pos.tp <= data['Close'] and pos.order_type == 'buy'):
                    pos.close_position(data['Date'], pos.tp)
                elif (pos.tp >= data['Close'] and pos.order_type == 'sell'):
                    pos.close_position(data['Date'], pos.tp)

    def has_open_positions(self):
        return any(pos.status == 'open' for pos in self.positions)

    def logic(self, data):
        if not self.has_open_positions():
            # Only open trades if ATR and RSI are not NaN
            if pd.notna(data['atr_14']) and pd.notna(data['rsi_14']):
                if data['rsi_14'] < 30:
                    open_datetime = data['Date']
                    open_price = data['Close']
                    order_type = 'buy'
                    volume = 10000
                    sl = open_price - 2 * data['atr_14']
                    tp = open_price + 2 * data['atr_14']
                    self.add_position(Position(open_datetime, open_price, order_type, volume, sl, tp))
                elif data['rsi_14'] > 70:
                    open_datetime = data['Date']
                    open_price = data['Close']
                    order_type = 'sell'
                    volume = 10000
                    sl = open_price + 2 * data['atr_14']
                    tp = open_price - 2 * data['atr_14']
                    self.add_position(Position(open_datetime, open_price, order_type, volume, sl, tp))

    def run(self):
        for i, data in self.data.iterrows():
            self.close_tp_sl(data)
            self.logic(data)
        return self.get_positions_df()
    
    def __init__(self, open_datetime, open_price, order_type, volume, sl, tp):
        self.open_datetime = open_datetime
        self.open_price = open_price
        self.order_type = order_type
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.close_datetime = None
        self.close_price = None
        self.profit = None
        self.status = 'open'

    def close_position(self, close_datetime, close_price):
        self.close_datetime = close_datetime
        self.close_price = close_price
        if self.order_type == 'buy':
            self.profit = (self.close_price - self.open_price) * self.volume
        else:
            self.profit = (self.open_price - self.close_price) * self.volume
        self.status = 'closed'

    def _asdict(self):
        return {
            'open_datetime': self.open_datetime,
            'open_price': self.open_price,
            'order_type': self.order_type,
            'volume': self.volume,
            'sl': self.sl,
            'tp': self.tp,
            'close_datetime': self.close_datetime,
            'close_price': self.close_price,
            'profit': self.profit,
            'status': self.status,
        }
    
def bayesian_rsi_signal(rsi_value):
    # Prior: equal probability for each action
    probs = {'buy': 1/3, 'hold': 1/3, 'sell': 1/3}
    # Likelihoods based on RSI value
    if rsi_value < 30:
        probs['buy'] *= 0.8
        probs['hold'] *= 0.15
        probs['sell'] *= 0.05
    elif rsi_value > 70:
        probs['buy'] *= 0.05
        probs['hold'] *= 0.15
        probs['sell'] *= 0.8
    else:
        probs['buy'] *= 0.15
        probs['hold'] *= 0.7
        probs['sell'] *= 0.15
    # Normalize
    total = sum(probs.values())
    for k in probs:
        probs[k] /= total
    # Choose the action with the highest probability
    signal = max(probs, key=probs.get)
    return signal, probs

def gbm(S0, mu, sigma, T = 1., N = 10, M= 1000): 
    
    dt = T/ float(N) 
    S= np.array([S0]*(N+1)*M, dtype='float32').reshape(N+1, M)  
       
    for i in range(N):      
        dS = S[i,]*(mu*dt +  sigma*np.sqrt(dt)*np.random.randn(M))
        S[i+1,]=S[i,] + dS 
    
    return S

def bsformula(cp, s, k, rf, t, v, div):
        """ Price an option using the Black-Scholes model.
        cp: +1/-1 for call/put
        s: initial stock price
        k: strike price
        t: expiration time
        v: volatility
        rf: risk-free rate
        div: dividend
        """

        d1 = (np.log(s/k)+(rf-div+0.5*v*v)*t)/(v*np.sqrt(t))
        d2 = d1 - v*np.sqrt(t)

        optprice = (cp*s*np.exp(-div*t)*st.norm.cdf(cp*d1)) - (cp*k*np.exp(-rf*t)*st.norm.cdf(cp*d2))
        delta = cp*st.norm.cdf(cp*d1)
        vega  = s*np.sqrt(t)*st.norm.pdf(d1)
        return optprice, delta, vega

def update_beliefs_with_data(features, targets, prior_beliefs, prior_uncertainty, data_noise_level):
    """
    Updates prior beliefs about model parameters using Bayesian Linear Regression.

    Arguments:
    - features: Matrix of input data (n_samples x n_features)
    - targets: Vector of observed outcomes (length n_samples)
    - prior_beliefs: Our initial guess for the model parameters (length n_features)
    - prior_uncertainty: How uncertain we are about those initial guesses (matrix)
    - data_noise_level: Variance in the observed data (scalar, e.g., market noise)

    Returns:
    - updated_beliefs: New expected values for the model parameters
    - updated_uncertainty: New uncertainty (covariance) about the parameters
    """
    # Ensure all inputs have the correct shape
    features = np.atleast_2d(features)
    targets = np.atleast_1d(targets)
    prior_beliefs = np.atleast_1d(prior_beliefs)

    # Invert prior uncertainty to express it in "precision" (confidence)
    prior_precision = np.linalg.inv(prior_uncertainty)

    # Combine prior confidence with the data’s information content
    total_precision = prior_precision + (1 / data_noise_level) * features.T @ features
    updated_uncertainty = np.linalg.inv(total_precision)

    # Combine prior expectations and observed targets into a weighted update
    weighted_prior = prior_precision @ prior_beliefs
    weighted_data = (1 / data_noise_level) * features.T @ targets
    updated_beliefs = updated_uncertainty @ (weighted_prior + weighted_data)

    return updated_beliefs, updated_uncertainty

def g1(x, theta):
    # If x > 0 => theta; else => theta + 0.35
    return np.where(x > 0, theta, theta + 0.35)

def g0(x, theta):
    return 1 - g1(x, theta)

def likelihood(G, X, theta):
    probs = g1(X, theta)**G * g0(X, theta)**(1 - G)
    return np.prod(probs)

def log_likelihood(G, X, theta):
    g1_vals = g1(X, theta)
    g0_vals = g0(X, theta)
    return np.sum(G * np.log(g1_vals) + (1 - G) * np.log(g0_vals))

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    """
    Radial Basis Function (RBF) kernel — measures similarity between points.
    
    Args:
        X1, X2: Input arrays of shape (n_samples, n_features)
        length_scale: Controls how far the influence of a point extends
        variance: Controls the height (vertical stretch) of the kernel

    Returns:
        Kernel matrix of shape (n_samples_X1, n_samples_X2)
    """
    # Compute pairwise squared Euclidean distances
    sqdist = (
        np.sum(X1**2, axis=1).reshape(-1, 1) +  # (n, 1)
        np.sum(X2**2, axis=1) -                # (m,)
        2 * X1 @ X2.T                          # (n, m)
    )
    
    # Apply the RBF formula
    return variance * np.exp(-0.5 * sqdist / length_scale**2)

def gp_predict(X_train, y_train, X_test, noise_var=1e-4, kernel_func=rbf_kernel, **kernel_params):
    """
    Predict the output at new inputs using a Gaussian Process model.
    
    Args:
        X_train: Known input features (n_train, d)
        y_train: Observed outputs (n_train,)
        X_test: New input points to predict at (n_test, d)
        noise_var: Observation noise variance σ²_n
        kernel_func: Function to compute similarity (defaults to RBF)
        kernel_params: Extra hyperparameters like length_scale, variance
    
    Returns:
        pred_mean: Predicted mean at test points (n_test,)
        pred_var: Predictive variance (uncertainty) at test points (n_test,)
    """
    # 1. Compute the covariance (similarity) between training points
    K_train = kernel_func(X_train, X_train, **kernel_params)
    K_train += noise_var * np.eye(len(X_train))  # Add noise to diagonal for numerical stability

    # 2. Compute the covariance between test and training points
    K_cross = kernel_func(X_test, X_train, **kernel_params)

    # 3. Compute the covariance among test points
    K_test = kernel_func(X_test, X_test, **kernel_params)

    # 4. Invert the training covariance matrix
    K_train_inv = np.linalg.inv(K_train)

    # 5. Compute the predicted mean: how much each train point pulls on the test point
    pred_mean = K_cross @ K_train_inv @ y_train

    # 6. Compute uncertainty in prediction (posterior covariance)
    pred_cov = K_test - K_cross @ K_train_inv @ K_cross.T
    pred_var = np.diag(pred_cov)  # Only care about variance along the diagonal

    return pred_mean, pred_var

def display_portfolio_ohlcv_heads(ticker, period, interval):

    """
    Display OHLCV data heads for each individual stock in portfolio
    """
    from IPython.display import display  # <-- Add this import
    print("📊 PORTFOLIO OHLCV DATA - Individual Stock Analysis")
    print("=" * 70)
    successful_stocks = 0
    failed_stocks = 0

    for i, stock_ticker in enumerate(ticker):
        print(f"\n🏢 {i+1}/{len(ticker)}: {stock_ticker.upper()}")
        print("-" * 50)
        try:
            stock_data = yf.download(
                stock_ticker.upper(),
                period=period,
                interval=interval,
                progress=False
            )
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = [col[0] for col in stock_data.columns]
            if not stock_data.empty and len(stock_data) > 0:
                print(f"📈 Data Shape: {stock_data.shape}")
                print(f"📅 Date Range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
                print(f"📋 Columns: {list(stock_data.columns)}")
                print(f"\n📊 {stock_ticker.upper()} - Last 5 Days OHLCV:")
                display_data = stock_data.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']]
                display(display_data)  # <-- Use display() for DataFrame
                latest = stock_data.iloc[-1]
                print(f"\n💰 Latest Trading Day:")
                print(f"   Open: ${latest['Open']:8.2f}")
                print(f"   High: ${latest['High']:8.2f}")
                print(f"   Low:  ${latest['Low']:8.2f}")
                print(f"   Close:${latest['Close']:8.2f}")
                try:
                    volume_display = f"{latest['Volume']:,}" if pd.notna(latest['Volume']) and latest['Volume'] > 0 else "N/A"
                    print(f"   Volume: {volume_display}")
                except (ValueError, TypeError):
                    print(f"   Volume: N/A")
                if len(stock_data) > 1:
                    prev_close = stock_data.iloc[-2]['Close']
                    daily_change = latest['Close'] - prev_close
                    daily_change_pct = (daily_change / prev_close) * 100
                    print(f"   Daily Change: ${daily_change:+.2f} ({daily_change_pct:+.2f}%)")
                    successful_stocks += 1
                else:
                    print(f"❌ No data available for {stock_ticker.upper()}")
                    failed_stocks += 1
            else:
                print(f"❌ No data available for {stock_ticker.upper()}")
                failed_stocks += 1
        except Exception as e:
            print(f"❌ Error fetching {stock_ticker.upper()}: {str(e)}")
            failed_stocks += 1
    print("\n" + "=" * 70)
    print(f"📊 SUMMARY: ✅ {successful_stocks} successful, ❌ {failed_stocks} failed")
    print(f"✅ Successfully analyzed {successful_stocks} out of {len(ticker)} stocks")

def analyze_volatility_regimes(ticker_list, period, interval, window):
    """
    For each ticker, computes rolling volatility, splits into high/low regimes,
    estimates P(up | high/low volatility) using Bayesian inference, and plots histogram.
    great for understanding how volatility affects market movements. and really useful for trading strategies.
    bc it helps you see how often the market goes up or down when volatility is high or low.
    """
    import yfinance as yf
    import matplotlib.pyplot as plt
    import numpy as np

    ticker_upper = [t.upper() for t in ticker_list]
    data = yf.download(ticker_upper, period=period, interval=interval, group_by='ticker')

    for symbol in ticker_upper:
        print(f"\n=== {symbol} ===")
        df = data[symbol].copy().dropna()
        if df.empty:
            print("No data for this ticker.")
            continue

        df['DailyReturn'] = df['Close'].pct_change()
        df = df.dropna()
        df['RollingVolatility'] = df['DailyReturn'].rolling(window=window).std()
        median_volatility_value = df['RollingVolatility'].median()
        df['IsHighVolatility'] = (df['RollingVolatility'] > median_volatility_value).astype(int)
        df['IsUpDay'] = (df['DailyReturn'] > 0).astype(int)

        up_days_when_high_volatility = df[df['IsHighVolatility'] == 1]['IsUpDay']
        up_days_when_low_volatility= df[df['IsHighVolatility'] == 0]['IsUpDay']

        # Bayesian inference with Beta(1,1) prior
        up_high_volatility = up_days_when_high_volatility.sum() + 1
        down_high_volatility = len(up_days_when_high_volatility) - up_days_when_high_volatility.sum() + 1
        up_low_volatility = up_days_when_low_volatility.sum() + 1
        down_low_volatility = len(up_days_when_low_volatility) - up_days_when_low_volatility.sum() + 1

        probability_up_with_high_volatility = up_high_volatility / (up_high_volatility + down_high_volatility)
        probability_up_with_low_volatility = up_low_volatility / (up_low_volatility + down_low_volatility)

        print(f"P(up | high volatility): {probability_up_with_high_volatility:.2%}")
        print(f"P(up | low volatility): {probability_up_with_low_volatility:.2%}")

        plt.hist(df['RollingVolatility'].dropna(), bins=30, alpha=0.5, label='Volatility')
        plt.axvline(median_volatility_value, color='red', linestyle='--', label='Median Volatility')
        plt.xlabel('Volatility (std. dev. of returns)')
        plt.ylabel('Frequency (number of days)')
        plt.legend()
        plt.title(f'Volatility Regimes for {symbol}')
        plt.show()

def forecast_next_day_return(ticker, period="200d", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty or len(df) < 20:
        print(f"Not enough data for {ticker}")
        return
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df = rsi_calculator(df)
    df['RSI'] = df['rsi_14']

    for lag in range(1, 6):
        df[f'Return_lag{lag}'] = df['Return'].shift(lag)
    
    df = df.dropna()

    # 🧱 Defensive: ensure enough data remains
    if df.empty or len(df) < 5:
        print(f"⚠️ Not enough usable data after cleaning for {ticker}")
        return

    feature_cols = [f'Return_lag{lag}' for lag in range(1, 6)] + ['Volatility', 'RSI']
    X = df[feature_cols].values
    y = df['Return'].values
    prior_beliefs = np.zeros(X.shape[1])
    prior_uncertainty = np.eye(X.shape[1]) * 0.1
    data_noise_level = 0.01
    posterior_mean, posterior_cov = update_beliefs_with_data(X, y, prior_beliefs, prior_uncertainty, data_noise_level)
    y_pred = X @ posterior_mean
    y_std = np.sqrt(np.sum(X @ posterior_cov * X, axis=1) + data_noise_level)

    # ⚠️ Another check before plotting/using results
    if len(y_pred) == 0:
        print(f"⚠️ Empty prediction output for {ticker}")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, y, label='Actual Return')
    plt.plot(df.index, y_pred, label='Predicted Return')
    plt.fill_between(df.index, y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2, label='95% CI')
    plt.title(f'{ticker}: Bayesian Linear Regression Next-Day Return Forecast (with RSI)')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.show()

    last_pred = y_pred[-1]
    last_std = y_std[-1]
    lower, upper = last_pred - 2*last_std, last_pred + 2*last_std

    rsi_value = df['RSI'].iloc[-1]
    rsi_signal, rsi_probs = bayesian_rsi_signal(rsi_value)

    if lower > 0 and rsi_signal == "buy":
        advice = "STRONG BUY"
    elif upper < 0 and rsi_signal == "sell":
        advice = "STRONG SELL"
    elif rsi_signal == "hold":
        advice = "HOLD"
    else:
        advice = "WEAK SIGNAL"

    print(f"{ticker} | {df.index[-1].date()} | Predicted Return: {last_pred:.4f} | 95% CI: [{lower:.4f}, {upper:.4f}] | Advice: {advice}")

def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands and z-score for price positioning.
    
    Returns:
        df with added columns: BB_Upper, BB_Middle, BB_Lower, BB_Z_Score, BB_Width
    """
    df = df.copy()
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (num_std * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (num_std * df['BB_Std'])
    
    # Z-score: how many standard deviations is current price from mean?
    df['BB_Z_Score'] = (df['Close'] - df['BB_Middle']) / (df['BB_Std'] + 1e-8)
    
    # Band width (volatility measure)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-8)
    
    return df


def unified_bayesian_gp_forecast(ticker, period="200d", interval="1d", num_lags=10):
    
    """
    Combined Bayesian Linear Regression + Gaussian Process forecasting system.
    Uses both approaches and creates an ensemble prediction with uncertainty quantification.
    
    NEW: Incorporates Bollinger Bands z-scores as features for better signal quality.
    """
    # Download and prepare data
    df = yf.download(ticker, period=period, interval=interval)
    if df.empty or len(df) < max(20, num_lags + 1):
        print(f"❌ Not enough data for {ticker}")
        return None
    
    # Feature engineering
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df = rsi_calculator(df)
    df['RSI'] = df['rsi_14']
    
    # Add Bollinger Bands and z-scores
    df = calculate_bollinger_bands(df, window=20, num_std=2)
    
    # Add Bollinger Bands and z-scores
    df = calculate_bollinger_bands(df, window=20, num_std=2)
    
    # Create lagged features for both models
    for lag in range(1, max(6, num_lags + 1)):
        df[f'Return_lag{lag}'] = df['Return'].shift(lag)
    
    df = df.dropna()
    
    if df.empty or len(df) < 10:
        print(f"⚠️ Not enough usable data after cleaning for {ticker}")
        return None
    
    # ==============================
    # 🧠 METHOD 1: BAYESIAN LINEAR REGRESSION (Enhanced with BB z-scores)
    # ==============================
    print(f"\n🔵 Running Bayesian Linear Regression for {ticker}...")
    
    # Bayesian features: lagged returns + volatility + RSI + Bollinger Band metrics
    bayesian_features = [f'Return_lag{lag}' for lag in range(1, 6)] + ['Volatility', 'RSI', 'BB_Z_Score', 'BB_Width']
    X_bayesian = df[bayesian_features].values
    y_bayesian = df['Return'].values
    
    # Bayesian inference
    prior_beliefs = np.zeros(X_bayesian.shape[1])
    prior_uncertainty = np.eye(X_bayesian.shape[1]) * 0.1
    data_noise_level = 0.01
    
    posterior_mean, posterior_cov = update_beliefs_with_data(
        X_bayesian, y_bayesian, prior_beliefs, prior_uncertainty, data_noise_level
    )
    
    # Bayesian predictions
    y_pred_bayesian = X_bayesian @ posterior_mean
    y_std_bayesian = np.sqrt(np.sum(X_bayesian @ posterior_cov * X_bayesian, axis=1) + data_noise_level)
    
    # Latest Bayesian prediction
    bayesian_forecast = y_pred_bayesian[-1]
    bayesian_std = y_std_bayesian[-1]
    bayesian_ci_lower = bayesian_forecast - 2 * bayesian_std
    bayesian_ci_upper = bayesian_forecast + 2 * bayesian_std
    
    # ==============================
    # 🟠 METHOD 2: GAUSSIAN PROCESS
    # ==============================
    print(f"🟠 Running Gaussian Process for {ticker}...")
    
    # GP features: just lagged returns (pure time series)
    gp_features = [f'Return_lag{i}' for i in range(1, num_lags + 1)]
    X_gp = df[gp_features].values
    y_gp = df['Return'].values
    
    # GP training/testing split
    X_gp_train, y_gp_train = X_gp[:-1], y_gp[:-1]
    X_gp_test = X_gp[-1:].reshape(1, -1)
    
    # GP prediction using custom function
    gp_pred_mean, gp_pred_var = gp_predict(
        X_gp_train, y_gp_train, X_gp_test,
        noise_var=1e-4,
        kernel_func=rbf_kernel,
        length_scale=1.0,
        variance=1.0
    )
    
    # GP results
    gp_forecast = gp_pred_mean[0]
    gp_std = np.sqrt(gp_pred_var[0])
    gp_ci_lower = gp_forecast - 2 * gp_std
    gp_ci_upper = gp_forecast + 2 * gp_std
    
    # ==============================
    # 📈 RSI TECHNICAL SIGNAL + BOLLINGER BANDS
    # ==============================
    current_rsi = df['RSI'].iloc[-1]
    rsi_signal, rsi_probs = bayesian_rsi_signal(current_rsi)
    
    # Bollinger Band position signal
    bb_z = df['BB_Z_Score'].iloc[-1]
    if bb_z < -1.5:
        bb_signal = "buy"  # Price near lower band (oversold)
    elif bb_z > 1.5:
        bb_signal = "sell"  # Price near upper band (overbought)
    else:
        bb_signal = "hold"
    
    # ==============================
    # 🎯 ENSEMBLE COMBINATION (Enhanced with BB signal)
    # ==============================
    
    # Weight models by their uncertainty (inverse variance weighting)
    bayesian_weight = 1 / (bayesian_std**2 + 1e-6)
    gp_weight = 1 / (gp_std**2 + 1e-6)
    total_weight = bayesian_weight + gp_weight
    
    # Ensemble prediction
    ensemble_forecast = (
        (bayesian_weight * bayesian_forecast + gp_weight * gp_forecast) / total_weight
    )
    
    # Combined uncertainty (conservative approach)
    ensemble_std = np.sqrt(
        (bayesian_weight * (bayesian_std**2 + bayesian_forecast**2) + 
         gp_weight * (gp_std**2 + gp_forecast**2)) / total_weight - ensemble_forecast**2
    )
    
    ensemble_ci_lower = ensemble_forecast - 2 * ensemble_std
    ensemble_ci_upper = ensemble_forecast + 2 * ensemble_std
    
    # ==============================
    # 🎪 TRADING SIGNAL GENERATION (Enhanced with BB)
    # ==============================
    
    def generate_ensemble_signal(bayesian_pred, gp_pred, ensemble_pred, rsi_signal, bb_signal):
        """Generate trading signal from ensemble of models including Bollinger Bands"""
        
        # Count bullish signals
        signals = []
        if bayesian_ci_lower > 0: signals.append("bayesian_bull")
        elif bayesian_ci_upper < 0: signals.append("bayesian_bear")
        
        if gp_ci_lower > 0: signals.append("gp_bull")
        elif gp_ci_upper < 0: signals.append("gp_bear")
        
        if ensemble_ci_lower > 0: signals.append("ensemble_bull")
        elif ensemble_ci_upper < 0: signals.append("ensemble_bear")
        
        # RSI signal
        if rsi_signal == "buy": signals.append("rsi_bull")
        elif rsi_signal == "sell": signals.append("rsi_bear")
        
        # Bollinger Band signal
        if bb_signal == "buy": signals.append("bb_bull")
        elif bb_signal == "sell": signals.append("bb_bear")
        
        # Count bullish vs bearish
        bull_count = len([s for s in signals if "bull" in s])
        bear_count = len([s for s in signals if "bear" in s])
        
        if bull_count >= 3:
            return "STRONG BUY", 0.9
        elif bull_count >= 2:
            return "BUY", 0.75
        elif bear_count >= 3:
            return "STRONG SELL", 0.9
        elif bear_count >= 2:
            return "SELL", 0.75
        else:
            return "HOLD", 0.5
    
    final_signal, confidence = generate_ensemble_signal(
        bayesian_forecast, gp_forecast, ensemble_forecast, rsi_signal, bb_signal
    )
    
    # ==============================
    # 📊 VISUALIZATION
    # ==============================
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Model Predictions Comparison
    plt.subplot(3, 2, 1)
    recent_dates = df.index[-50:]  # Last 50 days
    recent_returns = df['Return'].iloc[-50:].values
    recent_bayesian = y_pred_bayesian[-50:]
    
    plt.plot(recent_dates, recent_returns, 'k-', label='Actual Returns', alpha=0.7)
    plt.plot(recent_dates, recent_bayesian, 'b-', label='Bayesian Prediction', alpha=0.8)
    plt.fill_between(recent_dates, 
                     recent_bayesian - 2*y_std_bayesian[-50:],
                     recent_bayesian + 2*y_std_bayesian[-50:],
                     color='blue', alpha=0.2, label='Bayesian 95% CI')
    plt.title(f'{ticker}: Bayesian Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: GP Performance on recent data
    plt.subplot(3, 2, 2)
    gp_pred_recent, gp_std_recent = gp_predict(
        X_gp_train[-30:], y_gp_train[-30:], X_gp_train[-30:],
        noise_var=1e-4, kernel_func=rbf_kernel, length_scale=1.0, variance=1.0
    )
    
    plt.plot(recent_dates[-30:], y_gp_train[-30:], 'k-', label='Actual', alpha=0.7)
    plt.plot(recent_dates[-30:], gp_pred_recent, 'orange', label='GP Prediction', alpha=0.8)
    plt.fill_between(recent_dates[-30:], 
                     gp_pred_recent - 2*gp_std_recent,
                     gp_pred_recent + 2*gp_std_recent,
                     color='orange', alpha=0.2, label='GP 95% CI')
    plt.title(f'{ticker}: Gaussian Process')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: RSI
    plt.subplot(3, 2, 3)
    plt.plot(recent_dates, df['RSI'].iloc[-50:], 'purple', linewidth=2)
    plt.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    plt.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    plt.axhline(current_rsi, color='orange', linewidth=3, label=f'Current: {current_rsi:.1f}')
    plt.title(f'RSI (Signal: {rsi_signal.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Ensemble Forecast Comparison
    plt.subplot(3, 2, 4)
    methods = ['Bayesian', 'GP', 'Ensemble']
    forecasts = [bayesian_forecast, gp_forecast, ensemble_forecast]
    uncertainties = [bayesian_std, gp_std, ensemble_std]
    colors = ['blue', 'orange', 'green']
    
    bars = plt.bar(methods, forecasts, color=colors, alpha=0.7)
    plt.errorbar(methods, forecasts, yerr=[2*u for u in uncertainties], 
                fmt='none', color='black', capsize=5)
    plt.title('Forecast Comparison')
    plt.ylabel('Predicted Return')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, forecast, std in zip(bars, forecasts, uncertainties):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                f'{forecast:.4f}', ha='center', va='bottom')
    
    # Plot 5: Signal Confidence
    plt.subplot(3, 2, 5)
    signal_colors = {'STRONG BUY': 'darkgreen', 'BUY': 'green', 
                    'HOLD': 'gray', 'SELL': 'red', 'STRONG SELL': 'darkred'}
    
    plt.bar(['Final Signal'], [confidence], 
           color=signal_colors.get(final_signal, 'gray'), alpha=0.8)
    plt.ylim(0, 1)
    plt.title(f'Trading Signal: {final_signal}')
    plt.ylabel('Confidence')
    
    # Plot 6: Uncertainty Comparison
    plt.subplot(3, 2, 6)
    plt.bar(methods, [2*u for u in uncertainties], color=colors, alpha=0.7)
    plt.title('Model Uncertainty (95% CI width)')
    plt.ylabel('Uncertainty Range')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ==============================
    # 📋 RESULTS SUMMARY
    # ==============================
    
    print(f"\n📊 {ticker} - Unified Forecast Summary ({df.index[-1].date()})")
    print("=" * 70)
    print(f"🔵 Bayesian Forecast: {bayesian_forecast:.4f} ± {2*bayesian_std:.4f}")
    print(f"🟠 GP Forecast:       {gp_forecast:.4f} ± {2*gp_std:.4f}")
    print(f"🟢 Ensemble Forecast: {ensemble_forecast:.4f} ± {2*ensemble_std:.4f}")
    print(f"📈 RSI Signal:        {rsi_signal.upper()} (RSI: {current_rsi:.1f})")
    print(f"📊 BB Signal:         {bb_signal.upper()} (Z-Score: {bb_z:.2f})")
    print(f"🎯 Final Signal:      {final_signal} (Confidence: {confidence:.1%})")
    print(f"💰 Recommendation:    {final_signal}")
    
    return {
        'ticker': ticker,
        'date': df.index[-1].date(),
        'bayesian': {
            'forecast': bayesian_forecast,
            'std': bayesian_std,
            'ci': (bayesian_ci_lower, bayesian_ci_upper)
        },
        'gp': {
            'forecast': gp_forecast,
            'std': gp_std,
            'ci': (gp_ci_lower, gp_ci_upper)
        },
        'ensemble': {
            'forecast': ensemble_forecast,
            'std': ensemble_std,
            'ci': (ensemble_ci_lower, ensemble_ci_upper),
            'z_score': ensemble_forecast / (ensemble_std + 1e-8)  # Z-score of forecast
        },
        'rsi': {
            'value': current_rsi,
            'signal': rsi_signal
        },
        'bollinger_bands': {
            'z_score': bb_z,
            'signal': bb_signal,
            'upper': df['BB_Upper'].iloc[-1],
            'middle': df['BB_Middle'].iloc[-1],
            'lower': df['BB_Lower'].iloc[-1]
        },
        'final_signal': final_signal,
        'confidence': confidence,
        'final_confidence': confidence,  # Alias for paper_trading.ipynb compatibility
        'model_weights': {
            'bayesian': bayesian_weight / total_weight,
            'gp': gp_weight / total_weight
        }
    }
