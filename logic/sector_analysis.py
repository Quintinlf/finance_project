"""
Sector Analysis Module

Analyzes sector ETFs to determine which sectors are worth investing in.
Provides automated sector rotation strategy recommendations.

Features:
- Download and analyze sector ETF data (XLK, XLF, XLE, etc.)
- Generate forecasts for each sector using Bayesian/GP models
- Rank sectors by attractiveness
- Auto-trade best sectors with user confirmation

Major S&P 500 Sector ETFs (SPDR):
- XLK: Technology
- XLF: Financials  
- XLE: Energy
- XLV: Health Care
- XLI: Industrials
- XLY: Consumer Discretionary
- XLP: Consumer Staples
- XLU: Utilities
- XLRE: Real Estate
- XLC: Communication Services
- XLB: Materials
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path


# Import forecasting functions
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# Sector ETF definitions
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
    'XLB': 'Materials'
}


def analyze_all_sectors(
    period: str = "200d",
    min_confidence: float = 0.65,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze all major sector ETFs and rank by investment attractiveness.
    
    Args:
        period: Historical data period (default "200d")
        min_confidence: Minimum confidence threshold for BUY signals
        verbose: Print detailed output
    
    Returns:
        DataFrame with sector rankings, sorted by score
    """
    from trading_functions import unified_bayesian_gp_forecast
    
    results = []
    
    if verbose:
        print("="*70)
        print("🌐 SECTOR ANALYSIS - S&P 500 Sectors")
        print("="*70)
    
    for etf, name in SECTOR_ETFS.items():
        if verbose:
            print(f"\n📊 Analyzing {etf} ({name})...")
        
        try:
            # Run forecast
            forecast = unified_bayesian_gp_forecast(etf, period=period, interval="1d")
            
            if forecast is None:
                if verbose:
                    print(f"   ⚠️ Skipping {etf} - insufficient data")
                continue
            
            # Extract metrics
            signal = forecast.get('final_signal', 'HOLD')
            confidence = forecast.get('confidence', 0.0)
            ensemble_forecast = forecast['ensemble']['forecast']
            ensemble_z = forecast['ensemble'].get('z_score', 0.0)
            rsi_value = forecast['rsi']['value']
            bb_z_score = forecast['bollinger_bands']['z_score']
            
            # Calculate attractiveness score (0-100)
            score = 0
            
            # Component 1: Signal strength (0-40 points)
            if signal in ['STRONG BUY', 'BUY']:
                score += confidence * 40
            elif signal in ['STRONG SELL', 'SELL']:
                score -= confidence * 40
            
            # Component 2: Forecast magnitude (0-25 points)
            if ensemble_forecast > 0:
                score += min(ensemble_forecast * 500, 25)  # Cap at 25
            else:
                score += max(ensemble_forecast * 500, -25)
            
            # Component 3: Statistical edge (0-20 points)
            if abs(ensemble_z) > 0:
                score += min(abs(ensemble_z) * 10, 20) * np.sign(ensemble_forecast)
            
            # Component 4: Technical indicators (0-15 points)
            if rsi_value < 40:  # Oversold = good buy
                score += 15
            elif rsi_value > 60:  # Overbought = caution
                score -= 15
            
            # Normalize to 0-100
            score = max(0, min(100, score + 50))
            
            results.append({
                'etf': etf,
                'sector': name,
                'signal': signal,
                'confidence': confidence,
                'score': score,
                'forecast_return': ensemble_forecast,
                'z_score': ensemble_z,
                'rsi': rsi_value,
                'bb_z': bb_z_score,
                'meets_threshold': confidence >= min_confidence and signal in ['BUY', 'STRONG BUY']
            })
            
            if verbose:
                print(f"   Signal: {signal} ({confidence:.1%})")
                print(f"   Score: {score:.1f}/100")
                print(f"   Forecast: {ensemble_forecast:.4f}")
        
        except Exception as e:
            if verbose:
                print(f"   ❌ Error analyzing {etf}: {str(e)}")
            continue
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(results)
    
    if df.empty:
        if verbose:
            print("\n⚠️ No sectors successfully analyzed")
        return df
    
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    if verbose:
        print("\n" + "="*70)
        print("📈 SECTOR RANKINGS (Best to Worst)")
        print("="*70)
        print(df[['etf', 'sector', 'signal', 'confidence', 'score']].to_string(index=False))
        print("="*70)
    
    return df


def get_top_sectors(n: int = 3, min_confidence: float = 0.65) -> List[str]:
    """
    Get top N sectors to invest in.
    
    Args:
        n: Number of top sectors to return
        min_confidence: Minimum confidence for BUY signals
    
    Returns:
        List of ETF symbols (e.g., ['XLK', 'XLF', 'XLV'])
    """
    df = analyze_all_sectors(min_confidence=min_confidence, verbose=False)
    
    if df.empty:
        return []
    
    # Filter for BUY signals meeting threshold
    buy_sectors = df[df['meets_threshold']]
    
    if buy_sectors.empty:
        print("⚠️ No sectors meet BUY criteria")
        return []
    
    top_n = buy_sectors.head(n)
    
    return top_n['etf'].tolist()


def auto_trade_sectors(
    top_n: int = 3,
    min_confidence: float = 0.70,
    notional_per_sector: float = 500,
    require_confirmation: bool = True
) -> Dict:
    """
    Automated sector trading: analyze, rank, and trade top sectors.
    
    Args:
        top_n: Number of top sectors to trade
        min_confidence: Minimum confidence for trades
        notional_per_sector: Dollar amount to invest per sector
        require_confirmation: Ask user before executing trades
    
    Returns:
        Dict with trade results
    """
    from alpaca_exercises import connect_trading_client, market_order
    
    print("🤖 Automated Sector Trading")
    print("="*70)
    
    # Analyze sectors
    df = analyze_all_sectors(min_confidence=min_confidence, verbose=True)
    
    if df.empty:
        return {'status': 'error', 'message': 'No sectors analyzed successfully'}
    
    # Get top sectors
    buy_sectors = df[df['meets_threshold']].head(top_n)
    
    if buy_sectors.empty:
        return {'status': 'no_trades', 'message': 'No sectors meet BUY criteria'}
    
    print(f"\n🎯 Top {len(buy_sectors)} Sectors to Trade:")
    for _, row in buy_sectors.iterrows():
        print(f"   {row['etf']} ({row['sector']}): Score {row['score']:.1f}/100, Conf {row['confidence']:.1%}")
    
    # Confirmation
    if require_confirmation:
        print(f"\n💰 Will invest ${notional_per_sector:.2f} in each sector (Total: ${notional_per_sector * len(buy_sectors):.2f})")
        response = input("Proceed with trades? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("❌ Trades cancelled by user")
            return {'status': 'cancelled', 'message': 'User declined to proceed'}
    
    # Execute trades
    print("\n🚀 Executing trades...")
    trading_client = connect_trading_client()
    
    trade_results = []
    for _, row in buy_sectors.iterrows():
        etf = row['etf']
        
        try:
            # Place market order
            order = market_order(
                client=trading_client,
                symbol=etf,
                side="buy",
                notional=notional_per_sector
            )
            
            trade_results.append({
                'etf': etf,
                'status': 'success',
                'order_id': order.id if hasattr(order, 'id') else None
            })
            
            print(f"   ✅ {etf}: Order placed")
        
        except Exception as e:
            trade_results.append({
                'etf': etf,
                'status': 'error',
                'error': str(e)
            })
            print(f"   ❌ {etf}: Error - {str(e)}")
    
    print("="*70)
    
    return {
        'status': 'completed',
        'num_trades': len([r for r in trade_results if r['status'] == 'success']),
        'trades': trade_results
    }


def sector_momentum_score(etf: str, lookback_days: int = 20) -> float:
    """
    Calculate momentum score for a sector ETF.
    
    Args:
        etf: Sector ETF symbol
        lookback_days: Days to look back for momentum
    
    Returns:
        Momentum score (positive = uptrend, negative = downtrend)
    """
    df = yf.download(etf, period=f"{lookback_days + 10}d", progress=False)
    
    if df.empty or len(df) < lookback_days:
        return 0.0
    
    # Calculate rate of change
    close_now = df['Close'].iloc[-1]
    close_past = df['Close'].iloc[-lookback_days]
    
    roc = (close_now - close_past) / close_past * 100
    
    # Calculate trend strength (linear regression slope)
    x = np.arange(lookback_days)
    y = df['Close'].iloc[-lookback_days:].values
    
    slope = np.polyfit(x, y, 1)[0]
    normalized_slope = slope / close_past * 100  # As percentage
    
    # Combined momentum score
    momentum = (roc * 0.6) + (normalized_slope * 0.4)
    
    return float(momentum)


def compare_sector_momentum() -> pd.DataFrame:
    """
    Compare momentum across all sectors.
    
    Returns:
        DataFrame with momentum rankings
    """
    print("🔄 Calculating sector momentum...")
    
    results = []
    for etf, sector in SECTOR_ETFS.items():
        try:
            momentum = sector_momentum_score(etf, lookback_days=20)
            results.append({
                'etf': etf,
                'sector': sector,
                'momentum_20d': momentum
            })
        except Exception as e:
            print(f"⚠️ Error calculating momentum for {etf}: {e}")
    
    df = pd.DataFrame(results).sort_values('momentum_20d', ascending=False)
    
    print("\n📊 Sector Momentum Rankings:")
    print(df.to_string(index=False))
    
    return df
