"""
Prediction Backtesting and Historical Tracking System
=====================================================

This module stores historical predictions, compares them against actual outcomes,
and calculates rolling performance metrics (MAE, MAPE, directional accuracy).

Key Features:
- Store predictions in CSV with timestamps for audit trail
- Calculate rolling MAE/MAPE over different time windows
- Track which model (Bayesian/GP/Ensemble) performs best per stock
- Generate accuracy reports and trend analysis
- Enable adaptive model selection based on historical performance

Usage Example:
    from prediction_backtester import PredictionBacktester
    
    backtester = PredictionBacktester()
    
    # Store a prediction
    backtester.store_prediction(
        symbol='AAPL',
        date='2026-01-21',
        bayesian_pred=0.015,
        gp_pred=0.012,
        ensemble_pred=0.013,
        current_close=150.00
    )
    
    # Next day: record actual outcome
    backtester.update_actual_close('AAPL', '2026-01-21', actual_close=151.50)
    
    # Get performance metrics
    metrics = backtester.get_metrics_summary('AAPL')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from logic.evaluation_metrics import (
    calculate_mae, calculate_mape, calculate_rmse,
    calculate_directional_accuracy, calculate_all_metrics
)


class PredictionBacktester:
    """
    Manages historical predictions and calculates rolling performance metrics.
    """
    
    def __init__(self, storage_dir: str = "prediction_history"):
        """
        Initialize the backtester with storage directory.
        
        Args:
            storage_dir: Directory to store prediction CSV files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.predictions_file = self.storage_dir / "predictions.csv"
        self.metrics_cache_file = self.storage_dir / "metrics_cache.json"
        
        # Load existing predictions if available
        self.predictions_df = self._load_predictions()
    
    def _load_predictions(self) -> pd.DataFrame:
        """Load predictions from CSV file."""
        if self.predictions_file.exists():
            df = pd.read_csv(self.predictions_file, parse_dates=['date', 'timestamp'])
            return df
        else:
            # Create empty DataFrame with schema
            return pd.DataFrame(columns=[
                'symbol', 'date', 'timestamp',
                'current_close', 'actual_close',
                'bayesian_return_pred', 'gp_return_pred', 'ensemble_return_pred',
                'bayesian_price_pred', 'gp_price_pred', 'ensemble_price_pred',
                'bayesian_ci_lower', 'bayesian_ci_upper',
                'gp_ci_lower', 'gp_ci_upper',
                'ensemble_ci_lower', 'ensemble_ci_upper',
                'final_signal', 'confidence',
                'rsi_value', 'bb_z_score',
                'actual_updated'
            ])
    
    def _save_predictions(self):
        """Save predictions DataFrame to CSV."""
        self.predictions_df.to_csv(self.predictions_file, index=False)
    
    def store_prediction(self,
                        symbol: str,
                        date: str,
                        current_close: float,
                        bayesian_return: float,
                        gp_return: float,
                        ensemble_return: float,
                        bayesian_price: float,
                        gp_price: float,
                        ensemble_price: float,
                        bayesian_ci: Tuple[float, float] = None,
                        gp_ci: Tuple[float, float] = None,
                        ensemble_ci: Tuple[float, float] = None,
                        final_signal: str = None,
                        confidence: float = None,
                        rsi_value: float = None,
                        bb_z_score: float = None) -> bool:
        """
        Store a new prediction in the database.
        
        Args:
            symbol: Stock ticker symbol
            date: Prediction date (YYYY-MM-DD)
            current_close: Current closing price at prediction time
            bayesian_return: Bayesian model's predicted return
            gp_return: GP model's predicted return
            ensemble_return: Ensemble model's predicted return
            bayesian_price: Bayesian model's predicted closing price
            gp_price: GP model's predicted closing price
            ensemble_price: Ensemble model's predicted closing price
            bayesian_ci: Bayesian confidence interval (lower, upper)
            gp_ci: GP confidence interval
            ensemble_ci: Ensemble confidence interval
            final_signal: Trading signal (BUY/SELL/HOLD)
            confidence: Signal confidence (0-1)
            rsi_value: RSI indicator value
            bb_z_score: Bollinger Bands z-score
            
        Returns:
            bool: True if stored successfully
        """
        # Check if prediction already exists for this symbol/date
        existing = self.predictions_df[
            (self.predictions_df['symbol'] == symbol) &
            (self.predictions_df['date'] == date)
        ]
        
        if len(existing) > 0:
            print(f"⚠️  Prediction already exists for {symbol} on {date}, skipping")
            return False
        
        # Create new row
        new_row = {
            'symbol': symbol,
            'date': pd.to_datetime(date),
            'timestamp': datetime.now(),
            'current_close': current_close,
            'actual_close': np.nan,  # Will be filled later
            'bayesian_return_pred': bayesian_return,
            'gp_return_pred': gp_return,
            'ensemble_return_pred': ensemble_return,
            'bayesian_price_pred': bayesian_price,
            'gp_price_pred': gp_price,
            'ensemble_price_pred': ensemble_price,
            'bayesian_ci_lower': bayesian_ci[0] if bayesian_ci else np.nan,
            'bayesian_ci_upper': bayesian_ci[1] if bayesian_ci else np.nan,
            'gp_ci_lower': gp_ci[0] if gp_ci else np.nan,
            'gp_ci_upper': gp_ci[1] if gp_ci else np.nan,
            'ensemble_ci_lower': ensemble_ci[0] if ensemble_ci else np.nan,
            'ensemble_ci_upper': ensemble_ci[1] if ensemble_ci else np.nan,
            'final_signal': final_signal,
            'confidence': confidence,
            'rsi_value': rsi_value,
            'bb_z_score': bb_z_score,
            'actual_updated': False
        }
        
        # Append to DataFrame
        self.predictions_df = pd.concat([
            self.predictions_df,
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        # Save to disk
        self._save_predictions()
        
        return True
    
    def store_prediction_from_forecast(self, forecast_result: Dict) -> bool:
        """
        Store prediction directly from unified_bayesian_gp_forecast() output.
        
        Args:
            forecast_result: Dictionary output from unified_bayesian_gp_forecast()
            
        Returns:
            bool: True if stored successfully
        """
        symbol = forecast_result['ticker']
        date = str(forecast_result['date'])
        current_close = forecast_result['current_close']
        
        # Extract return predictions
        bayesian_return = forecast_result['bayesian']['forecast']
        gp_return = forecast_result['gp']['forecast']
        ensemble_return = forecast_result['ensemble']['forecast']
        
        # Extract price predictions
        bayesian_price = forecast_result['bayesian']['next_day_close']
        gp_price = forecast_result['gp']['next_day_close']
        ensemble_price = forecast_result['ensemble']['next_day_close']
        
        # Extract confidence intervals (returns)
        bayesian_ci = forecast_result['bayesian']['ci']
        gp_ci = forecast_result['gp']['ci']
        ensemble_ci = forecast_result['ensemble']['ci']
        
        # Extract other features
        final_signal = forecast_result['final_signal']
        confidence = forecast_result['confidence']
        rsi_value = forecast_result['rsi']['value']
        bb_z_score = forecast_result['bollinger_bands']['z_score']
        
        return self.store_prediction(
            symbol=symbol,
            date=date,
            current_close=current_close,
            bayesian_return=bayesian_return,
            gp_return=gp_return,
            ensemble_return=ensemble_return,
            bayesian_price=bayesian_price,
            gp_price=gp_price,
            ensemble_price=ensemble_price,
            bayesian_ci=bayesian_ci,
            gp_ci=gp_ci,
            ensemble_ci=ensemble_ci,
            final_signal=final_signal,
            confidence=confidence,
            rsi_value=rsi_value,
            bb_z_score=bb_z_score
        )
    
    def update_actual_close(self, symbol: str, date: str, actual_close: float) -> bool:
        """
        Update the actual closing price for a prediction.
        
        Args:
            symbol: Stock ticker
            date: Prediction date
            actual_close: Actual closing price observed
            
        Returns:
            bool: True if updated successfully
        """
        mask = (self.predictions_df['symbol'] == symbol) & \
               (self.predictions_df['date'] == pd.to_datetime(date))
        
        if mask.sum() == 0:
            print(f"⚠️  No prediction found for {symbol} on {date}")
            return False
        
        # Update actual close and mark as updated
        self.predictions_df.loc[mask, 'actual_close'] = actual_close
        self.predictions_df.loc[mask, 'actual_updated'] = True
        
        # Save to disk
        self._save_predictions()
        
        return True
    
    def bulk_update_actuals(self, lookback_days: int = 30) -> int:
        """
        Bulk update actual closing prices for recent predictions using yfinance.
        
        Args:
            lookback_days: Number of days to look back for updates
            
        Returns:
            int: Number of predictions updated
        """
        import yfinance as yf
        
        # Get predictions that need updating
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        needs_update = self.predictions_df[
            (self.predictions_df['actual_updated'] == False) &
            (self.predictions_df['date'] >= cutoff_date)
        ].copy()
        
        if len(needs_update) == 0:
            print("✅ All recent predictions already have actual prices")
            return 0
        
        updated_count = 0
        
        # Group by symbol to minimize API calls
        for symbol in needs_update['symbol'].unique():
            symbol_preds = needs_update[needs_update['symbol'] == symbol]
            
            # Get historical data for this symbol
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{lookback_days + 5}d")
                
                if hist.empty:
                    continue
                
                # Update each prediction
                for idx, row in symbol_preds.iterrows():
                    pred_date = pd.to_datetime(row['date'])
                    # Actual close is the NEXT day's close after prediction
                    next_date = pred_date + timedelta(days=1)
                    
                    # Find the actual close (may be a few days later due to weekends)
                    future_data = hist[hist.index > pred_date]
                    
                    if len(future_data) > 0:
                        actual_close = future_data['Close'].iloc[0]
                        self.update_actual_close(symbol, str(pred_date.date()), actual_close)
                        updated_count += 1
            
            except Exception as e:
                print(f"⚠️  Error updating {symbol}: {e}")
                continue
        
        print(f"✅ Updated {updated_count} predictions with actual prices")
        return updated_count
    
    def calculate_errors(self, symbol: Optional[str] = None, 
                        model: str = 'ensemble') -> pd.DataFrame:
        """
        Calculate prediction errors for completed predictions.
        
        Args:
            symbol: Filter by symbol (None = all symbols)
            model: Which model to evaluate ('bayesian', 'gp', 'ensemble')
            
        Returns:
            DataFrame with errors calculated
        """
        # Filter to predictions with actual values
        completed = self.predictions_df[self.predictions_df['actual_updated'] == True].copy()
        
        if symbol:
            completed = completed[completed['symbol'] == symbol]
        
        if len(completed) == 0:
            return pd.DataFrame()
        
        # Select the appropriate prediction column
        pred_col = f"{model}_price_pred"
        
        if pred_col not in completed.columns:
            raise ValueError(f"Model '{model}' not found in predictions")
        
        # Calculate errors
        completed['price_error'] = completed['actual_close'] - completed[pred_col]
        completed['abs_error'] = np.abs(completed['price_error'])
        completed['pct_error'] = (completed['price_error'] / completed['actual_close']) * 100
        completed['abs_pct_error'] = np.abs(completed['pct_error'])
        
        # Calculate actual and predicted returns
        completed['actual_return'] = (completed['actual_close'] - completed['current_close']) / completed['current_close']
        completed['predicted_return'] = completed[f"{model}_return_pred"]
        completed['return_error'] = completed['actual_return'] - completed['predicted_return']
        
        # Direction accuracy
        completed['actual_direction'] = np.sign(completed['actual_return'])
        completed['predicted_direction'] = np.sign(completed['predicted_return'])
        completed['direction_correct'] = (completed['actual_direction'] == completed['predicted_direction'])
        
        return completed
    
    def get_metrics_summary(self, symbol: Optional[str] = None,
                           window_days: int = None) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive metrics summary for all models.
        
        Args:
            symbol: Filter by symbol (None = all symbols)
            window_days: Rolling window in days (None = all time)
            
        Returns:
            dict: Metrics for each model
        """
        results = {}
        
        for model in ['bayesian', 'gp', 'ensemble']:
            errors_df = self.calculate_errors(symbol=symbol, model=model)
            
            if len(errors_df) == 0:
                results[model] = {'error': 'No completed predictions'}
                continue
            
            # Apply time window filter
            if window_days:
                cutoff_date = datetime.now() - timedelta(days=window_days)
                errors_df = errors_df[errors_df['date'] >= cutoff_date]
            
            if len(errors_df) == 0:
                results[model] = {'error': f'No predictions in last {window_days} days'}
                continue
            
            # Calculate metrics
            actual_prices = errors_df['actual_close'].values
            predicted_prices = errors_df[f'{model}_price_pred'].values
            actual_returns = errors_df['actual_return'].values
            predicted_returns = errors_df['predicted_return'].values
            
            metrics = {
                'n_predictions': len(errors_df),
                'mae': calculate_mae(actual_prices, predicted_prices),
                'rmse': calculate_rmse(actual_prices, predicted_prices),
                'mape': calculate_mape(actual_prices, predicted_prices),
                'directional_accuracy': calculate_directional_accuracy(actual_returns, predicted_returns),
                'avg_price_error': errors_df['price_error'].mean(),
                'avg_abs_error': errors_df['abs_error'].mean(),
                'avg_pct_error': errors_df['pct_error'].mean(),
                'avg_abs_pct_error': errors_df['abs_pct_error'].mean(),
                'correct_directions': errors_df['direction_correct'].sum(),
                'total_predictions': len(errors_df)
            }
            
            results[model] = metrics
        
        return results
    
    def print_metrics_report(self, symbol: Optional[str] = None, window_days: int = None):
        """
        Print a formatted metrics report.
        
        Args:
            symbol: Filter by symbol
            window_days: Rolling window in days
        """
        metrics = self.get_metrics_summary(symbol=symbol, window_days=window_days)
        
        symbol_str = f" for {symbol}" if symbol else " (All Symbols)"
        window_str = f" - Last {window_days} days" if window_days else " - All Time"
        
        print("=" * 80)
        print(f"  📊 MODEL PERFORMANCE REPORT{symbol_str}{window_str}")
        print("=" * 80)
        
        for model_name, model_metrics in metrics.items():
            print(f"\n🔹 {model_name.upper()} MODEL")
            print("-" * 80)
            
            if 'error' in model_metrics:
                print(f"   {model_metrics['error']}")
                continue
            
            print(f"   Predictions:          {model_metrics['n_predictions']}")
            print(f"   MAE:                  ${model_metrics['mae']:.4f}")
            print(f"   RMSE:                 ${model_metrics['rmse']:.4f}")
            print(f"   MAPE:                 {model_metrics['mape']:.2f}%")
            print(f"   Directional Accuracy: {model_metrics['directional_accuracy']:.1f}%")
            print(f"   Avg Price Error:      ${model_metrics['avg_price_error']:.4f}")
            print(f"   Avg Abs % Error:      {model_metrics['avg_abs_pct_error']:.2f}%")
            
            # Rating
            if model_metrics['directional_accuracy'] >= 60:
                rating = "✅ EXCELLENT"
            elif model_metrics['directional_accuracy'] >= 55:
                rating = "✅ GOOD"
            elif model_metrics['directional_accuracy'] >= 52:
                rating = "⚠️  MARGINAL"
            else:
                rating = "❌ POOR"
            
            print(f"   Rating:               {rating}")
        
        print("\n" + "=" * 80)
    
    def get_best_model_per_stock(self) -> pd.DataFrame:
        """
        Determine which model performs best for each stock.
        
        Returns:
            DataFrame with best model per stock
        """
        symbols = self.predictions_df['symbol'].unique()
        results = []
        
        for symbol in symbols:
            metrics = self.get_metrics_summary(symbol=symbol)
            
            # Find model with lowest MAE
            best_model = None
            best_mae = float('inf')
            
            for model_name, model_metrics in metrics.items():
                if 'error' not in model_metrics:
                    if model_metrics['mae'] < best_mae:
                        best_mae = model_metrics['mae']
                        best_model = model_name
            
            if best_model:
                results.append({
                    'symbol': symbol,
                    'best_model': best_model,
                    'mae': best_mae,
                    'mape': metrics[best_model]['mape'],
                    'directional_accuracy': metrics[best_model]['directional_accuracy'],
                    'n_predictions': metrics[best_model]['n_predictions']
                })
        
        return pd.DataFrame(results).sort_values('mae')
    
    def get_prediction_count(self, symbol: Optional[str] = None) -> int:
        """Get total number of predictions stored."""
        if symbol:
            return len(self.predictions_df[self.predictions_df['symbol'] == symbol])
        return len(self.predictions_df)
    
    def get_completed_count(self, symbol: Optional[str] = None) -> int:
        """Get number of predictions with actual outcomes."""
        df = self.predictions_df[self.predictions_df['actual_updated'] == True]
        if symbol:
            df = df[df['symbol'] == symbol]
        return len(df)


if __name__ == "__main__":
    # Example usage
    print("Prediction Backtester - Example Usage\n")
    
    backtester = PredictionBacktester(storage_dir="prediction_history_test")
    
    # Simulate storing a prediction
    backtester.store_prediction(
        symbol='AAPL',
        date='2026-01-21',
        current_close=150.00,
        bayesian_return=0.015,
        gp_return=0.012,
        ensemble_return=0.013,
        bayesian_price=152.25,
        gp_price=151.80,
        ensemble_price=151.95,
        bayesian_ci=(0.010, 0.020),
        gp_ci=(0.008, 0.016),
        ensemble_ci=(0.009, 0.017),
        final_signal='BUY',
        confidence=0.75,
        rsi_value=58.5,
        bb_z_score=-0.5
    )
    
    # Simulate updating with actual outcome
    backtester.update_actual_close('AAPL', '2026-01-21', 151.50)
    
    # Get metrics
    backtester.print_metrics_report(symbol='AAPL')
    
    print("\n✅ Backtester initialized successfully!")
    print(f"   Predictions stored: {backtester.get_prediction_count()}")
    print(f"   Completed predictions: {backtester.get_completed_count()}")
