"""
Evaluation Metrics for Algorithmic Trading Models
==================================================

This module provides comprehensive error metrics for evaluating price predictions
and trading model performance, including MAE, MSE, MAPE, directional accuracy,
and specialized trading metrics.

Key Metrics:
- Mean Absolute Error (MAE): Average absolute difference between predictions and actuals
- Mean Squared Error (MSE) / RMSE: Emphasizes larger errors more heavily
- Mean Absolute Percentage Error (MAPE): Error as percentage of actual values
- Directional Accuracy: Percentage of correct up/down movement predictions
- Relative MAE: MAE compared to a baseline (e.g., naive forecast)

Usage Example:
    from evaluation_metrics import calculate_mae, calculate_directional_accuracy
    
    actual_prices = [100, 102, 101, 105]
    predicted_prices = [101, 103, 100, 106]
    
    mae = calculate_mae(actual_prices, predicted_prices)
    direction_acc = calculate_directional_accuracy(actual_prices, predicted_prices)
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict


def calculate_mae(actual: Union[np.ndarray, list, pd.Series], 
                  predicted: Union[np.ndarray, list, pd.Series]) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE measures the average magnitude of errors in a set of predictions,
    without considering their direction. All individual differences have
    equal weight in the average.
    
    Formula: MAE = (1/n) * Σ|actual_i - predicted_i|
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        float: Mean Absolute Error
        
    Example:
        >>> actual = [100, 102, 101, 105]
        >>> predicted = [101, 103, 100, 106]
        >>> calculate_mae(actual, predicted)
        1.0
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
    
    if len(actual) == 0:
        return np.nan
    
    return float(np.mean(np.abs(actual - predicted)))


def calculate_mse(actual: Union[np.ndarray, list, pd.Series], 
                  predicted: Union[np.ndarray, list, pd.Series]) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    MSE measures the average of the squares of errors. It emphasizes
    larger errors more than MAE, making it useful when large errors
    are particularly undesirable.
    
    Formula: MSE = (1/n) * Σ(actual_i - predicted_i)²
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        float: Mean Squared Error
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
    
    if len(actual) == 0:
        return np.nan
    
    return float(np.mean((actual - predicted) ** 2))


def calculate_rmse(actual: Union[np.ndarray, list, pd.Series], 
                   predicted: Union[np.ndarray, list, pd.Series]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE is the square root of MSE, bringing the error metric back to
    the same units as the original data. It's more interpretable than MSE.
    
    Formula: RMSE = √(MSE) = √[(1/n) * Σ(actual_i - predicted_i)²]
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        float: Root Mean Squared Error
    """
    mse = calculate_mse(actual, predicted)
    return float(np.sqrt(mse)) if not np.isnan(mse) else np.nan


def calculate_mape(actual: Union[np.ndarray, list, pd.Series], 
                   predicted: Union[np.ndarray, list, pd.Series],
                   epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    MAPE expresses error as a percentage of actual values, making it
    scale-independent and easier to interpret across different price ranges.
    
    Formula: MAPE = (100/n) * Σ|actual_i - predicted_i| / |actual_i|
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        float: Mean Absolute Percentage Error (as percentage, e.g., 2.5 for 2.5%)
        
    Note:
        MAPE can be misleading when actual values are close to zero.
        Use epsilon to handle near-zero values.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
    
    if len(actual) == 0:
        return np.nan
    
    # Avoid division by zero
    actual_safe = np.where(np.abs(actual) < epsilon, epsilon, actual)
    
    return float(100 * np.mean(np.abs((actual - predicted) / actual_safe)))


def calculate_directional_accuracy(actual: Union[np.ndarray, list, pd.Series], 
                                   predicted: Union[np.ndarray, list, pd.Series]) -> float:
    """
    Calculate directional accuracy (percentage of correct up/down predictions).
    
    This metric measures how often the model correctly predicts the direction
    of price movement (up or down), which is crucial for trading decisions.
    
    Formula: DA = (Number of correct directions) / (Total predictions) * 100
    
    Args:
        actual: Array of actual values (or returns)
        predicted: Array of predicted values (or returns)
        
    Returns:
        float: Directional accuracy as percentage (0-100)
        
    Example:
        >>> actual = [1, -1, 1, -1, 1]  # Returns: up, down, up, down, up
        >>> predicted = [1, -1, -1, -1, 1]  # Predicted: up, down, down, down, up
        >>> calculate_directional_accuracy(actual, predicted)
        80.0  # 4 out of 5 correct
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
    
    if len(actual) == 0:
        return np.nan
    
    # Get direction (sign) of actual and predicted values
    actual_direction = np.sign(actual)
    predicted_direction = np.sign(predicted)
    
    # Count correct directions
    correct = (actual_direction == predicted_direction).sum()
    
    return float(100 * correct / len(actual))


def calculate_relative_mae(actual: Union[np.ndarray, list, pd.Series], 
                           predicted: Union[np.ndarray, list, pd.Series],
                           baseline_predicted: Union[np.ndarray, list, pd.Series] = None) -> float:
    """
    Calculate Relative MAE compared to a baseline prediction.
    
    Relative MAE shows improvement over a naive baseline (e.g., previous price).
    A value < 1.0 means the model beats the baseline; > 1.0 means it's worse.
    
    Formula: Relative MAE = MAE(model) / MAE(baseline)
    
    Args:
        actual: Array of actual values
        predicted: Array of model predictions
        baseline_predicted: Array of baseline predictions (if None, uses previous actual value)
        
    Returns:
        float: Relative MAE ratio
        
    Example:
        If baseline MAE is 2.0 and model MAE is 1.5, relative MAE = 0.75 (25% improvement)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if baseline_predicted is None:
        # Use previous value as naive baseline
        baseline_predicted = np.roll(actual, 1)
        baseline_predicted[0] = actual[0]  # First prediction = first actual
    else:
        baseline_predicted = np.array(baseline_predicted)
    
    model_mae = calculate_mae(actual, predicted)
    baseline_mae = calculate_mae(actual, baseline_predicted)
    
    if baseline_mae == 0 or np.isnan(baseline_mae):
        return np.nan
    
    return float(model_mae / baseline_mae)


def calculate_smape(actual: Union[np.ndarray, list, pd.Series], 
                    predicted: Union[np.ndarray, list, pd.Series]) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
    sMAPE is a variation of MAPE that is symmetric and bounded between 0% and 200%.
    It handles zero values better than MAPE.
    
    Formula: sMAPE = (100/n) * Σ(2 * |actual_i - predicted_i|) / (|actual_i| + |predicted_i|)
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        
    Returns:
        float: Symmetric MAPE as percentage
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
    
    if len(actual) == 0:
        return np.nan
    
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    return float(100 * np.mean(numerator / denominator))


def calculate_all_metrics(actual: Union[np.ndarray, list, pd.Series], 
                          predicted: Union[np.ndarray, list, pd.Series],
                          baseline_predicted: Union[np.ndarray, list, pd.Series] = None) -> Dict[str, float]:
    """
    Calculate all evaluation metrics at once.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
        baseline_predicted: Optional baseline predictions for relative MAE
        
    Returns:
        dict: Dictionary containing all metrics
        
    Example:
        >>> metrics = calculate_all_metrics([100, 102, 101], [101, 103, 100])
        >>> print(metrics['mae'], metrics['directional_accuracy'])
    """
    return {
        'mae': calculate_mae(actual, predicted),
        'mse': calculate_mse(actual, predicted),
        'rmse': calculate_rmse(actual, predicted),
        'mape': calculate_mape(actual, predicted),
        'smape': calculate_smape(actual, predicted),
        'directional_accuracy': calculate_directional_accuracy(actual, predicted),
        'relative_mae': calculate_relative_mae(actual, predicted, baseline_predicted)
    }


def print_metrics_summary(metrics: Dict[str, float], title: str = "Model Performance Metrics") -> None:
    """
    Print a formatted summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from calculate_all_metrics()
        title: Title for the summary
    """
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  MAE (Mean Absolute Error):        ${metrics['mae']:.4f}")
    print(f"  RMSE (Root Mean Squared Error):   ${metrics['rmse']:.4f}")
    print(f"  MAPE (Mean Abs % Error):          {metrics['mape']:.2f}%")
    print(f"  sMAPE (Symmetric MAPE):           {metrics['smape']:.2f}%")
    print(f"  Directional Accuracy:             {metrics['directional_accuracy']:.1f}%")
    
    if not np.isnan(metrics['relative_mae']):
        improvement = (1 - metrics['relative_mae']) * 100
        print(f"  Relative MAE vs Baseline:         {metrics['relative_mae']:.3f} ({improvement:+.1f}%)")
    
    print("=" * 70)


def evaluate_price_predictions(actual_prices: pd.Series,
                               predicted_prices: pd.Series,
                               dates: pd.Series = None,
                               verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate price predictions with detailed analysis.
    
    Args:
        actual_prices: Series of actual closing prices
        predicted_prices: Series of predicted closing prices
        dates: Optional dates for the predictions
        verbose: If True, print detailed summary
        
    Returns:
        dict: Comprehensive metrics dictionary
    """
    metrics = calculate_all_metrics(actual_prices, predicted_prices)
    
    if verbose:
        print_metrics_summary(metrics, "Price Prediction Evaluation")
        
        if dates is not None:
            print(f"\n  Evaluation Period: {dates.iloc[0]} to {dates.iloc[-1]}")
            print(f"  Number of Predictions: {len(actual_prices)}")
    
    return metrics


def evaluate_return_predictions(actual_returns: pd.Series,
                                predicted_returns: pd.Series,
                                verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate return predictions (optimized for trading).
    
    Returns are more relevant for trading than absolute prices.
    Directional accuracy is especially important.
    
    Args:
        actual_returns: Series of actual returns (e.g., daily % changes)
        predicted_returns: Series of predicted returns
        verbose: If True, print detailed summary
        
    Returns:
        dict: Comprehensive metrics dictionary
    """
    metrics = calculate_all_metrics(actual_returns, predicted_returns)
    
    if verbose:
        print_metrics_summary(metrics, "Return Prediction Evaluation")
        
        # Additional trading-specific insights
        correct_directions = (np.sign(actual_returns) == np.sign(predicted_returns)).sum()
        total = len(actual_returns)
        
        print(f"\n  Trading Insights:")
        print(f"    Correct Directions: {correct_directions}/{total}")
        print(f"    Signal Reliability: {metrics['directional_accuracy']:.1f}%")
        
        if metrics['directional_accuracy'] >= 60:
            print(f"    ✅ STRONG: Model shows profitable signal quality")
        elif metrics['directional_accuracy'] >= 52:
            print(f"    ✅ GOOD: Model has edge over random (50%)")
        else:
            print(f"    ⚠️  WARNING: Model below 52% directional accuracy")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("Evaluation Metrics Module - Example Usage\n")
    
    # Simulate some predictions
    np.random.seed(42)
    actual = np.array([100, 102, 101, 105, 107, 106, 110, 112])
    predicted = actual + np.random.normal(0, 1.5, len(actual))
    
    print("Price Prediction Example:")
    print(f"Actual:    {actual}")
    print(f"Predicted: {predicted.round(2)}\n")
    
    # Calculate all metrics
    metrics = calculate_all_metrics(actual, predicted)
    print_metrics_summary(metrics)
    
    # Returns example
    print("\n\nReturn Prediction Example:")
    actual_returns = np.array([0.02, -0.01, 0.04, 0.02, -0.01, 0.04, 0.02])
    predicted_returns = np.array([0.025, -0.015, 0.035, 0.025, 0.005, 0.045, 0.015])
    
    metrics_returns = evaluate_return_predictions(
        pd.Series(actual_returns), 
        pd.Series(predicted_returns),
        verbose=True
    )
