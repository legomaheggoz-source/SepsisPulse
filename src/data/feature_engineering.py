"""
Feature Engineering Module for SepsisPulse.

This module creates engineered features from raw time series data for the XGBoost-TS model.
It generates approximately 200 features from 41 raw variables through lag features,
rolling statistics, and rate of change calculations.
"""

from typing import List, Optional

import pandas as pd


def create_lag_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 3, 6],
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create lagged versions of each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data.
    lags : List[int], optional
        List of lag periods to create. Default is [1, 3, 6].
    columns : List[str], optional
        Specific columns to create lag features for. If None, uses all numeric columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus lagged features.
        New columns are named like {column}_lag_{lag} (e.g., HR_lag_1, HR_lag_3).

    Notes
    -----
    Edge cases at the beginning of time series are handled by pandas shift(),
    which fills with NaN values for unavailable lags.

    Examples
    --------
    >>> df = pd.DataFrame({'HR': [80, 82, 85, 88], 'BP': [120, 118, 122, 125]})
    >>> result = create_lag_features(df, lags=[1, 2])
    >>> result.columns.tolist()
    ['HR', 'BP', 'HR_lag_1', 'HR_lag_2', 'BP_lag_1', 'BP_lag_2']
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            result[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return result


def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [3, 6, 12],
    columns: Optional[List[str]] = None,
    statistics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create rolling window statistics for each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data.
    windows : List[int], optional
        List of window sizes for rolling calculations. Default is [3, 6, 12].
    columns : List[str], optional
        Specific columns to create rolling features for. If None, uses all numeric columns.
    statistics : List[str], optional
        Statistics to compute. Options: 'mean', 'std', 'min', 'max'.
        Default is ['mean', 'std', 'min', 'max'].

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus rolling features.
        New columns are named like {column}_roll_{stat}_{window}
        (e.g., HR_roll_mean_3, HR_roll_std_6).

    Notes
    -----
    - Rolling windows use min_periods=1 to handle edge cases at the beginning
      of time series, allowing calculations even with fewer observations than
      the window size.
    - Standard deviation for windows with fewer than 2 observations will be NaN.

    Examples
    --------
    >>> df = pd.DataFrame({'HR': [80, 82, 85, 88, 90]})
    >>> result = create_rolling_features(df, windows=[3], statistics=['mean', 'std'])
    >>> result.columns.tolist()
    ['HR', 'HR_roll_mean_3', 'HR_roll_std_3']
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    if statistics is None:
        statistics = ['mean', 'std', 'min', 'max']

    stat_functions = {
        'mean': lambda x: x.mean(),
        'std': lambda x: x.std(),
        'min': lambda x: x.min(),
        'max': lambda x: x.max(),
    }

    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            rolling = df[col].rolling(window=window, min_periods=1)
            for stat in statistics:
                if stat in stat_functions:
                    result[f"{col}_roll_{stat}_{window}"] = rolling.agg(stat_functions[stat])

    return result


def create_rate_of_change(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    periods: int = 1,
) -> pd.DataFrame:
    """
    Create first difference (delta) features representing rate of change.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data.
    columns : List[str], optional
        Specific columns to create delta features for. If None, uses all numeric columns.
    periods : int, optional
        Number of periods for the difference calculation. Default is 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus delta features.
        New columns are named like {column}_delta (e.g., HR_delta).

    Notes
    -----
    Edge cases at the beginning of time series are handled by pandas diff(),
    which fills the first row(s) with NaN.

    Examples
    --------
    >>> df = pd.DataFrame({'HR': [80, 82, 85, 88]})
    >>> result = create_rate_of_change(df)
    >>> result['HR_delta'].tolist()
    [nan, 2.0, 3.0, 3.0]
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        result[f"{col}_delta"] = df[col].diff(periods=periods)

    return result


def create_all_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 3, 6],
    windows: List[int] = [3, 6, 12],
    columns: Optional[List[str]] = None,
    include_lag: bool = True,
    include_rolling: bool = True,
    include_delta: bool = True,
) -> pd.DataFrame:
    """
    Create all engineered features by combining lag, rolling, and rate of change features.

    This function generates approximately 200 features from 41 raw variables by applying:
    - Lag features for each specified lag period
    - Rolling statistics (mean, std, min, max) for each window size
    - First difference (delta) features

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data.
    lags : List[int], optional
        List of lag periods. Default is [1, 3, 6].
    windows : List[int], optional
        List of window sizes for rolling calculations. Default is [3, 6, 12].
    columns : List[str], optional
        Specific columns to create features for. If None, uses all numeric columns.
    include_lag : bool, optional
        Whether to include lag features. Default is True.
    include_rolling : bool, optional
        Whether to include rolling features. Default is True.
    include_delta : bool, optional
        Whether to include delta features. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus all engineered features.

    Notes
    -----
    Feature count estimation for N numeric columns:
    - Original features: N
    - Lag features: N * len(lags) = N * 3 = 3N (default)
    - Rolling features: N * len(windows) * 4 = N * 3 * 4 = 12N (default)
    - Delta features: N * 1 = N

    Total: N + 3N + 12N + N = 17N features
    For 41 raw variables: 17 * 41 = 697 features (theoretical maximum)

    Edge cases at the beginning of time series are handled appropriately:
    - Lag and delta features will have NaN values for early rows
    - Rolling features use min_periods=1 for graceful degradation

    Examples
    --------
    >>> df = pd.DataFrame({'HR': [80, 82, 85, 88, 90], 'BP': [120, 118, 122, 125, 128]})
    >>> result = create_all_features(df, lags=[1], windows=[3])
    >>> len(result.columns)
    12
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    result = df.copy()

    if include_lag:
        lag_features = create_lag_features(df, lags=lags, columns=columns)
        # Get only the new lag columns (exclude original columns)
        lag_cols = [c for c in lag_features.columns if '_lag_' in c]
        result = pd.concat([result, lag_features[lag_cols]], axis=1)

    if include_rolling:
        rolling_features = create_rolling_features(df, windows=windows, columns=columns)
        # Get only the new rolling columns (exclude original columns)
        rolling_cols = [c for c in rolling_features.columns if '_roll_' in c]
        result = pd.concat([result, rolling_features[rolling_cols]], axis=1)

    if include_delta:
        delta_features = create_rate_of_change(df, columns=columns)
        # Get only the new delta columns (exclude original columns)
        delta_cols = [c for c in delta_features.columns if '_delta' in c]
        result = pd.concat([result, delta_features[delta_cols]], axis=1)

    return result


def get_feature_names(
    columns: List[str],
    lags: List[int] = [1, 3, 6],
    windows: List[int] = [3, 6, 12],
    statistics: List[str] = ['mean', 'std', 'min', 'max'],
) -> List[str]:
    """
    Get the list of feature names that would be generated for given columns.

    Parameters
    ----------
    columns : List[str]
        List of original column names.
    lags : List[int], optional
        List of lag periods. Default is [1, 3, 6].
    windows : List[int], optional
        List of window sizes. Default is [3, 6, 12].
    statistics : List[str], optional
        List of rolling statistics. Default is ['mean', 'std', 'min', 'max'].

    Returns
    -------
    List[str]
        List of all feature names including original and engineered features.

    Examples
    --------
    >>> names = get_feature_names(['HR'], lags=[1], windows=[3], statistics=['mean'])
    >>> names
    ['HR', 'HR_lag_1', 'HR_roll_mean_3', 'HR_delta']
    """
    feature_names = list(columns)

    # Lag feature names
    for col in columns:
        for lag in lags:
            feature_names.append(f"{col}_lag_{lag}")

    # Rolling feature names
    for col in columns:
        for window in windows:
            for stat in statistics:
                feature_names.append(f"{col}_roll_{stat}_{window}")

    # Delta feature names
    for col in columns:
        feature_names.append(f"{col}_delta")

    return feature_names
