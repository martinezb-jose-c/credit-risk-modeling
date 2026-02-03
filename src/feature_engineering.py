"""
Feature engineering functions for credit risk modeling.
"""
import pandas as pd
import numpy as np


def create_fico_score(df: pd.DataFrame,
                      low_col: str = 'fico_range_low',
                      high_col: str = 'fico_range_high') -> pd.Series:
    """
    Calculate average FICO score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with FICO columns
    low_col : str
        Name of the column with low FICO range
    high_col : str
        Name of the column with high FICO range

    Returns
    -------
    pd.Series
        Average FICO score
    """
    return (df[low_col] + df[high_col]) / 2


def create_fico_band(fico_score: pd.Series) -> pd.Series:
    """
    Create FICO bands according to regulatory standards.

    Parameters
    ----------
    fico_score : pd.Series
        Continuous FICO score

    Returns
    -------
    pd.Series
        Categorical FICO bands
    """
    bins = [0, 579, 669, 739, 799, 850]
    labels = ['Very Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
    return pd.cut(fico_score, bins=bins, labels=labels, include_lowest=True)


def log_transform(series: pd.Series, shift: float = 1.0) -> pd.Series:
    """
    Apply logarithmic transformation (log1p by default).

    Parameters
    ----------
    series : pd.Series
        Series to transform
    shift : float
        Value to add before log (to avoid log(0))

    Returns
    -------
    pd.Series
        Transformed series
    """
    return np.log1p(series) if shift == 1.0 else np.log(series + shift)


def calculate_credit_history_years(df: pd.DataFrame,
                                   date_col: str = 'earliest_cr_line',
                                   reference_col: str = 'issue_d') -> pd.Series:
    """
    Calculate years of credit history.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date columns
    date_col : str
        Name of the column with first credit line date
    reference_col : str
        Name of the column with reference date (loan issue date)

    Returns
    -------
    pd.Series
        Years of credit history
    """
    earliest_date = pd.to_datetime(df[date_col], format='%b-%Y', errors='coerce')
    reference_date = pd.to_datetime(df[reference_col], format='%b-%Y', errors='coerce')

    return (reference_date - earliest_date).dt.days / 365.25


def create_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary risk indicator variables.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with new binary columns
    """
    df = df.copy()
    df['has_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)
    df['has_public_records'] = (df['pub_rec'] > 0).astype(int)
    df['has_bankruptcy'] = (df['pub_rec_bankruptcies'] > 0).astype(int)
    return df


def create_vintage_features(df: pd.DataFrame,
                            date_col: str = 'issue_d') -> pd.DataFrame:
    """
    Create vintage (cohort) features for portfolio analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with issue date column
    date_col : str
        Name of the column with loan issue date

    Returns
    -------
    pd.DataFrame
        DataFrame with vintage features
    """
    df = df.copy()
    issue_date = pd.to_datetime(df[date_col], format='%b-%Y', errors='coerce')

    df['vintage_year'] = issue_date.dt.year
    df['vintage_quarter'] = issue_date.dt.to_period('Q').astype(str)
    df['vintage_month'] = issue_date.dt.to_period('M').astype(str)

    return df


def create_utilization_ratio(df: pd.DataFrame,
                             balance_col: str = 'revol_bal',
                             limit_col: str = 'revol_util') -> pd.Series:
    """
    Calculate credit utilization ratio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with balance and limit columns
    balance_col : str
        Name of the column with revolving balance
    limit_col : str
        Name of the column with revolving utilization

    Returns
    -------
    pd.Series
        Credit utilization ratio
    """
    return df[limit_col] / 100  # revol_util is already a percentage


def create_dti_adjusted(df: pd.DataFrame,
                        dti_col: str = 'dti',
                        installment_col: str = 'installment',
                        income_col: str = 'annual_inc') -> pd.Series:
    """
    Calculate adjusted DTI including the new loan installment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DTI, installment, and income columns
    dti_col : str
        Name of the DTI column
    installment_col : str
        Name of the installment column
    income_col : str
        Name of the annual income column

    Returns
    -------
    pd.Series
        Adjusted DTI ratio
    """
    monthly_income = df[income_col] / 12
    current_debt = (df[dti_col] / 100) * monthly_income
    new_debt = current_debt + df[installment_col]

    return (new_debt / monthly_income) * 100



def calculate_iv(df, feature, target):
    """
    Information Value for feature selection in credit risk modeling

    IV < 0.02: Not predictive
    IV 0.02-0.1: Weak predictor
    IV 0.1-0.3: Medium predictor
    IV 0.3-0.5: Strong predictor
    IV > 0.5: Suspicious (possible data leakage)

    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    feature : str
        Feature name to calculate IV
    target : str
        Target variable name (binary: 0/1)

    Returns:
    --------
    float : Information Value
    """
    # Create a copy to avoid modifying original data
    df_iv = df[[feature, target]].copy()

    # Handle numeric features: bin them into groups
    if df_iv[feature].dtype in ['float64', 'int64']:
        # Create bins for continuous variables
        df_iv[feature] = pd.qcut(df_iv[feature], q=10, duplicates='drop')

    # Group by feature values
    grouped = df_iv.groupby(feature)[target].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Bad']
    grouped['Good'] = grouped['Total'] - grouped['Bad']

    # Calculate totals
    total_good = grouped['Good'].sum()
    total_bad = grouped['Bad'].sum()

    # Calculate distribution percentages
    grouped['Dist_Good'] = grouped['Good'] / total_good
    grouped['Dist_Bad'] = grouped['Bad'] / total_bad

    # Avoid division by zero
    grouped['Dist_Good'] = grouped['Dist_Good'].replace(0, 0.0001)
    grouped['Dist_Bad'] = grouped['Dist_Bad'].replace(0, 0.0001)

    # Calculate WoE (Weight of Evidence)
    grouped['WoE'] = np.log(grouped['Dist_Good'] / grouped['Dist_Bad'])

    # Calculate IV for each bin
    grouped['IV'] = (grouped['Dist_Good'] - grouped['Dist_Bad']) * grouped['WoE']

    # Sum IV across all bins
    iv = grouped['IV'].sum()

    return iv
