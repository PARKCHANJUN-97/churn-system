import numpy as np
import pandas as pd

NUM_FEATURES = [
    'Customer_Age', 'Total_Relationship_Count', 'Credit_Limit',
    'Total_Ct_Chng_Q4_Q1', 'Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1',
    'Avg_Utilization_Ratio', 'Avg_Trans_Amt',
    'Monthly_Trans_Amt', 'Monthly_Trans_Ct'
]
CAT_FEATURES = ['Gender']

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Avg_Trans_Amt'] = df['Total_Trans_Amt'] / df['Total_Trans_Ct'].replace(0, np.nan)
    df['Monthly_Trans_Amt'] = df['Total_Trans_Amt'] / df['Months_on_book'].replace(0, np.nan)
    df['Monthly_Trans_Ct'] = df['Total_Trans_Ct'] / df['Months_on_book'].replace(0, np.nan)
    df[['Avg_Trans_Amt','Monthly_Trans_Amt','Monthly_Trans_Ct']] = df[
        ['Avg_Trans_Amt','Monthly_Trans_Amt','Monthly_Trans_Ct']
    ].fillna(0)
    return df

def build_target(df: pd.DataFrame) -> pd.Series:
    return (df['Attrition_Flag'].astype(str) == 'Attrited Customer').astype(int)

def build_X(df: pd.DataFrame) -> pd.DataFrame:
    X_raw = df[NUM_FEATURES + CAT_FEATURES].copy()
    X = pd.get_dummies(X_raw, drop_first=True)
    return X

def build_single_input_row(user_input: dict) -> pd.DataFrame:
    return pd.DataFrame([user_input])
